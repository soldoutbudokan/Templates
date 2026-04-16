# The TILT

**How much does a player tilt the game?**

Live site: [thetilt-rust.vercel.app](https://thetilt-rust.vercel.app)

TILT measures **Win Probability Added (WPA)** per ball in IPL cricket. For every single delivery bowled in the IPL's history, we calculate how much the win probability shifted and attribute that shift to the batter and bowler involved. Aggregate across all matches and you get each player's impact rating — their TILT.

---

## Quick Start

```bash
cd TheTilt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (downloads data, trains model, exports rankings)
python pipeline/run_pipeline.py

# Start the local website
cd api && flask --app index.py run --port 5002
```

Open [http://localhost:5002](http://localhost:5002)

---

## The Idea

Traditional cricket stats like batting average and economy rate tell you *what* a player did, but not *when* they did it. A batter who scores 30 off 20 balls when chasing 15 off the last over is very different from one who scores 30 off 20 when the team is cruising at 50/0 in the powerplay.

**TILT captures context.** It answers: *did this player make their team more or less likely to win, and by how much?*

A six off the last ball to win the match? Massive positive TILT. A golden duck in the first over of a 200-run chase? Small negative TILT (there were still 119 balls left). A wicket in the death overs when the opposition needed 30 off 12? Huge positive TILT for the bowler.

The stat is denominated in **win probability percentage points per match**. A TILT of +5.00% means that player adds 5 percentage points of win probability to their team every match they play, on average.

---

## How It Works: The Full Pipeline

### Step 1: Data (Cricsheet)

All data comes from [Cricsheet](https://cricsheet.org), an open cricket data project that provides ball-by-ball JSON files for every IPL match ever played.

- **1,183 matches** parsed (2008-2026)
- **280,000+ individual deliveries** with batter, bowler, runs, wickets, and match outcome
- Each delivery becomes a row with full match context

### Step 2: Feature Engineering

For each ball, we compute the **match state before the delivery is bowled**:

| Feature | Description |
|---------|-------------|
| `innings` | 1st or 2nd innings |
| `balls_remaining` | Legal deliveries left in the innings (max 120) |
| `wickets_in_hand` | Wickets remaining (0-10) |
| `runs_scored` | Runs scored so far this innings |
| `run_rate` | Current scoring rate (runs per over) |
| `target` | Chase target (innings 2 only, 0 for innings 1) |
| `runs_needed` | Runs still required (innings 2 only) |
| `required_run_rate` | Needed scoring rate to win (innings 2 only) |
| `is_powerplay` | Overs 1-6 (fielding restrictions) |
| `is_middle` | Overs 7-15 |
| `is_death` | Overs 16-20 (final phase) |
| `recent_run_rate` | Scoring rate over the last 3 overs |
| `recent_wickets` | Wickets fallen in the last 3 overs |
| `venue` | Match venue (categorical — LightGBM native handling) |
| `batting_team_chose_to_bat` | 1 if batting team won toss and chose to bat |
| `season_numeric` | IPL season year (2008-2026) for era adjustment |
| `opponent_bowler_economy` | Career bowling economy of the bowler (opponent quality proxy) |

**Key design decision:** We use a single model for both innings. For innings 1, the target/required run rate features are set to 0. This lets the model learn that innings 1 is about *setting a total* while innings 2 is about *chasing*. The `innings` feature acts as a switch.

### Step 3: Win Probability Model

We train a **LightGBM gradient-boosted classifier** to predict: *will the team currently batting win this match?*

```
Input:  17 match-state features (before each ball)
Output: P(batting team wins) ∈ [0, 1]
```

**Why LightGBM?**
- Handles mixed feature types (categorical + continuous) natively
- Fast to train on 276K rows
- Excellent probability calibration out of the box
- Easy to inspect feature importances

**Training details:**
- Train/test split: **80/20 by match** (not by ball — splitting by ball would leak information since balls within the same match are correlated)
- 2000 trees (with early stopping), learning rate 0.03, max depth 4, num_leaves 16
- Strong L2 regularization (reg_lambda=5.0, min_child_samples=500) to prevent sharp split boundary cliffs that cause unrealistic per-ball volatility
- Venue names deduplicated (59 raw → 38 canonical) to reduce categorical overfitting
- DLS-affected matches excluded from training (22 matches)
- No post-hoc calibration needed — the regularized model is well-calibrated out of the box (max calibration error ~4.5%)

**Model performance:**

| Metric | Value |
|--------|-------|
| Brier Score | 0.198 (lower is better, perfect = 0) |
| AUC | 0.756 |

**Feature importances** (what matters most for predicting the winner):

```
venue                  ████████████████████████████████  (1022)
required_run_rate      ████████████                     ( 422)
wickets_in_hand        ████████████                     ( 413)
target                 ███████████                      ( 402)
runs_needed            █████████                        ( 337)
run_rate               ████████                         ( 305)
runs_scored            ███████                          ( 259)
season_numeric         █████                            ( 179)
innings                █████                            ( 170)
opponent_bowler_econ   ██                               (  94)
```

Venue is the most important feature — confirming that ground conditions significantly affect win probability. Season captures era effects.

### Step 4: Computing TILT

For every ball in the dataset:

```
win_prob_before = model.predict(match_state_before_ball)
win_prob_after  = model.predict(match_state_after_ball)
delta_wp        = win_prob_after - win_prob_before
```

**Attribution:**
- **Batter credit:** `+delta_wp` (positive means the batter helped their batting team)
- **Bowler credit:** `-delta_wp` (positive means the bowler helped their bowling team)

*Example:* A six in the death overs that takes the batting team's win probability from 40% to 55% produces a `delta_wp` of +0.15. The batter gets +15% credited, the bowler gets -15%.

**Aggregation:**

For each player across their career:
- `batting_tilt_per_match` = total batting delta / matches batted
- `bowling_tilt_per_match` = total bowling delta / matches bowled
- `total_tilt_per_match` = batting + bowling combined

**Bayesian Shrinkage:**

Raw per-match averages are noisy for players with few matches. We apply empirical Bayes shrinkage:

```
shrunk_tilt = (n / (n + k)) * raw_tilt + (k / (n + k)) * population_mean
```

Where `k` is estimated from the ratio of within-player to between-player variance (k ≈ 5.3 for IPL data). This means:
- A player with 5 matches: 50% raw, 50% population mean
- A player with 50 matches: 90% raw, 10% population mean
- A player with 188 matches (Narine): 97% raw, 3% population mean

**Bayesian Posterior Confidence Intervals:**

Rankings use the 90% CI lower bound ("TILT floor") as the default sort. The CI uses Bayesian posterior variance (`var / (n + k)`) rather than the frequentist standard error (`var / n`), which is theoretically consistent with the shrinkage model — shrinkage itself reduces uncertainty.

### Step 5: Export & Website

Rankings and per-player breakdowns are exported as static JSON files and served on a Vercel-hosted website with sortable tables, search, player detail pages, match replays with win probability charts, and GOAT performance rankings.

---

## Results & Findings

Rankings use **Bayesian shrinkage** (empirical Bayes) to stabilize small-sample estimates. Players with few matches are pulled toward the population mean; high-match players keep their raw values. Confidence levels: green (100+ matches), yellow (30-100), gray (<30).

### Top Overall Players (ranked by TILT Floor — 90% CI lower bound)

| Rank | Player | TILT/Match | Raw | Confidence | Matches |
|------|--------|------------|-----|------------|---------|
| 1 | Sunil Narine | +5.59% | +5.79% | **high** | 188 |
| 2 | Rashid Khan | +5.85% | +6.14% | **high** | 136 |
| 3 | Philip Salt | +6.57% | +7.76% | medium | 36 |
| 4 | Lasith Malinga | +4.46% | +4.73% | **high** | 120 |
| 5 | Yashasvi Jaiswal | +5.20% | +5.73% | medium | 68 |
| 6 | Priyansh Arya | +9.77% | +13.04% | low | 18 |
| 7 | Ruturaj Gaikwad | +4.52% | +4.97% | medium | 73 |
| 8 | AB de Villiers | +3.98% | +4.16% | **high** | 166 |
| 9 | KL Rahul | +3.84% | +4.06% | **high** | 133 |
| 10 | Yuzvendra Chahal | +3.30% | +3.46% | **high** | 171 |

### Notable Observations

- **Sunil Narine** (#1) tops the floor ranking across 188 matches — the most consistent all-round impact player in IPL history
- **Rashid Khan** (#2) is elite in both batting and bowling TILT
- **Malinga** (#4) with high confidence at 120 matches — pure bowling impact
- The floor ranking naturally rewards consistency: Priyansh Arya has the highest raw TILT but ranks #6 because his 18-match sample produces a wider confidence interval
- **Venue matters most**: The venue feature has the highest importance, confirming that ground conditions significantly affect match outcomes
- **Era adjustment works**: Old-era players are not disproportionately penalized — the season_numeric feature captures evolving T20 scoring rates

---

## Architecture

```
TheTilt/
├── pipeline/                  # Offline Python pipeline
│   ├── download_data.py       # Cricsheet IPL ZIP download
│   ├── download_people.py     # Player registry + Wikidata full name resolution
│   ├── parse_matches.py       # JSON → DataFrame (BallEvent dataclass)
│   ├── build_features.py      # Match state features per delivery
│   ├── train_win_prob.py      # LightGBM win probability model
│   ├── compute_tilt.py        # Win prob deltas → per-player TILT
│   ├── export_json.py         # Static JSON for website (rankings, players, matches, GOATs)
│   ├── scrape_player_meta.py  # ESPNcricinfo batting hand / bowling type scraper
│   └── run_pipeline.py        # End-to-end orchestrator
├── api/
│   └── index.py               # Flask app for Vercel (serves static files)
├── public/
│   ├── index.html             # Leaderboard (sortable, searchable)
│   ├── player.html            # Player detail page (stats, charts, career trend)
│   ├── match.html             # Match replay (win prob chart, key moments, scorecards)
│   ├── goats.html             # GOAT performances (top match/season rankings)
│   ├── about.html             # Methodology page
│   ├── style.css              # Dark/light monospace theme
│   └── data/                  # Pre-computed static JSON
│       ├── tilt_rankings.json # All qualified players
│       ├── players/*.json     # Per-player detail files
│       ├── matches/*.json     # Per-match ball-by-ball data
│       ├── goats.json         # Top match/season performances
│       └── meta.json          # Data coverage metadata
├── .github/workflows/
│   └── refresh-tilt-data.yml  # Weekly automated data refresh
├── config/
│   └── pipeline_config.yaml   # Model hyperparameters, paths
├── data/                      # Raw + processed (gitignored)
├── models/                    # Saved model (gitignored)
└── vercel.json                # Deployment config
```

**Data flow:**
1. Pipeline runs locally (~2.5 minutes), downloads from Cricsheet, processes everything
2. Outputs static JSON to `public/data/`
3. Website serves JSON directly — no computation at request time
4. Deployed to Vercel (Python Flask for routing + static file serving)
5. GitHub Actions runs the pipeline weekly and auto-commits updated data

---

## Tech Stack

- **Data Pipeline:** Python 3, pandas, LightGBM, pyarrow, scikit-learn
- **Website:** Vanilla HTML/CSS/JS + Chart.js for interactive visualizations
- **API:** Flask (minimal — mostly just routes to static files)
- **Hosting:** Vercel (serverless Python + static files)
- **Aesthetic:** Dark monospace terminal style, matching [PlainCricket](https://plaincricket.vercel.app)

---

## Limitations & Future Work

### Current Limitations

1. **Fielding is invisible** — The model can't attribute catches, run-outs, or misfields to specific fielders.
2. **No batting position context** — An opener facing the new ball operates in a different context than a #6 batter.
3. **No bowler type classification** — The model doesn't know if a bowler is pace or spin. PaceOrSpin × Venue interactions could improve predictions. This requires external data or manual classification.
4. **Opponent quality is approximate** — We use career bowling economy as a proxy, but a full iterative system (using TILT-derived quality) would be more accurate.
5. **Second innings TILT asymmetry** — Win probability swings are inherently larger in the 2nd innings because the target is known and the match resolves ball by ball. On average, 2nd innings balls produce **1.54x** larger |delta_wp| than 1st innings balls. The effect is concentrated in death overs (2.5x) while powerplay overs are nearly equal (1.07x). This inflates single-match TILT for 2nd innings performances — 94% of the top-50 batting and 98% of the top-50 bowling GOAT performances come from the 2nd innings. The GOAT page now provides innings-filtered views to enable fair within-innings comparisons. Career-level rankings are largely unaffected (Spearman ρ = 0.99 between raw and innings-normalized career TILT). See `notebooks/innings_bias_analysis.py` for the full diagnostic.

### What's Been Addressed

- ~~Small sample bias~~ → **Bayesian shrinkage** with empirical Bayes (k ≈ 5.3) + posterior CIs
- ~~Single model for all eras~~ → **Season year** as a continuous feature
- ~~No venue context~~ → **Venue** as categorical feature (deduplicated, highest importance)
- ~~No toss context~~ → **Toss** parsed and used as `batting_team_chose_to_bat`
- ~~No opponent quality~~ → **Bowling economy** proxy added as feature
- ~~DLS matches~~ → Excluded from training (22 matches)
- ~~Model too volatile~~ → Strong L2 regularization, reduced depth, removed Platt calibration
- ~~No GOAT rankings~~ → Top match and season performances page
- ~~Manual data refresh~~ → GitHub Actions weekly cron job
- ~~Run-out attribution~~ → Key moments now show the correct dismissed player
- ~~Win prob chart flipped~~ → Consistent perspective from innings 1 batting team
- ~~GOAT page innings bias~~ → **Innings-filtered views** on match batting/bowling tabs with explanatory note

### Future Work

- **Batting hand / bowling type display** — scraper written (`scrape_player_meta.py`), 80/780 players scraped from ESPNcricinfo before hitting Akamai rate limits. The scraper is resumable — run `python pipeline/scrape_player_meta.py` after the rate limit clears (typically a few hours). Once `data/processed/player_meta.json` is complete, the data needs to be wired into the player JSON export and displayed on player pages.
- **Bowler type as model feature** — PaceOrSpin x Venue interactions (pending bowler type data from above)
- Impact sub analysis (data parsed, analysis pending)

---

## Data Source

All ball-by-ball data from [Cricsheet](https://cricsheet.org) under their open data license. Cricsheet is maintained by [Stephen Rushe](https://twitter.com/srushe) and is the gold standard for open cricket data.

---

## Running the Pipeline

### Full pipeline (download + train + export):
```bash
python pipeline/run_pipeline.py
```

### Individual steps:
```bash
python -m pipeline.download_data    # Download Cricsheet data
python -m pipeline.parse_matches    # Parse to DataFrame
python -m pipeline.build_features   # Add match state features
python -m pipeline.train_win_prob   # Train model
python -m pipeline.compute_tilt     # Compute player TILT
python -m pipeline.export_json      # Export website JSON
```

### Re-running with new data:
```bash
python pipeline/run_pipeline.py     # Will skip download if data exists
# To force re-download:
# python -c "from pipeline.download_data import download_cricsheet_data; download_cricsheet_data(force=True)"
```
