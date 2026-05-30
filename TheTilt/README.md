# The TILT

**How much does a player tilt the game?**

Live site: [thetilt-rust.vercel.app](https://thetilt-rust.vercel.app)

TILT measures **Win Probability Added (WPA)** per ball in IPL cricket. For every single delivery bowled in the IPL's history, we calculate how much the win probability shifted and attribute that shift to the batsman and bowler involved. Aggregate across all matches and you get each player's impact rating — their TILT.

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

Traditional cricket stats like batting average and economy rate tell you *what* a player did, but not *when* they did it. A batsman who scores 30 off 20 balls when chasing 15 off the last over is very different from one who scores 30 off 20 when the team is cruising at 50/0 in the powerplay.

**TILT captures context.** It answers: *did this player make their team more or less likely to win, and by how much?*

A six off the last ball to win the match? Massive positive TILT. A golden duck in the first over of a 200-run chase? Small negative TILT (there were still 119 balls left). A wicket in the death overs when the opposition needed 30 off 12? Huge positive TILT for the bowler.

The stat is denominated in **win probability percentage points per match**. A TILT of +5.00% means that player adds 5 percentage points of win probability to their team every match they play, on average.

---

## How It Works: The Full Pipeline

### Step 1: Data (Cricsheet)

All data comes from [Cricsheet](https://cricsheet.org), an open cricket data project that provides ball-by-ball JSON files for every IPL match ever played.

- **1,232 matches** parsed (2008-2026)
- **294,255 individual deliveries** with batsman, bowler, runs, wickets, and match outcome
- Each delivery becomes a row with full match context
- Team-name aliases normalized (e.g. "Royal Challengers Bengaluru" → "Royal Challengers Bangalore")

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
| `over` | Over number 0-19 (continuous — lets LightGBM learn phase effects more granularly than dummies) |
| `recent_wickets` | Wickets fallen in the last 18 balls (3 overs) |
| `venue` | Match venue (categorical — LightGBM native handling) |
| `batting_team_chose_to_bat` | 1 if batting team won toss and chose to bat |
| `season_numeric` | IPL season year (2008-2026) for era adjustment |
| `opponent_bowler_economy` | Career bowling economy of the bowler (opponent quality proxy) |
| `batting_team_nrr` | Season-to-date net run rate of the batting team (prior matches only — no leakage) |

**Key design decision:** We use a single model for both innings. For innings 1, the target/required run rate features are set to 0. This lets the model learn that innings 1 is about *setting a total* while innings 2 is about *chasing*. The `innings` feature acts as a switch.

### Step 3: Win Probability Model

We train a **K=100 LightGBM ensemble** to predict: *will the team currently batting win this match?*

```
Input:  15 match-state features (before each ball)
Output: averaged P(batting team wins) ∈ [0, 1]   ← mean of 100 LGBM members
```

**Why LightGBM?**
- Handles mixed feature types (categorical + continuous) natively
- Fast to train on 256K rows
- Excellent probability calibration out of the box
- Easy to inspect feature importances

**Why K=100 ensemble?** A single LightGBM run on this dataset is fitted enough that adding one or two new matches to Cricsheet can flip the career top-10 (issue #111: AB de Villiers fell from #3 to #9 across a single retrain on a +2-match delta). Train/holdout reshuffles, the boosting trajectory diverges, and individual ball predictions can swing by **5+ percentage points** for the same input. K=100 members trained on the same fixed split with random states 42…141 average that variance away. Brier and AUC change by less than 0.01 against a single member; what changes is *stability across retrains*.

**Training details:**
- Train/test split: **90/10 by match**, locked at `random_state=42` forever — every retrain validates on the same 119 matches so Brier/AUC numbers are directly comparable
- 100 ensemble members, each: 2000 trees (with early stopping), learning rate 0.03, max depth 4, num_leaves 16
- Heavy L2 regularization (reg_lambda=5.0, min_child_samples=500) — keeps individual trees smooth so member-to-member disagreement is genuine trajectory variance, not leaf noise
- Venue names deduplicated (59 raw → 38 canonical) to reduce categorical overfitting
- DLS-affected matches excluded from training (22 matches); their balls are still scored using the per-innings DLS allocation so revised chases aren't treated as full T20s
- No post-hoc calibration needed — the LightGBM probabilities are well-calibrated (max calibration error ~5%)
- `RETRAIN=1` env-var guardrail required to overwrite the committed pickle (prevents ad-hoc local pipeline runs from silently swapping in a noisier model)

**Model performance:**

| Metric | Value |
|--------|-------|
| Brier Score | 0.191 (lower is better, perfect = 0) |
| AUC | 0.780 |
| K-member disagreement (std) | median 1.4pp · p95 3.2pp · max 9.8pp |

**Feature importances** (mean across 100 ensemble members):

```
venue                        ████████████████████████████████   (654)
target                       █████████████                      (278)
batting_team_nrr             █████████████                      (269)
wickets_in_hand              ███████████                        (224)
run_rate                     ███████████                        (224)
runs_needed                  ██████████                         (203)
required_run_rate            ██████████                         (202)
innings                      █████████                          (187)
runs_scored                  █████████                          (175)
season_numeric               ██████                             (121)
opponent_bowler_economy      █                                   (32)
recent_wickets               █                                   (25)
over                         █                                   (22)
balls_remaining              ▏                                   (17)
batting_team_chose_to_bat    ▏                                   (17)
```

Venue dominates — ground conditions significantly affect win probability. Chase mechanics (target, wickets, run rate, required run rate) cluster together in the next tier. Team-strength (batting_team_nrr) is up there as well. Season captures era effects.

### Step 4: Computing TILT

For every ball in the dataset:

```
win_prob_before = model.predict(match_state_before_ball)
win_prob_after  = model.predict(match_state_after_ball)
delta_wp        = win_prob_after - win_prob_before
```

**Attribution:**
- **Batsman credit:** `+delta_wp` (positive means the batsman helped their batting team)
- **Bowler credit:** `-delta_wp` (positive means the bowler helped their bowling team)

*Example:* A six in the death overs that takes the batting team's win probability from 40% to 55% produces a `delta_wp` of +0.15. The batsman gets +15% credited, the bowler gets -15%.

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

*As of 2026-05-08 (post K=100 ensemble, post Step-3 removal, post `recent_wickets` rolling-window fix).*

| Rank | Player | TILT/Match | Raw | Confidence | Matches |
|------|--------|------------|-----|------------|---------|
| 1 | Sunil Narine | +4.88% | +5.07% | **high** | 195 |
| 2 | Jasprit Bumrah | +4.88% | +5.12% | **high** | 155 |
| 3 | Lasith Malinga | +4.35% | +4.64% | **high** | 122 |
| 4 | AB de Villiers | +4.39% | +4.60% | **high** | 168 |
| 5 | Yuzvendra Chahal | +3.08% | +3.24% | **high** | 179 |
| 6 | Nicholas Pooran | +4.15% | +4.50% | medium | 95 |
| 7 | Rashid Khan | +3.57% | +3.79% | **high** | 146 |
| 8 | Jos Buttler | +3.46% | +3.70% | **high** | 129 |
| 9 | Phil Salt | +4.58% | +5.48% | medium | 40 |
| 10 | KL Rahul | +3.08% | +3.28% | **high** | 144 |

### Notable Observations

- **Sunil Narine** tops both the floor ranking and the raw career total TILT (+9.89 lifetime, ahead of Bumrah at +7.93) — the most consistent all-round impact player in IPL history
- **AB de Villiers #4 floor / #3 by total** rises after the May 2026 `recent_wickets` rolling-window fix to `compute_state_after`. The previous logic over-counted recent wickets in the post-ball state, inflating bowling credit and depressing batting wp; the fix redistributed ~1 TILT back to top batters.
- **Buttler #8, Rahul #10** enter the floor top 10 alongside ABD — top batters reclaim space the wicket-window inflation had been giving to bowlers.
- **Bumrah, Malinga, Rashid Khan, Chahal** still anchor the bowler tier — now without inflated wicket-window credit. Floors are 0.5–1.0pp lower than the pre-fix table.
- **Pooran, Salt at the medium-confidence ranks** are the floor ranking behaving as designed under medium-sample players posting extreme raw numbers — shrinkage pulls them hard but the lower bound still clears most veterans
- **Venue matters most**: the venue feature has the highest importance, confirming that ground conditions significantly affect match outcomes
- **Era adjustment works**: old-era players are not disproportionately penalized — the season_numeric feature captures evolving T20 scoring rates

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
│   ├── refresh-tilt-data.yml  # 2×/day data-only refresh (no retrain)
│   └── retrain-tilt-model.yml # Seasonal (Mar 1) + manual full retrain
├── config/
│   └── pipeline_config.yaml   # Model hyperparameters, paths
├── data/                      # Raw + processed (gitignored)
├── models/                    # win_prob_lgbm.pkl is committed; others ignored
└── vercel.json                # Deployment config
```

**Data flow:**
1. Pipeline runs locally (~10 minutes for a full retrain — K=100 ensemble training is ~2 min of that; data download/parse/build/export accounts for the rest)
2. Outputs static JSON to `public/data/`
3. Website serves JSON directly — no computation at request time
4. Deployed to Vercel (Python Flask for routing + static file serving)
5. GitHub Actions refreshes data twice daily (07:00 / 14:00 UTC) reusing the committed model pickle; full retrains happen on March 1 or on-demand via `workflow_dispatch`

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
2. **No batting position context** — An opener facing the new ball operates in a different context than a #6 batsman.
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
- ~~"Royal Challengers Bengaluru" split from "Royal Challengers Bangalore"~~ → Normalized to a single canonical name at parse time
- ~~Career rankings flipping under retrain~~ → **K=100 LightGBM ensemble** averages trajectory variance across members, plus a `RETRAIN=1` env-var guardrail to prevent silent retrains (issue #111)

### Future Work

- **Batting hand / bowling type display** — scraper written (`scrape_player_meta.py`), 80/780 players scraped from ESPNcricinfo before hitting Akamai rate limits. The scraper is resumable — run `python pipeline/scrape_player_meta.py` after the rate limit clears (typically a few hours). Once `data/processed/player_meta.json` is complete, the data needs to be wired into the player JSON export and displayed on player pages.
- **Bowler type as model feature** — PaceOrSpin x Venue interactions (pending bowler type data from above)
- Impact sub analysis (data parsed, analysis pending)

---

## Data Source

All ball-by-ball data from [Cricsheet](https://cricsheet.org) under their open data license. Cricsheet is maintained by [Stephen Rushe](https://twitter.com/srushe) and is the gold standard for open cricket data.

---

## Running the Pipeline

### Full pipeline (download + train K=100 ensemble + export):
```bash
RETRAIN=1 python pipeline/run_pipeline.py
```

The `RETRAIN=1` guardrail (issue #111) prevents ad-hoc local pipeline runs from silently overwriting the committed model pickle. The data-only refresh below doesn't retrain and doesn't need it.

### Data-only refresh (skip training, reuse committed pickle):
```bash
python -c "from pipeline.run_pipeline import refresh_tilt_data_only; refresh_tilt_data_only()"
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
