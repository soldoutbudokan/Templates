# The TILT

**How much does a player tilt the game?**

Live site: [thetilt-rust.vercel.app](https://thetilt-rust.vercel.app)

TILT measures **Win Probability Added (WPA)** per ball in IPL cricket. For every single delivery bowled in the IPL's history, we calculate how much the win probability shifted and attribute that shift to the batter and bowler involved. Aggregate across all matches and you get each player's impact rating — their TILT.

Inspired by basketball analytics like [EPM](https://dunksandthrees.com/epm) and [RAPTOR](https://projects.fivethirtyeight.com/nba-player-ratings/), adapted for cricket's unique ball-by-ball structure.

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

- **1,159 matches** parsed (2008-2026)
- **276,512 individual deliveries** with batter, bowler, runs, wickets, and match outcome
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
- 1000 trees (with early stopping at ~135), learning rate 0.03, max depth 5
- DLS-affected matches excluded from training (22 matches)
- Platt scaling applied for calibration
- Trained in ~5 seconds

**Model performance:**

| Metric | Value |
|--------|-------|
| Brier Score | 0.203 (lower is better, perfect = 0) |
| AUC | 0.747 |
| Log Loss | 0.588 |

**Calibration** — the model's predicted probabilities closely match actual outcomes:

```
Predicted → Actual Win Rate
  6%  →   9%
 15%  →  16%
 26%  →  29%
 35%  →  42%
 45%  →  51%    (well-calibrated in the middle)
 55%  →  48%
 65%  →  55%
 75%  →  72%
 85%  →  82%
 93%  →  95%
```

**Sanity checks pass:**

| Scenario | Predicted Win Prob |
|----------|--------------------|
| Innings 2, need 2 off 6 balls, 8 wickets in hand | 89.0% |
| Innings 2, need 60 off 6 balls, 2 wickets in hand | 10.7% |
| Start of match (0/0, ball 1) | 42.3% |
| Innings 1, scored 200/2 off 18 overs | 77.4% |

**Feature importances** (what matters most for predicting the winner):

```
venue                  ████████████████████████████████  (1173)
required_run_rate      ████████████                     ( 444)
target                 ███████████                      ( 419)
wickets_in_hand        ██████████                       ( 375)
runs_needed            ████████                         ( 310)
run_rate               ████████                         ( 306)
runs_scored            ██████                           ( 255)
innings                █████                            ( 187)
season_numeric         ███                              ( 137)
opponent_bowler_econ   ██                               (  84)
balls_remaining        █                                (  50)
```

Venue is now the most important feature — confirming that where the match is played significantly affects win probability (e.g., Chinnaswamy's high-scoring conditions vs Chepauk's spin-friendly tracks). Season captures era effects.

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

Where `k` is estimated from the ratio of within-player to between-player variance (k ≈ 6 for IPL data). This means:
- A player with 6 matches: 50% raw, 50% population mean
- A player with 60 matches: 91% raw, 9% population mean
- A player with 188 matches (Narine): 97% raw, 3% population mean

### Step 5: Export & Website

Rankings and per-player breakdowns are exported as static JSON files and served on a Vercel-hosted website with sortable tables, search, and player detail pages.

---

## Results & Findings

Rankings use **Bayesian shrinkage** (empirical Bayes) to stabilize small-sample estimates. Players with few matches are pulled toward the population mean; high-match players keep their raw values. Confidence levels: green (100+ matches), yellow (30-100), gray (<30).

### Top Overall Players (Shrunk TILT, 10+ matches)

| Rank | Player | TILT/Match | Raw | Confidence | Matches |
|------|--------|------------|-----|------------|---------|
| 1 | Priyansh Arya | +11.48% | +15.83% | low | 18 |
| 2 | HM Amla | +6.89% | +10.11% | low | 16 |
| 3 | DE Bollinger | +5.51% | +7.11% | low | 27 |
| 4 | Rashid Khan | +4.98% | +5.28% | **high** | 136 |
| 5 | ML Hayden | +4.72% | +5.97% | medium | 31 |
| 6 | SP Narine | +4.59% | +4.79% | **high** | 188 |
| 7 | TM Head | +4.47% | +5.40% | medium | 40 |
| 8 | V Sehwag | +4.29% | +4.65% | **high** | 102 |
| 9 | RD Gaikwad | +4.20% | +4.69% | medium | 73 |
| 10 | YBK Jaiswal | +4.14% | +4.66% | medium | 68 |

### Top Bowlers

| Rank | Player | Bowl TILT/Match | Confidence | Matches |
|------|--------|-----------------|------------|---------|
| 4 | Rashid Khan | +4.33% | **high** | 136 |
| 6 | SP Narine | +3.60% | **high** | 188 |
| 25 | YS Chahal | +3.14% | **high** | 171 |
| 26 | SL Malinga | +3.03% | **high** | 120 |
| 34 | JJ Bumrah | +2.75% | **high** | 144 |

### Notable Observations

- **Rashid Khan** (#4) is the highest-ranked high-confidence player — elite in both batting and bowling
- **SP Narine** (#6) across 188 matches — the most consistent all-round impact player in IPL history
- **Bumrah** (#34) with +2.90% across 144 matches — pure bowling consistency. His raw and shrunk values are almost identical (high match count = minimal shrinkage)
- **Sehwag** (#8) at +4.29% with high confidence — his aggressive opening style created enormous win probability swings
- **Venue matters most**: The venue feature has the highest importance (1173), confirming that ground conditions significantly affect match outcomes
- **Era adjustment works**: Old-era bowlers (McGrath, Vettori) no longer disproportionately dominate — the season_numeric feature captures evolving T20 scoring rates
- **Shrinkage effect**: Priyansh Arya drops from +15.83% raw to +11.48% shrunk (18 matches). Rashid Khan barely moves: +5.28% → +4.98% (136 matches). The shrinkage constant k ≈ 6 means you need ~6 matches before the model trusts your data 50%.

---

## Architecture

```
TheTilt/
├── pipeline/                  # Offline Python pipeline
│   ├── download_data.py       # Cricsheet IPL ZIP download
│   ├── parse_matches.py       # JSON → DataFrame (BallEvent dataclass)
│   ├── build_features.py      # Match state features per delivery
│   ├── train_win_prob.py      # LightGBM win probability model
│   ├── compute_tilt.py        # Win prob deltas → per-player TILT
│   ├── export_json.py         # Static JSON for website
│   └── run_pipeline.py        # End-to-end orchestrator
├── api/
│   └── index.py               # Flask app for Vercel (serves static files)
├── public/
│   ├── index.html             # Leaderboard (sortable, searchable)
│   ├── player.html            # Player detail page
│   ├── about.html             # Methodology page
│   ├── style.css              # Dark monospace theme
│   └── data/                  # Pre-computed static JSON
│       ├── tilt_rankings.json # All qualified players
│       ├── players/*.json     # Per-player detail files
│       └── meta.json          # Data coverage metadata
├── config/
│   └── pipeline_config.yaml   # Model hyperparameters, paths
├── data/                      # Raw + processed (gitignored)
├── models/                    # Saved model (gitignored)
└── vercel.json                # Deployment config
```

**Data flow:**
1. Pipeline runs locally (~37 seconds), downloads from Cricsheet, processes everything
2. Outputs static JSON to `public/data/`
3. Website serves JSON directly — no computation at request time
4. Deployed to Vercel (Python Flask for routing + static file serving)

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

### What's Been Addressed

- ~~Small sample bias~~ → **Bayesian shrinkage** with empirical Bayes (k ≈ 6)
- ~~Single model for all eras~~ → **Season year** as a continuous feature
- ~~No venue context~~ → **Venue** as categorical feature (highest importance)
- ~~No toss context~~ → **Toss** parsed and used as `batting_team_chose_to_bat`
- ~~No opponent quality~~ → **Bowling economy** proxy added as feature
- ~~DLS matches~~ → Excluded from training (22 matches)
- ~~Recent run rate bug~~ → Fixed approximation in `compute_tilt.py`

### Future Work

- **Bowler type classification** (pace/spin) + PaceOrSpin × Venue interactions
- Expand to all T20 internationals (21,000+ matches on Cricsheet)
- Player comparison tool
- Automated pipeline for new Cricsheet data releases
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
