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

**Key design decision:** We use a single model for both innings. For innings 1, the target/required run rate features are set to 0. This lets the model learn that innings 1 is about *setting a total* while innings 2 is about *chasing*. The `innings` feature acts as a switch.

### Step 3: Win Probability Model

We train a **LightGBM gradient-boosted classifier** to predict: *will the team currently batting win this match?*

```
Input:  13 match-state features (before each ball)
Output: P(batting team wins) ∈ [0, 1]
```

**Why LightGBM?**
- Handles mixed feature types (categorical + continuous) natively
- Fast to train on 276K rows
- Excellent probability calibration out of the box
- Easy to inspect feature importances

**Training details:**
- Train/test split: **80/20 by match** (not by ball — splitting by ball would leak information since balls within the same match are correlated)
- 500 trees, learning rate 0.05, max depth 6
- Trained in ~5 seconds

**Model performance:**

| Metric | Value |
|--------|-------|
| Brier Score | 0.202 (lower is better, perfect = 0) |
| AUC | 0.759 |
| Log Loss | 0.603 |

**Calibration** — the model's predicted probabilities closely match actual outcomes:

```
Predicted → Actual Win Rate
  3%  →  14%    (slightly under-confident at extremes)
 15%  →  25%
 25%  →  33%
 36%  →  35%    
 46%  →  43%    (well-calibrated in the middle)
 54%  →  50%
 65%  →  61%
 75%  →  73%    
 85%  →  74%
 97%  →  89%    (slightly over-confident at extremes)
```

**Sanity checks pass:**

| Scenario | Predicted Win Prob |
|----------|--------------------|
| Innings 2, need 2 off 6 balls, 8 wickets in hand | 99.9% |
| Innings 2, need 60 off 6 balls, 2 wickets in hand | 0.0% |
| Start of match (0/0, ball 1) | 47.7% |
| Innings 1, scored 200/2 off 18 overs | 94.5% |

**Feature importances** (what matters most for predicting the winner):

```
target                 ████████████████████████████████  (4373)
run_rate               ██████████████████               (2457)
wickets_in_hand        ████████████                     (1685)
required_run_rate      ███████████                      (1596)
runs_scored            ██████████                       (1378)
recent_run_rate        ████████                         (1144)
runs_needed            ███████                          ( 949)
recent_wickets         ████                             ( 506)
balls_remaining        ███                              ( 459)
innings                ██                               ( 222)
```

The target (chase total) dominates, which makes sense — knowing what you're chasing is the single most important piece of information in an IPL match.

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

### Step 5: Export & Website

Rankings and per-player breakdowns are exported as static JSON files and served on a Vercel-hosted website with sortable tables, search, and player detail pages.

---

## Results & Findings

### Top Overall Players (10+ matches)

| Rank | Player | Team | TILT/Match | Matches |
|------|--------|------|------------|---------|
| 14 | JC Buttler | Gujarat Titans | +6.45% | 121 |
| 17 | YBK Jaiswal | Rajasthan Royals | +6.11% | 68 |
| 20 | KL Rahul | Delhi Capitals | +5.89% | 133 |
| 23 | DA Warner | Sunrisers Hyderabad | +5.47% | 180 |
| 24 | AB de Villiers | Royal Challengers Bangalore | +5.40% | 166 |

### Top Bowlers

| Rank | Player | Team | Bowl TILT/Match | Matches |
|------|--------|------|-----------------|---------|
| 1 | GD McGrath | Delhi Daredevils | +6.98% | 14 |
| 2 | DL Vettori | Royal Challengers Bangalore | +5.52% | 34 |
| 4 | SL Malinga | Mumbai Indians | +5.78% | 120 |
| 6 | SP Narine | Kolkata Knight Riders | +3.73% | 188 |
| 8 | Rashid Khan | Gujarat Titans | +4.04% | 136 |

### Notable Observations

- **Rashid Khan** is the best all-rounder by TILT — positive in both batting (+1.57%) and bowling (+4.04%)
- **Bumrah** has a bowling TILT of +2.71% across 144 matches — elite consistency
- **Kohli** (+2.46%) ranks lower than you might expect — he's extremely consistent but TILT rewards *clutch moments* and *game-changing innings* more than volume
- **AB de Villiers** (+5.40%) ranks much higher — his explosive style created larger win probability swings per innings
- Small-sample players dominate the very top (V Suryavanshi at +18% in 10 matches). Phase 2 will add Bayesian shrinkage to handle this.

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
- **Website:** Vanilla HTML/CSS/JS (no framework needed — it's a static data site)
- **API:** Flask (minimal — mostly just routes to static files)
- **Hosting:** Vercel (serverless Python + static files)
- **Aesthetic:** Dark monospace terminal style, matching [PlainCricket](https://plaincricket.vercel.app)

---

## Limitations & Future Work

### Current Limitations

1. **No opponent quality adjustment** — A boundary off Bumrah should count for more than a boundary off a part-timer. Currently both are weighted equally.
2. **Small sample bias** — Players with 10-15 matches can have extreme TILT values that are more noise than signal.
3. **Fielding is invisible** — The model can't attribute catches, run-outs, or misfields to specific fielders.
4. **No batting position context** — An opener facing the new ball operates in a different context than a #6 batter.
5. **Single model for all eras** — IPL scoring rates have changed over 18 years, but we use one model for everything.

### Phase 2 (Planned)

- **Opponent quality adjustment:** Two-pass system where pass 1 computes raw TILT, then pass 2 re-weights based on opponent strength
- **Bayesian shrinkage:** Regress small-sample players toward the population mean
- **Confidence intervals:** Bootstrap-based uncertainty bounds
- **Richer breakdowns:** Season-by-season trends, venue effects, matchup analysis

### Phase 3 (Future)

- Expand to all T20 internationals (21,000+ matches on Cricsheet)
- Player comparison tool
- Automated pipeline for new Cricsheet data releases
- More sophisticated model (pitch conditions, toss, home advantage)

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
