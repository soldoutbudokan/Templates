# The TILT — Progress Tracker

## Status: Phase 1 (MVP) — Complete

### Pipeline
- [x] Project structure and config
- [x] `download_data.py` — download Cricsheet IPL data (1,183 match files)
- [x] `parse_matches.py` — parse JSON to ball-by-ball DataFrame (276,512 balls from 1,159 matches)
- [x] `build_features.py` — match state features per delivery
- [x] `train_win_prob.py` — LightGBM win probability model (Brier: 0.202, AUC: 0.759)
- [x] `compute_tilt.py` — per-player TILT scores from win prob deltas (781 players)
- [x] `export_json.py` — static JSON for website (443 qualified players with 10+ matches)
- [x] `run_pipeline.py` — end-to-end orchestrator (37s total)

### Website
- [x] Leaderboard page (sortable table, search/filter, role tabs)
- [x] Player detail pages (season breakdown, phase breakdown, best/worst matches)
- [x] About/methodology page
- [x] Dark monospace styling (PlainCricket aesthetic)
- [x] Flask API + Vercel deployment config
- [ ] Deploy to Vercel

### Verification
- [x] Model Brier score ~0.20 (0.2020)
- [x] Sanity checks pass (need 2 off 6 = 99.9%, impossible chase = 0.0%)
- [x] Top TILT players are recognizable (ABD #24, Warner #23, KL Rahul #20, Buttler #14, Rashid Khan #33)
- [ ] Live on Vercel

### Notable Results
- Top batters: PD Salt, Travis Head, Buttler, Jaiswal, Warner, ABD, KL Rahul
- Top all-rounder: Rashid Khan (bat +0.016, bowl +0.040)
- Top bowler: Bumrah (bowl +0.027)
- Small-sample players dominate top 5 — Bayesian shrinkage needed in Phase 2

---

## Phase 2 (Planned)
- Opponent quality adjustment (two-pass TILT)
- Bayesian shrinkage for small samples
- Confidence intervals
- Phase/season/venue breakdowns in rankings view

## Phase 3 (Future)
- All T20 internationals
- Season-over-season trends
- Player comparison tool
- Automated data refresh pipeline
