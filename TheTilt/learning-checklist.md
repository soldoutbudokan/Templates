# TheTilt — Learning Checklist

A running map of what you should be able to explain (and *demonstrate*, not just
recite) by the end of this teaching session. Boxes get ticked only once you've
shown the understanding back to me — high level **and** low level.

**Legend:** `[ ]` not yet · `[~]` partial / in progress · `[x]` demonstrated

---

## 🎓 Graduation bar (the /goal)

The session isn't done until you can explain, **unprompted**:

- [ ] Why TILT beats traditional stats (average / economy / strike rate)
- [ ] How a single ball becomes a number on the leaderboard (end-to-end)
- [ ] Why the K=100 ensemble exists
- [ ] Why the floor-ranking (90% CI lower bound) exists instead of raw TILT
- [ ] Exactly what you'd have to refresh if you changed the model or the feature set

---

## Stage 1 — Motivation: *what, not when*

- [ ] The core problem: traditional stats measure output, not **leverage/context**
- [ ] TILT = Win Probability Added (WPA) per ball, in win-prob percentage points / match
- [ ] Attribution convention: batter gets `+Δwp`, bowler gets `−Δwp`
- [ ] Why context (balls/wickets/target remaining) *is* the entire point
- [ ] Worked intuition: same 30 runs, different game states → different TILT

## Stage 2 — The win-probability model

- [ ] The 15 before-the-ball features (and that they're all **pre-delivery** state)
- [ ] Single model for both innings; innings-1 chase features = 0; `innings` is the switch — why not two models
- [ ] Why LightGBM (mixed types, calibration, speed, importances)
- [ ] The K=100 ensemble: what varies (random_state 42…141), what's frozen (the 90/10 split @ seed 42)
- [ ] *Why* the ensemble exists: single-fit instability (ABD #3→#9 on +2 matches), trajectory variance, ~5pp ball swings
- [ ] Heavy regularization (reg_lambda=5, min_child_samples=500, depth 4, leaves 16) and *why* (genuine trajectory variance, not leaf noise)
- [ ] Monotone constraints (wickets_in_hand +1, runs_needed −1, required_run_rate −1) and why
- [ ] Performance: Brier 0.191, AUC 0.780; what each means
- [ ] Feature importances: venue dominates → chase mechanics → team strength → era; *why venue?*

## Stage 3 — From win prob to TILT

- [ ] `wp_before` vs `wp_after`: how the after-state is constructed (`compute_state_after`)
- [ ] `delta_wp = wp_after − wp_before`; the +/− attribution
- [ ] Per-match aggregation → per-match TILT → career per-match TILT
- [ ] Bayesian shrinkage: `shrunk = n/(n+k)·raw + k/(n+k)·pop_mean`; what k is (within/between var ≈ 5.3)
- [ ] Why shrinkage (small-sample noise); the 5 / 50 / 188-match intuition
- [ ] Ranking by the 90% CI lower bound ("TILT floor")
- [ ] Posterior variance `var/(n+k)` vs frequentist `var/n` — why the posterior one is internally consistent
- [ ] (code-beats-prose) what the code *actually* uses for the CI variance

## Stage 4 — Edge cases

- [ ] Innings asymmetry: 2nd-innings |Δwp| ≈ 1.57×, death overs 2.12×
- [ ] Boundary calibration (issue #62): the innings-break cliff, isotonic + midpoint fix
- [ ] The `recent_wickets` rolling-window after-state fix (issue #110/#185)
- [ ] DLS matches: excluded from **training**, still **scored**; per-innings allocation
- [ ] Ties / Super Over: eliminator credited as winner, SO innings dropped (issue #84)
- [ ] No-result matches: injected from raw + supplement YAML (issues #75/#83)
- [ ] The `RETRAIN=1` guardrail — what it protects against (issue #111)

## Stage 5 — The website / system

- [ ] Static-JSON architecture: everything precomputed offline, Flask serves files, zero compute at request
- [ ] The pipeline DAG and the artifact at each stage (raw → parquet → pkl → deltas → public/data/*.json)
- [ ] Which JSON feeds which page
- [ ] Two-tier dependency discipline: tier-1 auto-refreshed vs tier-2 hand-maintained embedded numbers
- [ ] *Why* tier-2 exists (the "site says 1.54x but script says 1.61x" stale-number bug)
- [ ] The reverse "if you change X, refresh Y" map
- [ ] The two cron workflows: twice-daily data refresh vs Mar-1/manual full retrain

---

*Maintained during the session. Items are ticked as you demonstrate them.*
