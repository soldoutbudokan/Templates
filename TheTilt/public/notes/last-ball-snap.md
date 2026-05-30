# Snapping the Final Ball — and Why We Reverted It

**The match-terminal snap looked like a clean fix. It turned into a windfall machine for whoever happened to be bowling the last ball. We replaced it with a more aggressive model and a DLS-allocation fix that solve the underlying problem without dumping the entire correction onto a single delivery.**

The original post (preserved below in spirit) argued that the win-probability classifier never saw the literal end of a match, so its `wp_after` on the final ball of innings 2 averaged 0.928 on winning chases (truth: 1.0) and 0.161 on losing ones (truth: 0.0). The fix it shipped: snap `wp_after` on each match's final innings-2 ball to its actual outcome (1.0 / 0.0 / 0.5 for a regulation tie) and recompute `delta_wp` for that ball.

That fix was wrong in a way the original post didn't anticipate.

---

## What broke

The snap concentrates the entire ~7–16pp model-vs-truth gap onto **one ball, credited to one bowler**. That's fine when the gap is small. It's catastrophic when the gap is large — which happens whenever the model is confidently wrong about the chase state.

The clearest example: **Trent Boult, RR vs DC, 2 May 2018** ([match 1136592](match.html?id=1136592)). Rajasthan were chasing a DLS-revised target of 197 in 12 overs. They got to 145/5 with one ball left and lost by 50 runs. Boult bowled the last ball (a single).

Pre-snap, the model's `wp_before` on that ball was **0.89** — it thought RR were almost certainly going to win. They weren't even close. The model was fooled because `balls_remaining` in the feature pipeline ignored `innings_allocation` and defaulted to 120: the model saw "needs 52 from 46 balls with 5 wickets in hand" and shrugged, when in reality there was one ball left in a curtailed chase.

The snap then forced `wp_after = 0`, producing a `delta_wp = -0.89` charged to the batting team and `+0.89` to the bowler. Boult's match TILT ballooned to **+0.97** for a 3/26 spell — the all-time #1 single-game bowling performance in the dataset, of which **+0.89 was pure snap windfall**.

A decomposition of every bowler's match TILT into "snap credit" (the wp_before on the last ball, if they bowled it) vs. the rest:

| | Match TILT | Snap credit | Real bowling |
|:--|--:|--:|--:|
| TA Boult, 2018-05-02 (DC vs RR) | +0.97 | +0.89 | +0.08 |
| I Sharma, 2008-05-08 (KKR vs RCB) | +0.95 | +0.91 | +0.04 |
| L Balaji, 2009-05-07 (CSK vs PBKS) | +0.83 | +0.87 | −0.04 |
| Z Khan, 2013-05-18 (RCB vs CSK) | +0.78 | +0.83 | −0.05 |
| B Kumar, 2015-04-22 (SRH vs KKR) | +0.67 | +0.62 | +0.04 |

Every entry in the post-snap top single-game bowling list was dominated by snap credit, not by what the bowler actually did with the ball. **24 matches** had a snap credit greater than +0.50 going to one bowler.

This wasn't a quirk. It was a structural consequence of the snap design: when the model is wrong by 80pp at the end of a one-sided chase, snapping forces the entire 80pp into one player's tilt regardless of whether they did anything special on that ball.

---

## What we did instead

Three changes, applied together:

**1. Fixed the DLS allocation bug in `build_features.py`.** `balls_remaining` now reads `innings_allocation` from the parsed match data rather than defaulting to 120. A 12-over chase is 72 balls; a 6-over slog is 36. The Boult match's `wp_before` on the final ball drops from 0.89 to **0.16** with no other changes — the model now correctly sees a one-ball-left, target-not-close state.

**2. Loosened the LightGBM hyperparameters in `config/pipeline_config.yaml`.** Old: `max_depth=4, num_leaves=16, min_child_samples=500, reg_lambda=5.0`. New: `max_depth=6, num_leaves=64, min_child_samples=100, reg_lambda=1.0`. The old settings were so heavily regularized that even with `balls_remaining=1`, the model couldn't carve a tight enough leaf to push wp toward 0 or 1 in resolved chase states. The new settings let the model express the confidence the data actually supports. Brier improves from 0.198 to 0.178; AUC from 0.761 to 0.820. *(May 2026 update: the looser hyperparameters were reverted in commit `c6cc83e1` after they fell out of the K=100 ensemble retune. Current production runs the original `max_depth=4 / num_leaves=16 / min_child_samples=500 / reg_lambda=5.0` settings on K=100 averaged members; ensemble Brier 0.191, AUC 0.780.)*

**3. Removed `apply_match_terminal_snap` from the tilt pipeline.** Final-ball `wp_after` is now whatever the model says. Across the dataset that's **0.87 on winning chases** (was 1.0 with snap, ~0.93 without snap and old model) and **0.18 on losing chases** (was 0.0 with snap, ~0.16 without snap and old model). The residual gap is a real model-confidence shortfall, but it's now spread across the last few balls of each match rather than dumped onto whoever bowled the literal last ball — and individual leaf predictions of 0.95 / 0.05 are common at terminal states, not capped at 0.93 / 0.16.

---

## What changed in the rankings

The snap's pathological top-of-leaderboard entries are gone. Top single-game bowling now reads like real bowling:

| Rank | Bowler | Match | Figures | TILT |
|:--|:--|:--|:--|--:|
| 1 | Mohammed Shami | 2019-04-10 KXIP vs MI | 3/21 (4 ov) | +0.71 |
| 2 | Bhuvneshwar Kumar | 2017-04-17 SRH vs KXIP | 5/20 (4 ov) | +0.62 |
| 3 | Dale Steyn | 2012-05-20 DC vs RCB | 3/8 (4 ov) | +0.61 |
| 4 | Jofra Archer | 2026-05-24 RR vs MI | 3/17 (4 ov) | +0.61 |
| 5 | Rashid Khan | 2018-04-24 SRH vs MI | 2/11 (4 ov) | +0.60 |

Career rankings reshuffle in the opposite direction from the original snap-era post. **DJ Bravo**, who entered the snap-era all-time top 10 from #62 on the strength of his death-overs reputation, drops out again — the snap was inflating his match-end deliveries with credit the model never actually disagreed with. **Dhoni and Miller**, who lost 34 spots each under the snap, partially recover.

---

## What we'd do differently

The original snap post was correct that the model is structurally under-confident at terminal states. The mistake was the prescription: a hard snap that concentrates correction onto one player.

The right shape of fix, in retrospect:

- **Fix any feature bugs that make the model wrong about the state** (the DLS allocation thing). Anything else is layering corrections on top of corrections.
- **Give the model enough capacity to push toward 0/1 when the chase is mathematically over.** Heavy regularization makes the leaderboard "smooth" in a way that obscures actual gaps in predictive accuracy at the boundary.
- **If a residual gap remains after that, smooth the correction across the last several balls** rather than snapping one. The model is already mildly under-confident at ball-from-end ∈ {0, 1, 2, 3} on tight finishes; a soft per-ball blend across that window would distribute any leftover correction across the players who were actually involved in those moments. We didn't ship this — the residual after the model retune is small enough not to require it — but it's the shape we'd reach for if we needed more.

The general lesson: **single-row hard corrections to model output produce single-row hard credit windfalls**. If a fix touches 1 row per match and shifts that row's prediction by 20+ percentage points, the credit lands on whichever player happens to be on that row. That's not a methodology — it's a lottery.

---

*Original publication: 2026-05-06 (snap shipped). Revised: 2026-05-06 (snap reverted, post rewritten as post-mortem).*
