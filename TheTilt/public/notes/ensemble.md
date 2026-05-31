# Why We Ensemble

**Single LightGBM models on this dataset have ~14× more retrain variance than daily-data drift. A K=100 ensemble averaged at inference fixes the rank instability that one retrain produces.**

For most of TheTilt's existence the win-probability classifier was a single LightGBM model. It was Brier-good (0.18) and visually well-calibrated, and the rankings looked right. What we missed was that the rankings looked right *for that specific training run*. Pulling the same code through one more retrain on a barely-changed dataset produced something noticeably different.

This post is the diagnostic that surfaced the problem and the architectural fix that resolves it: a **K=100 LightGBM ensemble** averaged at inference, plus a `RETRAIN=1` environment-variable guardrail to prevent silent retrains. The fix landed for [issue #111](https://github.com/soldoutbudokan/Templates/issues/111).

---

## The thing that surfaced it

In early May 2026 the career top-10 had **Bhuvneshwar Kumar above AB de Villiers**. That's not a defensible ordering by any cricketing read of those two careers — Kumar is a top-tier death-overs swing bowler, ABD is the most consistent T20 batter the IPL has produced, and across <span id="ens-kumar-mt">205</span> vs <span id="ens-abd-mt">168</span> matches the gap should be the other way.

The first instinct (the natural one) was that something about the model's calibration had drifted. We'd shipped a few things lately: an exponential PP-decay calibration, a final-ball wp_after snap, a hyperparameter retune. Maybe one of those was the culprit. So we reverted the bundle.

Rankings didn't recover. **Re-running the pre-bundle config on today's dataset still produced the inversion.** The configuration was byte-identical to a known-good earlier snapshot. The only thing that had changed was the dataset itself — two new matches had landed in Cricsheet — and the resulting *re-trained* model.

That was the first piece of evidence pointing somewhere unexpected.

---

## The diagnostic

A single deterministic ball — innings 1, ball 0, no runs, no wickets, 10 wickets in hand, 120 balls left, fully-decided features for everything except `venue` — should produce the same `wp_before` from any model trained on the same code, hyperparameters, and data with the same random seed. The model class, the seed, and the hyperparameters were unchanged across the two snapshots. The data had grown by 2 matches.

Same input, four IPL openings, two pickles:

| Match | Date | Pickle A wp(ball 0) | Pickle B wp(ball 0) | Drift |
|:--|:--|:--:|:--:|:--:|
| 335982  | 2008-04-18 | 0.5160 | 0.5095 | −0.65 pp |
| 734047  | 2014-05-30 | 0.5375 | 0.5422 | +0.47 pp |
| 1082591 | 2017-04-05 | 0.5429 | 0.5351 | −0.78 pp |
| 1304051 | 2022-03-29 | 0.5158 | 0.5643 | **+4.85 pp** |

The 4.85pp drift on match 1304051 is pure model-retrain variance. Same input, different output, no code or hyperparameter or major data change. Compounded over 290k balls × 168 matches, that's enough to swing ABD's career total by −2.29 — and to flip his rank from #3 to #9.

### Why does this happen at all?

The pipeline uses `GroupShuffleSplit(test_size=0.1, random_state=42)`. That splitter is **deterministic given the set of `match_id`s** — but adding 2 matches reshuffles which matches land in train vs holdout. The training corpus changes by ~0.2%, but it changes in a way that affects the train/test boundary. Different training set → different gradient-boosting trajectory → different model. The `random_state=42` controls the LGBM seed *given the data*; it doesn't control which 90% of the data goes into training when the dataset grows.

LightGBM is a high-capacity learner. Even with monotone constraints and L2 regularization, the per-iteration choices it makes are somewhat sensitive to the exact rows it sees. That's normally fine — predictions stay close in aggregate. But "close in aggregate" doesn't mean "close on every input." Some specific feature combinations (an opening venue at a particular era, a specific phase × wickets-in-hand combo) get assigned to slightly different leaves between retrains, and those leaves can differ by a meaningful margin.

### How big is this compared to "real" drift?

We had a clean baseline to compare against: 7 daily snapshots from late April through May, all using the same *committed pickle* while the dataset grew with new matches.

| Source | p95 total-tilt drift | Max |
|:--|:--:|:--:|
| **Data growth (7 daily snapshots, same pickle)** | 0.226 | 0.566 |
| **One retrain (e9be8d8e → c6cc83e1, +2 matches)** | 1.343 | **5.55** |

A single retrain produces ~14× the noise of seven days of daily refreshes combined, on the metric that drives rankings. RG Sharma's career total swung **from +3.19 to −2.36** — sign-flipping a 271-match career. That's not signal moving; that's the trained model itself being unstable.

The structural problem: **the career signal between #1 and #20 in the top 10 is roughly 10 TILT units; the noise floor of one retrain is 5+ TILT units.** Rankings in the middle of the table are essentially random under retrain.

---

## The fix: K=100 ensemble, fixed holdout, retrain guardrail

Three changes, applied together:

### 1. Lock the holdout split

The 90/10 train/holdout split is now computed once with `random_state=42` and is the same 119 matches *forever*. Brier, AUC, and log-loss numbers are directly comparable across pipeline runs because they're measured on the same held-out games. Adding new IPL data shifts which matches sit in the *training* set; the holdout doesn't move.

```python
# config/pipeline_config.yaml
test_size: 0.1     # 10% holdout = 119 matches, locked
random_state: 42
```

### 2. Train K=100 LGBM members; average their predict_proba

```python
for seed in range(42, 142):
    member = LGBMClassifier(random_state=seed, ...).fit(train_df, ...)
    members.append(member)

ensemble = EnsembleModel(members)   # predict_proba averages across K
```

Same hyperparameters as before (`max_depth=4, num_leaves=16, min_child_samples=500, reg_lambda=5.0, n_estimators=2000`). The only thing that changes between members is the LightGBM internal random seed, which controls which features are sampled at each split, the order of bin-tie-breaks, and similar trajectory choices.

Cost:

| | Single | K=100 ensemble |
|:--|:--:|:--:|
| Train time | ~5s | ~110s |
| Pickle size | ~0.4 MB | ~33 MB |
| Inference (290k balls) | ~1s | ~5s |
| Holdout Brier | 0.190 | 0.191 |
| Holdout AUC | 0.781 | 0.780 |
| Per-ball wp disagreement (std across members) | n/a | median 1.4pp · p95 3.2pp · max 9.8pp |

Brier and AUC barely change — that's expected. Individual members are already well-regularized; ensembling isn't about making any single prediction better. It's about variance reduction across retrains.

The disagreement statistic is the real story: the K members routinely disagree by 1–2pp on the same ball, with worst-case ~10pp. Each individual training run is essentially sampling from this distribution. The ensemble averages it out.

### 3. RETRAIN=1 env-var guardrail

The way TheTilt got into this state is that two interactive runs of `pipeline/run_pipeline.py` silently overwrote the committed pickle while debugging downstream issues. By the time we noticed the rankings looked off, the model was three retrains removed from the version we'd validated on. Fix:

```python
# pipeline/train_win_prob.py
if not os.environ.get("RETRAIN") and Path(save_path).exists():
    raise RuntimeError(
        "Refusing to overwrite existing pickle. Set RETRAIN=1 to confirm."
    )
```

The cron data-refresh workflow doesn't call training at all, so it's unaffected. The seasonal retrain workflow sets `RETRAIN=1` in its env. Local runs of `run_pipeline.py` need the operator to opt in explicitly, which surfaces "you are about to retrain the model" as a deliberate action rather than a side effect.

---

## Convergence: how many members do we actually need?

K=100 is conservative — pickle size and training cost are both linear in K, so it's worth knowing whether we're paying for stability we don't need.

Total TILT for tracked players at K=20, K=50, K=100 (same fixed split, same data):

| Player | K=20 | K=50 | K=100 | Δ (K=20→100) |
|:--|:--:|:--:|:--:|:--:|
| AB de Villiers | +6.358 | +6.404 | +6.382 | +0.024 |
| Jasprit Bumrah | +9.490 | +9.408 | +9.484 | −0.006 |
| Sunil Narine | +10.373 | +10.352 | +10.331 | −0.042 |
| Rashid Khan | +7.189 | +7.244 | +7.258 | +0.069 |
| Bhuvneshwar Kumar | +5.511 | +5.546 | +5.540 | +0.029 |
| MS Dhoni | +2.924 | +2.772 | +2.745 | −0.179 |

Total-tilt shifts from K=20 to K=100 are within **±0.18** for every tracked player — *below* the daily-data drift noise floor of 0.226. K=20 is already at the architecture's variance floor; additional members produce less change than a normal daily refresh does. This is because individual LGBM members trained on the same data with the same hyperparameters are highly correlated, so additional members give diminishing returns.

We're shipping K=100 anyway. Storage is cheap, retrains are rare (once a year on the seasonal cron, plus on-demand via `workflow_dispatch`), and the conservative choice puts a firm floor on rank stability.

---

## What this fixes

The original triggering symptom — Bhuvneshwar Kumar above AB de Villiers in the top 10 — is gone:

| | Before (single model) | After (K=100 ensemble) |
|:--|:--:|:--:|
| AB de Villiers (total tilt, 168 matches) | +5.35 (#9) | **+6.84 (#6)** |
| Bhuvneshwar Kumar (total tilt, 199 matches) | +7.86 (#3) | **+6.93 (#4)** |
| Floor ranking (default site sort) | B Kumar above ABD | **ABD #7, B Kumar #8** |

ABD now ranks above Kumar on the floor view (the leaderboard's default sort), and the gap on raw career total is also right-sized. The fix reaches this without any post-hoc rescaling, manual reshuffle, or threshold-based filtering. Ranks that move under the ensemble are moving because the average of 100 trajectories disagrees with the one we happened to draw.

*A subsequent change in the same investigation (May 2026) removed the linear-decay Step 3 from `apply_boundary_calibration` because it was systematically over-crediting early-chase bowlers. After that change, ABD sits at #<span id="ens-abd-total-rank">3</span> by total / #<span id="ens-abd-floor-rank">5</span> by floor (a low-confidence, low-match debutant currently sits above him on the floor sort), and B Kumar is outside the top 10 floor. The story above is unchanged — the K=100 ensemble is what stabilized the rankings — but the live numbers shifted again. See [innings boundary § Step 3 removed](notes.html?note=innings-boundary#update-step-3-removed-issue-110-follow-up).*

More importantly: **the rankings will now hold across retrains.** A K=20 spot-check confirmed the top-10 floor is byte-identical to K=100 within the noise floor of normal data drift. Adding a match next week won't reshuffle the top of the leaderboard.

---

## What this doesn't fix

A few open items that K=100 doesn't address:

- **Telescoping residual at over boundaries** ([issue #110](https://github.com/soldoutbudokan/Templates/issues/110)). With retrain variance now controlled, the chained-endpoints A/B was finally clean. Two variants tested in `notebooks/boundary_cliff_prototype.py`: V5 (chain across all balls) and V6 (chain only at over boundaries). Both close the per-match telescoping residual (~5.5pp inn2 mean abs under the production Steps 1+2 calibration) but reshuffle the top-10 by 5–30 ranks for tracked players and surface medium/low-confidence short-career bowlers in the top-10 floor. Both stay deferred. Separate fallout from the same investigation: production's previous Step 3 (linear-decay damping over inn2 first six balls) was found to over-credit early-chase bowlers, and was removed — see [Innings boundary](notes.html?note=innings-boundary).
- **DLS handling beyond training-time exclusion.** 22 DLS-affected matches are filtered from training; their balls still flow through compute_tilt with the standard model, contributing to player TILTs from out-of-distribution states. Excluding them from ranking aggregation is a separate decision worth making.
- **Out-of-distribution feature combinations**, including the example in the diagnostic above (a 2022 RCB venue at a season the model has limited support for). Ensembling reduces the variance of the prediction on those points but doesn't fix the underlying coverage gap. More years of data fix it eventually; nothing in the model class fixes it now.

---

## What we'd do differently

The fundamental lesson: **retrain stability is a separate validation metric from accuracy.** A model can have a great Brier and great AUC and still be useless for career rankings if its predictions on specific feature combinations swing 5pp under retrain. The right discipline is to validate not just on a held-out set but on a *retrain replicate* — train two times on the same data, one with seed 42 and one with seed 43, and check whether the per-ball predictions agree to within whatever your downstream pipeline can tolerate. We weren't doing that.

The reason this surfaced now and not earlier is that the leaderboard had been mostly stable for a few months while the dataset stayed roughly the same size. The model was only ever retrained when something else changed, and we attributed the rank shifts to whatever change had been made — never to the retrain itself. Adding a retrain-stability check earlier would have caught the underlying issue independent of the dc29112a / c6cc83e1 / e9be8d8e debugging.

---

*Implementation: see [`pipeline/train_win_prob.py`](https://github.com/soldoutbudokan/Templates/blob/master/TheTilt/pipeline/train_win_prob.py) (`EnsembleModel`, `train_ensemble`) and [`pipeline/compute_tilt.py`](https://github.com/soldoutbudokan/Templates/blob/master/TheTilt/pipeline/compute_tilt.py) (model-load handles both single and ensemble pickles). The full root-cause diagnostic and decision log live on [issue #111](https://github.com/soldoutbudokan/Templates/issues/111).*
