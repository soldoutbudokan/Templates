# Fixing the Innings-Boundary Jump

**A two-step calibration that closes the visible cliff in the win-probability chart and corrects the artificial TILT it was producing.**

For most of TheTilt's existence, the match-page win-probability chart had a visible discontinuity at the innings break. The line would leap by five, ten, sometimes twenty percentage points across a moment when no ball had been bowled. We documented it ([previous version of this post](#technical-details)), tried four different model rewrites to fix it, shipped a cosmetic chart smoother that hid the jump visually but left the underlying TILT data distorted — and parked the real fix as an open issue.

This post describes the real fix that landed in [issue #62](https://github.com/soldoutbudokan/Templates/issues/62): a two-step boundary calibration that runs as a post-processing layer on the existing model. Median per-match cliff went from **8.4pp to 0pp** (mathematically zero). The model, the features, and the deltas on every non-boundary ball are byte-for-byte unchanged. Only one ball at the end of innings 1 and one ball at the start of innings 2 — about 1.4% of all deliveries — get a deterministic post-hoc correction. Career rankings reshuffle modestly in the direction you'd expect: players whose tilt was being inflated by the cliff lose a little, players whose tilt was being suppressed by it gain a little.

---

## The problem in one paragraph

The win-probability classifier sees the innings boundary as two different feature vectors of the same world state. End of innings 1: `innings=1`, `target=0`, `runs_needed=0`, `required_run_rate=0`, `balls_remaining≈1`. Start of innings 2: `innings=2`, `target=actual`, `runs_needed=actual`, `required_run_rate=concrete`, `balls_remaining=120`, `wickets_in_hand=10`. Six features move in one step. The model — quite reasonably — learned different decision-tree branches for the two states, and they don't agree. Across all 1,169 IPL matches with both innings, the median absolute disagreement was **8.4 percentage points** in the batting-first team's win probability. The mean signed disagreement was **−5.1 pp**: the model was systematically more pessimistic about the batting-first team's chances at the start of the chase than at the end of its own innings.

We weren't going to retrain the model away from this. The earlier analysis ([the original write-up of this post](#technical-details)) explored that direction and found four dead ends — every "structural" rewrite that suppressed the cliff also flattened the inn1/inn2 signal that distinguishes anchor batters from finishers, and ranked Ashwin and AB de Villiers in places that no cricket fan would defend.

The trick was to stop treating it as a model problem and start treating it as a calibration problem.

---

## The fix: isotonic + per-match midpoint

Two steps, both applied only to the two boundary balls per match:

### Step 1 — Per-side isotonic calibration

The model's raw output at the boundary is biased: at inn1's end, average BF-POV WP is 0.469 when the actual BF win rate at those states is closer to 0.445; at inn2's start, average BF-POV WP is 0.422 when the truth is also closer to 0.445. So we fit two `IsotonicRegression`s — monotonic step functions — on every past match:

```
iso_inn1: raw_wp_at_inn1_end       →  empirical bf-win rate
iso_inn2: raw_wp_at_inn2_start_bf  →  empirical bf-win rate
```

Trained over all 1,160-ish non-DLS matches with `batting_team_won` as the target. Each iso learns a calibrated mapping in its own input range. Applied side-by-side, this kills the average bias (mean signed cliff drops from −5.1pp to +0.3pp). But per-match disagreement actually grows — each side calibrates independently, so the two endpoints don't have to agree on any specific match. Median |cliff| moved from 8.4pp → 11.5pp. Bias dies, variance grows. Step 1 alone is not enough.

### Step 2 — Per-match midpoint bridge

For each match, after step 1, snap both endpoints to their BF-POV midpoint:

```
mid_bf = 0.5 * (iso_inn1(inn1_end_bf_raw) + iso_inn2(1 - inn2_start_chase_raw))
inn1.last.wp_after  = mid_bf
inn2.first.wp_before = 1 - mid_bf
delta_wp recomputed for both balls
```

This forces the model to agree with itself across the boundary, by construction. Cliff goes to **exactly 0** on every match. The chart line is naturally continuous — no cosmetic bridge, no dashed-marker fudge. Just the data.

### Why this combination

Step 1 alone is what the original issue called "option 2" and recommended as the surgical fix. Implementing it revealed the variance problem above. Step 2 alone (averaging without isotonic) would close the visual cliff but bake in the model's calibration error — both endpoints would converge on a biased midpoint. Together: step 1 calibrates each side to truth, step 2 enforces per-match continuity. The "A2" name in the analysis log distinguished it from "A" (isotonic only) and "B" (the failed two-model rewrite).

---

## What changed in the rankings

The whole point of doing this carefully was to make sure the fix doesn't violate basic cricketing intuition. Here's the actual top-10 movement.

### Top 10 by total career TILT

| Rank | Before fix | After fix (now) |
|:--|:--|:--|
| 1 | SP Narine | **SP Narine** |
| 2 | Rashid Khan | **JJ Bumrah** ⬆ |
| 3 | JJ Bumrah | Rashid Khan ⬇ |
| 4 | SL Malinga | **SL Malinga** |
| 5 | AB de Villiers | **AB de Villiers** |
| 6 | B Kumar | YS Chahal ⬆ |
| 7 | YS Chahal | **CH Gayle** 🆕 |
| 8 | R Ashwin | JC Buttler ⬆ |
| 9 | JC Buttler | B Kumar ⬇ |
| 10 | Harbhajan Singh | DW Steyn 🆕 |

Top 5 essentially stable. The middle of the list reshuffles: Bumrah climbs to #2 (his bowling tilt was slightly suppressed by inn1-end overconfidence — recalibration helps him), Gayle and Steyn enter, R Ashwin and Harbhajan exit. **B Kumar drops from #6 to #9** — see below for why.

### Top 10 by batting career TILT

| Rank | Before fix | After fix (now) |
|:--|:--|:--|
| 1 | AB de Villiers | **AB de Villiers** |
| 2 | JC Buttler | CH Gayle ⬆ |
| 3 | CH Gayle | JC Buttler ⬇ |
| 4 | MS Dhoni | V Sehwag ⬆ |
| 5 | DA Warner | RR Pant ⬆ |
| 6 | N Pooran | **YBK Jaiswal** 🆕 |
| 7 | V Sehwag | DA Warner ⬇ |
| 8 | Shubman Gill | N Pooran ⬇ |
| 9 | RR Pant | **DR Smith** 🆕 |
| 10 | GJ Maxwell | Shubman Gill ⬇ |

Dhoni and Maxwell drop out of the top 10. Jaiswal and DR Smith enter. ABD's #1 spot is unaffected — his career boundary share was only ~2% of his total tilt, so the recalibration barely touches him.

### Top 10 by bowling career TILT

| Rank | Before fix | After fix (now) |
|:--|:--|:--|
| 1 | SP Narine | **SP Narine** |
| 2 | Rashid Khan | JJ Bumrah ⬆ |
| 3 | JJ Bumrah | Rashid Khan ⬇ |
| 4 | B Kumar | **SL Malinga** |
| 5 | SL Malinga | YS Chahal ⬆ |
| 6 | R Ashwin | Harbhajan Singh ⬆ |
| 7 | Harbhajan Singh | B Kumar ⬇ |
| 8 | YS Chahal | R Ashwin ⬇ |
| 9 | A Mishra | DW Steyn ⬆ |
| 10 | DW Steyn | A Mishra ⬇ |

The headline movement: B Kumar drops from #4 to #7. About 13% of his pre-fix career bowling tilt came from the first ball of innings 2 — a place where the model used to be wildly pessimistic about the chase, so any wicket on that ball generated outsized credit. Recalibration removes that artificial credit. He's still a top-10 bowler; he's just no longer over-credited for being the death-knell-deliverer of the model's chase pessimism.

---

## What didn't change

- **The model itself.** No retraining. The committed `models/win_prob_lgbm.pkl` is identical.
- **All non-boundary deltas.** Every powerplay, middle-overs, and death-overs delta uses the original raw model output. Only the two boundary balls per match have their `wp_after` / `wp_before` overwritten.
- **The match-volatility analysis.** [The Second Innings Problem](notes.html?note=innings-bias) is a different diagnostic — it documents why career rankings are roughly innings-balanced even though single-match TILT shifts skew toward the chase. That post stands as written; the boundary fix doesn't interact with it.

---

## What it cost

About 30 lines added to `pipeline/compute_tilt.py` (a new `apply_boundary_calibration` function called between `compute_ball_deltas` and `aggregate_player_tilt`). About 50 lines removed from `public/match.html` (the cosmetic chart-bridge JavaScript is now redundant and would be a double correction). Two scikit-learn `IsotonicRegression` fits at pipeline time, both trained in well under a second on ~1,200 data points each. The fix is fully reversible — drop the function call and the system reverts to the pre-fix state.

---

## Technical details

The original analysis that drove this fix is preserved as the diagnostic notebook `notebooks/innings_boundary_analysis.py`. It can be re-run against the current pipeline output and will report a median |cliff| of 0.0pp, mean signed cliff of 0.0pp, and zero matches with |cliff| ≥ 5pp — confirming the fix is working as intended. The pre-calibration values are stashed in `deltas.parquet` as `wp_after_raw` and `wp_before_raw` (only populated on the two boundary rows per match) so the notebook can also produce before/after diagnostics.

The four model-rewrite attempts that failed, the per-team-carry experiment that ranked ABD #39, and the trade-off matrix between cliff-closure and ranking validity are all in [issue #62](https://github.com/soldoutbudokan/Templates/issues/62) for anyone who wants the full backstory.

---

## Update: alpha-decay across the first over (issue #71)

The two-step calibration above kills the inn1/inn2 cliff but leaves a fresh discontinuity at the next ball. inn1.last and inn2.first agree on a midpoint (mathematically zero gap), but inn2.first.wp_after and the rest of the chase still come from the raw model — and the model's natural "chase has just started" state is systematically a few percentage points off the calibrated midpoint. The chart smooths the boundary, then jumps a few points on the next ball.

The fix shipped for [issue #71](https://github.com/soldoutbudokan/Templates/issues/71) is a third step layered on top of the two above: a linear-decay blend of wp_before and wp_after across balls 1–6 of innings 2, with weight `alpha = (6 − ball)/5`. Ball 1 stays at the calibrated midpoint (alpha = 1.0, full bridge — same behavior as before). Balls 2 through 5 progressively let the raw model take over. Ball 6 is the raw model unchanged. By the start of over 2 the calibration has zero influence, and by ball 1's delta_wp ends at exactly 0 by construction.

### What changed in the rankings (post-#71)

Top-10 movement is small once you sort by the website's primary leaderboard ordering (`tilt_ci_lower_90`). The biggest shift is GD McGrath rising from #30 to #3 in bowling — his short 2008–2009 stint had high point-estimate TILT but high variance, and the calibration's influence on early-chase variance reduces his uncertainty interval. Beyond that, Sohail Tanvir enters the top 10 overall (was #26), and most others move 0–2 spots.

Ranked by **total career TILT** (the same lens the tables above use):

| Rank | After #71 |
|:--|:--|
| 1 | SP Narine |
| 2 | JJ Bumrah |
| 3 | AB de Villiers ⬆ |
| 4 | SL Malinga |
| 5 | Rashid Khan ⬇ |
| 6 | YS Chahal |
| 7 | B Kumar ⬆ |
| 8 | JC Buttler |
| 9 | CH Gayle ⬇ |
| 10 | Harbhajan Singh ⬆ |

B Kumar climbs from #9 back to #7 — the alpha-decay returns some of the early-chase credit that the per-match midpoint snap was zeroing out. The names that exit (DW Steyn) had benefited from the model's chase pessimism on a few specific deliveries that no longer carry full weight.

The fix is reversible: drop the third step (lines marked "Step 3" in `apply_boundary_calibration`) and the system reverts to the post-#62 calibration above.

---

## Update: exponential decay across the powerplay + chained endpoints

The linear-decay-over-6-balls fix above closed the visible second-over jump but left two issues. First, by the end of over 1 the alpha had snapped from 0.2 to 0.0 and the model's "after-7-balls chase" prediction would still differ a few percentage points from the calibrated trajectory — a smaller, but still visible, jump. Second, an internal bookkeeping issue: each ball's `wp_before` and `wp_after` were blended *independently*, so within balls 1–6 of innings 2, `wp_after(k) ≠ wp_before(k+1)`. The cumulative gap (~0.05–0.10 percentage points per match) flowed into a residual when summing inn2 deltas — i.e., `(inn2_start) + Σ(inn2 deltas) ≠ inn2_terminal`.

The new third step fixes both at once:

- **Exponential decay across the full powerplay** (balls 1–36 of inn2). `alpha_k = exp(−(k−1) / τ)` with `τ = 7`. Ball 1 still has `alpha = 1.0` (full bridge to midpoint, identical behavior to the previous version). By ball 36 `alpha ≈ 0.007`, so the calibration's influence has effectively faded by the end of the powerplay — no visible jump at any point in the over.
- **Chained endpoints**. Define a single `boundary_wp(k)` value at each ball boundary; set `wp_after(k) := boundary_wp(k)` and `wp_before(k+1) := boundary_wp(k)`. Within balls 1–36 of inn2, `wp_after(k) = wp_before(k+1)` exactly by construction. The blend region telescopes cleanly.

The remaining residual when summing inn2 deltas is now ~0.038 percentage points per match (down from 0.086 with the linear-over-6 blend), and the leftover is structural: between-over model output gaps where a new bowler comes on and the model's "before-state" features differ slightly from its "after-state" features for the previous bowler. Closing that gap would require redefining `wp_before(k) := wp_after(k−1)` across all balls, which is a deeper methodology change deferred to a future pass.

### What changed in the rankings (post-PP-decay)

Powerplay batters gain visibly. Their early-PP impact had been damped most aggressively by the linear-over-6 blend (alpha collapsed from 1.0 to 0.0 in five balls); the smoother exponential blend distributes that damping more honestly across the full powerplay. The biggest shifts among 50+ match players, ranked by total career TILT:

| Player | Pre-PP | Post-PP | Δ |
|:--|:--:|:--:|:--:|
| V Kohli | 141 | 90 | +51 |
| RV Uthappa | 84 | 55 | +29 |
| AC Gilchrist | 61 | 36 | +25 |
| G Gambhir | 115 | 93 | +22 |
| M Vijay | 166 | 145 | +21 |

Mirror image: bowlers who closed out tight chases give back a portion of their last-ball-snap gains as the credit redistributes more evenly:

| Player | Pre-PP | Post-PP | Δ |
|:--|:--:|:--:|:--:|
| MM Sharma | 32 | 68 | −36 |
| AD Russell | 27 | 56 | −29 |
| JA Morkel | 78 | 103 | −25 |
| L Balaji | 87 | 109 | −22 |
| TG Southee | 40 | 62 | −22 |

Top of the leaderboard is essentially unchanged — Narine #1, Bumrah #2, ABD/Malinga swap #3 and #4. DJ Bravo (who entered the top 10 in the [last-ball snap](notes.html?note=last-ball-snap)) drops from #10 to #12 as some of his early-chase damping gets smoothed out.

The fix is reversible: revert this commit and the system returns to the linear-decay-over-6-balls state.

---

## Update: K=100 ensemble (issue #111) — variance under retraining

The boundary calibration above is unchanged in the May 2026 ensemble work, but its motivating diagnostic — the 0.038pp telescoping residual between blended and post-blend balls — is now small enough to live below a different and bigger noise source: **the retrain variance of the underlying LightGBM model**.

Issue #111 traced career-rank instability (AB de Villiers bouncing between #3 and #9 across daily refreshes; Bhuvneshwar Kumar above ABD on a +2-match dataset growth) to a single-model retrain producing a substantively different model on the same code, same hyperparameters, same data — only the random train/holdout reshuffle drove the change. On a fully-deterministic ball-0 feature vector (`innings=1, balls_remaining=120, wickets_in_hand=10, runs_scored=0`), two consecutive retrains predicted **0.5158 vs 0.5643** for the same input. That 4.85pp drift, compounded over 290k balls × 168 ABD matches, was enough to swing his career total by −2.29 TILT — about 14× the noise floor of seven days of daily refreshes combined.

The fix shipped in [issue #111](https://github.com/soldoutbudokan/Templates/issues/111) is a **K=100 LightGBM ensemble** averaged at inference (random states 42…141 on a fixed 90/10 holdout split locked at seed=42 forever), plus a `RETRAIN=1` env-var guardrail to prevent ad-hoc local pipeline runs from silently overwriting the committed pickle. Brier and AUC change by less than 0.01 against a single member; what changes is *stability across retrains*.

**Why this affects the boundary fix:** with retrain variance under control, the deeper telescoping methodology change discussed above (redefining `wp_before(k) := wp_after(k−1)` across all balls, [issue #110](https://github.com/soldoutbudokan/Templates/issues/110)) becomes A/B-testable for the first time. Previously, any "before/after" comparison of a chained-endpoints fix would have been swamped by the model's own retrain variance — the rankings drift from one retrain alone exceeded the residual the deeper fix is supposed to close. Under K=100, that's no longer true. Whether to ship the deeper fix is a separate decision based on how the rankings move; the methodology now permits a clean read on it.

The boundary calibration itself (steps 1, 2, 3 above) is unchanged. Median per-match cliff is still mathematically zero. The residual numbers in the *PP-decay* update section (0.086 → 0.038 abs mean per match, telescoping in the blend region exact by construction) hold under K=100 too — those numbers are fundamentally about the calibration logic, not the model behind it.

See [Why we ensemble](notes.html?note=ensemble) for the full diagnostic of the retrain-variance issue and the K=100 fix.
