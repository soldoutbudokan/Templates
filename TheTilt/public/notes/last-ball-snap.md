# Snapping the Final Ball

**The win-probability model never knew the match was over. A six off the last ball that wins the chase scored +5pp instead of +50pp against TILT — until now.**

The win-probability classifier underlying TILT is trained on intra-innings states. It sees `runs needed`, `balls remaining`, `wickets in hand`, and the rest, and predicts a probability that the batting team will win. What it doesn't see — and was never trained on — is the *terminal* state: balls remaining = 0, match over. So when the chasing team hits the winning run on the very last ball, the model returns a `wp_after` of "this state looks like ~93% to win" rather than the actual answer of 100%.

Across all 1,206 IPL matches with two innings, the model's `wp_after` on the final ball averages **0.928** when the chase succeeded and **0.161** when it failed. Truth: 1.0 and 0.0 respectively. So a winning hit was credited with a `delta_wp` ~7 percentage points smaller than reality, and a chase-ending wicket was credited ~16pp smaller. The most consequential ball of every match was being systematically undercounted.

This post describes the fix: a deterministic snap of `wp_after` on the final ball of innings 2 to its actual outcome. Touches one ball per match (~1,200 balls). Career rankings reshuffle in exactly the directions you'd expect — death-overs specialists rise, "finishers who didn't finish" fall.

---

## The fix in one paragraph

After the model has scored every ball and the boundary calibration has aligned the inn1↔inn2 seam, set `wp_after` on each match's final innings-2 ball to:

- `1.0` if the chasing team won
- `0.0` if the chasing team lost
- `0.5` if the regulation result was a tie (these go to a super over, but the regulation match is genuinely a tie at the moment its 240th legal ball is bowled)

Recompute `delta_wp = wp_after − wp_before` for that one ball. Done. The structurally analogous step exists already at the inn1↔inn2 seam ([the boundary calibration](notes.html?note=innings-boundary)) — this just extends the same idea to the match terminus.

Detection of regulation ties is: `inn1 total runs == inn2 final cumulative runs`. There are 16 such matches in the data.

---

## What changed in the rankings

### Top 10 by Total Career TILT

| Rank | Before | After |
|:--|:--|:--|
| 1 | SP Narine | **SP Narine** |
| 2 | JJ Bumrah | **JJ Bumrah** |
| 3 | AB de Villiers | SL Malinga ⬆ |
| 4 | SL Malinga | AB de Villiers ⬇ |
| 5 | Rashid Khan | B Kumar ⬆ |
| 6 | YS Chahal | Rashid Khan ⬇ |
| 7 | CH Gayle | YS Chahal ⬇ |
| 8 | B Kumar | JC Buttler ⬆ |
| 9 | JC Buttler | CH Gayle ⬇ |
| 10 | DW Steyn | **DJ Bravo** 🆕 |

DJ Bravo entering the all-time top 10 from a previous rank of #62 is the headline. He was IPL's most famous death-overs specialist and the model was systematically undercrediting him by exactly the magnitude this fix corrects: every closed-out tight win that ended on his bowling now gets the full "chased team's WP went from 70% to 0%" credit instead of "70% to 16%."

### Biggest gainers (50+ matches)

| Player | Before | After | Δ |
|:--|:--:|:--:|:--:|
| R Vinay Kumar | 119 | 62 | +57 |
| Avesh Khan | 131 | 76 | +55 |
| HV Patel | 116 | 66 | +50 |
| JD Unadkat | 142 | 96 | +46 |
| **DJ Bravo** | **55** | **10** | **+45** |
| L Balaji | 130 | 87 | +43 |
| AD Russell | 58 | 27 | +31 |
| TG Southee | 70 | 40 | +30 |

Every name on this list is a death- or closing-overs specialist — bowlers who pitched up at the end of tight chases. The pattern is mechanical: their highest-WP-impact moments were exactly the moments the model was failing to snap to the true outcome.

### Biggest losers (50+ matches)

| Player | Before | After | Δ |
|:--|:--:|:--:|:--:|
| RV Uthappa | 28 | 84 | −56 |
| **MS Dhoni** | **33** | **67** | **−34** |
| **DA Miller** | **35** | **69** | **−34** |
| MA Agarwal | 107 | 145 | −38 |
| AR Patel | 78 | 116 | −38 |
| JP Duminy | 108 | 143 | −35 |
| S Dube | 97 | 132 | −35 |

Dhoni and Miller dropping 34 spots each is the other half of the story. The "finisher" reputation rests partly on tight wins, but partly on tight losses where the player was at the crease at the end. The model used to credit them with a chasing-team WP of ~16% at the moment the match ended in defeat — credit that doesn't actually exist. The fix reassigns that ~16pp per closing failure to where it belongs (which is nowhere — it's the chasing team's loss, full stop).

This isn't a verdict on Dhoni or Miller as players. It's a correction to a specific arithmetic error in how their losing-chase appearances were being scored.

---

## Why this is a real bug

It's worth being explicit about why the pre-fix state was wrong rather than just "an approximation."

The win-probability model is trained as a binary classifier on `(state, batting_team_won)`. Its training labels are the actual match outcomes. So at every state during a chase, the model is *learning* to predict the eventual binary outcome. The closer to the match end, the closer the model's prediction *should* converge to the actual outcome — and it almost does, but not quite. It maxes out at ~93% because:

- The model never sees the post-final-ball state during training (there is no such ball-level row).
- The model's features at the *true* final ball don't encode "we're done" — they encode "1 ball remaining" or "0 balls remaining + score reached" indirectly through `runs_needed` and `required_run_rate`.
- LightGBM's tree splits aren't designed to push individual leaf predictions to exactly 1.0 or 0.0; they max out at the dataset's empirical winning rate within the relevant leaf, which on close matches is high but never 100%.

So the fix isn't fighting the model — it's adding the one piece of information the model can't have: *the match is over and this is what happened*. Which is exactly the same kind of edit the boundary calibration makes at the innings break.

---

## What didn't change

- **The model itself.** No retraining. The committed `models/win_prob_lgbm.pkl` is identical.
- **All non-final balls.** Every ball except the literal last delivery of innings 2 in each match retains its raw model output.
- **The cliff at the innings boundary.** Already zero ([per #62 + #71](notes.html?note=innings-boundary)).
- **The chart on match pages.** It already shows the snapped `wp_after` because the chart reads from the same calibrated `deltas.parquet`. The visual consequence: the chart's final point now lands at exactly 100% or 0% (or 50% on ties), which is what it should always have done.

---

## What it cost

About 50 lines added to `pipeline/compute_tilt.py` (a new `apply_match_terminal_snap` function, called between `apply_boundary_calibration` and `aggregate_player_tilt`). Touches one row per match; reversible by dropping the function call. No model retrain, no feature changes, no new training data. The pipeline runtime is unchanged.

---

## What's next

This sanity check incidentally exposed a separate issue: the boundary calibration's alpha-decay across balls 1–6 of innings 2 introduces ~0.05–0.10pp telescoping gaps between adjacent ball pairs, because each ball's blended endpoints are independent rather than chained. The next methodology pass will switch to an exponential decay across the full powerplay (balls 1–36) and chain the blend so that `wp_after(ball k) == wp_before(ball k+1)` exactly. That should drive the inn2 telescoping residual to zero and remove the second-over visual jump that motivated this whole investigation.
