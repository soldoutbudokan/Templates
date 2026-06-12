# The Importance of Venue

**How much of a batsman's TILT is coming from *where* they played? We relocate six player-cohorts' home matches into the model, one venue swap at a time, and read off the delta.**

Cricket fans carry strong priors about grounds. M Chinnaswamy is a batsman's paradise; Chepauk rewards patience and spin; Wankhede swings in the evening. The TILT model has `venue` as a feature in its win-probability classifier, so those narratives are implicitly priced in. This post makes the pricing explicit.

---

## What we're actually measuring

The experiment is simple, but it's worth being precise about what it does and doesn't do.

We take a cohort — say, RCB's eight home matches in IPL 2016 — and in the feature frame, overwrite `venue` to a different ground (Chepauk, Wankhede, anything on a whitelist). Then we re-run the exact same trained classifier through `compute_ball_deltas`, the function underneath the TILT numbers on the leaderboard. Every ball-state — over, runs_scored, wickets_in_hand, required_run_rate, recent_wickets — is unchanged. The only thing that moves is the venue label.

This is a *model-sensitivity* experiment, not a causal one. We are not re-simulating alternate matches; we are asking the trained model "if you'd been told these deliveries happened at Chepauk instead of Chinnaswamy, what would you have scored them as?" The result measures the model's **reliance on the venue feature** — which is correlated with but distinct from real-world venue effects.

A few constraints we enforce:

- **9-venue whitelist**: M Chinnaswamy, Wankhede, Eden Gardens, Chepauk, Rajiv Gandhi (Uppal), PCA Mohali, Arun Jaitley, Sawai Mansingh, Narendra Modi. All have substantial pre-2021 data and are in the model's learned category list (the model carries 38 venue categories in total; all 9 targets are present). We exclude COVID bio-bubble venues (DY Patil, Sharjah, Dubai, Sheikh Zayed) as swap targets because their `season_numeric × venue` interactions are unstable.
- **DLS-affected matches** are dropped from every cohort (4,131 DLS-affected rows are filtered out of the feature frame before anything runs).
- **Raw (unshrunk) TILT/match** throughout. The empirical-Bayes shrinkage (k ≈ 4.8) used on the main leaderboard is monotone in the observed value at fixed n, so rankings and signs don't flip; shrunk figures would compress magnitudes ~10–15%.
- **Pre-boundary-calibration baseline.** The venue figures are the model's *pre-boundary-calibration* re-score: `compute_ball_deltas` is run without the per-match boundary-recalibration post-step (issue #62) that the published leaderboard applies. The measured quantity is the **delta** between a venue and the swap, an apples-to-apples comparison where the calibration step would cancel anyway. The consequence is that absolute *first-innings* figures here match the player pages, but absolute *second-innings* (chase) figures can read lower than the calibrated per-match TILT shown elsewhere on the site — Kohli's May-7 chase is the clearest example below.
- **Replay-sanity gate**: before any counterfactual, we re-run the model on the untouched cohort and confirm our delta-sum matches the committed `deltas.parquet` to within 1e-9. It does (ABD 2016 cohort: 0.00e+00 difference).

---

## Start with the example everyone asks about: ABD, RCB 2016, moved to Chepauk

In IPL 2016, RCB played eight home matches at M Chinnaswamy. AB de Villiers faced 216 legal deliveries across those eight games, scored 379 runs, and produced a raw batting TILT per match of **+11.40%** — a genuinely elite eight-match run.

Relocate those deliveries to Chepauk and re-score. His TILT/match *rises* to **+12.20%**. A lift of **+0.80 percentage points**.

![ABD match-by-match, real vs swapped to Chepauk](plots/venue_abd_2016_match_by_match.png)

That is the opposite of what the cricketing caricature predicts. The caricature says Chepauk, with its slower surface and spin, would punish a dominant middle-overs hitter; we'd expect ABD's number to collapse. Instead it edges *up*, and only modestly. His 129(52) against Gujarat Lions on May 14 actually slips, from +26.53% to +23.97%; his 79(47) on May 24 edges up, from +68.71% to +70.12%. The match-level moves are small and mixed — the cohort sum nets out to a gentle lift, not a collapse.

The modesty gets more interesting once you see the full sweep. Across the nine whitelisted targets, his TILT ranges from **+9.81%** (Sawai Mansingh) to **+14.48%** (Mohali). Chinnaswamy sits just below the middle of that band, with Wankhede a hair under it and Chepauk above. The venues that *drop* him are the slower or lower-scoring ones (Sawai Mansingh, Uppal); the venues that *lift* him are the northern, typically higher-scoring grounds (Mohali, Narendra Modi, Arun Jaitley). The lever for ABD 2016 spans roughly 4.7pp across those nine grounds — real, but Chepauk lands close to where Chinnaswamy does.

---

## The same matches, for Kohli: a bigger lift in the same direction

Kohli batted in those same eight matches. His raw TILT/match at Chinnaswamy is **+8.08%** — well below ABD's, and well documented elsewhere as one of his lower per-match rates for a famously high-volume season.

Relocate those eight matches to Chepauk and his TILT climbs to **+10.57%**. A rise of **+2.50pp** — the same sign as ABD's lift, but roughly three times the magnitude.

![ABD vs Kohli venue lever — same 8 matches, same swap](plots/venue_abd_vs_kohli_2016.png)

Across the full nine-venue sweep, Kohli's TILT/match ranges from **+8.19%** (Sawai Mansingh) up to **+10.97%** (Arun Jaitley), with his actual home ground (+8.08%) sitting at the very bottom of the band. Every alternate target lifts him. The biggest single-match movement: his 108(58) chase against Rising Pune on May 7 (the one the [paradox post](kohli-2016-paradox.md) calls his "chase masterclass") gains **+16.39pp** of TILT from the swap — moving from a +46.81% baseline to +63.20% if relocated to Chepauk. (That +46.81% is the *pre-calibration* baseline; the published, boundary-calibrated figure on Kohli's player page reads higher — around +66% — so read this row as a venue *delta* of sixteen-odd points, not as a competing absolute.) Sixteen points, from one venue swap, on a match that was already his best of the season.

So what does it mean that the same swap lifts both batsmen, but Kohli three times as much?

Roughly, this: the model has learned that Chinnaswamy is so high-scoring that each run there carries a smaller WP shift; moving runs to a *harder* venue inflates their per-ball weight. ABD's 2016 innings were mostly *already* dominant — by the time a run lands, WP is often already near 1, so relocating those late runs to a harder venue has little room to help (it can even hurt, as on May 24). Kohli's 2016 matches, full of slow starts and gradual climbs, benefit enormously from a venue where each ball counts more: his scoring gets *more* room to move WP before the innings caps out. Both get lifted because Chinnaswamy is the friendliest prior in the band; Kohli gets lifted more because the shape of his innings leaves more headroom.

The venue lever, in other words, is not a uniform multiplier on performance. It's specific to the shape of the innings, which is specific to the player.

---

## Gayle: the sweep spans 3.5 percentage points

Gayle played 38 home matches for RCB across his career, 2011 to 2017. Real career TILT at Chinnaswamy is **+0.69%/match**. Relocate to Wankhede: **−0.56%** (−1.25pp). Relocate to Chepauk: **+2.94%** (+2.25pp).

His full nine-venue sweep:

| Target venue | Gayle TILT/match |
|:---|:---:|
| M Chinnaswamy Stadium (home) | **+0.69%** |
| Wankhede Stadium | −0.56% |
| Sawai Mansingh Stadium | +0.62% |
| Arun Jaitley Stadium | +1.36% |
| Eden Gardens | +1.38% |
| PCA IS Bindra Stadium, Mohali | +1.88% |
| Rajiv Gandhi International, Uppal | +1.89% |
| Narendra Modi Stadium, Ahmedabad | +2.84% |
| MA Chidambaram Stadium, Chepauk | **+2.94%** |

The range is 3.5pp — not small for a career-level cohort with 38 matches. And the shape is odd: the venue that looks structurally most similar to Chinnaswamy (flat, batsman-friendly Wankhede) *suppresses* Gayle's TILT below his home figure; the lower-scoring venues lift him. Again the logic is about where the WP ceiling sits relative to his scoring: Gayle on a Chinnaswamy-clone spends a lot of balls at states the model already considers resolved.

---

## Dhoni at Chepauk: the fortress that isn't

MS Dhoni played 65 matches at Chepauk as a CSK player. His real TILT/match there is **+3.14%**. Swap to Eden: **+2.33%** (−0.81pp). Swap to Wankhede: **+2.31%**. Swap to Chinnaswamy: **+2.36%**. The full sweep spans **+2.22% to +3.61%** — 1.4pp, with Chepauk near the top of his band (only Mohali reads higher).

Essentially, the venue lever doesn't find Dhoni. Whatever TILT he generates, he generates largely from match state and game phase, not from the ground. The "fortress Chepauk" narrative is intuitive to anyone who watched CSK win there routinely, but the routine winning came from the team and Dhoni's role within it — and what little the model *does* attribute to the ground mostly points the conventional way (his home reads near the top), inside a thin 1.4pp band.

---

## Rohit at Wankhede: the home counter-example

Rohit Sharma's 88 career home matches for Mumbai Indians at Wankhede produce a raw TILT/match of **+0.08%**. Swap those matches around and:

| Target venue | Rohit TILT/match |
|:---|:---:|
| **Wankhede Stadium (home)** | **+0.08%** |
| Eden Gardens | +0.13% |
| Sawai Mansingh Stadium | +0.47% |
| M Chinnaswamy Stadium | +0.58% |
| Arun Jaitley Stadium | +0.60% |
| Rajiv Gandhi International, Uppal | +0.86% |
| Narendra Modi Stadium, Ahmedabad | +0.98% |
| PCA IS Bindra Stadium, Mohali | +1.01% |
| MA Chidambaram Stadium, Chepauk | +1.45% |

All eight alternate venues lift him. His actual home ground sits at the very bottom of the sweep. By the model's reckoning, Rohit's Wankhede "home advantage" is, at the margin, a drag on his TILT — he'd score higher on the leaderboard if the same 88 matches had been played at Chepauk (+1.37pp).

This is not a claim that Rohit *is* worse at Wankhede; his actual runs happened there and the counterfactual can't do anything about that. It's a claim that the model reads his ball-state trajectories as slightly more impressive against a harder venue prior than against Wankhede's.

---

## A data-driven pick: TH David, the sporadic home-ground case

To widen the roster past famous names we filtered the dataset for players with 30–150 career matches, a primary home venue in the 9-whitelist, and a non-trivial home-ball share — then ranked by observed home-minus-away per-ball TILT gap. Top of the list this time: Tim David (Mumbai Indians, 59 matches, Wankhede as home).

He has 13 home matches in the cohort — still modest, caveat emptor — but the lever is real and visible.

| Target venue | TH David TILT/match |
|:---|:---:|
| Eden Gardens | +12.94% |
| M Chinnaswamy Stadium | +13.20% |
| **Wankhede Stadium (home)** | **+13.21%** |
| Sawai Mansingh Stadium | +13.37% |
| Narendra Modi Stadium, Ahmedabad | +13.41% |
| MA Chidambaram Stadium, Chepauk | +13.98% |
| PCA IS Bindra Stadium, Mohali | +15.33% |
| Rajiv Gandhi International, Uppal | +16.09% |
| Arun Jaitley Stadium | +16.33% |

His actual home ground sits in the lower half. Swap to Chepauk: **+13.98%**, a lift of **+0.77pp** off the +13.21% home figure. Swap to Arun Jaitley: +16.33%, the top of the band. The full sweep spans **+12.94% to +16.33%** — a 3.4pp range on thirteen relocated matches, and like the 2016 RCB cohort, the *harder* venues are the ones that lift this Wankhede regular.

---

## All six, in one picture

![Dumbbell: venue lever across all six cohorts](plots/venue_dumbbell_six_cohorts.png)

Sorted by absolute delta. Kohli 2016 at the top — a +2.50pp lift on the cohort whose headline number reads worst. Rohit's +1.37pp Chepauk lift and Gayle's −1.25pp Wankhede drop follow, then Dhoni's −0.81pp Eden swap, ABD 2016's gentle +0.80pp, and TH David's +0.77pp Chepauk lift.

The thing worth staring at is the two rows for the 2016 RCB cohort. They are *the same eight matches*, swapped to *the same target* (Chepauk), and they move in *the same direction* — both up — but by very different amounts: Kohli gains roughly three times what ABD does. The venue lever is one signed force here; its size is set by the shape of each batsman's innings.

---

## What this does and doesn't measure

The venue lever quantified here is the model's reliance on the venue categorical. That is related to, but not the same as, real-world venue effects. Four specific limits worth stating:

- **Opposition is held constant in the model's view but not in reality**. ABD's Chinnaswamy matches were against whatever bowlers RCB faced at home — which is not a random sample of IPL bowling attacks. Relocating those matches holds opposition fixed, which is convenient for the counterfactual but not what would actually happen if RCB played their home slate somewhere else.
- **Era drift is baked in**. The model has more 2017–2024 data for Narendra Modi, Ekana, Holkar; earlier-era venues dominate pre-2020 splits. When a 2016 cohort gets swapped to Narendra Modi, you're partly measuring the 2021+ scoring regime, not the 2016 one. Hence the 9-venue whitelist — but even within it, the signal isn't era-neutral.
- **"Relocating the match" ≠ "relocating the player"**. When we swap venue, we swap it for both innings. We're asking the model to rescore the *match*, not to imagine the player in a different team context.
- **It's a model probe, not a physics experiment**. The venue categorical encodes whatever correlation the model's tree splits found useful — pitch behaviour, yes, but also squad composition at that ground, local bowler economies, toss decisions, and the specific batsmen who happened to play there. The lever is measuring the model's belief about venue, which is correlated with but distinct from venue itself.

---

## Takeaway

Venue is a real feature in TILT, and its lever is player-specific, often small, and occasionally counterintuitive.

For Dhoni the lever barely exists, and what there is points the conventional way (Chepauk tops his band by a hair). For Rohit it exists but points the wrong way — his Wankhede home isn't where the model rates him highest. For Kohli 2016 it's large and points up; for ABD 2016 it's small and points up too, in the very direction the spin-track stereotype says it shouldn't. For Gayle the lever's magnitude depends entirely on which alternate venue you pick: a 3.9pp range across the nine-venue sweep, with the lower-scoring targets *lifting* him.

The cleanest methodological finding isn't in any individual number. It's that the same eight matches, under the same venue swap, lift two batsmen's TILT by very different amounts — Kohli roughly three times ABD. That's the signature of a lever that's being filtered through the shape of each player's innings, not a lever that scales uniformly with "how friendly is this ground." It also exposes a quieter surprise: the model reads Chinnaswamy as so batsman-friendly that almost *any* harder venue inflates the per-ball WP weight, so relocating RCB's 2016 home slate to Chepauk lifts both men rather than punishing either.

If you came to this post looking for a clean story about venue mattering, the cleanest version is: venue matters for TILT, but it matters to each player in their own way, and sometimes — for a dominant Chinnaswamy cohort — a "harder" ground reads as *more* impressive, not less.

---

*Source: `notebooks/venue_importance_analysis.py`. Every number above is printed by the notebook's summary block. The venue figures are the model's pre-boundary-calibration re-score, used as a clean apples-to-apples baseline for each swap; the measured quantity is the delta. Replay-sanity gate (commit-equivalence to `deltas.parquet`) passed at 0.00e+00 before any counterfactual was run.*
