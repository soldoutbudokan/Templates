# Why TILT Underrates All-Rounders

**The most famous IPL all-rounder ranks 293rd. Maxwell, the highest-rated career all-rounder, is at 92. Zero all-rounders make the top 50. This is structural, not bias.**

If you sort the TILT leaderboard and scroll, something is missing. Russell, Pollard, Stokes, Watson, Pandya — the players who decide T20 matches by their dual-threat — sit far below where their reputation suggests they belong. A model that is supposed to measure how much a player tilts the game produces a top-50 with batters and bowlers and almost no all-rounders.

This is not a bug. It's the math of measuring two roles at once.

---

## The Pattern

Of the 458 players with at least ten matches, 45 (9.8%) qualify as all-rounders under TILT's classification — at least 50 balls faced and 50 balls bowled, and a bat:bowl ratio between 0.3 and 3.0. If all-rounders performed like the rest of the pool, you'd expect about five of them in the top 50.

There is exactly one. JR Hopes, ranked 42nd, on a 20-match sample.

| Top-N | All-rounders | Expected at pool share | Ratio |
|:--|:--|:--|:--|
| 10 | 0 | 1.0 | 0.0x |
| 25 | 0 | 2.4 | 0.0x |
| 50 | 1 | 4.9 | **0.2x** |
| 100 | 4 | 9.8 | 0.4x |
| 200 | 14 | 19.6 | 0.7x |

![Top-N share by role](plots/all_rounders_topn_share.png)

The chart shows the gap. Across every depth — 10, 25, 50, 100, 200 — all-rounders are underrepresented. The longer the list gets, the closer they come to their pool share, but they never catch up.

---

## The Distributions

![TILT distribution by role](plots/all_rounders_distribution.png)

Look at the boxes. Batters and bowlers have wide distributions with long upper tails — their best players sit well above the median. The all-rounder box is compressed: the 75th percentile is barely positive and the median is below zero.

| Role | n | Mean TILT/match | p75 | p90 | Top-quintile mean |
|:--|--:|--:|--:|--:|--:|
| Batter | 177 | -0.0060 | 0.014 | 0.030 | **0.040** |
| All-rounder | 45 | -0.0128 | 0.004 | 0.014 | **0.016** |
| Bowler | 236 | -0.0033 | 0.019 | 0.038 | **0.041** |

The top quintile of each role tells the story most cleanly. The best 35 batters average 0.040 TILT/match. The best 47 bowlers average 0.041. The best 9 all-rounders average **0.016** — about 40% of either specialist cohort.

That's the headline. An all-rounder at the top of their role is not even close to a specialist at the top of theirs.

---

## Why? The Cancellation Problem

![Cancellation scatter](plots/all_rounders_cancellation.png)

This is every match an all-rounder has played, plotted by their batting TILT (x-axis) versus their bowling TILT (y-axis). The dashed line is the cancellation line — points below it have a positive bat and negative bowl, or vice versa, where the two contributions partially cancel.

44.3% of all-rounder matches are mixed-sign. They have a good batting day and a bad bowling day, or the reverse. Specialists never have this problem — a pure batter only has a bat row, a pure bowler only a bowl row.

The drag is quantifiable. If you treat each all-rounder match as if the bat and bowl contributions belonged to two different specialists, the average absolute impact is **0.126 TILT**. The actual combined value (where the contributions add up, including cancellation) is **0.105 TILT**. That's a **16.9% reduction** — pure cancellation drag, not a measurement issue.

| Match outcome | Share of all-rounder matches |
|:--|--:|
| Both bat and bowl positive | 9.7% |
| Both bat and bowl negative | **45.9%** |
| Mixed sign (cancellation) | 44.3% |

The 45.9% both-negative number is even worse than the cancellation. Almost half of an all-rounder's matches are *bad in both roles*. A specialist only has one role to be bad at; an all-rounder has two surfaces to fail on. Their floor is lower because they're exposed twice.

---

## The Counterintuitive Twist

This part is interesting. If you only look at the top tail of single-match performances, all-rounders aren't underrepresented at all.

![Match-TILT distribution](plots/all_rounders_match_distribution.png)

| Role | % matches with TILT ≥ 0.30 (GOAT-tier) |
|:--|--:|
| Batter | 3.24% |
| **All-rounder** | **3.78%** |
| Bowler | 2.77% |

All-rounders actually produce GOAT-tier matches at a slightly higher rate than batters or bowlers. When everything clicks — they bat 50 off 25 *and* take 2/15 — the combined TILT is enormous, and they get more "everything clicks" days than specialists do "perfect bat" or "perfect bowl" days.

So the underrating is not in their best matches. It's in the floor. All-rounders' explosive matches are normal-frequency, but their bad matches are worse.

You can see the same thing in the per-match standard deviation:

| Role | Match-TILT mean | Match-TILT std | p99 |
|:--|--:|--:|--:|
| Batter | -0.0013 | 0.130 | 0.459 |
| All-rounder | -0.0064 | **0.151** | **0.515** |
| Bowler | 0.0069 | 0.149 | 0.423 |

All-rounders have the highest variance and the highest p99. They are not boring middle-of-the-pack contributors. They are *boom-or-bust* contributors whose mean is dragged below zero by deep busts.

---

## Where the Stars Actually Sit

For a sense of scale, here's where the famous IPL all-rounders rank in career TILT/match:

| Rank | Player | Matches | TILT/match |
|:--|:--|--:|--:|
| 92 | GJ Maxwell | 139 | 0.0199 |
| 141 | AD Russell | 133 | 0.0116 |
| 144 | KA Pollard | 179 | 0.0113 |
| 153 | BA Stokes | 45 | 0.0089 |
| 168 | SR Watson | 144 | 0.0063 |
| 293 | **HH Pandya** | 157 | -0.005 |

Maxwell — at rank 92, with the only career-length sample above 100 matches in the top all-rounder block — is the leaderboard's best argument *for* an all-rounder. Hardik Pandya, the IPL's most decorated active all-rounder, sits at 293, below the median of qualified players. Russell and Pollard, two of the most decisive death-overs hitters in IPL history, both rank in the 140s.

By comparison, the top of the leaderboard is full of specialists who concentrate their impact: Sohail Tanvir's bowling, AB Mhatre's and Vaibhav Suryavanshi's batting cameos, Priyansh Arya's small-sample heroics. None of them are diversified across roles. None of them are dragged down by a bad bowling day after a good bat.

---

## The Philosophical Reading

What does TILT actually measure? It measures how much a player's actions move win probability. That's a single-axis measurement. When a batter walks off with 80 not out chasing 200, the model rolls all of his impact into one number. When an all-rounder bats 25 off 18 *and* bowls four overs for 35, the same model splits him across both phases. Each contribution is competing against a more concentrated specialist on its own axis, and the all-rounder's bat day is being averaged with their bowl day.

Cricket fans reward versatility because a great all-rounder is genuinely harder to build a team around — he's a slot freed up, a batting depth bonus, a fifth bowler when the captain needs one. None of that shows up in a per-ball win-probability swing. The model sees only the events. And in events, focused specialists win.

You can argue this is a feature (a top-10 should reward concentrated dominance) or a bug (the rankings should value role flexibility). The data here doesn't take a side. It just makes clear that **the gap is structural** — half from cancellation in mixed-sign matches, half from a worse floor across both roles. It is not the model picking favourites or having a feature missing.

If you want to see all-rounders fairly, the per-match GOAT lists are the place to look. Their best games are right there with the specialists. The career averages just have a different equation working against them.

---

*The numbers above come from `notebooks/all_rounders_analysis.py`. Cohort sizes and headline ratios will drift slightly across model retrains; the structural pattern is robust.*
