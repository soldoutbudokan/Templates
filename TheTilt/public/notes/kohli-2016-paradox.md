# The 2016 Kohli Dilemma: Why His Greatest Season Survives the TILT Test

**973 runs at 152 strike rate. Four centuries. Seven fifties. And a TILT of +4.91% per match.**

Virat Kohli's 2016 IPL season is widely considered the greatest individual batting season in tournament history. The counting stats are staggering. The natural worry with a metric like TILT is that it would punish exactly this kind of season: huge volume, much of it first-innings, on a stacked lineup where credit gets shared. It doesn't. By TILT, 2016 stands as one of Kohli's best impact seasons too, clearly ahead of his celebrated 2019 (+3.44% per match), where he scored less than half the runs at a lower strike rate.

In this post, we dig into why a season that *should* have triggered every one of TILT's structural penalties came through them anyway, and what that says about the metric.

---

## The Numbers

| | 2016 | 2019 |
|:--|:--|:--|
| **Runs** | 973 | 439 |
| **Balls** | 640 | 321 |
| **Strike Rate** | 152.0 | 136.8 |
| **Matches** | 16 | 13 |
| **Batting TILT/match** | **+4.91%** | **+3.44%** |
| **Total WPA** | +78% | +45% |
| **TILT per ball** | +0.123% | +0.139% |

The 2016 season generated **three-quarters more total win probability added** than 2019, from twice the balls. And the per-ball rate is close: +0.123% vs +0.139%, a slight edge to 2019. Kohli wasn't trading much efficiency for volume in 2016. He was nearly as impactful per delivery as in his vaunted 2019, and he sustained it across twice as many of them.

That is the dilemma in a sentence. The conventional read says 2016 was the volume year and 2019 was the efficient year. TILT says they were nearly equally efficient per ball, and 2016 simply did far more of it.

---

## What Happened: Match by Match

![Kohli match-by-match TILT](plots/kohli_match_by_match.png)

Each bar is one match. Green is positive batting TILT, red is negative. Labels above each bar show runs(balls). The 2016 chart tells a volatile story: the highest highs in the database alongside a couple of heavily negative matches. The remarkable thing is that the highs were big enough to absorb the lows and still leave the season comfortably positive.

### 2016 Match Log

| Date | Inn | Runs | TILT | Note |
|:-----|:---:|-----:|-----:|:-----|
| [Apr 12](match.html?id=980907) | 1st | 75(51) | +2.99% | |
| [Apr 17](match.html?id=980921) | 1st | 79(48) | -3.13% | |
| [Apr 20](match.html?id=980927) | 1st | 33(30) | -0.48% | |
| [Apr 22](match.html?id=980931) | 1st | 80(63) | +7.23% | |
| [Apr 24](match.html?id=980937) | 1st | 100(63) | +21.17% | Century in low-scoring game |
| [Apr 30](match.html?id=980953) | 2nd | 14(17) | -17.51% | Failed chase, caught early |
| [May 02](match.html?id=980959) | 1st | 52(44) | +4.10% | |
| [May 07](match.html?id=980969) | 2nd | 108(58) | +61.70% | Best match, chase masterclass |
| **[May 09](match.html?id=980977)** | **1st** | **20(21)** | **-21.93%** | **Slow start, team rescued by KL Rahul** |
| [May 11](match.html?id=980981) | 1st | 7(7) | -7.00% | |
| [May 14](match.html?id=980987) | 1st | 109(55) | +5.58% | Century alongside ABD's 129 |
| [May 16](match.html?id=980995) | 2nd | 75(51) | +6.65% | |
| [May 18](match.html?id=980999) | 1st | 113(50) | +5.57% | Century, solidly positive TILT |
| [May 22](match.html?id=981011) | 2nd | 54(45) | +6.48% | |
| [May 24](match.html?id=981013) | 2nd | 0(2) | -5.16% | Golden duck |
| [May 29](match.html?id=981019) | 2nd | 54(35) | +12.28% | |

The chase masterclass on May 7, 108(58) for +61.70% TILT, is the single most valuable batting performance in the season and one of the largest in the database. It is exactly the kind of innings the counting stats and TILT agree on completely.

---

## The Credit Sharing Problem

![Credit sharing in Kohli centuries](plots/kohli_credit_sharing.png)

If anything was going to drag 2016 down, it was credit sharing. RCB's batting card that year read Kohli, ABD, Gayle, Watson. When two world-class batsmen tear an innings apart together, the marginal win probability of each run is lower than if one of them had carried the load alone. The chart above shows where the win-probability credit went in each century innings, with Kohli highlighted. In two of the four centuries another batsman at the other end captured a large share of what was on offer, and yet Kohli still came out positive in both.

### 109(55) vs Gujarat Lions, TILT +5.58%

AB de Villiers scored 129(52) in the same innings. By the time ABD reached his hundred, RCB's win probability was already up to 79%. Kohli at that moment was 51 off 40, with +2.64% of his TILT for the match banked; his remaining runs added another +2.94%, even with ABD having already shifted the match decisively and less WP left to capture.

This isn't because Kohli batted poorly. It's because ABD happened to be on strike for the balls that moved win probability the most. In a ball-by-ball model, whoever faces the delivery gets the credit, even when partnership dynamics created the opportunity. The notable thing is that Kohli's impact was spread across the whole innings — roughly half banked before ABD's hundred, half after — rather than evaporating once ABD took over.

### 113(50) vs KXIP, TILT +5.57%

Chris Gayle scored 73(35) at the top and captured a large slice of the early innings delta. Even so, Kohli's century registered a clearly positive +5.57% TILT. He built through the middle overs and then accelerated late, and the model rewards the runs he added even on top of Gayle's start. This is the case that most cleanly refutes the idea that a century on a stacked side is worth nothing by TILT: Gayle took the biggest slice, and Kohli still came out solidly positive.

The honest version of the credit-sharing story, then, is not that brilliance gets erased on a stacked team. It's that it gets *split*. The WP "pie" for an innings is finite, and when Gayle and ABD are taking the biggest slices, a century registers less than it would on a weaker side. But "less" still left Kohli net positive in every century he scored in 2016, including the two he shared.

---

## The -21.93% Catastrophe

On May 9th, Kohli scored 20(21), a strike rate of 95.2, before being caught early in the innings. Here's the ball-by-ball:

He started with a lucky boundary, but a sequence of dots and singles through the powerplay and into the middle overs steadily eroded RCB's position. Each dot ball in the powerplay is a small negative WP shift. Twenty-one of them compound.

KL Rahul then came in and scored 42(25) for +17.5% TILT. RCB won the match. Kohli's slow start was the problem that someone else had to fix.

This single match cost roughly as much TILT as a top-tier match earns, and it happened because the model captures exactly what matters: Kohli consumed 21 balls at a below-par rate during the phase of the innings when scoring rates matter most. What is striking is that the rest of the season was strong enough that even a -21.93% match (plus a -17.51% failed chase in April) couldn't pull the average below +4.91%.

---

## The Phase Story

![Phase breakdown](plots/kohli_phase_breakdown.png)

| Phase | 2016 Runs | 2016 TILT | 2019 Runs | 2019 TILT |
|:------|----------:|----------:|----------:|----------:|
| Powerplay | 278(230) SR 121 | **-17.14%** | 214(165) SR 130 | **+28.24%** |
| Middle | 462(304) SR 152 | +40.46% | 148(123) SR 120 | -10.82% |
| Death | 233(106) SR 220 | +55.23% | 77(33) SR 233 | +27.35% |

The powerplay is the one phase where 2016 leaks: at SR 121 across 230 powerplay balls, the accumulated dots dragged WPA down by 17.1 percentage points. But Kohli more than made it back in the middle and death overs, which added +40.5 and +55.2 points respectively, the two phases together accounting for essentially all of his +78% season WPA. The 2016 profile is a slow-ish start paid back with interest later; the 2019 profile is the mirror image, a fast powerplay followed by a sluggish middle. The two phase distributions are nearly opposite, which is part of why the volume comparison is so misleading: these were different-shaped seasons, not a high-volume one and an efficient one.

---

## The Dot Ball Tax

![Dot ball tax](plots/kohli_dot_ball_tax.png)

| | 2016 | 2019 |
|:--|:--|:--|
| Dot balls | 169 (26.4%) | 92 (28.7%) |
| Dot ball TILT penalty | **-261.3%** | **-117.4%** |
| Scoring ball TILT | +341.4% | +164.2% |

The dot percentages are similar; 2019 actually had the slightly higher dot rate. Because Kohli faced *roughly twice as many total balls* in 2016, he ate roughly twice as many dots, and the gross dot-ball penalty is correspondingly about twice as large (-261.3% vs -117.4%). But his scoring-ball TILT scaled the same way (+341.4% vs +164.2%), so the two effects net out to a similar per-ball rate. The dot-ball tax in 2016 was real and large in absolute terms; it just wasn't large *relative to the boundaries* that paid it off.

---

## It's Not Just Kohli

Plenty of high-volume seasons in IPL history do collapse under TILT, scoring lots of runs in low-leverage situations or at sub-par rates. 2016 Kohli is interesting precisely because it doesn't. It sits near the top of any per-match impact list, alongside other all-time TILT seasons. The comparison below isn't meant to show 2016 was bad; it's meant to show what separates a high-volume season that *keeps* its impact from one that bleeds it away.

### The Comparison Table

| Player | Season | Runs | SR | Matches | TILT/match | Dot % | Inn 2 % |
|:-------|:-------|-----:|---:|--------:|-----------:|------:|--------:|
| **V Kohli** | **2016** | **973** | **152.0** | **16** | **<span id="kp-kohli-2016-tpm">+4.91%</span>** | **26.4** | **32.5** |
| V Kohli | 2019 | 439 | 136.8 | 13 | <span id="kp-kohli-2019-tpm">+3.44%</span> | 28.7 | 37.1 |
| DA Warner | 2016 | 848 | 151.4 | 17 | <span id="kp-warner-2016-tpm">+7.31%</span> | 32.9 | 52.7 |
| KL Rahul | 2018 | 659 | 158.4 | 14 | <span id="kp-rahul-2018-tpm">+13.89%</span> | 34.1 | 67.8 |
| F du Plessis | 2021 | 633 | 138.2 | 16 | <span id="kp-faf-2021-tpm">+0.15%</span> | 33.4 | 36.0 |
| AM Rahane | 2013 | 488 | 106.6 | 18 | <span id="kp-rahane-2013-tpm">-4.80%</span> | 40.0 | 57.4 |
| JH Kallis | 2012 | 409 | 106.5 | 17 | <span id="kp-kallis-2012-tpm">-3.68%</span> | 37.8 | 69.0 |
| Ishan Kishan | 2022 | 418 | 120.1 | 14 | <span id="kp-ishan-2022-tpm">-6.08%</span> | 40.2 | 63.5 |

### Warner 2016: Same Volume, More TILT

Warner scored 848 runs at SR 151, on a par with Kohli's counting stats, and posted an even higher +7.31% per match. The gap isn't about efficiency per run, it's about leverage:

- **More 2nd innings batting**: Warner batted in the 2nd innings 52.7% of the time (vs Kohli's 32.5%). Second-innings balls carry structurally larger WP swings, so the same quality of batting converts to more TILT.
- **Smaller catastrophes**: Warner's worst match was less severe than Kohli's -21.93% May-9 collapse, so less of his ceiling was clawed back by disasters.

Kohli's 2016 lands a notch below Warner's chiefly because two-thirds of his balls came in the first innings, where the WP pie is smaller, not because his batting was lower-impact ball for ball.

### KL Rahul 2018: The TILT Machine

659 runs, <span id="kp-rahul-2018-tpm-prose">+13.89%</span> per match, one of the highest single-season TILT rates in the database. Rahul batted in the 2nd innings 67.8% of the time, more than two-thirds of his balls in higher-leverage chase situations. This is partly a function of KXIP's batting order and toss decisions, but it compounds: more chase innings = more TILT opportunity. It is the clearest illustration of the lever Kohli's first-innings-heavy 2016 didn't get to pull.

### du Plessis 2021: The Orange Cap Mirage

Faf du Plessis won the Orange Cap in 2021 with 633 runs at SR 138. His TILT was **+0.15% per match**, essentially zero. He scored a lot of runs, but disproportionately in low-leverage situations. His dot rate (33.4%) and a batting-first lean (36% inn 2 balls) meant his volume translated to almost no impact by TILT's measure. This is the high-volume-collapses-under-TILT pattern that 2016 Kohli pointedly avoided.

### Rahane 2013 & Kallis 2012: The Accumulator Penalty

Both scored 400+ runs at sub-107 strike rates with 38-40% dot ball percentages. They accumulated, and the model correctly identifies that at SR 106, you're hurting your team on most deliveries. Every ball at below the expected scoring rate is a small negative WP shift. The contrast with Kohli's SR 152 is the whole point: volume only survives TILT if the rate clears the bar.

---

## Runs vs Impact: The Full Picture

![Runs vs TILT scatter](plots/runs_vs_tilt_scatter.png)

Every dot is an IPL season with 400+ runs. The correlation between runs scored and TILT per match is **weak positive** (r = 0.27). Scoring lots of runs helps, but the *rate* and *timing* matter far more than the volume.

Kohli 2016 sits in the upper region of the cloud: high runs and high TILT together, which is a rarer combination than it sounds. The seasons up there tend to be from players who scored fast, chased often, and didn't have too many catastrophic matches. 2016 qualifies on the first two and survives despite the third.

---

## Kohli's Career Arc

![Kohli career timeline](plots/kohli_career_timeline.png)

The purple line shows runs by season. The bars show batting TILT per match (2016 highlighted). The 2016 runs peak corresponds to one of his strongest TILT seasons as well; in this case the counting peak and the impact peak largely line up, which is not something we can say for every great-by-the-numbers campaign.

---

## The Verdict: Is the Model Right?

TILT is picking up something real and, in this case, flattering about 2016. The 108(58) chase, the SR 220 in the death overs, the consistently positive centuries: the model rewards all of it because the runs came at rates and moments that actually moved matches. The catastrophic matches are real too. 20(21) in the powerplay when your team needs a fast start is genuinely harmful, and the heavy penalty the model applies isn't the model misbehaving, it's the model doing exactly what it's designed to do. And credit sharing reflects reality: when ABD scores 129 and Kohli scores 109 in the same innings, the marginal impact of each run is lower than if one of them had carried the load alone, even though Kohli still came out positive.

Where TILT arguably still leaves value on the table is on the first-innings and volume sides. The first innings has a structural ceiling: the total WP available in innings 1 is limited and gets divided across all batsmen, so on a stacked 2016 RCB lineup the pie is split several ways while a single dominant batsman on a weaker team can capture everything. Two-thirds of Kohli's 2016 balls came in that lower-ceiling first innings, which is most of the gap to Warner. And volume should count for something in its own right: adding +0.123% per ball across 640 balls is a different achievement from adding +0.139% across 321, and a per-match number doesn't fully capture that. Both of these, if anything, suggest TILT may still slightly *understate* 2016 rather than overstate it.

---

## So Is 2016 Overrated?

No, and TILT agrees. It was a historically great season by the counting stats, and it remains one of Kohli's best by impact too: nearly as efficient per ball as his celebrated 2019, sustained across twice the workload, and net positive even after two genuinely costly failures.

The old worry, that an impact metric would expose 2016 as empty volume, simply doesn't hold up. The credit-sharing, dot-ball, and first-innings-ceiling effects are all real, and they shave something off the top. But they don't erase the season; they explain why it lands a notch below the most leverage-rich campaigns like Warner's 2016 or Rahul's 2018 rather than at the very summit.

The counting stats and TILT mostly agree here. Where they differ, on the value of first-innings volume, the disagreement is small and points the same way. That rare consensus is what makes 2016 the cleanest case for the metric, not a paradox against it.
