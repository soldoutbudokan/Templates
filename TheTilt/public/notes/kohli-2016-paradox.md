# The 2016 Kohli Dilemma: Why His Greatest Season Isn't His Best TILT Season

**973 runs at 148 strike rate. Four centuries. Seven fifties. And a TILT of +4.31% per match.**

Virat Kohli's 2016 IPL season is widely considered the greatest individual batting season in tournament history. The counting stats are staggering. But his TILT ranks it below his 2019 season (+6.63%), where he scored less than half the runs at a lower strike rate.

This isn't a bug. It's TILT doing what it's designed to do — and exposing some genuine blind spots in the process.

---

## The Numbers

| | 2016 | 2019 |
|:--|:--|:--|
| **Runs** | 973 | 439 |
| **Balls** | 655 | 325 |
| **Strike Rate** | 148.5 | 135.1 |
| **Matches** | 16 | 13 |
| **Batting TILT/match** | **+4.31%** | **+6.63%** |
| **Total WPA** | +69.0% | +86.2% |
| **TILT per ball** | +0.105% | +0.265% |

The 2019 season generated **more total win probability added** from half the balls. Kohli's TILT *per ball* was 2.5x more efficient in 2019.

---

## What Happened: Match by Match

![Kohli match-by-match TILT](plots/kohli_match_by_match.png)

The 2016 chart tells a volatile story. The highest highs — but also two catastrophic negative matches that dragged the average down. The 2019 chart is spikier but with fewer deep negatives.

### 2016 Match Log

| Date | Inn | Runs | TILT | Note |
|:-----|:---:|-----:|-----:|:-----|
| Apr 12 | 1st | 75(51) | +5.17% | |
| Apr 17 | 1st | 79(48) | +10.66% | |
| Apr 20 | 1st | 33(34) | +1.43% | |
| Apr 22 | 1st | 80(70) | +10.23% | |
| Apr 24 | 1st | 100(64) | +23.84% | Century in low-scoring game |
| Apr 30 | 2nd | 14(17) | -16.79% | Failed chase, caught early |
| May 02 | 1st | 52(44) | +1.52% | |
| May 07 | 2nd | 108(58) | +43.65% | Best match — chase masterclass |
| **May 09** | **1st** | **20(21)** | **-39.77%** | **Slow start, team rescued by KL Rahul** |
| May 11 | 1st | 7(8) | -1.46% | |
| May 14 | 1st | 109(55) | +2.18% | Century overshadowed by ABD's 129 |
| May 16 | 2nd | 75(51) | +22.07% | |
| May 18 | 1st | 113(51) | -0.07% | Century, zero TILT (Gayle took all credit) |
| May 22 | 2nd | 54(45) | +9.87% | |
| May 24 | 2nd | 0(2) | -3.22% | Golden duck |
| May 29 | 2nd | 54(36) | -0.36% | |

Two matches define the paradox: **two centuries that registered near-zero TILT**.

---

## The Credit Sharing Problem

![Credit sharing in Kohli centuries](plots/kohli_credit_sharing.png)

### 109(55) vs Gujarat Lions — TILT: +2.18%

AB de Villiers scored 129(53) in the same innings. The total innings WP delta was +31.55%. ABD captured +33.06% of it. Kohli, despite a century off 55 balls, got +2.18%.

This isn't because Kohli batted poorly. It's because ABD happened to be on strike for the balls that shifted win probability the most. In a ball-by-ball model, whoever faces the delivery gets credit — even if the partnership dynamics created those opportunities.

### 113(51) vs KXIP — TILT: -0.07%

Chris Gayle scored 73(35) at the top and captured +33.92% of the innings delta. By the time Kohli was accelerating, RCB's win probability was already high. His century moved an already-dominant position very little.

**A century for literally zero TILT.**

This is TILT's most counterintuitive property: on a stacked team, individual brilliance gets diluted. The WP "pie" for an innings is finite. When Gayle and ABD are taking the biggest slices, even a century barely registers.

---

## The -39.77% Catastrophe

On May 9th, Kohli scored 20(21) — a strike rate of 95.2 — before being caught at over 7.5. Here's the ball-by-ball:

He started with a lucky boundary (4 runs, +24.12% WP) but then hit 0.3 and the model registered -18.56%. From there, a sequence of dots and singles through 7 overs steadily eroded RCB's position. Each dot ball in the powerplay is a small negative. Twenty-one of them compound.

KL Rahul then came in and scored 42(26) for +40.47% TILT. RCB won the match. Kohli's slow start was the problem that someone else had to fix.

This single match wiped out the TILT equivalent of a good century. And it happened because the model captures exactly what matters: Kohli consumed 21 balls at a below-par rate during the phase of the innings when scoring rates matter most.

---

## The Phase Story

![Phase breakdown](plots/kohli_phase_breakdown.png)

| Phase | 2016 Runs | 2016 TILT | 2019 Runs | 2019 TILT |
|:------|----------:|----------:|----------:|----------:|
| Powerplay | 278(233) SR 119 | **-23.55%** | 214(167) SR 128 | **+57.62%** |
| Middle | 490(333) SR 147 | +35.75% | 153(131) SR 117 | -2.83% |
| Death | 205(89) SR 230 | +56.77% | 72(27) SR 267 | +31.44% |

The powerplay is where 2016 bleeds. Kohli scored 278 runs at SR 119 — good by conventional standards — but **each dot ball in the powerplay carries a WP penalty**. Across 233 powerplay balls, the dots accumulated to -23.55% of total TILT. In 2019, he was more aggressive from ball one and his powerplay TILT was +57.62%.

In 2016, Kohli was the anchor who accelerated in the middle and death overs. That's a valid strategy — but the model says the early-innings caution cost more than the late-innings aggression recovered.

---

## The Dot Ball Tax

![Dot ball tax](plots/kohli_dot_ball_tax.png)

| | 2016 | 2019 |
|:--|:--|:--|
| Dot balls | 184 (28.1%) | 96 (29.5%) |
| Dot ball TILT penalty | **-232.5%** | **-129.4%** |
| Scoring ball TILT | +301.5% | +215.7% |

The dot ball percentages are similar. But in 2016, Kohli faced *twice as many total balls*, which means twice as many dots, each one chipping away at TILT. When you face 655 balls, even a 28% dot rate means 184 small negative events that your boundaries need to overcome.

---

## It's Not Just Kohli

This pattern — high volume, modest TILT — shows up across IPL history. Here are the most instructive comparisons:

### The Comparison Table

| Player | Season | Runs | SR | Matches | TILT/match | Dot % | Inn 2 % |
|:-------|:-------|-----:|---:|--------:|-----------:|------:|--------:|
| **V Kohli** | **2016** | **973** | **148.5** | **16** | **+4.31%** | **28.1** | **31.9** |
| V Kohli | 2019 | 439 | 135.1 | 13 | +6.63% | 29.5 | 37.2 |
| DA Warner | 2016 | 848 | 146.5 | 17 | +11.97% | 35.1 | 52.5 |
| KL Rahul | 2018 | 659 | 154.7 | 14 | +17.79% | 35.7 | 68.3 |
| F du Plessis | 2021 | 633 | 134.4 | 16 | -4.00% | 35.2 | 35.0 |
| AM Rahane | 2013 | 488 | 103.4 | 18 | -4.87% | 41.7 | 57.4 |
| JH Kallis | 2012 | 409 | 104.1 | 17 | -5.50% | 39.2 | 69.7 |
| Ishan Kishan | 2022 | 418 | 117.7 | 14 | -6.12% | 41.4 | 63.7 |

### Warner 2016: Same Volume, Triple the TILT

Warner scored 848 runs at SR 146.5 — slightly below Kohli's counting stats. But his TILT was +11.97% per match, nearly 3x Kohli's. Why?

- **Zero catastrophic matches**: Warner's worst match was -5.73%. Kohli's was -39.77%. Warner never had a single match below -10%.
- **More 2nd innings batting**: Warner batted in the 2nd innings 52.5% of the time (vs Kohli's 31.9%). Second innings balls carry structurally larger WP swings.
- **Consistency**: Warner's match TILT standard deviation was 16.79% vs Kohli's 18.06%. Similar spread, but no outlier disasters.

### KL Rahul 2018: The TILT Machine

659 runs, +17.79% per match. Rahul batted in the 2nd innings 68.3% of the time. More than two-thirds of his balls were in higher-leverage situations. This is partly a function of KXIP's batting order and toss decisions, but it compounds: more chase innings = more TILT opportunity.

### du Plessis 2021: The Orange Cap Mirage

Faf du Plessis won the Orange Cap in 2021 with 633 runs at SR 134. His TILT was **-4.00% per match**. He scored a lot of runs, but disproportionately in low-leverage situations. His dot rate (35.2%) and a batting-first lean (only 35% inn 2 balls) meant his volume translated to negative impact by TILT's measure.

### Rahane 2013 & Kallis 2012: The Accumulator Penalty

Both scored 400+ runs at sub-105 strike rates with 40%+ dot ball percentages. They accumulated — and the model correctly identifies that at SR 103, you're hurting your team on most deliveries. Every ball at below the expected scoring rate is a small negative WP shift.

---

## Runs vs Impact: The Full Picture

![Runs vs TILT scatter](plots/runs_vs_tilt_scatter.png)

Every dot is an IPL season with 400+ runs. The correlation between runs scored and TILT per match is **weak positive** (r = 0.27). Scoring lots of runs helps, but the *rate* and *timing* matter far more than the volume.

Kohli 2016 sits in the middle of the cloud — genuinely good but not as impactful as the raw numbers suggest. The seasons in the top-right (high runs AND high TILT) tend to be from players who scored fast, chased often, and didn't have many catastrophic matches.

---

## Kohli's Career Arc

![Kohli career timeline](plots/kohli_career_timeline.png)

The purple line shows runs. The bars show TILT. The 2016 peak in runs clearly doesn't correspond to the TILT peak. His best TILT seasons are scattered — 2019, some earlier years where he scored fewer runs but in more decisive situations.

---

## The Verdict: Is the Model Right?

### Where TILT is correct

1. **Context matters more than volume.** Some of Kohli's 2016 runs came when RCB were already cruising. A century at 180/2 in the 18th over doesn't shift much.

2. **The catastrophic matches are real.** 20(21) in the powerplay when your team needs a fast start is genuinely harmful. The model is right to penalize this heavily.

3. **Credit sharing reflects reality.** When ABD scores 129 and Kohli scores 109 in the same innings, the marginal impact of each run is lower than if one player carried the load alone.

### Where TILT has blind spots

1. **Partnership dynamics are invisible.** Kohli anchoring allowed ABD and Gayle to play with freedom. That enabling role doesn't register in ball-by-ball WP deltas. A batter who occupies the crease and lets others attack around them provides value that TILT can't see.

2. **First innings has a structural ceiling.** The total WP available in innings 1 is limited and gets divided among all batters. On a stacked 2016 RCB lineup (Kohli, ABD, Gayle, Watson), the pie is split four ways. On a weaker team, one dominant batter captures everything.

3. **No batting position adjustment.** Kohli opening against the new ball with fielding restrictions is harder than a middle-order batter walking in at 120/2. The model doesn't adjust for difficulty of role.

4. **Dot ball accounting is harsh.** In T20, even the best batters have a 25-30% dot rate. But each dot ball is penalized the same whether it's a leave outside off or a mistimed drive. Some dots are strategic (getting the eye in early); the model treats them all as failures.

5. **Volume should count for something.** Playing 16 matches, batting 655 balls, and still being net positive is a remarkable achievement. TILT measures rate, not volume — a player who adds +4.31% over 16 matches arguably provides more total value than one who adds +6.63% over 13.

---

## So Is 2016 Overrated?

No. It was a historically great season by any measure. But TILT reveals that some of those runs were less impactful than they looked — scored on a dominant team, alongside other world-class batters, often in the first innings where WP shifts are smaller, and partly offset by a few costly failures.

The truth is somewhere between the counting stats and TILT. The counting stats overrate 2016 because they ignore context. TILT underrates it because it can't see the enabling value of an anchor who creates conditions for others to attack.

Both are incomplete. That's what makes the comparison interesting.
