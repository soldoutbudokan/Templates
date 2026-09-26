# Card Counting Trainer

A Next.js app for learning a steady Hi-Lo count and accurate blackjack basic strategy. Quick play makes short practice easy to start; guided sessions and full-table tests provide deeper, inspectable feedback.

**Live:** [counting-trainer-sob.vercel.app](https://counting-trainer-sob.vercel.app/)

## Quick Start

Requires Node.js 20 or later.

```bash
npm ci
npm run dev
```

Open http://localhost:3000.

```bash
npm test
npm run typecheck
npm run build
```

## Quick play

The opening screen offers one-tap **Count**, **Strategy**, or **Mixed** practice runs. Each run provides 60 seconds of answering time; coaching feedback and pauses do not consume that time. Mixed runs alternate cumulative pair counting and basic-strategy choices. Counting carries a running total across Count questions, with the corrected total supplied after each first answer. Strategy cards belong to a separate drill and do not enter that total; the full-table test integrates every visible card.

Correct first answers earn 10 XP, plus a combo bonus that rises by 2 XP every three consecutive correct answers, capped at 10 bonus XP. Mistakes earn no XP and reset the combo. Incorrect answers hold the explanation until the next question is requested; correct answers advance after a short reward beat unless held for review. Runs finish on a results screen with separate skill accuracy, a personal best, and an optional rematch; they never restart automatically. Pausing or hiding the tab covers the cards and freezes the clock.

Daily practice streaks use the browser’s local calendar date. A completed run needs at least five answers to earn daily credit; a run that reaches the 200-question cap also counts as complete. Early exits retain first-answer records but do not earn daily credit. Streaks and XP measure practice activity, not mastery; uninterrupted full tests remain separate. Quick-play progress is stored on this device, alongside the other training histories.

## Guided training

- **Practice:** a 78-card segment from a six-deck shoe. Cards appear in pairs and disappear before each checkpoint. Feedback supplies the corrected count before the next round.
- **Assessment:** 234 cards, reaching a 75% cut card in the same six-deck shoe. Answers and replays stay hidden until the attempt ends. Dealing pauses at checkpoints; this is a counting assessment, not a complete blackjack table simulation.
- **Focus:** running count alone, or running count plus true-count conversion. The conversion task supplies a half-deck estimate; it does not assess visual deck estimation.
- **Pace:** 1.2, 0.9, 0.65, or 0.45 seconds per card. Pair exposure lasts twice the per-card interval.
- **Lost count:** record the loss honestly. Practice offers reconstruction and a corrected count; assessment records the response without revealing an answer.
- **Replay:** step through an already completed segment from its known starting count. Replay never advances the shoe or changes the first submitted answer.
- **Interruptions:** pausing or hiding the browser tab covers the cards and stops dealing. An interrupted assessment can continue as practice but cannot regain assessment eligibility. Paused response time is excluded. The same pair reappears after a dealing pause and must not be counted twice.
- **Progress:** the latest 24 ended attempts, including partial attempts, are stored on this device. History preserves rules, pace, answers, and replayable checkpoints. Active attempts are held in memory until ended; reloading the page discards an active attempt.
- **Next exercise:** recommendations inspect the last three completed, uninterrupted sessions. Fewer than eight observations, running-count accuracy below 90%, or a reported loss of count recommends retention practice. Otherwise the recommendation adds conversion. These are transparent product heuristics, not validated mastery criteria.

The results count exact **checkpoints**, not supposedly correct individual cards. Running and true-count accuracy stay separate. A complete balanced deck's final zero is never the only assessment target. Profit and simulated winnings do not enter the score.

## Full table test

**Full test** combines playing decisions with continuous Hi-Lo counting. Choose a 10-round sample or a six-deck shoe ending after the round that reaches the 75% cut card. Add two other players or play heads-up; set automatic card pace from 0.45 to 1.2 seconds. A new session explicitly starts a new shoe at zero; no mid-round shuffle occurs.

- Play hit, stand, double, split, and surrender, plus an explicit insurance decision before an ace-up peek. Wrong but legal choices are played as chosen.
- Six decks, H17, DAS, US peek, late surrender, 3:2 natural blackjack. Maximum four hands, no resplitting aces, one card to split aces; split 21 is not a natural.
- Cards appear one at a time. The hidden dealer card contributes nothing until revealed. This trainer always reveals the dealer hole card at round end, including when every player has busted or surrendered. If no hand remains to contest the dealer, no extra dealer draws occur.
- The table clears after settlement. Running and true counts are submitted together against frozen targets; the supplied half-deck estimate tests conversion, not physical deck estimation.
- Test mode withholds correctness until the session ends. Coached mode explains each choice and gives the corrected cumulative count after each round. Winnings never affect grades.
- Strategy, exact running-count, and exact true-count scores use separate fractions. Review includes decision snapshots, category coverage, count checkpoints, and an ordered replay of every exposed card.
- Pausing or hiding the browser tab covers the table, stops automatic play, and permanently marks the attempt interrupted. Resuming shows the same cards; they must not be counted again. Ending at an unanswered checkpoint records it as skipped.
- The latest 24 ended table attempts, including partial and interrupted attempts, appear in Progress. Loading saved history replays inputs from the seed and recalculates targets and scores. Active sessions are held in memory and are discarded on reload.

“Perfect” here means exact **basic strategy** for the displayed rules and legal actions. It does not mean count-dependent optimal play. Betting, deviations, and bankroll management are outside this test. A clean result requires a completed, uninterrupted test with at least one playing decision and every strategy/count answer correct; it describes the situations actually observed, not unobserved skills.

## Perfect strategy

The dedicated **Perfect strategy** section provides Hard / Soft / Pairs / Mixed sessions, 20- or 30-decision tests, immediate explanations in practice, and reference charts. Questions balance chart situations rather than natural shoe frequency. Tests preserve first answers, hide the chart and feedback until completion, and mark hidden-tab interruptions. Results show category sample sizes and missed situations; missed hands can become a coached repair drill. Strategy-only reviews last until you leave the section; table results persist in Progress.

## Counting and strategy conventions

Hi-Lo values: 2–6 = +1; 7–9 = 0; 10/J/Q/K/A = −1. The running count includes every card first exposed since the current shuffle.

True-count exercises use remaining decks rounded to the nearest half deck, with a minimum divisor of 0.5. Divide the running count by that estimate, then **floor toward negative infinity**: +1.75 becomes +1 and −1.5 becomes −2. Inputs must be whole numbers; no ±1 grading tolerance applies.

Basic-strategy practice uses one explicit total-dependent profile:

- Six decks; dealer hits soft 17.
- Double on any initial two cards; double after split allowed.
- Late surrender after a negative dealer blackjack check.
- A natural blackjack is resolved, rather than presented as a strategy question.
- The evaluator includes unavailable-action fallbacks. The table engine supplies the current hand’s actual legal actions before grading.

The example 1–2–4–8–10 betting schedule is for execution practice only. It is not calibrated to a bankroll, risk tolerance, table limits, or game profitability.

References: [QFIT true-count conventions](https://www.qfit.com/CalculatingTrueCounts.htm), [Shackleford's 4–8 deck strategy and H17 amendments](https://wizardofodds.com/games/blackjack/strategy/4-decks/), [QFIT integrated practice](https://www.qfit.com/book/ModernBlackjackPage94.htm).

## Isolated practice drills

The original five drills remain available under **Practice drills**:

- **Spread:** independent card-batch arithmetic; explicitly starts a fresh shoe for each batch.
- **Speed drill:** independent flashes; single-deck sequences stop at 51 cards, avoiding a forced-zero endpoint and implicit second shoe. Hiding the tab abandons the drill.
- **True count:** cumulative counting across rounds with strict half-deck/floor conversion and explicit new-shoe notice.
- **Table spread:** an independent layout with both dealer cards exposed. It is labeled as arithmetic practice, not realistic table play.
- **Basic strategy:** the fixed six-deck profile above, without decorative zero-count betting advice.

Free-practice streaks are separate from guided-session history. Layouts adapt to narrow screens, count inputs have accessible labels, and guided inputs include a sign button for mobile keyboards. Reduced-motion preferences disable decorative animation.

## Implementation

- `lib/deck.ts`: one shoe model, seeded random source support, distinct physical IDs across shuffles, separate drawing/exposure, idempotent exposure, cumulative count, atomic exhaustion rejection, and explicit round-boundary shuffling.
- `lib/countPolicy.ts`: strict number parsing, half-deck estimation, and the shared true-count convention.
- `lib/training.ts`: reproducible checkpoint plans, frozen targets, independent counting/conversion grades, replay counts, validated bounded history, and recommendations.
- `components/TrainingSession.tsx`: coached/assessment lifecycle, pause behavior, first-attempt submission guard, replay and review.
- `lib/quickPlay.ts` and `components/QuickPlay.tsx`: finite practice runs, first-answer combos, local-calendar activity, and bounded history.
- `lib/blackjack.ts`: immutable, seeded full-round engine and separate strategy/count records.
- `lib/tableHistory.ts`: bounded table history rebuilt from learner inputs.
- `components/FullTableTest.tsx`: table lifecycle, concealed test answers, interruption handling, and review.
- `lib/strategyPractice.ts` and `components/StrategyPractice.tsx`: balanced strategy questions, reference charts, test and repair flow.
- `components/FreePractice.tsx`: retained isolated drills.
- `app/page.tsx`: training, practice, learning references, and device-local progress.

No new runtime dependencies, account system, or remote storage are required. History is versioned; loading it reconstructs targets from the seed and recomputes grades instead of trusting stored correctness flags.

## Verification

`npm test` uses Node's test runner and the existing TypeScript compiler. It covers 340 independently transcribed strategy chart cells and legal-action fallbacks; insurance, hole-card exposure, naturals, splits, surrender, H17, deterministic full shoes, partial sessions, and replayed table history; plus guided counting, strict conversion, corrupt history, assessment eligibility, and recommendations. `npm run typecheck` checks the application. The Counting GitHub Actions workflow runs all three verification commands, including the production build.

## Later stages

Full blackjack rounds and dedicated strategy testing are implemented. Visual discard-tray estimation, count-based deviation drills, spaced scheduling, and bankroll/variance simulations remain future work. They should reuse the same count model and keep assisted practice separate from assessment.
