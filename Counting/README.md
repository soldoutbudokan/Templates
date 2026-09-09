# Card Counting Trainer

A Next.js app for learning a steady Hi-Lo count. Guided practice carries one running count across successive groups of cards, checks it at unpredictable points, and turns mistakes into a replay and a concrete next exercise.

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

## Counting and strategy conventions

Hi-Lo values: 2–6 = +1; 7–9 = 0; 10/J/Q/K/A = −1. The running count includes every card first exposed since the current shuffle.

True-count exercises use remaining decks rounded to the nearest half deck, with a minimum divisor of 0.5. Divide the running count by that estimate, then **floor toward negative infinity**: +1.75 becomes +1 and −1.5 becomes −2. Inputs must be whole numbers; no ±1 grading tolerance applies.

Basic-strategy practice uses one explicit total-dependent profile:

- Six decks; dealer hits soft 17.
- Double on any initial two cards; double after split allowed.
- Late surrender after a negative dealer blackjack check.
- A natural blackjack is resolved, rather than presented as a strategy question.
- The evaluator includes unavailable-action fallbacks. It is not a complete round or split-hand engine.

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
- `components/FreePractice.tsx`: retained isolated drills.
- `app/page.tsx`: training, practice, learning references, and device-local progress.

No new runtime dependencies, account system, or remote storage are required. History is versioned; loading it reconstructs targets from the seed and recomputes grades instead of trusting stored correctness flags.

## Verification

`npm test` uses Node's test runner and the existing TypeScript compiler. It covers independent H17 strategy fixtures, exposure and shuffle invariants, strict conversion boundaries, reproducible checkpoints, error attribution, corrupt history, assessment eligibility, and recommendations. `npm run typecheck` checks the application. The Counting GitHub Actions workflow runs all three verification commands, including the production build.

## Later stages

The first release focuses on correct continuous counting, useful feedback and valid checkpoint assessment. Full blackjack rounds with playing decisions, visual discard-tray estimation, deviation drills, spaced scheduling, and bankroll/variance simulations remain future work. They should reuse the same count model and keep assisted practice separate from assessment.
