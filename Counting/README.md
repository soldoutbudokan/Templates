# Card Counting Trainer

A web app to practice Hi-Lo card counting for blackjack. Features four training modes, realistic multi-deck shoe simulation, streak tracking, and betting strategy tips.

**Live Demo:** https://counting-trainer-sob.vercel.app

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Features

### Classic Mode
- Displays 15-30 cards at once
- Submit your running count and check your answer
- Cards animate in with a flip effect

### Speed Drill Mode
- Cards flash one at a time at configurable speeds (1s, 0.75s, 0.5s, 0.25s, 0.15s, 0.1s)
- **Hard Mode**: 40-60 card sequences with casino visual distractions (floating chips, background pulses, scrolling ticker)
- Enter your count after the sequence completes

### True Count Trainer
- Two-step challenge: first enter the running count, then convert to true count
- Shows decks remaining to help calculate (true count = running count ÷ decks remaining)
- Accepts answers within ±1 of exact value for rounding tolerance
- Detailed breakdown shown after each round

### Multi-Hand Simulation
- Simulates a real blackjack table with 2-4 player hands plus a dealer
- Cards dealt in realistic casino order (one round left-to-right, then second round)
- Count all visible cards across all hands

### Betting Strategy Tips
- Toggle on/off in the sidebar
- Shows recommended bet sizing based on true count after each round
- Visual indicator bar from minimum bet to maximum spread

### Streak Tracking
- Current streak of consecutive correct answers
- Best streak persisted across sessions (localStorage)
- Visual indicators at 5+ and 10+ streaks

### Realistic Deck Simulation
- Choose from 1, 2, 6, or 8 deck shoes
- No duplicate cards until reshuffle
- Automatic reshuffle when ~75% of shoe is dealt (cut card)
- Visual progress bar showing cards remaining

## Hi-Lo Counting System

| Cards | Count Value |
|-------|-------------|
| 2-6   | +1          |
| 7-9   | 0           |
| 10-A  | -1          |

## Keyboard Shortcuts

| Key   | Action      |
|-------|-------------|
| Enter | Submit guess |
| Space | New hand / Start drill |
| Esc   | Stop drill (speed mode) |

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Deployed on Vercel

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deployment

Deployed to Vercel. To redeploy:

```bash
vercel --prod
```

Or push to the connected GitHub repository for automatic deployments.

## Project Structure

```
Counting/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main game page (state hub)
│   └── globals.css         # Tailwind + custom animations
├── components/
│   ├── BettingAdvice.tsx   # Betting strategy display
│   ├── Card.tsx            # Card with flip animation
│   ├── CasinoNoise.tsx     # Hard mode visual distractions
│   ├── Controls.tsx        # Settings panel (4-mode selector)
│   ├── GameBoard.tsx       # Classic mode
│   ├── MultiHandBoard.tsx  # Multi-hand table simulation
│   ├── SpeedDrill.tsx      # Speed drill mode
│   └── TrueCountTrainer.tsx # True count practice
├── lib/
│   ├── betting.ts          # Betting strategy logic
│   ├── deck.ts             # Deck class with shuffle logic
│   ├── types.ts            # Shared TypeScript types
│   └── usePersistedState.ts # localStorage persistence hook
├── package.json
├── next.config.js
├── tailwind.config.js
└── tsconfig.json
```
