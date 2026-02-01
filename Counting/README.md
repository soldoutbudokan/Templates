# Card Counting Trainer

A web app to practice Hi-Lo card counting for blackjack. Features realistic multi-deck shoe simulation and speed drill mode for building counting speed.

**Live Demo:** https://counting-trainer-sob.vercel.app

## Features

### Classic Mode
- Displays 15-30 cards at once
- Submit your running count and check your answer
- Cards animate in with a flip effect

### Speed Drill Mode
- Cards flash one at a time at configurable speeds
- Practice speeds: 1s, 0.75s, 0.5s, 0.25s per card
- Enter your count after the sequence completes

### Realistic Deck Simulation
- Choose from 1, 2, 6, or 8 deck shoes (common casino configurations)
- No duplicate cards until reshuffle
- Automatic reshuffle when ~75% of shoe is dealt (cut card)
- Visual indicator showing cards remaining in shoe

### Session Tracking
- Rounds played counter
- Accuracy percentage

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

Open http://localhost:3000 in your browser.

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
│   ├── layout.tsx      # Root layout with metadata
│   ├── page.tsx        # Main game page
│   └── globals.css     # Tailwind + custom animations
├── components/
│   ├── Card.tsx        # Card with flip animation
│   ├── Controls.tsx    # Settings panel
│   ├── GameBoard.tsx   # Classic mode
│   └── SpeedDrill.tsx  # Speed drill mode
├── lib/
│   └── deck.ts         # Deck class with shuffle logic
├── package.json
├── next.config.js
├── tailwind.config.js
└── tsconfig.json
```
