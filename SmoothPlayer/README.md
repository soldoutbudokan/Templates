# SmoothPlayer

A smooth MP4 video player web app. Paste a link to an MP4 stream and play it with polished controls and keyboard shortcuts.

**Live:** [smoothplayer.vercel.app](https://smoothplayer.vercel.app/)

## Quick Start

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Keyboard Shortcuts

| Key | Action |
|---|---|
| Space / K | Play / Pause |
| ← → | Skip 2 seconds |
| Shift + ← → | Skip 10 seconds |
| ↑ ↓ | Volume up / down |
| M | Mute / Unmute |
| F | Fullscreen |
| 0-9 | Jump to 0%-90% |
| < > | Playback speed |
| ? | Show shortcuts |

## Features

- Custom video controls with seek bar, volume, fullscreen
- Buffered progress indicator
- Smooth 60fps seek bar (requestAnimationFrame)
- Auto-hiding controls and cursor after 3s inactivity
- Click to play/pause, double-click for fullscreen
- Dark theme optimized for video watching
- No URLs or user data stored anywhere

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Deployed on Vercel
