# PlainCricket

A minimal, no-clutter cricket scores website. Live scores without the ads, popups, and chaos of typical cricket sites.

**Live site: https://plaincricket.vercel.app**

## Features

- **Match List**: Live/recent matches with team names, scores, venue, and time (EST/IST)
- **Full Scorecard**: Click any match to see both innings with complete batting and bowling stats
- **Did Not Bat**: Shows players who didn't bat for incomplete innings
- **Auto-refresh**: Scores update every 30 seconds when viewing a match
- **Dark Mode**: Easy on the eyes, monospace terminal aesthetic

## Tech Stack

- **Backend**: Python/Flask API that scrapes Cricbuzz
- **Frontend**: Vanilla HTML/CSS/JS - no frameworks, no bloat
- **Hosting**: Vercel (serverless Python)

## API Endpoints

- `GET /api/matches` - List of current matches with scores, venue, time
- `GET /api/score?id=<match_id>` - Full scorecard for a specific match

## Local Development

```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
flask --app index.py run --port 5001
```

Then open http://127.0.0.1:5001

## Deploy to Vercel

```bash
vercel --prod
```

## How It Works

1. Scrapes Cricbuzz's live scores page for match list
2. Fetches scorecard pages in parallel for scores/venue/time
3. Parses HTML with BeautifulSoup (Cricbuzz is heavily JS-rendered, so we extract what's in static HTML)
4. Returns clean JSON that the frontend renders

## Limitations

- Data depends on Cricbuzz's HTML structure (may break if they change it)
- Some match details only available after scorecard is published
- Rate limited by Cricbuzz (uses random user agents to avoid blocks)
