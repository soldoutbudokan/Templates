# PlainCricket

A minimal, no-clutter cricket scores website. Live scores without the ads, popups, and chaos of typical cricket sites.

**Live site: https://plaincricket.vercel.app**

## Quick Start

```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
flask --app index.py run --port 5001
```

Then open http://127.0.0.1:5001

## Features

- **Match List**: Live matches first, then upcoming (by start time), then finished. Team names, scores, venue, and start time (US Eastern/IST, DST-aware)
- **Live indicator**: In-progress matches get a red dot, driven by the API's match-state classification
- **Full Scorecard**: Both innings with batting, bowling, extras, and test-match notation (`245 & 91-2 (24 Ov)`)
- **Did Not Bat / Yet to Bat**: Shows players still to come, with the correct label for in-progress innings
- **Auto-refresh**: Scorecard refreshes every 30s and the match list every 60s while the tab is visible, with an "updated Xs ago" indicator — no flicker, content swaps in place
- **Dark & light mode**: Follows the system preference; monospace terminal aesthetic either way
- **Degraded-mode notice**: If the scraper stops finding scores (usually a Cricbuzz markup change), the UI says so instead of silently showing nothing

## Tech Stack

- **Backend**: Python/Flask API that scrapes Cricbuzz
- **Frontend**: Vanilla HTML/CSS/JS — no frameworks, no bloat, all scraped data HTML-escaped before rendering
- **Hosting**: Vercel (serverless Python)

## API Endpoints

- `GET /api` — API info
- `GET /api/matches` — list of current matches. Each match has `id`, `title`, `team1`/`team2`, `score1`/`score2` (matched to teams by name, not innings order), `status`, `state` (`live` | `upcoming` | `complete`), `venue`, `startTime` (ISO), `timeEST`, `timeLocal`. The payload also carries a `degraded` flag.
- `GET /api/score?id=<match_id>` — full scorecard: innings with batting/bowling/extras, did-not-bat (with label), playing XI for teams yet to bat, plus `status` and `state`. The id must be numeric.

The API is served same-origin for the frontend (no CORS headers).

## Caching

Responses carry `Cache-Control: public, s-maxage=…, stale-while-revalidate=…` headers (30s for the match list, 20s for scorecards), so Vercel's edge cache absorbs most traffic — page loads usually never hit Python, and Cricbuzz sees only a trickle of requests. A small in-memory TTL cache backs this up on warm instances and in local dev.

## Tests

The parsing logic is covered by tests using synthetic HTML fixtures that mirror Cricbuzz's markup (innings headers, batting/bowling grids, JSON-LD):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

If Cricbuzz changes their markup, save the new pages as fixtures and adjust the parsers until the suite passes again.

## Deploy to Vercel

```bash
vercel --prod
```

## How It Works

1. Scrapes Cricbuzz's live scores page for the match list (links + JSON-LD metadata, matched by word-bounded team names)
2. Fetches scorecard pages in parallel for scores/venue/time
3. Parses HTML with BeautifulSoup (Cricbuzz is heavily JS-rendered, so we extract what's in static HTML)
4. Pairs innings scores to teams **by name** (innings order is whoever batted first, which is unrelated to title order)
5. Returns clean JSON that the frontend renders

## Limitations

- Data depends on Cricbuzz's HTML structure (may break if they change it — the `degraded` flag and the test suite are the early-warning systems)
- Some match details only available after the scorecard is published
- Rate limited by Cricbuzz (mitigated by edge caching and randomized user agents)
