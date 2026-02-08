import random
import requests
import re
import os
from time import time as time_func
from bs4 import BeautifulSoup as bs
from markupsafe import escape
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import pytz

# Simple TTL cache for match data
_cache = {}
CACHE_TTL = 180  # seconds


def clean_status_text(status):
    """Remove 'b' suffix from ball counts (e.g., '36b' -> '36')"""
    if not status:
        return status
    # Remove 'b' only when it follows a number at word boundary
    return re.sub(r'(\d+)b\b', r'\1', status)

app = Flask(__name__, static_folder='../public', static_url_path='')
app.json.sort_keys = False
CORS(app)


@app.route('/')
def serve_frontend():
    """Serve the frontend HTML"""
    return send_from_directory(app.static_folder, 'index.html')

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


def get_soup(url):
    """Fetch URL and return BeautifulSoup object"""
    session = requests.Session()
    headers = {"User-Agent": random.choice(user_agents)}
    response = session.get(url, headers=headers, timeout=10)
    return bs(response.content, "html.parser")


def safe_text(element, default="--"):
    """Safely extract text from BeautifulSoup element"""
    return element.text.strip() if element else default


def format_match_times(iso_time_str):
    """Convert ISO time to EST and local (IST) times"""
    try:
        # Parse ISO format like "2026-01-31T16:00:00.000Z"
        utc_time = datetime.fromisoformat(iso_time_str.replace("Z", "+00:00"))

        # Convert to EST
        est_tz = pytz.timezone("America/New_York")
        est_time = utc_time.astimezone(est_tz)
        est_str = est_time.strftime("%b %d, %I:%M %p EST")

        # Convert to IST (common local time for cricket)
        ist_tz = pytz.timezone("Asia/Kolkata")
        ist_time = utc_time.astimezone(ist_tz)
        ist_str = ist_time.strftime("%b %d, %I:%M %p IST")

        return est_str, ist_str
    except:
        return "", ""


@app.route("/api")
def api_info():
    return jsonify({
        "app": "PlainCricket API",
        "version": "1.0.0",
        "endpoints": {
            "/api": "API info",
            "/api/matches": "List of live/recent matches",
            "/api/score?id=<match_id>": "Score for specific match"
        }
    })


def get_match_details(match_id):
    """Fetch scores, venue, date for a single match"""
    try:
        url = f"https://www.cricbuzz.com/live-cricket-scorecard/{match_id}"
        soup = get_soup(url)
        text = soup.get_text(separator="\n", strip=True)
        lines = text.split("\n")

        # Get venue and date
        venue = ""
        match_date = ""
        for i, line in enumerate(lines):
            if line.strip() in ["Venue:", "Venue"] and i + 1 < len(lines):
                venue = lines[i + 1].strip()
            if line.strip() in ["Date:", "Date"] and i + 1 < len(lines):
                match_date = lines[i + 1].strip()

        # Get scores from DOM containers (correct order, no index mismatch)
        innings_headers = soup.select('div[id^="team-"][id*="-innings-"]')
        seen_ids = set()
        scores = []
        for hdr in innings_headers:
            hdr_id = hdr.get("id", "")
            if not hdr_id or hdr_id in seen_ids:
                continue
            seen_ids.add(hdr_id)

            # Team name (prefer full name)
            full_name_div = hdr.select_one("div.hidden.tb\\:block.font-bold")
            short_name_div = hdr.select_one("div.tb\\:hidden.font-bold")
            team_name = safe_text(full_name_div, "") or safe_text(short_name_div, "")

            # Score and overs from score area
            score_div = hdr.select_one("div.flex.gap-4 > div")
            score_text = safe_text(score_div, "")
            if score_text and team_name:
                score_match = re.match(r'^(\d{1,4}-\d{1,2})\s*\((.+?)\)$', score_text)
                if score_match:
                    scores.append({
                        "team": team_name,
                        "score": score_match.group(1),
                        "overs": f"({score_match.group(2)})"
                    })

        return {"scores": scores[:4], "venue": venue, "date": match_date}
    except:
        return {"scores": [], "venue": "", "date": ""}


@app.route("/api/matches")
def matches():
    """Get list of current matches"""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Check cache first
    now = time_func()
    if 'matches' in _cache and now - _cache['matches']['time'] < CACHE_TTL:
        return jsonify({"matches": _cache['matches']['data']})

    try:
        soup = get_soup("https://www.cricbuzz.com/cricket-match/live-scores")
        match_list = []
        seen_ids = set()

        # Try to get rich data from JSON-LD
        json_ld_matches = {}
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            try:
                data = json.loads(script.string)
                if data.get("@type") == "WebPage" and "mainEntity" in data:
                    items = data["mainEntity"].get("itemListElement", [])
                    for item in items:
                        if item.get("@type") == "SportsEvent":
                            competitors = item.get("competitor", [])
                            if len(competitors) >= 2:
                                team1 = competitors[0].get("name", "")
                                team2 = competitors[1].get("name", "")
                                status = item.get("eventStatus", "")
                                name = item.get("name", "")
                                venue = item.get("location", "").rstrip(", ")
                                start_time = item.get("startDate", "")
                                json_ld_matches[name] = {
                                    "team1": team1,
                                    "team2": team2,
                                    "status": status,
                                    "venue": venue,
                                    "startTime": start_time
                                }
            except:
                pass

        # Common abbreviation to full name mapping
        abbr_to_full = {
            "WI": "West Indies", "RSA": "South Africa", "NZ": "New Zealand",
            "IND": "India", "PAK": "Pakistan", "AUS": "Australia",
            "ENG": "England", "SL": "Sri Lanka", "BAN": "Bangladesh",
            "AFG": "Afghanistan", "ZIM": "Zimbabwe", "IRE": "Ireland",
            "UAE": "UAE", "SCO": "Scotland", "NEP": "Nepal", "USA": "USA",
            "INDU19": "India U19", "PAKU19": "Pakistan U19", "ENGU19": "England U19",
            "AUSU19": "Australia U19", "NZU19": "New Zealand U19"
        }

        # Find match links with class="block mb-3" (main content cards)
        links = soup.find_all("a", href=re.compile(r'/live-cricket-scores/\d+'), class_="block")
        if not links:
            # Fallback: find all links with the href pattern
            links = soup.find_all("a", href=re.compile(r'/live-cricket-scores/\d+'))

        for link in links:
            href = link.get("href", "")

            # Extract match ID from URL
            parts = href.split("/")
            match_id = None
            for i, part in enumerate(parts):
                if part == "live-cricket-scores" and i + 1 < len(parts):
                    match_id = parts[i + 1]
                    break

            if not match_id or match_id in seen_ids:
                continue
            seen_ids.add(match_id)

            # Use title attribute for rich data (has full team names + status)
            title_attr = link.get("title", "").strip()
            text = link.text.strip()
            if not title_attr and (not text or len(text) < 3):
                continue

            # Parse title attribute: "Team1 vs Team2, Match Type - Status"
            title = text if text else title_attr
            status = ""
            team1, team2 = "", ""

            if title_attr:
                # Extract status after last " - "
                if " - " in title_attr:
                    base, raw_status = title_attr.rsplit(" - ", 1)
                    status = clean_status_text(raw_status.strip())
                    # Extract teams from "Team1 vs Team2, Match Type"
                    vs_match = re.match(r'^(.+?)\s+vs\s+(.+?)(?:,\s+.+)?$', base, re.IGNORECASE)
                    if vs_match:
                        team1, team2 = vs_match.group(1).strip(), vs_match.group(2).strip()
                else:
                    vs_match = re.match(r'^(.+?)\s+vs\s+(.+?)(?:,\s+.+)?$', title_attr, re.IGNORECASE)
                    if vs_match:
                        team1, team2 = vs_match.group(1).strip(), vs_match.group(2).strip()

            # Construct display title from link text or title attr
            if " - " in text:
                parts_text = text.split(" - ", 1)
                title = parts_text[0].strip()
                if not status:
                    status = clean_status_text(parts_text[1].strip())

            # Initialize venue and time
            venue = ""
            start_time = ""
            time_est = ""
            time_local = ""

            # Try to find richer data from JSON-LD
            title_upper = (title_attr or title).upper()
            for name, data in json_ld_matches.items():
                t1, t2 = data["team1"], data["team2"]
                t1_abbr = t1[:3].upper() if len(t1) >= 3 else t1.upper()
                t2_abbr = t2[:3].upper() if len(t2) >= 3 else t2.upper()

                match1 = t1_abbr in title_upper or t1.upper() in title_upper
                match2 = t2_abbr in title_upper or t2.upper() in title_upper

                full_to_abbr = {v.upper(): k for k, v in abbr_to_full.items()}
                if t1.upper() in full_to_abbr and full_to_abbr[t1.upper()] in title_upper:
                    match1 = True
                if t2.upper() in full_to_abbr and full_to_abbr[t2.upper()] in title_upper:
                    match2 = True

                if match1 and match2:
                    if not team1:
                        team1 = t1
                    if not team2:
                        team2 = t2
                    if not status and data["status"]:
                        status = clean_status_text(data["status"])
                    venue = data.get("venue", "")
                    start_time = data.get("startTime", "")
                    break

            # Fallback: expand abbreviations in team names
            if team1:
                team1 = abbr_to_full.get(team1.upper(), team1)
            if team2:
                team2 = abbr_to_full.get(team2.upper(), team2)

            # Fallback: parse team names from title
            if not team1 or not team2:
                vs_match = re.match(r'^(.+?)\s+vs\s+(.+?)(?:\s+\d+(?:st|nd|rd|th)\s+|$)', title, re.IGNORECASE)
                if vs_match:
                    t1_raw, t2_raw = vs_match.group(1).strip(), vs_match.group(2).strip()
                    team1 = abbr_to_full.get(t1_raw.upper(), t1_raw)
                    team2 = abbr_to_full.get(t2_raw.upper(), t2_raw)

            # Format times
            if start_time:
                time_est, time_local = format_match_times(start_time)

            match_list.append({
                "id": match_id,
                "title": title,
                "team1": team1,
                "team2": team2,
                "status": status,
                "score1": "",
                "score2": "",
                "venue": venue,
                "timeEST": time_est,
                "timeLocal": time_local,
            })

            if len(match_list) >= 15:
                break

        # Only fetch scores for matches that have started (skip Preview/upcoming)
        matches_needing_scores = [
            m for m in match_list
            if m["status"].lower() not in ("preview", "upcoming")
        ]

        def fetch_details(match):
            details = get_match_details(match["id"])
            scores = details["scores"]
            if len(scores) >= 1:
                match["score1"] = f"{scores[0]['score']} {scores[0]['overs']}"
            if len(scores) >= 2:
                match["score2"] = f"{scores[1]['score']} {scores[1]['overs']}"
            if not match.get("venue") and details["venue"]:
                match["venue"] = details["venue"]
            if not match.get("timeEST") and details["date"]:
                match["timeEST"] = details["date"]
                match["timeLocal"] = ""
            return match

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_details, m): m for m in matches_needing_scores}
            for future in as_completed(futures):
                try:
                    future.result()
                except:
                    pass

        # Cache the results
        _cache['matches'] = {'data': match_list, 'time': time_func()}

        return jsonify({"matches": match_list})

    except Exception as e:
        return jsonify({"error": str(e), "matches": []}), 500


@app.route("/api/score")
def score():
    """Get detailed score for a specific match"""
    match_id = request.args.get("id")
    if not match_id:
        return jsonify({"error": "Match ID required. Use ?id=<match_id>"}), 400

    match_id = escape(match_id)

    try:
        # Use scorecard page - has more data in static HTML
        url = f"https://www.cricbuzz.com/live-cricket-scorecard/{match_id}"
        soup = get_soup(url)

        # Match title from h1
        h1 = soup.find("h1")
        title = h1.text.strip().replace(" - Scorecard", "") if h1 else "Unknown Match"

        # Get text as lines for parsing
        text = soup.get_text(separator="\n", strip=True)
        lines = text.split("\n")

        # Parse venue and date
        venue = ""
        match_date = ""
        for i, line in enumerate(lines):
            if line.strip() in ["Venue:", "Venue"] and i + 1 < len(lines):
                venue = lines[i + 1].strip()
            if line.strip() in ["Date:", "Date"] and i + 1 < len(lines):
                match_date = lines[i + 1].strip()

        # Parse innings from DOM containers (each innings is self-contained)
        # Pattern: div#team-{teamId}-innings-{n} (header) + div#scard-team-{teamId}-innings-{n} (body)
        innings_headers = soup.select('div[id^="team-"][id*="-innings-"]')

        # Deduplicate (page has mobile + desktop copies)
        seen_ids = set()
        unique_headers = []
        for hdr in innings_headers:
            hdr_id = hdr.get("id", "")
            if hdr_id and hdr_id not in seen_ids:
                seen_ids.add(hdr_id)
                unique_headers.append(hdr)

        response_innings = []
        for hdr in unique_headers:
            hdr_id = hdr.get("id", "")

            # Extract team name (prefer full name from hidden-on-mobile div)
            full_name_div = hdr.select_one("div.hidden.tb\\:block.font-bold")
            short_name_div = hdr.select_one("div.tb\\:hidden.font-bold")
            team_name = safe_text(full_name_div, "") or safe_text(short_name_div, "Unknown")

            # Extract score and overs from the score area
            score_div = hdr.select_one("div.flex.gap-4 > div")
            score_text = safe_text(score_div, "")
            inn_score = "--"
            inn_overs = ""
            if score_text:
                # Format is like "310-4 (50 Ov)" or "311-3 (41.1 Ov)"
                score_match = re.match(r'^(\d{1,4}-\d{1,2})\s*\((.+?)\)$', score_text)
                if score_match:
                    inn_score = score_match.group(1)
                    inn_overs = f"({score_match.group(2)})"

            # Find matching scorecard body: team-X-innings-N -> scard-team-X-innings-N
            scard_id = "scard-" + hdr_id
            scard = soup.find("div", id=scard_id)

            batsmen = []
            bowlers = []
            did_not_bat = []

            if scard:
                # Parse batsmen from scorecard-bat-grid rows
                bat_grids = scard.select("div.scorecard-bat-grid")
                for grid in bat_grids:
                    children = grid.find_all("div", recursive=False)
                    if not children:
                        continue
                    # Skip header row (contains "Batter" text in a font-bold div)
                    first_text = safe_text(children[0], "").strip()
                    if first_text == "Batter":
                        continue

                    # Batter row: cell[0]=name+dismissal, cell[1]=runs, cell[2]=balls, ...
                    if len(children) >= 3:
                        name_cell = children[0]
                        name_link = name_cell.find("a")
                        name = safe_text(name_link, "").strip() if name_link else first_text.split("\n")[0].strip()
                        name = name.split("(")[0].strip()

                        dismissal_div = name_cell.select_one("div.text-cbTxtSec")
                        dismissal = safe_text(dismissal_div, "").strip()
                        is_not_out = "not out" in dismissal.lower() or "batting" in dismissal.lower()

                        runs = safe_text(children[1], "--").strip()
                        balls = safe_text(children[2], "--").strip()

                        if name and runs != "--":
                            batsmen.append({
                                "name": name,
                                "runs": runs,
                                "balls": balls,
                                "isNotOut": is_not_out
                            })

                # Parse "Did not Bat" section within batting area
                dnb_section = None
                for div in scard.find_all("div", recursive=True):
                    bold_child = div.find("div", class_="font-bold")
                    if bold_child and "Did not Bat" in bold_child.get_text(strip=True):
                        dnb_section = div
                        break
                if dnb_section:
                    dnb_links = dnb_section.find_all("a")
                    if dnb_links:
                        did_not_bat = [a.get_text(strip=True).rstrip(",") for a in dnb_links if a.get_text(strip=True)]
                    else:
                        # Fallback: get text from the sibling div
                        dnb_children = dnb_section.find_all("div", recursive=False)
                        if len(dnb_children) >= 2:
                            dnb_text = dnb_children[1].get_text(strip=True)
                            did_not_bat = [n.strip() for n in dnb_text.split(",") if n.strip()]

                # Parse bowlers from scorecard-bowl-grid rows
                bowl_grids = scard.select("div.scorecard-bowl-grid")
                for grid in bowl_grids:
                    children = grid.find_all("div", recursive=False)
                    # Also check for direct <a> children (bowler name is an <a> tag)
                    first_child = grid.find(recursive=False)
                    if not first_child:
                        continue

                    # Skip header row
                    first_text = safe_text(first_child, "").strip()
                    if first_text == "Bowler":
                        continue

                    # Bowler row: cell[0]=name(a), cell[1]=overs, cell[2]=maidens, cell[3]=runs, cell[4]=wickets
                    all_cells = grid.find_all(recursive=False)
                    if len(all_cells) >= 5:
                        name_el = all_cells[0]
                        name = safe_text(name_el, "").strip()
                        overs_val = safe_text(all_cells[1], "--").strip()
                        runs_val = safe_text(all_cells[3], "--").strip()
                        wickets_val = safe_text(all_cells[4], "--").strip()

                        if name and not name.startswith("("):
                            bowlers.append({
                                "name": name,
                                "overs": overs_val,
                                "runs": runs_val,
                                "wickets": wickets_val
                            })

            response_innings.append({
                "team": team_name,
                "score": inn_score,
                "overs": inn_overs,
                "batsmen": batsmen,
                "bowlers": bowlers,
                "didNotBat": did_not_bat
            })

        # Try to find match status - look for result line
        status = ""
        for line in lines:
            line = line.strip()
            if any(kw in line.lower() for kw in ["won by", "lead by", "trail by", " need ", "match tied", "match drawn"]):
                if len(line) < 80 and len(line) > 5:
                    status = clean_status_text(line)
                    break

        return jsonify({
            "id": str(match_id),
            "title": title,
            "status": status,
            "venue": venue,
            "date": match_date,
            "innings": response_innings
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
