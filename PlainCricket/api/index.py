import random
import requests
import re
import os
from bs4 import BeautifulSoup as bs
from markupsafe import escape
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import pytz

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
    return bs(response.content, "lxml")


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

        # Get scores
        scores = []
        for i, line in enumerate(lines):
            if re.match(r'^\d{1,3}-\d{1,2}$', line):
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if re.match(r'^\(\d+', next_line):
                    team_name = lines[i - 1] if i > 0 else ""
                    if team_name not in ["Total", "Batter", "Bowler", "Did not Bat"]:
                        scores.append({"team": team_name, "score": line, "overs": next_line})

        # Dedupe scores
        seen = set()
        unique = []
        for s in scores:
            if s["team"] not in seen:
                seen.add(s["team"])
                unique.append(s)

        return {"scores": unique[:2], "venue": venue, "date": match_date}
    except:
        return {"scores": [], "venue": "", "date": ""}


@app.route("/api/matches")
def matches():
    """Get list of current matches"""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # Find all links to live cricket scores
        links = soup.find_all("a", href=True)

        for link in links:
            href = link.get("href", "")
            if "/live-cricket-scores/" not in href:
                continue

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

            # Parse title from link text
            text = link.text.strip()
            if not text or len(text) < 3:
                continue

            # Extract status if present
            title = text
            status = ""
            if " - " in text:
                parts = text.split(" - ", 1)
                title = parts[0].strip()
                status = parts[1].strip() if len(parts) > 1 else ""

            # Initialize venue and time
            venue = ""
            start_time = ""
            time_est = ""
            time_local = ""

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

            # Try to find richer data from JSON-LD
            team1, team2 = "", ""
            title_upper = title.upper()
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
                    team1, team2 = t1, t2
                    if not status and data["status"]:
                        status = data["status"]
                    venue = data.get("venue", "")
                    start_time = data.get("startTime", "")
                    break

            # Fallback: parse team names from title like "IND vs NZ" or "India vs New Zealand"
            if not team1 or not team2:
                # Match "Team1 vs Team2" allowing multi-word names, stopping at common suffixes
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

        # Fetch scores/venue/date in parallel for matches
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

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_details, m): m for m in match_list}
            for future in as_completed(futures):
                try:
                    future.result()
                except:
                    pass

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

        # Find innings - pattern: TeamCode, TeamName, Score, (Overs), Batter
        innings = []
        i = 0
        while i < len(lines) - 4:
            line = lines[i]
            # Look for score pattern like "271-5" or "225-10"
            if re.match(r'^\d{1,3}-\d{1,2}$', line):
                score = line
                # Check if next line has overs
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if re.match(r'^\(\d+(?:\.\d+)? Ov\)$', next_line):
                    overs = next_line
                    # Look back for team name
                    team_name = lines[i - 1] if i > 0 else "Unknown"
                    team_code = lines[i - 2] if i > 1 else ""

                    # Validate it's a team (not "Total" or other text)
                    if team_name not in ["Total", "Batter", "Bowler", "Did not Bat"] and len(team_name) > 2:
                        innings.append({
                            "team": team_name,
                            "code": team_code,
                            "score": score,
                            "overs": overs
                        })
            i += 1

        # Deduplicate innings (page has duplicates for mobile/desktop)
        seen_teams = set()
        unique_innings = []
        for inn in innings:
            if inn["team"] not in seen_teams:
                seen_teams.add(inn["team"])
                unique_innings.append(inn)
        innings = unique_innings[:2]  # Max 2 innings

        # Parse batting rows - split by "Batter" header
        bat_rows = soup.select("div[class*='scorecard-bat-grid']")
        all_batsmen = []
        current_innings_batsmen = []

        for row in bat_rows:
            row_text = row.get_text(separator="|", strip=True)
            parts = [p.strip() for p in row_text.split("|") if p.strip()]

            if not parts:
                continue

            if parts[0] == "Batter":
                # New innings header - save previous if exists
                if current_innings_batsmen:
                    all_batsmen.append(current_innings_batsmen)
                current_innings_batsmen = []
            elif len(parts) >= 3 and parts[0] not in ["Batter", "Extras", "Total", "Did not Bat"]:
                name = parts[0].split("(")[0].strip()
                # Find runs and balls
                runs = "--"
                balls = "--"
                for p in parts[1:]:
                    if p.isdigit():
                        if runs == "--":
                            runs = p
                        elif balls == "--":
                            balls = p
                            break
                if name and runs != "--":
                    current_innings_batsmen.append({"name": name, "runs": runs, "balls": balls})

        if current_innings_batsmen:
            all_batsmen.append(current_innings_batsmen)

        # Parse "Did not Bat" sections - associated with preceding innings total
        dnb_by_score = {}
        current_dnb = []
        last_score = None
        in_dnb = False

        for i, line in enumerate(lines):
            # Track the last innings score we saw
            if re.match(r'^\d{1,3}-\d{1,2}$', line):
                potential_score = line
                # Check if this is an innings total (followed by overs)
                if i + 1 < len(lines) and re.match(r'^\(\d+', lines[i + 1]):
                    last_score = potential_score

            if "Did not Bat" in line:
                if current_dnb and last_score:
                    dnb_by_score[last_score] = current_dnb
                current_dnb = []
                in_dnb = True
            elif in_dnb:
                if line in ["Bowler", "O", "M", "R", "W", "Batter", "Total", "Extras"]:
                    in_dnb = False
                    if current_dnb and last_score:
                        dnb_by_score[last_score] = current_dnb
                        current_dnb = []
                else:
                    name = line.rstrip(",").strip()
                    if name and len(name) > 1 and not name.isdigit():
                        current_dnb.append(name)

        if current_dnb and last_score:
            dnb_by_score[last_score] = current_dnb

        # Parse bowling rows similarly
        bowl_rows = soup.select("div[class*='scorecard-bowl-grid']")
        all_bowlers = []
        current_innings_bowlers = []

        for row in bowl_rows:
            row_text = row.get_text(separator="|", strip=True)
            parts = [p.strip() for p in row_text.split("|") if p.strip()]

            if not parts:
                continue

            if parts[0] == "Bowler":
                if current_innings_bowlers:
                    all_bowlers.append(current_innings_bowlers)
                current_innings_bowlers = []
            elif len(parts) >= 4 and parts[0] != "Bowler":
                name = parts[0]
                overs = parts[1] if len(parts) > 1 else "--"
                runs = parts[3] if len(parts) > 3 else "--"
                wickets = parts[4] if len(parts) > 4 else "--"
                if name and not name.startswith("("):
                    current_innings_bowlers.append({
                        "name": name,
                        "overs": overs,
                        "runs": runs,
                        "wickets": wickets
                    })

        if current_innings_bowlers:
            all_bowlers.append(current_innings_bowlers)

        # Deduplicate (page has mobile/desktop duplicates)
        if len(all_batsmen) > 2:
            all_batsmen = all_batsmen[:2]
        if len(all_bowlers) > 2:
            all_bowlers = all_bowlers[:2]

        # Try to find match status - look for result line
        status = ""
        for line in lines:
            line = line.strip()
            if any(kw in line.lower() for kw in ["won by", "lead by", "trail by", " need ", "match tied", "match drawn"]):
                if len(line) < 80 and len(line) > 5:
                    status = line
                    break

        # Build response with both innings
        response_innings = []
        for idx, inn in enumerate(innings):
            inn_data = {
                "team": inn["team"],
                "score": inn["score"],
                "overs": inn["overs"],
                "batsmen": all_batsmen[idx] if idx < len(all_batsmen) else [],
                "bowlers": all_bowlers[idx] if idx < len(all_bowlers) else [],
                "didNotBat": dnb_by_score.get(inn["score"], [])
            }
            response_innings.append(inn_data)

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
