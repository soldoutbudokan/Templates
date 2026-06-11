"""PlainCricket API.

Scrapes Cricbuzz pages and serves clean JSON for the PlainCricket frontend.

Endpoints:
    GET /api          - API info
    GET /api/matches  - list of live/upcoming/finished matches (live first)
    GET /api/score    - full scorecard for one match (?id=<match_id>)

Caching: responses carry ``Cache-Control: s-maxage`` headers so Vercel's edge
absorbs most traffic; a small in-memory TTL cache helps warm instances and
local development.
"""

import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import time as time_func
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, Tag
from flask import Flask, jsonify, request, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../public", static_url_path="")
app.json.sort_keys = False

CRICBUZZ_LIVE_URL = "https://www.cricbuzz.com/cricket-match/live-scores"
CRICBUZZ_SCORECARD_URL = "https://www.cricbuzz.com/live-cricket-scorecard/{match_id}"
CRICBUZZ_FACTS_URL = "https://www.cricbuzz.com/cricket-match-facts/{match_id}"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Innings score as rendered in scorecard headers: "310-4 (50 Ov)",
# "297 (49.2 Ov)" for all out, "657-7 d (158 Ov)" for declared.
SCORE_RE = re.compile(r"^(\d{1,4}(?:[-/]\d{1,2})?)\s*(d|dec\.?)?\s*\((.+?)\)$", re.IGNORECASE)

DNB_RE = re.compile(r"(did not bat|yet to bat)", re.IGNORECASE)

ABBR_TO_FULL: Dict[str, str] = {
    "WI": "West Indies", "RSA": "South Africa", "NZ": "New Zealand",
    "IND": "India", "PAK": "Pakistan", "AUS": "Australia",
    "ENG": "England", "SL": "Sri Lanka", "BAN": "Bangladesh",
    "AFG": "Afghanistan", "ZIM": "Zimbabwe", "IRE": "Ireland",
    "UAE": "UAE", "SCO": "Scotland", "NEP": "Nepal", "USA": "USA",
    "INDU19": "India U19", "PAKU19": "Pakistan U19", "ENGU19": "England U19",
    "AUSU19": "Australia U19", "NZU19": "New Zealand U19",
}
FULL_TO_ABBR: Dict[str, str] = {v.upper(): k for k, v in ABBR_TO_FULL.items()}

COMPLETE_KEYWORDS = (
    "won by", "match tied", "match drawn", "abandoned", "no result",
    "called off", "cancelled",
)
UPCOMING_KEYWORDS = (
    "preview", "upcoming", "match starts", "starts at", "yet to begin",
    "toss delayed",
)

MATCHES_CACHE_TTL = 30  # seconds
SCORE_CACHE_TTL = 20  # seconds
SCORE_CACHE_MAX_ENTRIES = 50
MAX_MATCHES = 15

_cache: Dict[str, Dict] = {}
_score_cache: Dict[str, Dict] = {}
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Per-thread session so warm invocations reuse connections safely."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def get_soup(url: str) -> BeautifulSoup:
    """Fetch URL and return a BeautifulSoup document. Raises on HTTP errors."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    response = _get_session().get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def safe_text(element: Optional[Tag], default: str = "--") -> str:
    """Safely extract stripped text from a BeautifulSoup element."""
    return element.text.strip() if element else default


def clean_status_text(status: Optional[str]) -> Optional[str]:
    """Expand Cricbuzz's ball-count shorthand ('36b' -> '36 balls')."""
    if not status:
        return status
    return re.sub(r"(\d+)b\b", r"\1 balls", status)


def format_match_times(iso_time_str: str) -> Tuple[str, str]:
    """Convert an ISO timestamp to US Eastern and IST display strings."""
    try:
        utc_time = datetime.fromisoformat(iso_time_str.replace("Z", "+00:00"))
        eastern = utc_time.astimezone(ZoneInfo("America/New_York"))
        ist = utc_time.astimezone(ZoneInfo("Asia/Kolkata"))
        # %Z renders EST or EDT depending on the date, and IST for Kolkata.
        return (
            eastern.strftime("%b %d, %I:%M %p %Z"),
            ist.strftime("%b %d, %I:%M %p %Z"),
        )
    except (ValueError, TypeError):
        return "", ""


def _start_epoch(iso_time_str: str) -> Optional[float]:
    """ISO timestamp -> epoch seconds, or None if unparseable."""
    try:
        return datetime.fromisoformat(iso_time_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def classify_status(status: Optional[str]) -> str:
    """Classify a Cricbuzz status line as 'live', 'upcoming' or 'complete'.

    Unknown/empty statuses default to 'live' so we still fetch scores for
    them; matches() downgrades to 'upcoming' if no scores turn up.
    """
    s = (status or "").strip().lower()
    if any(kw in s for kw in COMPLETE_KEYWORDS):
        return "complete"
    if any(kw in s for kw in UPCOMING_KEYWORDS):
        return "upcoming"
    return "live"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip()


def _same_team(a: str, b: str) -> bool:
    """Fuzzy team-name equality ('IND' vs 'India', 'India' vs 'India Women')."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _title_has_team(title_upper: str, team_name: str) -> bool:
    """True if the match title mentions this team as a whole word.

    Checks the full name and its known abbreviation. Word boundaries prevent
    false positives like 'IND' matching inside 'WEST INDIES'.
    """
    team_upper = (team_name or "").upper().strip()
    if not team_upper:
        return False
    candidates = {team_upper}
    abbr = FULL_TO_ABBR.get(team_upper)
    if abbr:
        candidates.add(abbr)
    return any(re.search(rf"\b{re.escape(c)}\b", title_upper) for c in candidates)


def parse_score_text(score_text: str) -> Tuple[str, str]:
    """Parse '310-4 (50 Ov)' / '297 (49.2 Ov)' / '657-7 d (158 Ov)'.

    Returns (score, overs) like ('657-7 d', '(158 Ov)'), or ('', '') if the
    text doesn't look like a score.
    """
    if not score_text:
        return "", ""
    match = SCORE_RE.match(score_text.strip())
    if not match:
        return "", ""
    score = match.group(1) + (" d" if match.group(2) else "")
    return score, f"({match.group(3)})"


# ---------------------------------------------------------------------------
# Scorecard parsing
# ---------------------------------------------------------------------------

def parse_innings_headers(soup: BeautifulSoup) -> List[Dict]:
    """Parse innings header divs (chronological order, deduplicated).

    Each innings is a div#team-{teamId}-innings-{n} holding the team name and
    the innings score; the page contains mobile + desktop copies, hence the
    dedup by id.
    """
    results: List[Dict] = []
    seen_ids = set()
    for hdr in soup.select('div[id^="team-"][id*="-innings-"]'):
        hdr_id = hdr.get("id", "")
        if not hdr_id or hdr_id in seen_ids:
            continue
        seen_ids.add(hdr_id)

        full_name_div = hdr.select_one("div.hidden.tb\\:block.font-bold")
        short_name_div = hdr.select_one("div.tb\\:hidden.font-bold")
        team_name = safe_text(full_name_div, "") or safe_text(short_name_div, "")

        score_div = hdr.select_one("div.flex.gap-4 > div")
        score, overs = parse_score_text(safe_text(score_div, ""))

        results.append({"id": hdr_id, "team": team_name, "score": score, "overs": overs})
    return results


def parse_venue_and_date(lines: List[str]) -> Tuple[str, str]:
    """Pull venue and date out of a scorecard page's text lines."""
    venue = ""
    match_date = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("Venue:", "Venue") and i + 1 < len(lines):
            venue = lines[i + 1].strip()
        if stripped in ("Date:", "Date") and i + 1 < len(lines):
            match_date = lines[i + 1].strip()
    return venue, match_date


def parse_batsmen(scard: Tag) -> List[Dict]:
    """Parse batting rows from a scorecard body (scorecard-bat-grid rows)."""
    batsmen: List[Dict] = []
    for grid in scard.select("div.scorecard-bat-grid"):
        children = grid.find_all("div", recursive=False)
        if not children:
            continue
        first_text = safe_text(children[0], "").strip()
        if first_text == "Batter":  # header row
            continue
        if len(children) < 3:
            continue

        name_cell = children[0]
        name_link = name_cell.find("a")
        name = safe_text(name_link, "").strip() if name_link else first_text.split("\n")[0].strip()
        name = name.split("(")[0].strip()

        dismissal = safe_text(name_cell.select_one("div.text-cbTxtSec"), "").strip()
        is_not_out = "not out" in dismissal.lower() or "batting" in dismissal.lower()

        runs = safe_text(children[1], "--").strip()
        balls = safe_text(children[2], "--").strip()

        if name and runs != "--":
            batsmen.append({"name": name, "runs": runs, "balls": balls, "isNotOut": is_not_out})
    return batsmen


def parse_bowlers(scard: Tag) -> List[Dict]:
    """Parse bowling rows from a scorecard body (scorecard-bowl-grid rows)."""
    bowlers: List[Dict] = []
    for grid in scard.select("div.scorecard-bowl-grid"):
        first_child = grid.find(recursive=False)
        if not first_child:
            continue
        if safe_text(first_child, "").strip() == "Bowler":  # header row
            continue

        all_cells = grid.find_all(recursive=False)
        if len(all_cells) < 5:
            continue
        name = safe_text(all_cells[0], "").strip()
        if not name or name.startswith("("):
            continue
        bowlers.append({
            "name": name,
            "overs": safe_text(all_cells[1], "--").strip(),
            "runs": safe_text(all_cells[3], "--").strip(),
            "wickets": safe_text(all_cells[4], "--").strip(),
        })
    return bowlers


def parse_extras(scard: Tag) -> str:
    """Find the extras line in a scorecard body, e.g. '12 (b 4, lb 2, w 5, nb 1)'.

    Text-based so it tolerates either 'value then breakdown' or the reverse.
    Returns '' when no extras line is present.
    """
    lines = [ln.strip() for ln in scard.get_text(separator="\n", strip=True).split("\n")]
    for i, line in enumerate(lines):
        if line.lower() == "extras" and i + 1 < len(lines):
            candidates = lines[i + 1:i + 3]
            value = next((c for c in candidates if re.match(r"^\d+$", c)), "")
            detail = next((c for c in candidates if c.startswith("(")), "")
            if value:
                return f"{value} {detail}".strip()
    return ""


def parse_did_not_bat(scard: Tag) -> Tuple[List[str], str]:
    """Parse the 'Did not bat' / 'Yet to bat' section.

    Returns (names, label) where label preserves which phrase Cricbuzz used,
    so a live innings can correctly say 'Yet to bat'.
    """
    section = None
    label = "Did not bat"

    for div in scard.find_all("div", recursive=True):
        bold_child = div.find("div", class_="font-bold")
        if bold_child:
            match = DNB_RE.search(bold_child.get_text(strip=True))
            if match:
                section, label = div, match.group(1)
                break

    if section is None:
        # The section can sit after the scorecard body rather than inside it.
        for sibling in scard.next_siblings:
            if not isinstance(sibling, Tag):
                continue
            sib_id = sibling.get("id", "")
            if sib_id.startswith(("team-", "scard-")):
                break
            bold_child = sibling.find("div", class_="font-bold")
            if bold_child:
                match = DNB_RE.search(bold_child.get_text(strip=True))
                if match:
                    section, label = sibling, match.group(1)
                    break

    if section is None:
        return [], "Did not bat"

    label = label[:1].upper() + label[1:].lower()

    names: List[str] = []
    links = section.find_all("a")
    if links:
        names = [a.get_text(strip=True).rstrip(",") for a in links if a.get_text(strip=True)]
    else:
        children = section.find_all("div", recursive=False)
        if len(children) >= 2:
            names = [n.strip() for n in children[1].get_text(strip=True).split(",") if n.strip()]
    return names, label


def build_scorecard(soup: BeautifulSoup, match_id: str) -> Dict:
    """Assemble the full /api/score payload from a scorecard page."""
    h1 = soup.find("h1")
    title = h1.text.strip().replace(" - Scorecard", "") if h1 else "Unknown Match"

    lines = soup.get_text(separator="\n", strip=True).split("\n")
    venue, match_date = parse_venue_and_date(lines)

    innings: List[Dict] = []
    for hdr in parse_innings_headers(soup):
        scard = soup.find("div", id=f"scard-{hdr['id']}")
        batsmen: List[Dict] = []
        bowlers: List[Dict] = []
        extras = ""
        did_not_bat: List[str] = []
        dnb_label = "Did not bat"
        if scard:
            batsmen = parse_batsmen(scard)
            bowlers = parse_bowlers(scard)
            extras = parse_extras(scard)
            did_not_bat, dnb_label = parse_did_not_bat(scard)

        innings.append({
            "team": hdr["team"] or "Unknown",
            "score": hdr["score"] or "--",
            "overs": hdr["overs"],
            "extras": extras,
            "batsmen": batsmen,
            "bowlers": bowlers,
            "didNotBat": did_not_bat,
            "didNotBatLabel": dnb_label,
        })

    status = ""
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped.lower() for kw in
               ("won by", "lead by", "trail by", " need ", "match tied", "match drawn",
                "abandoned", "no result")):
            if 5 < len(stripped) < 80:
                status = clean_status_text(stripped)
                break

    state = classify_status(status)
    if not status:
        state = "live" if innings else "upcoming"

    return {
        "id": match_id,
        "title": title,
        "status": status,
        "state": state,
        "venue": venue,
        "date": match_date,
        "innings": innings,
        "playingXI": {},
    }


# ---------------------------------------------------------------------------
# Match list parsing
# ---------------------------------------------------------------------------

def parse_json_ld_matches(soup: BeautifulSoup) -> Dict[str, Dict]:
    """Extract SportsEvent entries (teams, venue, start time) from JSON-LD."""
    results: Dict[str, Dict] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("@type") != "WebPage" or "mainEntity" not in data:
            continue
        for item in data["mainEntity"].get("itemListElement", []):
            try:
                if item.get("@type") != "SportsEvent":
                    continue
                competitors = item.get("competitor", [])
                if len(competitors) < 2:
                    continue
                results[item.get("name", "")] = {
                    "team1": competitors[0].get("name", ""),
                    "team2": competitors[1].get("name", ""),
                    "status": item.get("eventStatus", ""),
                    "venue": (item.get("location", "") or "").rstrip(", "),
                    "startTime": item.get("startDate", ""),
                }
            except (AttributeError, TypeError):
                continue
    return results


def parse_match_list(soup: BeautifulSoup) -> List[Dict]:
    """Parse the live-scores page into a list of match dicts (no scores yet)."""
    json_ld_matches = parse_json_ld_matches(soup)

    links = soup.find_all("a", href=re.compile(r"/live-cricket-scores/\d+"), class_="block")
    if not links:
        links = soup.find_all("a", href=re.compile(r"/live-cricket-scores/\d+"))

    match_list: List[Dict] = []
    seen_ids = set()
    for link in links:
        id_match = re.search(r"/live-cricket-scores/(\d+)", link.get("href", ""))
        if not id_match:
            continue
        match_id = id_match.group(1)
        if match_id in seen_ids:
            continue
        seen_ids.add(match_id)

        title_attr = link.get("title", "").strip()
        text = link.text.strip()
        if not title_attr and len(text) < 3:
            continue

        title = text if text else title_attr
        status = ""
        team1, team2 = "", ""

        # Title attribute looks like "Team1 vs Team2, Match Type - Status".
        if title_attr:
            base = title_attr
            if " - " in title_attr:
                base, raw_status = title_attr.rsplit(" - ", 1)
                status = clean_status_text(raw_status.strip())
            vs_match = re.match(r"^(.+?)\s+vs\s+(.+?)(?:,\s+.+)?$", base, re.IGNORECASE)
            if vs_match:
                team1, team2 = vs_match.group(1).strip(), vs_match.group(2).strip()

        if " - " in text:
            text_title, text_status = text.split(" - ", 1)
            title = text_title.strip()
            if not status:
                status = clean_status_text(text_status.strip())

        # Enrich with JSON-LD data (full team names, venue, start time).
        venue = ""
        start_time = ""
        title_upper = (title_attr or title).upper()
        for data in json_ld_matches.values():
            if _title_has_team(title_upper, data["team1"]) and _title_has_team(title_upper, data["team2"]):
                team1 = team1 or data["team1"]
                team2 = team2 or data["team2"]
                if not status and data["status"]:
                    status = clean_status_text(data["status"])
                venue = data["venue"]
                start_time = data["startTime"]
                break

        team1 = ABBR_TO_FULL.get(team1.upper(), team1) if team1 else team1
        team2 = ABBR_TO_FULL.get(team2.upper(), team2) if team2 else team2

        if not team1 or not team2:
            vs_match = re.match(r"^(.+?)\s+vs\s+(.+?)(?:\s+\d+(?:st|nd|rd|th)\s+|$)", title, re.IGNORECASE)
            if vs_match:
                t1_raw, t2_raw = vs_match.group(1).strip(), vs_match.group(2).strip()
                team1 = team1 or ABBR_TO_FULL.get(t1_raw.upper(), t1_raw)
                team2 = team2 or ABBR_TO_FULL.get(t2_raw.upper(), t2_raw)

        time_est, time_local = format_match_times(start_time) if start_time else ("", "")

        match_list.append({
            "id": match_id,
            "title": title,
            "team1": team1,
            "team2": team2,
            "status": status,
            "state": classify_status(status),
            "score1": "",
            "score2": "",
            "venue": venue,
            "startTime": start_time,
            "timeEST": time_est,
            "timeLocal": time_local,
        })
        if len(match_list) >= MAX_MATCHES:
            break
    return match_list


def format_team_scores(team_innings: List[Dict]) -> str:
    """Format one team's innings for the list view.

    Single innings: '245-3 (41.1 Ov)'. Multiple innings (tests) use the
    standard notation with overs on the current innings only: '245 & 91-2 (24 Ov)'.
    """
    if not team_innings:
        return ""
    if len(team_innings) == 1:
        inn = team_innings[0]
        return f"{inn['score']} {inn['overs']}".strip()
    joined = " & ".join(inn["score"] for inn in team_innings)
    last_overs = team_innings[-1]["overs"]
    return f"{joined} {last_overs}".strip() if last_overs else joined


def assign_scores_to_teams(match: Dict, innings: List[Dict]) -> None:
    """Attach innings scores to team1/team2 by team *name*, not innings order.

    Innings arrive in chronological order (whoever batted first), which is
    unrelated to the title order, so positional assignment puts scores next
    to the wrong team. If name matching can't account for every innings,
    fall back to grouping by the scorecard's own team names and overwrite
    team1/team2 so scores always sit next to the right name.
    """
    innings = [inn for inn in innings if inn.get("score")]
    if not innings:
        return

    team1, team2 = match.get("team1", ""), match.get("team2", "")
    bucket1 = [inn for inn in innings if _same_team(inn["team"], team1)]
    bucket2 = [inn for inn in innings if _same_team(inn["team"], team2)]

    if len(bucket1) + len(bucket2) != len(innings) or not team1 or not team2:
        order: List[str] = []
        groups: Dict[str, List[Dict]] = {}
        for inn in innings:
            key = _norm(inn["team"])
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(inn)
        bucket1 = groups[order[0]]
        bucket2 = groups[order[1]] if len(order) > 1 else []
        match["team1"] = bucket1[0]["team"]
        if bucket2:
            match["team2"] = bucket2[0]["team"]

    match["score1"] = format_team_scores(bucket1)
    match["score2"] = format_team_scores(bucket2)


def get_match_details(match_id: str) -> Dict:
    """Fetch innings scores, venue and date for a single match."""
    try:
        soup = get_soup(CRICBUZZ_SCORECARD_URL.format(match_id=match_id))
        lines = soup.get_text(separator="\n", strip=True).split("\n")
        venue, match_date = parse_venue_and_date(lines)
        innings = [inn for inn in parse_innings_headers(soup) if inn["score"] and inn["team"]]
        return {"innings": innings, "venue": venue, "date": match_date}
    except Exception:
        logger.exception("Failed to fetch details for match %s", match_id)
        return {"innings": [], "venue": "", "date": ""}


def get_playing_xi(match_id: str) -> Dict[str, List[str]]:
    """Scrape playing XI from the match facts page: {team_name: [players]}."""
    try:
        facts_soup = get_soup(CRICBUZZ_FACTS_URL.format(match_id=match_id))
        playing_xi: Dict[str, List[str]] = {}
        for players_div in facts_soup.find_all("div", class_="font-bold", string="Players"):
            # Direct <a> children of the sibling flex-wrap div (excludes bench).
            players_wrap = players_div.find_next_sibling("div")
            if not players_wrap:
                continue
            names: List[str] = []
            for a in players_wrap.find_all("a", recursive=False):
                name = a.get_text(strip=True).rstrip(",").strip()
                name = re.sub(r"\s*\([^)]*\)", "", name).rstrip(",").strip()
                if name:
                    names.append(name)

            row_grid = players_div.find_parent(class_=re.compile("facts-row-grid"))
            team_name = "Unknown"
            if row_grid:
                first_child = row_grid.find("div", recursive=False)
                if first_child:
                    team_name = first_child.get_text(strip=True).replace("squad", "").strip()

            if team_name != "Unknown" and names:
                playing_xi[team_name] = names
        return playing_xi
    except Exception:
        logger.exception("Failed to fetch playing XI for match %s", match_id)
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _json_response(payload: Dict, status: int = 200, max_age: int = 0, swr: int = 0):
    """JSON response with edge-cache headers (s-maxage) on cacheable 200s."""
    resp = jsonify(payload)
    resp.status_code = status
    if status == 200 and max_age > 0:
        resp.headers["Cache-Control"] = f"public, s-maxage={max_age}, stale-while-revalidate={swr}"
    else:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _sort_key(match: Dict) -> Tuple[int, float]:
    """Live first, then upcoming by start time, then finished (stable)."""
    order = {"live": 0, "upcoming": 1, "complete": 2}.get(match.get("state"), 1)
    if match.get("state") == "upcoming":
        ts = _start_epoch(match.get("startTime") or "")
        return order, ts if ts is not None else float("inf")
    return order, 0.0


@app.route("/")
def serve_frontend():
    """Serve the frontend HTML (local development; Vercel serves it statically)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api")
def api_info():
    return jsonify({
        "app": "PlainCricket API",
        "version": "2.0.0",
        "endpoints": {
            "/api": "API info",
            "/api/matches": "List of live/upcoming/finished matches (live first)",
            "/api/score?id=<match_id>": "Full scorecard for a specific match",
        },
    })


@app.route("/api/matches")
def matches():
    """Get the list of current matches, sorted live-first."""
    now = time_func()
    cached = _cache.get("matches")
    if cached and now - cached["time"] < MATCHES_CACHE_TTL:
        return _json_response(cached["payload"], max_age=30, swr=120)

    try:
        soup = get_soup(CRICBUZZ_LIVE_URL)
    except Exception:
        logger.exception("Failed to fetch Cricbuzz live scores page")
        return _json_response(
            {"error": "could not reach the scores source", "matches": [], "degraded": True},
            status=502,
        )

    try:
        match_list = parse_match_list(soup)
    except Exception:
        logger.exception("Failed to parse match list")
        return _json_response(
            {"error": "could not parse the scores source", "matches": [], "degraded": True},
            status=500,
        )

    def fetch_details(match: Dict) -> None:
        details = get_match_details(match["id"])
        assign_scores_to_teams(match, details["innings"])
        if not match.get("venue") and details["venue"]:
            match["venue"] = details["venue"]
        if not match.get("timeEST") and details["date"]:
            match["timeEST"] = details["date"]

    to_fetch = [m for m in match_list if m["state"] != "upcoming"]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_details, m) for m in to_fetch]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    logger.exception("Failed to fetch match details")

    # Self-correct classification: scores mean it's not upcoming; an empty
    # status with no scores means it hasn't started.
    for match in match_list:
        if match["state"] == "upcoming" and (match["score1"] or match["score2"]):
            match["state"] = "live"
        elif match["state"] == "live" and not match["status"] and not match["score1"] and not match["score2"]:
            match["state"] = "upcoming"

    match_list.sort(key=_sort_key)

    # Degraded = we parsed nothing, or started matches all came back scoreless,
    # which usually means Cricbuzz changed their markup.
    started = [m for m in match_list if m["state"] != "upcoming"]
    degraded = not match_list or (bool(started) and all(not m["score1"] and not m["score2"] for m in started))
    if degraded:
        logger.warning("Matches response is degraded: %d matches, %d started", len(match_list), len(started))

    payload = {"matches": match_list, "degraded": degraded}
    _cache["matches"] = {"payload": payload, "time": time_func()}
    return _json_response(payload, max_age=30, swr=120)


@app.route("/api/score")
def score():
    """Get the detailed scorecard for a specific match."""
    match_id = request.args.get("id", "")
    if not match_id.isdigit():
        return _json_response(
            {"error": "match id must be numeric, e.g. /api/score?id=12345"}, status=400,
        )

    now = time_func()
    cached = _score_cache.get(match_id)
    if cached and now - cached["time"] < SCORE_CACHE_TTL:
        return _json_response(cached["payload"], max_age=20, swr=60)

    try:
        soup = get_soup(CRICBUZZ_SCORECARD_URL.format(match_id=match_id))
    except Exception:
        logger.exception("Failed to fetch scorecard for match %s", match_id)
        return _json_response({"error": "could not reach the scores source"}, status=502)

    try:
        payload = build_scorecard(soup, match_id)
    except Exception:
        logger.exception("Failed to parse scorecard for match %s", match_id)
        return _json_response({"error": "could not parse the scorecard"}, status=500)

    # Show squads for teams that haven't batted yet.
    if len(payload["innings"]) < 2:
        payload["playingXI"] = get_playing_xi(match_id)

    if len(_score_cache) >= SCORE_CACHE_MAX_ENTRIES:
        oldest = min(_score_cache, key=lambda k: _score_cache[k]["time"])
        _score_cache.pop(oldest, None)
    _score_cache[match_id] = {"payload": payload, "time": time_func()}

    return _json_response(payload, max_age=20, swr=60)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
