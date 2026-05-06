# %% Imports
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


# %% Data classes
@dataclass
class BallEvent:
    match_id: str
    date: str
    venue: str
    batting_team: str
    bowling_team: str
    innings: int
    over: int
    ball: int
    batter: str
    bowler: str
    non_striker: str
    batter_id: Optional[str]
    bowler_id: Optional[str]
    non_striker_id: Optional[str]
    runs_batter: int
    runs_extras: int
    runs_total: int
    is_wide: bool
    is_noball: bool
    is_wicket: bool
    wicket_kind: Optional[str]
    player_dismissed: Optional[str]
    player_dismissed_id: Optional[str]
    wicket_fielders: Optional[str]
    winner: Optional[str]
    season: str
    toss_winner: Optional[str]
    toss_decision: Optional[str]
    dls_method: Optional[str]
    is_impact_sub_match: bool
    event_stage: Optional[str]
    event_match_number: Optional[int]
    # Per-innings overs allocation for this team. Default to info.overs (20 in
    # IPL T20). DLS-truncated chases use innings.target.overs; DLS-truncated
    # first innings (rain forces an early close before all-out) use the actual
    # overs played. Used by NRR (#107): bowled-out teams get the full
    # allocation as the denominator regardless of when they were dismissed.
    innings_allocation: int


# %% Team name normalization
# Franchise aliases (DD↔DC, KXIP↔PBKS, RPS spelling variants, etc.) are loaded
# from config/team_aliases.yaml. Every downstream artifact reads canonical
# names from the parquet — this module is the single chokepoint.
_TEAM_ALIASES_PATH = Path(__file__).parent.parent / "config" / "team_aliases.yaml"


def _load_team_aliases(path: Path = _TEAM_ALIASES_PATH) -> tuple:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_slug: Dict[str, str] = {}
    canonical_to_aliases: Dict[str, List[str]] = {}
    canonical_season_labels: Dict[str, list] = {}
    seen_slugs = {}
    for entry in cfg.get("teams", []):
        canonical = entry["canonical"]
        slug = entry["slug"]
        if slug in seen_slugs:
            raise ValueError(f"Duplicate team slug '{slug}' for '{canonical}' and '{seen_slugs[slug]}'")
        seen_slugs[slug] = canonical
        canonical_to_slug[canonical] = slug
        aliases = entry.get("aliases", []) or [canonical]
        canonical_to_aliases[canonical] = list(aliases)
        for alias in aliases:
            alias_to_canonical[alias] = canonical
        # Self-reference always resolves
        alias_to_canonical[canonical] = canonical
        canonical_season_labels[canonical] = entry.get("season_labels", []) or []
    return alias_to_canonical, canonical_to_slug, canonical_to_aliases, canonical_season_labels


_ALIAS_TO_CANONICAL, TEAM_SLUG, TEAM_ALIASES, _SEASON_LABELS = _load_team_aliases()


# Canonical season-string overrides (issue #73). Cricsheet labels three IPL
# seasons as cross-year (2007/08, 2009/10, 2020/21) even though each was
# played within a single calendar year; rewrite to single-year strings so
# URLs, picker chips, and per-season exports use the year people actually
# remember. All other Cricsheet season strings pass through unchanged.
_SEASON_RENAMES: Dict[str, str] = {
    "2007/08": "2008",
    "2009/10": "2010",
    "2020/21": "2020",
}


def normalize_season(season: Optional[str]) -> Optional[str]:
    if season is None:
        return None
    return _SEASON_RENAMES.get(str(season), str(season))


def normalize_team(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _ALIAS_TO_CANONICAL.get(name, name)


def season_team_label(canonical: Optional[str], season_year: Optional[int]) -> Optional[str]:
    """Return the display label for a canonical team in a given season year.

    Falls back to the canonical name when no season override applies.
    """
    if canonical is None:
        return None
    rules = _SEASON_LABELS.get(canonical, [])
    if season_year is not None:
        for rule in rules:
            through = rule.get("through_year")
            from_year = rule.get("from_year")
            if through is not None and season_year <= int(through):
                return rule["label"]
            if from_year is not None and season_year >= int(from_year):
                return rule["label"]
    return canonical


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Schema validation
def validate_match_data(data: dict, filepath: Path) -> bool:
    """Validate that a Cricsheet match JSON has required fields."""
    info = data.get("info")
    if not isinstance(info, dict):
        return False
    for key in ("dates", "teams", "outcome"):
        if key not in info:
            return False
    teams = info.get("teams", [])
    if len(teams) < 2:
        return False
    innings = data.get("innings")
    if not isinstance(innings, list) or len(innings) == 0:
        return False
    for inn in innings:
        if "team" not in inn or "overs" not in inn:
            return False
        for over in inn["overs"]:
            if "deliveries" not in over:
                return False
            for delivery in over["deliveries"]:
                if "batter" not in delivery or "bowler" not in delivery or "runs" not in delivery:
                    return False
    return True


# %% Parse single match
def parse_match(filepath: Path) -> List[BallEvent]:
    """Parse a single Cricsheet JSON match file into BallEvent records."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if not validate_match_data(data, filepath):
        return []

    info = data.get("info", {})

    # Skip non-standard matches (no result, abandoned, etc.). Ties decided by
    # Super Over carry `outcome.result == "tie"` and `outcome.eliminator` set
    # to the SO winner instead of the usual `outcome.winner`; credit the
    # eliminator so these matches flow through normally (issue #84).
    outcome = info.get("outcome", {})
    winner_raw = outcome.get("winner") or (
        outcome.get("eliminator") if outcome.get("result") == "tie" else None
    )
    if winner_raw is None:
        return []

    match_id = filepath.stem
    dates = info.get("dates", ["unknown"])
    date = dates[0] if dates else "unknown"
    venue = info.get("venue", "unknown")
    winner = normalize_team(winner_raw)
    teams = [normalize_team(t) for t in info.get("teams", [])]
    season = normalize_season(str(info.get("season", "unknown")))

    # Toss info
    toss = info.get("toss", {})
    toss_winner = normalize_team(toss.get("winner"))
    toss_decision = toss.get("decision")

    # DLS method
    dls_method = outcome.get("method")

    # Impact sub detection (teams with > 11 players listed)
    players_info = info.get("players", {})
    is_impact_sub_match = any(len(v) > 11 for v in players_info.values())

    # Event metadata: `stage` is set on playoff matches ("Qualifier 1",
    # "Eliminator", "Qualifier 2", "Final", historical "Semi Final 1/2",
    # "3rd Place Play-off"). `match_number` is the league-stage seq for
    # tie-break ordering inside a season.
    event = info.get("event", {}) or {}
    event_stage = event.get("stage")
    event_match_number = event.get("match_number")
    if event_match_number is not None:
        try:
            event_match_number = int(event_match_number)
        except (TypeError, ValueError):
            event_match_number = None

    # Player name → unique ID mapping from Cricsheet registry
    registry = info.get("registry", {}).get("people", {})

    events: List[BallEvent] = []

    # Default match overs allocation (20 in IPL). Used as the per-innings
    # default unless DLS revisions kick in (issue #107). Detect a rain-
    # affected match either by an explicit DLS-decided outcome OR by any
    # innings carrying a DLS-revised target — a match where rain only cut
    # inn 1 short and the chase still completed normally (e.g. KKR-MI 2024)
    # has no `outcome.method` but the inn-2 target.overs is reduced.
    default_allocation = int(info.get("overs", 20))
    # A match is DLS-truncated if either (a) DLS decided the win or (b) any
    # innings carries a target.overs *below* the default — cricsheet sets
    # target.overs on every chase (even non-DLS, with target.overs ==
    # default), so we have to compare to the default rather than just
    # checking presence.
    has_dls_in_match = (
        dls_method is not None
        or any(
            (i.get("target") or {}).get("overs") is not None
            and int(i["target"]["overs"]) < default_allocation
            for i in data.get("innings", [])
            if not i.get("super_over")
        )
    )

    for innings_idx, innings_data in enumerate(data.get("innings", []), start=1):
        # Drop Super Over innings (issue #84). They're a 1-over tiebreaker with
        # a known target from ball 1; including them would pollute career
        # stats and the regulation-cricket WP signal. IPL/Cricinfo convention
        # is to track SO performances separately, not in batting/bowling
        # totals.
        if innings_data.get("super_over"):
            continue
        batting_team = normalize_team(innings_data.get("team", ""))
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else "unknown"

        # Determine this innings' overs allocation for NRR (#107). Inn-2 with
        # a DLS-revised target uses target.overs; inn-1 in a DLS match where
        # rain shortened the innings uses the actual played overs; otherwise
        # default. Bowled-out teams fall through here using the full
        # allocation in the formula even though their actual overs were
        # fewer.
        target = innings_data.get("target") or {}
        target_overs = target.get("overs")
        if target_overs is not None and int(target_overs) < default_allocation:
            # DLS-revised chase allocation
            innings_allocation = int(target_overs)
        elif has_dls_in_match and innings_idx == 1:
            # Inn 1 of a rain-affected match: use actual overs played as
            # the allocation (rain stopped them short of the default 20).
            actual_overs_played = len(innings_data.get("overs", []))
            innings_allocation = (
                actual_overs_played
                if actual_overs_played < default_allocation
                else default_allocation
            )
        else:
            innings_allocation = default_allocation

        for over_data in innings_data.get("overs", []):
            over_num = over_data.get("over", 0)

            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                runs = delivery.get("runs", {})
                wickets = delivery.get("wickets", [])
                extras = delivery.get("extras", {})

                is_wicket = len(wickets) > 0
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                wicket_kind = wickets[0].get("kind") if is_wicket else None
                player_dismissed = wickets[0].get("player_out") if is_wicket else None
                if is_wicket:
                    fielder_list = wickets[0].get("fielders") or []
                    fielder_names = [
                        f.get("name") for f in fielder_list if isinstance(f, dict) and f.get("name")
                    ]
                    wicket_fielders = ", ".join(fielder_names) if fielder_names else None
                else:
                    wicket_fielders = None

                batter_name = delivery.get("batter", "")
                bowler_name = delivery.get("bowler", "")
                non_striker_name = delivery.get("non_striker", "")

                events.append(BallEvent(
                    match_id=match_id,
                    date=date,
                    venue=venue,
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    innings=innings_idx,
                    over=over_num,
                    ball=ball_idx,
                    batter=batter_name,
                    bowler=bowler_name,
                    non_striker=non_striker_name,
                    batter_id=registry.get(batter_name),
                    bowler_id=registry.get(bowler_name),
                    non_striker_id=registry.get(non_striker_name),
                    runs_batter=runs.get("batter", 0),
                    runs_extras=runs.get("extras", 0),
                    runs_total=runs.get("total", 0),
                    is_wide=is_wide,
                    is_noball=is_noball,
                    is_wicket=is_wicket,
                    wicket_kind=wicket_kind,
                    player_dismissed=player_dismissed,
                    player_dismissed_id=registry.get(player_dismissed) if player_dismissed else None,
                    wicket_fielders=wicket_fielders,
                    winner=winner,
                    season=season,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    dls_method=dls_method,
                    event_stage=event_stage,
                    event_match_number=event_match_number,
                    is_impact_sub_match=is_impact_sub_match,
                    innings_allocation=innings_allocation,
                ))

    return events


# %% Parse all matches
def parse_all_matches(raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Parse all Cricsheet JSON files into a single DataFrame."""
    config = load_config()
    raw_dir = Path(raw_dir or config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_dir.glob("*.json"))
    # Filter out non-match files (Cricsheet includes a README JSON sometimes)
    json_files = [f for f in json_files if f.stem.isdigit() or f.stem.startswith("1")]

    print(f"Parsing {len(json_files)} match files...")

    all_events: List[Dict] = []
    skipped = 0

    for i, filepath in enumerate(json_files):
        try:
            events = parse_match(filepath)
            if events:
                all_events.extend([e.__dict__ for e in events])
            else:
                skipped += 1
        except Exception as e:
            print(f"  Error parsing {filepath.name}: {e}")
            skipped += 1

        if (i + 1) % 200 == 0:
            print(f"  Parsed {i + 1}/{len(json_files)} files...")

    print(f"  Parsed {len(json_files) - skipped} matches, skipped {skipped}")
    print(f"  Total ball events: {len(all_events):,}")

    df = pd.DataFrame(all_events)

    # Save
    output_path = processed_dir / config["data"]["ball_events_file"]
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return df


# %% Parse no-result / abandoned matches
_NO_RESULTS_SUPPLEMENT_PATH = Path(__file__).parent.parent / "config" / "no_results_supplement.yaml"


def _load_no_results_supplement(path: Path = _NO_RESULTS_SUPPLEMENT_PATH) -> tuple:
    """Read the hand-maintained YAML of NR matches to add and cricsheet NRs to exclude.

    Returns (add_rows, exclude_match_ids). Both are empty when the file is
    missing. See `config/no_results_supplement.yaml` for the rationale.
    """
    if not path.exists():
        return [], set()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    add_rows: List[Dict] = []
    for entry in cfg.get("add", []) or []:
        teams = [normalize_team(entry.get("team1")), normalize_team(entry.get("team2"))]
        if not (teams[0] and teams[1]):
            continue
        season = normalize_season(str(entry.get("season", "")))
        date = entry.get("date")
        slug1 = (TEAM_SLUG.get(teams[0]) or teams[0].lower().replace(" ", "-"))
        slug2 = (TEAM_SLUG.get(teams[1]) or teams[1].lower().replace(" ", "-"))
        match_id = f"supp-{season}-{date}-{slug1}-{slug2}"
        add_rows.append({
            "match_id": match_id,
            "date": date,
            "season": season,
            "team1": teams[0],
            "team2": teams[1],
            "venue": entry.get("venue"),
            "event_stage": entry.get("event_stage"),
        })
    exclude_ids = {str(e.get("match_id")) for e in (cfg.get("exclude", []) or []) if e.get("match_id")}
    return add_rows, exclude_ids


def parse_no_results_from_raw(raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Scan raw Cricsheet JSONs for matches where `outcome.result == 'no result'`,
    then merge in hand-maintained supplement entries.

    Cricsheet does not publish a JSON for matches abandoned without a ball
    bowled, and occasionally records a "no result" for a match that was later
    rescheduled and replayed (the replay being the canonical fixture). The
    `config/no_results_supplement.yaml` companion file adds the missing rows
    and excludes the voided ones. Each surviving row contributes 1 NR to each
    of its two teams in the season standings.
    """
    config = load_config()
    raw_dir = Path(raw_dir or config["data"]["raw_dir"])

    add_rows, exclude_ids = _load_no_results_supplement()

    rows: List[Dict] = []
    for filepath in sorted(raw_dir.glob("*.json")):
        if not (filepath.stem.isdigit() or filepath.stem.startswith("1")):
            continue
        if filepath.stem in exclude_ids:
            continue
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        info = data.get("info", {}) or {}
        outcome = info.get("outcome", {}) or {}
        if outcome.get("result") != "no result":
            continue
        teams = [normalize_team(t) for t in info.get("teams", []) if t]
        if len(teams) != 2:
            continue
        season = normalize_season(str(info.get("season", "")))
        dates = info.get("dates", []) or []
        date = dates[0] if dates else None
        event = info.get("event", {}) or {}
        rows.append({
            "match_id": filepath.stem,
            "date": date,
            "season": season,
            "team1": teams[0],
            "team2": teams[1],
            "venue": info.get("venue"),
            "event_stage": event.get("stage"),
        })

    df = pd.DataFrame(rows + add_rows)
    if not df.empty:
        # Dedupe in case a supplement row collides with a cricsheet row
        # (canonicalize the team pair so order doesn't matter).
        def _pair(row):
            t1, t2 = row["team1"], row["team2"]
            return tuple(sorted([t1, t2]))
        df["_pair"] = df.apply(_pair, axis=1)
        df = df.drop_duplicates(subset=["season", "date", "_pair"]).drop(columns=["_pair"])
    print(f"  Found {len(df)} no-result matches "
          f"({len(rows)} from cricsheet, {len(add_rows)} from supplement, "
          f"{len(exclude_ids)} excluded)")
    return df


# %% Main
if __name__ == "__main__":
    df = parse_all_matches()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Sample:\n{df.head()}")
