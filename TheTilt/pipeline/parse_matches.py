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
    winner: Optional[str]
    season: str
    toss_winner: Optional[str]
    toss_decision: Optional[str]
    dls_method: Optional[str]
    is_impact_sub_match: bool


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

    # Skip non-standard matches (no result, abandoned, etc.)
    outcome = info.get("outcome", {})
    if "winner" not in outcome:
        return []

    match_id = filepath.stem
    dates = info.get("dates", ["unknown"])
    date = dates[0] if dates else "unknown"
    venue = info.get("venue", "unknown")
    winner = normalize_team(outcome.get("winner"))
    teams = [normalize_team(t) for t in info.get("teams", [])]
    season = str(info.get("season", "unknown"))

    # Toss info
    toss = info.get("toss", {})
    toss_winner = normalize_team(toss.get("winner"))
    toss_decision = toss.get("decision")

    # DLS method
    dls_method = outcome.get("method")

    # Impact sub detection (teams with > 11 players listed)
    players_info = info.get("players", {})
    is_impact_sub_match = any(len(v) > 11 for v in players_info.values())

    # Player name → unique ID mapping from Cricsheet registry
    registry = info.get("registry", {}).get("people", {})

    events: List[BallEvent] = []

    for innings_idx, innings_data in enumerate(data.get("innings", []), start=1):
        batting_team = normalize_team(innings_data.get("team", ""))
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else "unknown"

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
                    winner=winner,
                    season=season,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    dls_method=dls_method,
                    is_impact_sub_match=is_impact_sub_match,
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


# %% Main
if __name__ == "__main__":
    df = parse_all_matches()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Sample:\n{df.head()}")
