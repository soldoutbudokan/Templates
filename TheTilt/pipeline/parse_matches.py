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
    runs_batter: int
    runs_extras: int
    runs_total: int
    is_wicket: bool
    wicket_kind: Optional[str]
    player_dismissed: Optional[str]
    winner: Optional[str]
    season: str


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Parse single match
def parse_match(filepath: Path) -> List[BallEvent]:
    """Parse a single Cricsheet JSON match file into BallEvent records."""
    with open(filepath, "r") as f:
        data = json.load(f)

    info = data.get("info", {})

    # Skip non-standard matches (no result, abandoned, etc.)
    outcome = info.get("outcome", {})
    if "winner" not in outcome:
        return []

    match_id = filepath.stem
    dates = info.get("dates", ["unknown"])
    date = dates[0] if dates else "unknown"
    venue = info.get("venue", "unknown")
    winner = outcome.get("winner")
    teams = info.get("teams", [])
    season = str(info.get("season", "unknown"))

    events: List[BallEvent] = []

    for innings_idx, innings_data in enumerate(data.get("innings", []), start=1):
        batting_team = innings_data.get("team", "")
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else "unknown"

        for over_data in innings_data.get("overs", []):
            over_num = over_data.get("over", 0)

            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                runs = delivery.get("runs", {})
                wickets = delivery.get("wickets", [])

                is_wicket = len(wickets) > 0
                wicket_kind = wickets[0].get("kind") if is_wicket else None
                player_dismissed = wickets[0].get("player_out") if is_wicket else None

                events.append(BallEvent(
                    match_id=match_id,
                    date=date,
                    venue=venue,
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    innings=innings_idx,
                    over=over_num,
                    ball=ball_idx,
                    batter=delivery.get("batter", ""),
                    bowler=delivery.get("bowler", ""),
                    non_striker=delivery.get("non_striker", ""),
                    runs_batter=runs.get("batter", 0),
                    runs_extras=runs.get("extras", 0),
                    runs_total=runs.get("total", 0),
                    is_wicket=is_wicket,
                    wicket_kind=wicket_kind,
                    player_dismissed=player_dismissed,
                    winner=winner,
                    season=season,
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
