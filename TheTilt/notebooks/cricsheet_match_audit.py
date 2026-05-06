# %% Cricsheet match audit (issue #100)
"""Identify every Cricsheet IPL match that is in the raw dump but does
not flow through `parse_matches` into `ball_events.parquet`. Classify
each by the reason it was excluded so the user can decide whether to
patch via `config/no_results_supplement.yaml`.

Run from `TheTilt/`:
    ./venv/bin/python notebooks/cricsheet_match_audit.py
"""

# %% Imports
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Helpers
def _load_supplement_ids() -> tuple:
    """Return (added_keys, excluded_match_ids) from `config/no_results_supplement.yaml`.

    `add` rows don't carry a cricsheet match_id (they're for cricsheet gaps),
    so we key them by (season, date, sorted-team-pair). `exclude` rows do
    carry a cricsheet match_id.
    """
    path = PROJECT_ROOT / "config" / "no_results_supplement.yaml"
    if not path.exists():
        return set(), set()
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    added = set()
    for r in (doc.get("add") or []):
        season = str(r.get("season", ""))
        date = str(r.get("date", ""))
        teams = tuple(sorted([str(r.get("team1", "")), str(r.get("team2", ""))]))
        added.add((season, date, teams))
    excluded = {str(r.get("match_id")) for r in (doc.get("exclude") or []) if r.get("match_id")}
    return added, excluded


def _classify(raw: dict) -> dict:
    """Return diagnostic fields for a single cricsheet JSON dict."""
    info = raw.get("info") or {}
    outcome = info.get("outcome") or {}
    teams = info.get("teams") or []
    season_raw = info.get("season")
    dates = info.get("dates") or []
    venue = info.get("venue", "?")
    innings = raw.get("innings") or []

    has_winner = bool(outcome.get("winner"))
    has_eliminator = bool(outcome.get("eliminator"))
    result = outcome.get("result")
    method = outcome.get("method")
    n_innings = sum(1 for i in innings if not i.get("super_over"))
    has_super_over = any(i.get("super_over") for i in innings)
    only_super_over = (n_innings == 0 and has_super_over)

    if not teams or len(teams) < 2:
        reason = "missing_teams"
    elif n_innings < 2 and only_super_over:
        reason = "super_over_only"
    elif n_innings < 2:
        reason = "missing_innings"
    elif result == "no result" and not has_winner:
        reason = "no_result"
    elif result == "tie" and not has_winner and has_eliminator:
        reason = "tie_super_over_eliminator"
    elif not has_winner:
        reason = "no_winner"
    else:
        reason = "would_parse"

    return {
        "date": dates[0] if dates else "?",
        "season": str(season_raw) if season_raw is not None else "?",
        "teams": " vs ".join(teams),
        "venue": venue,
        "result": result or "",
        "method": method or "",
        "n_innings": n_innings,
        "has_super_over": has_super_over,
        "winner": outcome.get("winner") or outcome.get("eliminator") or "",
        "reason": reason,
    }


# %% Run
def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    parsed_path = PROJECT_ROOT / "data" / "processed" / "ball_events.parquet"

    raw_files = sorted(p for p in raw_dir.glob("*.json") if p.stem.isdigit() or p.stem.startswith("1"))
    raw_ids = [p.stem for p in raw_files]
    parsed_ids = set(pd.read_parquet(parsed_path)["match_id"].astype(str).unique())

    print(f"Cricsheet raw files: {len(raw_ids)}")
    print(f"Parsed into ball_events.parquet: {len(parsed_ids)}")
    print(f"Missing (raw but not parsed): {len(raw_ids) - len(parsed_ids)}")
    print()

    _, excluded_ids = _load_supplement_ids()

    rows = []
    for path in raw_files:
        if path.stem in parsed_ids:
            continue
        with open(path) as f:
            raw = json.load(f)
        info = _classify(raw)
        info["match_id"] = path.stem
        info["voided_in_supplement"] = path.stem in excluded_ids
        rows.append(info)

    df = pd.DataFrame(rows).sort_values(["season", "date"])
    print("Reason breakdown:")
    print(df["reason"].value_counts().to_string())
    print()
    print("Detail:")
    cols = ["match_id", "date", "season", "teams", "venue", "result", "method",
            "n_innings", "has_super_over", "winner", "reason",
            "voided_in_supplement"]
    print(df[cols].to_string(index=False))

    out_path = PROJECT_ROOT / "data" / "processed" / "missing_match_audit.csv"
    df[cols].to_csv(out_path, index=False)
    print(f"\nFull audit saved to {out_path}")


if __name__ == "__main__":
    main()
