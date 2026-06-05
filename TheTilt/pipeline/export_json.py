# %% Imports
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from pipeline.compute_tilt import make_slug, make_team_slug
from pipeline.parse_matches import TEAM_ALIASES, TEAM_SLUG, season_team_label


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_PLAYER_COUNTRY_CACHE: Optional[dict] = None


def load_player_countries(config_path: str = "config/player_countries.yaml") -> dict:
    """Load `{player_id: country_code}` from the hand-curated YAML.

    Default falls back to the `default_country` key (currently "IN") for any
    player_id not listed. The lookup is cached after first call so repeated
    exports don't re-parse the file. See `config/player_countries.yaml` and
    `/tmp/build_player_countries.py` for how the file is regenerated.
    """
    global _PLAYER_COUNTRY_CACHE
    if _PLAYER_COUNTRY_CACHE is not None:
        return _PLAYER_COUNTRY_CACHE
    try:
        with open(config_path, "r") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _PLAYER_COUNTRY_CACHE = {"_default": "IN", "_map": {}}
        return _PLAYER_COUNTRY_CACHE
    default = str(doc.get("default_country") or "IN")
    raw = doc.get("players") or {}
    mapping = {}
    for pid, entry in raw.items():
        if isinstance(entry, dict):
            c = entry.get("country")
        else:
            c = entry
        if c:
            mapping[str(pid)] = str(c)
    _PLAYER_COUNTRY_CACHE = {"_default": default, "_map": mapping}
    return _PLAYER_COUNTRY_CACHE


def country_for(player_id: Optional[str]) -> str:
    """Return ICC country code for a player_id, falling back to default_country."""
    cache = load_player_countries()
    if player_id is None:
        return cache["_default"]
    return cache["_map"].get(str(player_id), cache["_default"])


def load_completed_seasons(config_path: str = "config/seasons.yaml") -> set:
    """Hand-maintained list of seasons whose final has been played.

    Seasons not in this set are treated as in-progress: champion is exported
    as null even if the underlying data has a "last match winner" placeholder.
    """
    try:
        with open(config_path, "r") as f:
            doc = yaml.safe_load(f) or {}
        return set(str(s) for s in (doc.get("completed_seasons") or []))
    except FileNotFoundError:
        return set()


def _team_list(value, primary: str) -> list:
    """Normalize a player's `teams` cell (read from player_tilt.parquet) to a
    list of franchise names.

    pandas reads a parquet list column back as a numpy ndarray, not a Python
    list (pyarrow >= ~13 / pandas 2.x). A bare `isinstance(value, list)` check
    therefore evaluates False on the read-back and silently collapses every
    multi-franchise player down to their single primary team. Accept any
    non-string sequence; fall back to [primary] when the cell is missing, NaN,
    or empty.
    """
    if value is None or isinstance(value, str):
        return [primary]
    try:
        names = [t for t in value if isinstance(t, str) and t]
    except TypeError:
        return [primary]
    return names or [primary]


def _compute_team_season_nrr(deltas_df: pd.DataFrame, team: str, season: str) -> Optional[float]:
    """Season-final NRR for a (team, season), per IPL convention (issue #107).

    For each regular-season match the team played:
      * Runs scored / conceded use every-delivery totals (extras included).
      * Overs use legal deliveries only (wides + no-balls don't count toward
        overs faced/bowled, matching the IPL formula).
      * If a batting side is bowled out, its denominator is the full innings
        allocation (20 in T20, or the DLS-revised allocation for chases /
        rain-cut first innings — see `parse_matches.parse_match` for how
        `innings_allocation` is set per ball event).
      * If a chasing side wins early, its denominator is the actual overs
        faced — they didn't need their full allocation.
      * For DLS-decided chases (cricsheet flags `outcome.method == "D/L"`),
        both sides use the revised allocation as the denominator. The IPL
        playing conditions reference a "DLS-equivalent overs" formula for
        the team-batting-first that would shrink the residual further but
        requires the ICC's T20 Standard Edition resource table — that
        table is not publicly available, only the 50-over Standard table
        is published. We approximate with allocation-as-denominator on
        both sides; residuals against Wikipedia for DLS-heavy seasons are
        ≤0.083 NRR units and confined to the bowled-first team.

    Restricted to regular-season balls (issue #75) so the NRR matches the
    league-table convention. Returns None if the team didn't play any
    regular-season game.
    """
    df = deltas_df
    if "event_stage" in df.columns:
        df = df[df["event_stage"].isna()]
    df = df[df["season"] == season]
    if len(df) == 0:
        return None

    df = df.copy()
    df["_legal_ball"] = (~df["is_wide"] & ~df["is_noball"]).astype(int)

    def per_innings(side_mask: pd.Series) -> pd.DataFrame:
        sub = df[side_mask].groupby(["match_id", "innings"], as_index=False).agg(
            runs=("runs_total", "sum"),
            legal_balls=("_legal_ball", "sum"),
            allocation=("innings_allocation", "first"),
            batting_team=("batting_team", "first"),
            winner=("winner", "first"),
            dls_method=("dls_method", "first"),
        )
        sub["actual_overs"] = sub["legal_balls"] / 6.0
        team_batting_won = sub["winner"] == sub["batting_team"]
        ended_early = sub["actual_overs"] < sub["allocation"] - 0.01
        dls_decided = sub["dls_method"].notna()
        bowled_out_or_dls = ended_early & (
            (sub["innings"] == 1)
            | ((sub["innings"] == 2) & ~team_batting_won)
            | ((sub["innings"] == 2) & dls_decided)
        )
        sub["denominator"] = sub["actual_overs"]
        sub.loc[bowled_out_or_dls, "denominator"] = (
            sub.loc[bowled_out_or_dls, "allocation"].astype(float)
        )
        return sub

    bat = per_innings(df["batting_team"] == team)
    bowl = per_innings(df["bowling_team"] == team)
    if len(bat) == 0 or len(bowl) == 0:
        return None

    runs_for = float(bat["runs"].sum())
    overs_for = float(bat["denominator"].sum())
    runs_against = float(bowl["runs"].sum())
    overs_against = float(bowl["denominator"].sum())
    if overs_for == 0 or overs_against == 0:
        return None
    return round(runs_for / overs_for - runs_against / overs_against, 3)


def _season_position_map(team_season_tilt: pd.DataFrame, deltas_df: pd.DataFrame) -> dict:
    """{(team, season_str): position} for every team-season, ranked by IPL
    convention — points desc, then wins desc, then NRR desc — with genuine ties
    sharing a position. Single source so the team pages agree with the season
    standings (the team pages previously used a points-only rank) (#146).

    NOTE: this mirrors the ranking in export_seasons() (which assigns the season
    page's positions inline); keep the two tie-break rules in sync.
    """
    positions = {}
    for season in team_season_tilt["season"].unique():
        rows = []
        for _, r in team_season_tilt[team_season_tilt["season"] == season].iterrows():
            wins = int(r["wins"])
            no_results = int(r.get("no_results", 0))
            rows.append({
                "team": r["team"],
                "points": int(r.get("points", wins * 2 + no_results)),
                "wins": wins,
                "nrr": _compute_team_season_nrr(deltas_df, r["team"], str(season)),
            })
        rows.sort(key=lambda t: (-t["points"], -t["wins"], -(t["nrr"] if t["nrr"] is not None else -99)))
        prev_key = None
        prev_pos = 0
        for i, t in enumerate(rows, 1):
            key = (t["points"], t["wins"], t["nrr"])
            if key != prev_key:
                prev_pos = i
                prev_key = key
            positions[(t["team"], str(season))] = prev_pos
    return positions


# %% Legal-delivery flags
# A batter "faces" legal deliveries + no-balls (wides don't count — re-bowled, batter not credited).
# A bowler's over counts only legal deliveries (both wides and no-balls are re-bowled).

# Cricsheet wicket kinds credited to the bowler (lowercase, as parsed).
BOWLER_WICKET_KINDS = frozenset({
    "bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket",
})


def _add_legal_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["legal_bat"] = (~df["is_wide"]).astype(int)
    df["legal_bowl"] = (~df["is_wide"] & ~df["is_noball"]).astype(int)
    # Wickets credited to the bowler (purple-cap convention). Run outs, retired,
    # obstructing the field, etc. are dismissals but NOT bowling wickets (#123).
    # Every *bowler* wicket count uses this; team wickets-fallen and the
    # fall-of-wickets / dismissals list keep using is_wicket directly.
    df["is_bowler_wicket"] = (
        df["is_wicket"] & df["wicket_kind"].isin(BOWLER_WICKET_KINDS)
    )
    return df


# %% Counting-stat helpers (shared by player, team-season, season, leaders exports)
def _batting_counting_stats(
    bat_slice: pd.DataFrame,
    player_id: str,
    player_name: str,
    *,
    include_dismissals: bool = False,
) -> Optional[dict]:
    """Compute batting counting stats over a slice of legal-flagged ball data.

    Caller must have run `_add_legal_flags` already. Returns None if slice empty.
    `include_dismissals` controls whether the dismissals key is in the output
    (career view: yes; season view: no — preserves the existing JSON shape).
    """
    if len(bat_slice) == 0:
        return None
    runs = int(bat_slice["runs_batter"].sum())
    balls = int(bat_slice["legal_bat"].sum())
    innings = int(bat_slice["match_id"].nunique())
    if player_id:
        dismissals = int(bat_slice["player_dismissed_id"].eq(player_id).sum())
    else:
        dismissals = int(bat_slice["player_dismissed"].eq(player_name).sum())
    match_runs = bat_slice.groupby("match_id")["runs_batter"].sum()
    out = {
        "runs": runs,
        "innings": innings,
        "balls": balls,
        "avg": round(runs / dismissals, 2) if dismissals else None,
        "sr": round(runs / max(balls, 1) * 100, 2),
        "hs": int(match_runs.max()),
    }
    if include_dismissals:
        out["dismissals"] = dismissals
    out["not_outs"] = innings - dismissals
    # A "fifty" is 50–99; centuries are counted separately (#156).
    out["fifties"] = int(((match_runs >= 50) & (match_runs < 100)).sum())
    out["hundreds"] = int((match_runs >= 100).sum())
    return out


def _bowling_counting_stats(bowl_slice: pd.DataFrame) -> Optional[dict]:
    """Compute bowling counting stats over a legal-flagged slice. None if empty."""
    if len(bowl_slice) == 0:
        return None
    wickets = int(bowl_slice["is_bowler_wicket"].sum())
    balls = int(bowl_slice["legal_bowl"].sum())
    runs_conceded = int(bowl_slice["runs_total"].sum())
    innings = int(bowl_slice["match_id"].nunique())
    bowl_match = bowl_slice.groupby("match_id").agg(
        wickets=("is_bowler_wicket", "sum"),
        runs=("runs_total", "sum"),
    )
    best_idx = bowl_match["wickets"].idxmax()
    best_w = int(bowl_match.loc[best_idx, "wickets"])
    best_r = int(bowl_match.loc[best_idx, "runs"])
    return {
        "wickets": wickets,
        "innings": innings,
        "balls": balls,
        "runs_conceded": runs_conceded,
        "avg": round(runs_conceded / wickets, 2) if wickets else None,
        "economy": round(runs_conceded / max(balls, 1) * 6, 2),
        "best_figures": f"{best_w}/{best_r}",
    }


def _build_match_info_cache(deltas_df: pd.DataFrame) -> dict:
    """Pre-compute (date, venue, winner, season, teams, scores, toss, result_margin, stage) per match."""
    cache = {}
    for mid in deltas_df["match_id"].unique():
        mdf = deltas_df[deltas_df["match_id"] == mid]
        first = mdf.iloc[0]
        scores = {}
        innings_totals: dict[int, tuple] = {}
        for inn_num in sorted(mdf["innings"].unique()):
            inn_df = mdf[mdf["innings"] == inn_num]
            last_ball = inn_df.iloc[-1]
            total_r = int(last_ball["runs_scored"]) + int(last_ball["runs_total"])
            total_w = int(last_ball["wickets_fallen"]) + (1 if last_ball["is_wicket"] else 0)
            batting_team = str(inn_df.iloc[0]["batting_team"])
            scores[batting_team] = f"{total_r}/{total_w}"
            innings_totals[int(inn_num)] = (batting_team, total_r, total_w)

        winner = str(first["winner"]) if pd.notna(first["winner"]) else None
        result_margin = _compute_result_margin(innings_totals, winner)

        toss_winner = first.get("toss_winner")
        toss_decision = first.get("toss_decision")
        event_stage = first.get("event_stage")
        event_match_number = first.get("event_match_number")

        cache[mid] = {
            "date": str(first["date"]),
            "venue": str(first["venue"]),
            "winner": winner,
            "season": str(first["season"]),
            "teams": list(mdf["batting_team"].unique()),
            "scores": scores,
            "toss_winner": str(toss_winner) if pd.notna(toss_winner) else None,
            "toss_decision": str(toss_decision) if pd.notna(toss_decision) else None,
            "result_margin": result_margin,
            "event_stage": str(event_stage) if pd.notna(event_stage) else None,
            "event_match_number": int(event_match_number) if pd.notna(event_match_number) else None,
        }
    return cache


def _build_playoffs_block(season_match_ids: list, match_info: dict) -> list:
    """Return playoff matches for a season as a date-ordered list.

    Each entry: {stage, match_id, date, teams, scores, winner, result_margin}.
    Stage names come from Cricsheet `info.event.stage` (e.g. "Qualifier 1",
    "Eliminator", "Qualifier 2", "Final", historical "Semi Final 1/2",
    "3rd Place Play-off"). Empty list when a season has no playoff data
    (in-progress mid-season, or non-playoff format).
    """
    playoffs = []
    for mid in season_match_ids:
        mi = match_info.get(mid)
        if mi is None or not mi.get("event_stage"):
            continue
        playoffs.append({
            "stage": mi["event_stage"],
            "match_id": str(mid),
            "date": mi["date"],
            "venue": mi.get("venue"),
            "teams": mi["teams"],
            "scores": mi.get("scores", {}),
            "winner": mi.get("winner"),
            "result_margin": mi.get("result_margin"),
        })
    playoffs.sort(key=lambda p: p["date"])
    return playoffs


def _compute_result_margin(
    innings_totals: dict, winner: Optional[str]
) -> Optional[str]:
    """Format match result as e.g. "won by 7 wickets" or "won by 23 runs".

    Innings 1 batting team won → margin in runs (defending). Innings 2 batting
    team won → margin in wickets (chasing). Returns None when winner is unknown
    or innings totals are incomplete (e.g. abandoned matches that slipped past
    parser filters).
    """
    if winner is None or 1 not in innings_totals or 2 not in innings_totals:
        return None
    inn1_team, inn1_runs, _ = innings_totals[1]
    inn2_team, inn2_runs, inn2_wickets = innings_totals[2]
    if winner == inn1_team:
        margin = inn1_runs - inn2_runs
        return f"won by {margin} run{'s' if margin != 1 else ''}"
    if winner == inn2_team:
        wickets_left = 10 - inn2_wickets
        return f"won by {wickets_left} wicket{'s' if wickets_left != 1 else ''}"
    return None


def _build_player_slug_lookup(player_tilt: pd.DataFrame, *, include_empty_id: bool = False) -> dict:
    """Map player_id -> slug. `include_empty_id=True` mirrors the legacy GOATs
    behavior of also storing a slug under the empty-string key."""
    lookup = {}
    if player_tilt is None:
        return lookup
    for _, row in player_tilt.iterrows():
        pid = row.get("player_id", "")
        if pid:
            lookup[pid] = make_slug(row["player"], pid)
        elif include_empty_id:
            lookup[pid] = make_slug(row["player"], None)
    return lookup


def _build_player_name_lookup(player_tilt: pd.DataFrame) -> dict:
    """Map player_id -> full_name (falls back to display name)."""
    return {row.get("player_id", ""): row.get("full_name", row["player"]) for _, row in player_tilt.iterrows()}


# %% Export rankings
def export_rankings(
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
    min_matches: int = 10,
) -> Path:
    """Export player rankings to JSON for the website."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_matches = config["export"].get("min_matches", min_matches)

    # Filter by minimum matches
    qualified = player_tilt[player_tilt["total_matches"] >= min_matches].copy()
    print(f"  {len(qualified)} players with >= {min_matches} matches (of {len(player_tilt)} total)")

    # Classify player roles
    qualified["role"] = "batter"
    ball_ratio = qualified["bowling_balls"] / qualified["batting_balls"].replace(0, 1)
    qualified.loc[
        (qualified["bowling_balls"] >= 50) & (qualified["batting_balls"] >= 50) &
        ball_ratio.between(0.3, 3.0),
        "role",
    ] = "all-rounder"
    qualified.loc[
        (qualified["bowling_balls"] >= 100) &
        (qualified["bowling_balls"] > qualified["batting_balls"] * 1.5),
        "role",
    ] = "bowler"

    # Minimum balls threshold for per-role display
    min_role_balls = config["export"].get("min_role_balls", 50)

    # Build JSON records
    rankings = []
    for _, row in qualified.iterrows():
        player_id = row.get("player_id", "")
        bat_balls = int(row["batting_balls"])
        bowl_balls = int(row["bowling_balls"])
        bat_qualified = bat_balls >= min_role_balls
        bowl_qualified = bowl_balls >= min_role_balls
        rankings.append({
            "rank": 0,  # Will be set after sorting
            "player": row["player"],
            "full_name": row.get("full_name", row["player"]),
            "player_id": player_id,
            "slug": make_slug(row["player"], player_id if player_id else None),
            "country": country_for(player_id),
            "team": row["team"],
            "teams": _team_list(row.get("teams"), row["team"]),
            "role": row["role"],
            "total_tilt_per_match": round(row["total_tilt_per_match"], 5),
            "batting_tilt_per_match": round(row["batting_tilt_per_match"], 5) if bat_qualified else None,
            "bowling_tilt_per_match": round(row["bowling_tilt_per_match"], 5) if bowl_qualified else None,
            "total_tilt": round(row["total_tilt"], 5),
            "batting_total_tilt": round(row["batting_total_tilt"], 5),
            "bowling_total_tilt": round(row["bowling_total_tilt"], 5),
            "shrunk_total_tilt_per_match": round(row.get("shrunk_total_tilt_per_match", row["total_tilt_per_match"]), 5),
            "shrunk_batting_tilt_per_match": round(row.get("shrunk_batting_tilt_per_match", row["batting_tilt_per_match"]), 5) if bat_qualified else None,
            "shrunk_bowling_tilt_per_match": round(row.get("shrunk_bowling_tilt_per_match", row["bowling_tilt_per_match"]), 5) if bowl_qualified else None,
            "tilt_ci_lower": round(row.get("tilt_ci_lower", 0), 5),
            "tilt_ci_upper": round(row.get("tilt_ci_upper", 0), 5),
            "tilt_ci_lower_90": round(row.get("tilt_ci_lower_90", 0), 5),
            "tilt_ci_upper_90": round(row.get("tilt_ci_upper_90", 0), 5),
            "confidence": row.get("confidence", "low"),
            "total_matches": int(row["total_matches"]),
            "batting_balls": int(row["batting_balls"]),
            "bowling_balls": int(row["bowling_balls"]),
        })

    # Sort by 90% CI lower bound (penalizes small samples naturally)
    rankings.sort(key=lambda x: x["tilt_ci_lower_90"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    # Write
    rankings_path = output_dir / config["export"]["rankings_file"]
    with open(rankings_path, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"  Exported {len(rankings)} players to {rankings_path}")
    return rankings_path


# %% Export player details
def export_player_details(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Export per-player detail JSON files.

    Every player in `player_tilt` gets a file. The `min_matches` config gate
    applies only to the leaderboard (`export_rankings`); player pages must
    exist for every slug that any `matches/<id>.json` can link to (issue #91).
    """
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    players_dir = output_dir / config["export"]["players_dir"]
    players_dir.mkdir(parents=True, exist_ok=True)

    qualified = player_tilt

    deltas_df = _add_legal_flags(deltas_df)

    match_info_cache = _build_match_info_cache(deltas_df)

    exported = 0
    for _, row in qualified.iterrows():
        player_name = row["player"]
        player_id = row.get("player_id", "")
        slug = make_slug(player_name, player_id if player_id else None)

        # Get this player's ball-by-ball data using unique player ID
        if player_id:
            bat_df = deltas_df[deltas_df["batter_id"] == player_id]
            bowl_df = deltas_df[deltas_df["bowler_id"] == player_id]
        else:
            bat_df = deltas_df[deltas_df["batter"] == player_name]
            bowl_df = deltas_df[deltas_df["bowler"] == player_name]

        # Season breakdown (batting)
        season_batting = (
            bat_df.groupby("season")
            .agg(tilt=("delta_wp", "sum"), balls=("legal_bat", "sum"), matches=("match_id", "nunique"))
            .reset_index()
        )
        season_batting["tilt_per_match"] = season_batting["tilt"] / season_batting["matches"]

        # Season breakdown (bowling)
        bowl_df_copy = bowl_df.copy()
        bowl_df_copy["bowling_delta"] = -bowl_df_copy["delta_wp"]
        season_bowling = (
            bowl_df_copy.groupby("season")
            .agg(tilt=("bowling_delta", "sum"), balls=("legal_bowl", "sum"), matches=("match_id", "nunique"))
            .reset_index()
        )
        season_bowling["tilt_per_match"] = season_bowling["tilt"] / season_bowling["matches"]

        # Merge seasons: combine batting and bowling by season
        all_seasons = set(season_batting["season"].tolist()) | set(season_bowling["season"].tolist())
        bat_season_map = {r["season"]: r for _, r in season_batting.iterrows()}
        bowl_season_map = {r["season"]: r for _, r in season_bowling.iterrows()}

        merged_seasons = []
        for season in sorted(all_seasons, key=str):
            bat_s = bat_season_map.get(season)
            bowl_s = bowl_season_map.get(season)
            bat_tilt = float(bat_s["tilt_per_match"]) if bat_s is not None else 0.0
            bowl_tilt = float(bowl_s["tilt_per_match"]) if bowl_s is not None else 0.0
            bat_matches = int(bat_s["matches"]) if bat_s is not None else 0
            bowl_matches = int(bowl_s["matches"]) if bowl_s is not None else 0
            bat_balls = int(bat_s["balls"]) if bat_s is not None else 0
            bowl_balls = int(bowl_s["balls"]) if bowl_s is not None else 0

            # Per-season team and counting stats
            bat_season_df = bat_df[bat_df["season"] == season]
            bowl_season_df = bowl_df[bowl_df["season"] == season]
            season_bat_stats = _batting_counting_stats(bat_season_df, player_id, player_name, include_dismissals=True)
            season_bowl_stats = _bowling_counting_stats(bowl_season_df)
            season_team = None
            if len(bat_season_df) > 0:
                season_team = bat_season_df["batting_team"].mode().iloc[0]
            elif len(bowl_season_df) > 0:
                season_team = bowl_season_df["bowling_team"].mode().iloc[0]

            # Per-match breakdown for this season
            bat_match_ids = set(bat_season_df["match_id"].unique()) if len(bat_season_df) > 0 else set()
            bowl_match_ids = set(bowl_season_df["match_id"].unique()) if len(bowl_season_df) > 0 else set()
            all_match_ids = bat_match_ids | bowl_match_ids

            season_matches = []
            for mid in all_match_ids:
                mi = match_info_cache.get(mid)
                if mi is None:
                    continue

                # Determine player's team and opponent
                if mid in bat_match_ids:
                    player_team = str(bat_season_df[bat_season_df["match_id"] == mid]["batting_team"].iloc[0])
                else:
                    player_team = str(bowl_season_df[bowl_season_df["match_id"] == mid]["bowling_team"].iloc[0])
                opponent = [t for t in mi["teams"] if t != player_team]
                opponent = opponent[0] if opponent else player_team

                team_score = mi["scores"].get(player_team)
                opp_score = mi["scores"].get(opponent)

                # Batting stats for this match
                bat_m = bat_season_df[bat_season_df["match_id"] == mid] if mid in bat_match_ids else pd.DataFrame()
                bat_runs_val = int(bat_m["runs_batter"].sum()) if len(bat_m) > 0 else None
                bat_balls_val = int(bat_m["legal_bat"].sum()) if len(bat_m) > 0 else None
                bat_sr_val = round(bat_runs_val / max(bat_balls_val, 1) * 100, 1) if bat_runs_val is not None else None
                bat_tilt_val = round(float(bat_m["delta_wp"].sum()), 5) if len(bat_m) > 0 else None
                if len(bat_m) > 0:
                    if player_id:
                        bat_not_out_val = not bool(bat_m["player_dismissed_id"].eq(player_id).any())
                    else:
                        bat_not_out_val = not bool(bat_m["player_dismissed"].eq(player_name).any())
                else:
                    bat_not_out_val = None

                # Bowling stats for this match
                bowl_m = bowl_season_df[bowl_season_df["match_id"] == mid] if mid in bowl_match_ids else pd.DataFrame()
                bowl_wkts_val = int(bowl_m["is_bowler_wicket"].sum()) if len(bowl_m) > 0 else None
                bowl_balls_val = int(bowl_m["legal_bowl"].sum()) if len(bowl_m) > 0 else None
                bowl_runs_val = int(bowl_m["runs_total"].sum()) if len(bowl_m) > 0 else None
                bowl_econ_val = round(bowl_runs_val / max(bowl_balls_val, 1) * 6, 2) if bowl_runs_val is not None else None
                bowl_tilt_val = round(-float(bowl_m["delta_wp"].sum()), 5) if len(bowl_m) > 0 else None

                total_tilt_val = round((bat_tilt_val or 0) + (bowl_tilt_val or 0), 5)

                season_matches.append({
                    "match_id": str(mid),
                    "date": mi["date"],
                    "opponent": opponent,
                    "venue": mi["venue"],
                    "team_score": team_score,
                    "opponent_score": opp_score,
                    "winner": mi["winner"],
                    "bat_runs": bat_runs_val,
                    "bat_balls": bat_balls_val,
                    "bat_sr": bat_sr_val,
                    "bat_tilt": bat_tilt_val,
                    "bat_not_out": bat_not_out_val,
                    "bowl_wkts": bowl_wkts_val,
                    "bowl_balls": bowl_balls_val,
                    "bowl_runs": bowl_runs_val,
                    "bowl_econ": bowl_econ_val,
                    "bowl_tilt": bowl_tilt_val,
                    "total_tilt": total_tilt_val,
                })
            season_matches.sort(key=lambda x: x["date"])

            season_entry = {
                "season": str(season),
                "team": str(season_team) if season_team else None,
                "batting_tilt_per_match": round(bat_tilt, 5),
                "bowling_tilt_per_match": round(bowl_tilt, 5),
                "batting_matches": bat_matches,
                "bowling_matches": bowl_matches,
                "batting_balls": bat_balls,
                "bowling_balls": bowl_balls,
                "matches": season_matches,
            }
            if season_bat_stats:
                season_entry["batting_stats"] = season_bat_stats
            if season_bowl_stats:
                season_entry["bowling_stats"] = season_bowl_stats
            merged_seasons.append(season_entry)

        # Phase breakdown (batting)
        phase_batting = []
        for phase_name, col in [("powerplay", "is_powerplay"), ("middle", "is_middle"), ("death", "is_death")]:
            phase_df = bat_df[bat_df[col] == 1]
            if len(phase_df) > 0:
                phase_batting.append({
                    "phase": phase_name,
                    "tilt": round(phase_df["delta_wp"].sum(), 5),
                    "balls": int(phase_df["legal_bat"].sum()),
                    "avg_delta": round(phase_df["delta_wp"].mean(), 6),
                })

        # Phase breakdown (bowling)
        phase_bowling = []
        for phase_name, col in [("powerplay", "is_powerplay"), ("middle", "is_middle"), ("death", "is_death")]:
            phase_df = bowl_df_copy[bowl_df_copy[col] == 1]
            if len(phase_df) > 0:
                phase_bowling.append({
                    "phase": phase_name,
                    "tilt": round(phase_df["bowling_delta"].sum(), 5),
                    "balls": int(phase_df["legal_bowl"].sum()),
                    "avg_delta": round(phase_df["bowling_delta"].mean(), 6),
                })

        # Best/worst match performances (batting)
        if player_id:
            bat_with_dismissal = bat_df.assign(
                _self_dismissed=bat_df["player_dismissed_id"].eq(player_id)
            )
        else:
            bat_with_dismissal = bat_df.assign(
                _self_dismissed=bat_df["player_dismissed"].eq(player_name)
            )
        match_perf = (
            bat_with_dismissal.groupby(["match_id", "date", "bowling_team"])
            .agg(
                tilt=("delta_wp", "sum"),
                balls=("legal_bat", "sum"),
                runs=("runs_batter", "sum"),
                dismissed=("_self_dismissed", "any"),
            )
            .reset_index()
            .sort_values("tilt", ascending=False)
        )
        match_perf["not_out"] = ~match_perf["dismissed"]
        best_matches = match_perf.head(5).to_dict("records")
        worst_matches = match_perf.tail(5).sort_values("tilt").to_dict("records")

        # Best/worst match performances (bowling)
        if len(bowl_df_copy) > 0:
            bowl_match_perf = (
                bowl_df_copy.groupby(["match_id", "date", "batting_team"])
                .agg(tilt=("bowling_delta", "sum"), balls=("legal_bowl", "sum"), wickets=("is_bowler_wicket", "sum"))
                .reset_index()
                .sort_values("tilt", ascending=False)
            )
            bowling_best = bowl_match_perf.head(5).to_dict("records")
            bowling_worst = bowl_match_perf.tail(5).sort_values("tilt").to_dict("records")
        else:
            bowling_best = []
            bowling_worst = []

        # Career trend: cumulative TILT over time
        all_player_df = pd.concat([bat_df.assign(_role="bat"), bowl_df_copy.assign(_role="bowl")]) if len(bowl_df_copy) > 0 else bat_df.assign(_role="bat")
        if len(all_player_df) > 0:
            # Combine batting delta_wp and bowling -delta_wp per match
            match_tilt_trend = all_player_df.copy()
            match_tilt_trend["_tilt"] = match_tilt_trend.apply(
                lambda r: r["delta_wp"] if r["_role"] == "bat" else r.get("bowling_delta", -r["delta_wp"]), axis=1
            )
            match_trend = (
                match_tilt_trend.groupby(["match_id", "date"])
                .agg(tilt=("_tilt", "sum"))
                .reset_index()
                .sort_values("date")
            )
            match_trend["cumulative"] = match_trend["tilt"].cumsum()
            career_trend = [
                {"match": i + 1, "date": str(r["date"]), "tilt": round(r["tilt"], 5), "cumulative": round(r["cumulative"], 5)}
                for i, (_, r) in enumerate(match_trend.iterrows())
            ]
        else:
            career_trend = []

        # Team matchups: TILT per match vs each opponent
        if len(bat_df) > 0:
            opp_tilt = (
                bat_df.groupby("bowling_team")
                .agg(tilt=("delta_wp", "sum"), matches=("match_id", "nunique"))
                .reset_index()
            )
            opp_tilt["tilt_per_match"] = opp_tilt["tilt"] / opp_tilt["matches"]
            opp_tilt = opp_tilt.sort_values("tilt_per_match", ascending=False)
            team_matchups = [
                {"opponent": r["bowling_team"], "tilt_per_match": round(r["tilt_per_match"], 5), "matches": int(r["matches"])}
                for _, r in opp_tilt.iterrows()
            ]
        else:
            team_matchups = []

        # Bowling team matchups: bowling TILT per match vs each batting opponent
        if len(bowl_df_copy) > 0:
            bowl_opp_tilt = (
                bowl_df_copy.groupby("batting_team")
                .agg(tilt=("bowling_delta", "sum"), matches=("match_id", "nunique"))
                .reset_index()
            )
            bowl_opp_tilt["tilt_per_match"] = bowl_opp_tilt["tilt"] / bowl_opp_tilt["matches"]
            bowl_opp_tilt = bowl_opp_tilt.sort_values("tilt_per_match", ascending=False)
            bowling_team_matchups = [
                {"opponent": r["batting_team"], "tilt_per_match": round(r["tilt_per_match"], 5), "matches": int(r["matches"])}
                for _, r in bowl_opp_tilt.iterrows()
            ]
        else:
            bowling_team_matchups = []

        # Traditional counting stats (career-level)
        batting_stats = _batting_counting_stats(bat_df, player_id, player_name, include_dismissals=True)
        bowling_stats = _bowling_counting_stats(bowl_df)

        # Match TILT distribution (for histogram)
        match_tilt_distribution = []
        if len(career_trend) > 0:
            match_tilt_distribution = [round(t["tilt"], 5) for t in career_trend]

        # Build player detail JSON
        bat_balls = int(row["batting_balls"])
        bowl_balls = int(row["bowling_balls"])
        min_role_balls = config["export"].get("min_role_balls", 50)
        bat_qualified = bat_balls >= min_role_balls
        bowl_qualified = bowl_balls >= min_role_balls
        detail = {
            "player": player_name,
            "full_name": row.get("full_name", player_name),
            "player_id": player_id,
            "slug": slug,
            "country": country_for(player_id),
            "team": row["team"],
            "teams": _team_list(row.get("teams"), row["team"]),
            "total_tilt_per_match": round(row["total_tilt_per_match"], 5),
            "batting_tilt_per_match": round(row["batting_tilt_per_match"], 5) if bat_qualified else None,
            "bowling_tilt_per_match": round(row["bowling_tilt_per_match"], 5) if bowl_qualified else None,
            "shrunk_total_tilt_per_match": round(row.get("shrunk_total_tilt_per_match", row["total_tilt_per_match"]), 5),
            "shrunk_batting_tilt_per_match": round(row.get("shrunk_batting_tilt_per_match", row["batting_tilt_per_match"]), 5) if bat_qualified else None,
            "shrunk_bowling_tilt_per_match": round(row.get("shrunk_bowling_tilt_per_match", row["bowling_tilt_per_match"]), 5) if bowl_qualified else None,
            "tilt_ci_lower": round(row.get("tilt_ci_lower", 0), 5),
            "tilt_ci_upper": round(row.get("tilt_ci_upper", 0), 5),
            "tilt_ci_lower_90": round(row.get("tilt_ci_lower_90", 0), 5),
            "tilt_ci_upper_90": round(row.get("tilt_ci_upper_90", 0), 5),
            "confidence": row.get("confidence", "low"),
            "total_tilt": round(row["total_tilt"], 5),
            "total_matches": int(row["total_matches"]),
            "batting_balls": int(row["batting_balls"]),
            "bowling_balls": int(row["bowling_balls"]),
            "seasons": merged_seasons,
            "batting_phases": phase_batting,
            "bowling_phases": phase_bowling,
            "best_matches": [
                {
                    "match_id": str(r["match_id"]),
                    "date": str(r["date"]),
                    "vs": r["bowling_team"],
                    "tilt": round(r["tilt"], 5),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                    "not_out": bool(r["not_out"]),
                }
                for r in best_matches
            ],
            "worst_matches": [
                {
                    "match_id": str(r["match_id"]),
                    "date": str(r["date"]),
                    "vs": r["bowling_team"],
                    "tilt": round(r["tilt"], 5),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                    "not_out": bool(r["not_out"]),
                }
                for r in worst_matches
            ],
            "bowling_best_matches": [
                {
                    "match_id": str(r["match_id"]),
                    "date": str(r["date"]),
                    "vs": r["batting_team"],
                    "tilt": round(r["tilt"], 5),
                    "wickets": int(r["wickets"]),
                    "balls": int(r["balls"]),
                }
                for r in bowling_best
            ],
            "bowling_worst_matches": [
                {
                    "match_id": str(r["match_id"]),
                    "date": str(r["date"]),
                    "vs": r["batting_team"],
                    "tilt": round(r["tilt"], 5),
                    "wickets": int(r["wickets"]),
                    "balls": int(r["balls"]),
                }
                for r in bowling_worst
            ],
            "career_trend": career_trend,
            "team_matchups": team_matchups,
            "bowling_team_matchups": bowling_team_matchups,
            "batting_stats": batting_stats,
            "bowling_stats": bowling_stats,
            "match_tilt_distribution": match_tilt_distribution,
        }

        player_path = players_dir / f"{slug}.json"
        with open(player_path, "w") as f:
            json.dump(detail, f, indent=2)

        exported += 1

    print(f"  Exported {exported} player detail files to {players_dir}")
    return players_dir


# %% Export match details
def export_match_details(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Export per-match detail JSON files with scorecards and ball-by-ball WP."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    slug_lookup = _build_player_slug_lookup(player_tilt)

    deltas_df = _add_legal_flags(deltas_df)

    match_ids = deltas_df["match_id"].unique()
    print(f"  Exporting {len(match_ids)} match detail files...")

    # Pre-compute match_info for every match so result_margin / event_stage /
    # scores flow into both the per-match JSONs and match_index.json.
    match_info_cache = _build_match_info_cache(deltas_df)

    match_index = []

    for match_id in match_ids:
        mdf = deltas_df[deltas_df["match_id"] == match_id].sort_values(["innings", "ball_number"])

        # Match info
        first_row = mdf.iloc[0]
        teams = sorted(mdf["batting_team"].unique().tolist())
        winner = str(first_row["winner"]) if pd.notna(first_row["winner"]) else None
        cached = match_info_cache.get(match_id, {})

        match_info = {
            "match_id": str(match_id),
            "date": str(first_row["date"]),
            "venue": str(first_row["venue"]),
            "season": str(first_row["season"]),
            "teams": teams,
            "winner": winner,
            "toss_winner": str(first_row.get("toss_winner", "")) if pd.notna(first_row.get("toss_winner")) else None,
            "toss_decision": str(first_row.get("toss_decision", "")) if pd.notna(first_row.get("toss_decision")) else None,
            "result_margin": cached.get("result_margin"),
            "event_stage": cached.get("event_stage"),
            "scores": cached.get("scores"),
        }

        # Per-innings data
        innings_data = []
        for inn_num in sorted(mdf["innings"].unique()):
            inn_df = mdf[mdf["innings"] == inn_num].sort_values("ball_number").copy()
            # Legal-ball index within each over (1..6) for cricket over notation in
            # fall-of-wickets. Counts only legal deliveries (legal_bowl excludes
            # wides + no-balls), so it never exceeds .6 — the physical `ball` slot
            # does, since a wide/no-ball earlier in the over pushes `ball + 1` to
            # .7/.8 (#175). A wicket on an illegal delivery reports the legal balls
            # completed so far, matching scorecard convention.
            inn_df["legal_ball_in_over"] = inn_df.groupby("over")["legal_bowl"].cumsum()
            batting_team = str(inn_df.iloc[0]["batting_team"])
            bowling_team = str(inn_df.iloc[0]["bowling_team"])

            # Dismissal lookup keyed by dismissed player_id (one row per wicket).
            # Score-at-fall is the cumulative team score just AFTER this delivery,
            # which matches the conventional "67-3" notation. Wicket position
            # 1..N is the order of falls in the innings.
            dismissals_df = inn_df[inn_df["is_wicket"]].copy()
            dismissals_df = dismissals_df.sort_values("ball_number")
            dismissals_df["wicket_position"] = range(1, len(dismissals_df) + 1)
            fall_of_wickets = []
            dismissal_map = {}
            for _, r in dismissals_df.iterrows():
                fow_runs = int(r["runs_scored"]) + int(r["runs_total"])
                # Clamp to .1 minimum: a wicket on a wide/no-ball before any
                # legal ball in the over leaves legal_ball_in_over at 0, which
                # would render the non-cricket "X.0" notation (#180).
                over_label = f"{int(r['over'])}.{max(int(r['legal_ball_in_over']), 1)}"
                wicket_pos = int(r["wicket_position"])
                kind = r["wicket_kind"] if pd.notna(r["wicket_kind"]) else None
                fielders = r.get("wicket_fielders")
                fielders = str(fielders) if pd.notna(fielders) else None
                bowler_name = str(r["bowler"]) if pd.notna(r["bowler"]) else None
                dismissed_id = r["player_dismissed_id"] if pd.notna(r["player_dismissed_id"]) else None
                dismissed_name = r["player_dismissed"] if pd.notna(r["player_dismissed"]) else None
                bowler_credited = kind not in (None, "run out", "retired hurt", "retired out", "obstructing the field", "timed out")
                key = dismissed_id or dismissed_name
                if key is not None:
                    dismissal_map[key] = {
                        "kind": kind,
                        "bowler": bowler_name if bowler_credited else None,
                        "fielders": fielders,
                        "score_at_fall": fow_runs,
                        "wicket_position": wicket_pos,
                        "over": over_label,
                    }
                fall_of_wickets.append({
                    "wicket": wicket_pos,
                    "score": fow_runs,
                    "player": dismissed_name,
                    "over": over_label,
                })

            # Batting scorecard — sorted by order of appearance.
            # `balls` = legal_bat sum (excludes wides; no-balls still count — batter faced them).
            bat_card = (
                inn_df.groupby(["batter_id", "batter"])
                .agg(
                    runs=("runs_batter", "sum"),
                    balls=("legal_bat", "sum"),
                    tilt=("delta_wp", "sum"),
                    first_ball=("ball_number", "min"),
                )
                .reset_index()
                .sort_values("first_ball")
            )
            batting_scorecard = []
            for _, r in bat_card.iterrows():
                batter_id = r["batter_id"]
                batter_name = r["batter"]
                key = batter_id if batter_id else batter_name
                d = dismissal_map.get(key)
                batting_scorecard.append({
                    "player": batter_name,
                    "slug": slug_lookup.get(batter_id, ""),
                    "country": country_for(batter_id),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                    "sr": round(r["runs"] / max(r["balls"], 1) * 100, 1),
                    "tilt": round(r["tilt"], 5),
                    "dismissal_kind": d["kind"] if d else None,
                    "dismissal_bowler": d["bowler"] if d else None,
                    "dismissal_fielders": d["fielders"] if d else None,
                    "fall_of_wicket_score": d["score_at_fall"] if d else None,
                    "fall_of_wicket_over": d["over"] if d else None,
                    "wicket_position": d["wicket_position"] if d else None,
                    "not_out": d is None,
                })

            # Bowling scorecard — `legal_balls` excludes both wides and no-balls.
            bowl_card = (
                inn_df.groupby(["bowler_id", "bowler"])
                .agg(
                    runs=("runs_total", "sum"),
                    legal_balls=("legal_bowl", "sum"),
                    wickets=("is_bowler_wicket", "sum"),
                    tilt=("delta_wp", lambda x: -x.sum()),
                    first_ball=("ball_number", "min"),
                )
                .reset_index()
                .sort_values("first_ball")
            )

            bowling_scorecard = [
                {
                    "player": r["bowler"],
                    "slug": slug_lookup.get(r["bowler_id"], ""),
                    "country": country_for(r["bowler_id"]),
                    "overs": f"{int(r['legal_balls'] // 6)}.{int(r['legal_balls'] % 6)}",
                    "runs": int(r["runs"]),
                    "wickets": int(r["wickets"]),
                    "economy": round(r["runs"] / max(r["legal_balls"], 1) * 6, 2),
                    "tilt": round(r["tilt"], 5),
                }
                for _, r in bowl_card.iterrows()
            ]

            # Total score
            total_runs = int(inn_df["runs_scored"].iloc[-1] + inn_df["runs_total"].iloc[-1])
            total_wickets = int(inn_df["wickets_fallen"].iloc[-1] + (1 if inn_df.iloc[-1]["is_wicket"] else 0))

            innings_data.append({
                "innings": int(inn_num),
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "total": f"{total_runs}/{total_wickets}",
                "batting": batting_scorecard,
                "bowling": bowling_scorecard,
                "fall_of_wickets": fall_of_wickets,
            })

        # Ball-by-ball WP data
        balls = [
            {
                "inn": int(r["innings"]),
                "over": int(r["over"]),
                "ball": int(r["ball"]),
                "batter": r["batter"],
                "bowler": r["bowler"],
                "runs": int(r["runs_total"]),
                "score": f"{int(r['runs_scored']) + int(r['runs_total'])}/{int(r['wickets_fallen']) + (1 if r['is_wicket'] else 0)}",
                "wicket": bool(r["is_wicket"]),
                "wp": round(r["wp_before"], 4),
                "wp_after": round(r["wp_after"], 4),
                "delta": round(r["delta_wp"], 5),
            }
            for _, r in mdf.iterrows()
        ]

        # Key moments (top 5 by |delta_wp|). The very first ball of innings 2
        # (over 0, ball 0) carries the innings-boundary calibration residual in
        # its delta: wp_before is the calibrated chase-start midpoint (issue #62)
        # while wp_after is the model's raw forecast for the post-ball state, so
        # even a dot/single there can post a large |delta| that reflects the
        # boundary fix rather than anything the batter/bowler did. Drop that one
        # ball from the highlights so it isn't surfaced as a "key moment" (#144).
        # A wicket on that ball is kept — a genuine event outweighs the residual.
        km_candidates = mdf[
            ~(
                (mdf["innings"] == 2)
                & (mdf["over"] == 0)
                & (mdf["ball"] == 0)
                & (~mdf["is_wicket"].astype(bool))
            )
        ]
        mdf_sorted = km_candidates.reindex(
            km_candidates["delta_wp"].abs().sort_values(ascending=False).index
        )
        key_moments = [
            {
                "inn": int(r["innings"]),
                "over": int(r["over"]),
                "ball": int(r["ball"]),
                "batter": r["batter"],
                "bowler": r["bowler"],
                "runs": int(r["runs_total"]),
                "wicket": bool(r["is_wicket"]),
                "wicket_kind": r["wicket_kind"] if r["is_wicket"] else None,
                "player_dismissed": r["player_dismissed"] if r["is_wicket"] else None,
                # Delivery type so the UI can phrase wides/no-balls correctly
                # rather than crediting the batter with a "hit" (issue #165).
                "is_wide": bool(r["is_wide"]),
                "is_noball": bool(r["is_noball"]),
                "delta": round(r["delta_wp"], 5),
                "wp_after": round(r["wp_after"], 4),
            }
            for _, r in mdf_sorted.head(5).iterrows()
        ]

        match_detail = {
            **match_info,
            "innings": innings_data,
            "balls": balls,
            "key_moments": key_moments,
        }

        match_path = matches_dir / f"{match_id}.json"
        with open(match_path, "w") as f:
            json.dump(match_detail, f)

        # Add to index
        match_index.append({
            "match_id": str(match_id),
            "date": str(first_row["date"]),
            "season": str(first_row["season"]),
            "teams": teams,
            "venue": str(first_row["venue"]),
            "winner": winner,
            "result_margin": match_info.get("result_margin"),
            "event_stage": match_info.get("event_stage"),
            "scores": match_info.get("scores"),
        })

    # Sort index by date descending
    match_index.sort(key=lambda x: x["date"], reverse=True)
    index_path = output_dir / "match_index.json"
    with open(index_path, "w") as f:
        json.dump(match_index, f, indent=2)

    print(f"  Exported {len(match_ids)} match files to {matches_dir}")
    print(f"  Match index at {index_path}")
    return matches_dir


# %% Export metadata
def export_meta(
    deltas_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Export metadata JSON."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])

    meta = {
        "last_updated": str(date.today()),
        "matches_count": int(deltas_df["match_id"].nunique()),
        "balls_count": len(deltas_df),
        "seasons": sorted(deltas_df["season"].unique().tolist()),
        "data_source": "cricsheet.org",
        "model_version": "1.0",
        "stat_name": "TILT",
        "stat_description": "Win Probability Added per match — how much a player tilts the game",
    }

    meta_path = output_dir / config["export"]["meta_file"]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Exported metadata to {meta_path}")
    return meta_path


# %% Export GOAT performances
def export_goats(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Export top single-match and single-season performances by TILT."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])

    slug_lookup = _build_player_slug_lookup(player_tilt, include_empty_id=True)
    name_lookup = _build_player_name_lookup(player_tilt)

    deltas_df = _add_legal_flags(deltas_df)

    # --- Per-match dismissal + venue lookup so we can stamp not_out and venue
    # onto every GOAT row. dismissals: True if the batter was dismissed in
    # that (match_id, innings).
    dismissals_df = (
        deltas_df[deltas_df["player_dismissed_id"].notna()]
        [["match_id", "innings", "player_dismissed_id"]]
        .rename(columns={"player_dismissed_id": "batter_id"})
        .drop_duplicates()
        .assign(_dismissed=True)
    )
    venue_lookup = deltas_df.groupby("match_id")["venue"].first().to_dict()

    # --- Single-match batting ---
    bat_match = (
        deltas_df.groupby(["batter_id", "batter", "match_id", "date", "season", "batting_team", "innings"])
        .agg(tilt=("delta_wp", "sum"), runs=("runs_batter", "sum"), balls=("legal_bat", "sum"))
        .reset_index()
    )
    bat_match = bat_match.merge(
        dismissals_df, on=["batter_id", "match_id", "innings"], how="left"
    )
    bat_match["not_out"] = bat_match["_dismissed"].isna()
    bat_match = bat_match.drop(columns=["_dismissed"])

    def _bat_match_row(r):
        return {
            "player": name_lookup.get(r["batter_id"], r["batter"]),
            "slug": slug_lookup.get(r["batter_id"], ""),
            "country": country_for(r["batter_id"]),
            "team": r["batting_team"],
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
            "venue": venue_lookup.get(r["match_id"], ""),
            "tilt": round(r["tilt"], 5),
            "runs": int(r["runs"]),
            "balls": int(r["balls"]),
            "not_out": bool(r["not_out"]),
            "innings": int(r["innings"]),
        }

    # Top 50 per innings to populate filtered views, then top 50 overall for combined view
    top_bat_inn1 = bat_match[bat_match["innings"] == 1].nlargest(50, "tilt")
    top_bat_inn2 = bat_match[bat_match["innings"] == 2].nlargest(50, "tilt")
    top_bat_combined = pd.concat([top_bat_inn1, top_bat_inn2]).drop_duplicates(
        subset=["batter_id", "match_id"]
    )
    # Also include any top-50 overall entries not already covered
    top_bat_overall = bat_match.nlargest(50, "tilt")
    top_bat_match = pd.concat([top_bat_combined, top_bat_overall]).drop_duplicates(
        subset=["batter_id", "match_id"]
    ).nlargest(100, "tilt")

    goat_bat_match = [_bat_match_row(r) for _, r in top_bat_match.iterrows()]

    # --- Single-match bowling ---
    bowl_match = (
        deltas_df.groupby(["bowler_id", "bowler", "match_id", "date", "season", "bowling_team", "innings"])
        .agg(
            tilt=("delta_wp", lambda x: -x.sum()),
            wickets=("is_bowler_wicket", "sum"),
            runs_conceded=("runs_total", "sum"),
            balls=("legal_bowl", "sum"),
        )
        .reset_index()
    )

    def _bowl_match_row(r):
        return {
            "player": name_lookup.get(r["bowler_id"], r["bowler"]),
            "slug": slug_lookup.get(r["bowler_id"], ""),
            "country": country_for(r["bowler_id"]),
            "team": r["bowling_team"],
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
            "venue": venue_lookup.get(r["match_id"], ""),
            "tilt": round(r["tilt"], 5),
            "wickets": int(r["wickets"]),
            "runs_conceded": int(r["runs_conceded"]),
            "balls": int(r["balls"]),
            "innings": int(r["innings"]),
        }

    top_bowl_inn1 = bowl_match[bowl_match["innings"] == 1].nlargest(50, "tilt")
    top_bowl_inn2 = bowl_match[bowl_match["innings"] == 2].nlargest(50, "tilt")
    top_bowl_combined = pd.concat([top_bowl_inn1, top_bowl_inn2]).drop_duplicates(
        subset=["bowler_id", "match_id"]
    )
    top_bowl_overall = bowl_match.nlargest(50, "tilt")
    top_bowl_match = pd.concat([top_bowl_combined, top_bowl_overall]).drop_duplicates(
        subset=["bowler_id", "match_id"]
    ).nlargest(100, "tilt")

    goat_bowl_match = [_bowl_match_row(r) for _, r in top_bowl_match.iterrows()]

    # --- Single-match all-around (batting + bowling combined in same match) ---
    bat_by_match = bat_match[["batter_id", "match_id", "tilt"]].rename(
        columns={"batter_id": "player_id", "tilt": "bat_tilt"}
    )
    bowl_by_match = bowl_match[["bowler_id", "match_id", "tilt"]].rename(
        columns={"bowler_id": "player_id", "tilt": "bowl_tilt"}
    )
    combined_match = pd.merge(bat_by_match, bowl_by_match, on=["player_id", "match_id"], how="inner")
    combined_match["total_tilt"] = combined_match["bat_tilt"] + combined_match["bowl_tilt"]
    # Merge back match info
    match_info = deltas_df.groupby("match_id").first()[["date", "season"]].reset_index()
    combined_match = combined_match.merge(match_info, on="match_id", how="left")
    # Get player name and team
    player_names = deltas_df.groupby("batter_id").agg(player=("batter", "last")).reset_index().rename(
        columns={"batter_id": "player_id"}
    )
    combined_match = combined_match.merge(player_names, on="player_id", how="left")
    # Get team from batting data
    bat_teams = bat_match[["batter_id", "match_id", "batting_team"]].drop_duplicates(
        subset=["batter_id", "match_id"]
    ).rename(
        columns={"batter_id": "player_id", "batting_team": "team"}
    )
    combined_match = combined_match.merge(bat_teams, on=["player_id", "match_id"], how="left")

    top_allround_match = combined_match.nlargest(50, "total_tilt")
    goat_allround_match = [
        {
            "player": name_lookup.get(r["player_id"], r.get("player", "")),
            "slug": slug_lookup.get(r["player_id"], ""),
            "country": country_for(r["player_id"]),
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
            "venue": venue_lookup.get(r["match_id"], ""),
            "tilt": round(r["total_tilt"], 5),
            "bat_tilt": round(r["bat_tilt"], 5),
            "bowl_tilt": round(r["bowl_tilt"], 5),
        }
        for _, r in top_allround_match.iterrows()
    ]

    # --- Season batting ---
    bat_season = bat_match.groupby(["batter_id", "batter", "season"]).agg(
        total_tilt=("tilt", "sum"),
        matches=("match_id", "nunique"),
        runs=("runs", "sum"),
    ).reset_index()
    bat_season["tilt_per_match"] = bat_season["total_tilt"] / bat_season["matches"]
    # Get team for each season
    bat_season_teams = bat_match.groupby(["batter_id", "season"])["batting_team"].last().reset_index()
    bat_season = bat_season.merge(
        bat_season_teams.rename(columns={"batter_id": "batter_id", "batting_team": "team"}),
        on=["batter_id", "season"], how="left",
    )
    # Keep the top 50 by BOTH per-match and total TILT, so the goats page's
    # default total_tilt sort isn't missing high-total/lower-per-match seasons (#158).
    _bat_qual = bat_season[bat_season["matches"] >= 5]
    top_bat_season = pd.concat([
        _bat_qual.nlargest(50, "tilt_per_match"),
        _bat_qual.nlargest(50, "total_tilt"),
    ]).drop_duplicates(subset=["batter_id", "season"])
    goat_bat_season = [
        {
            "player": name_lookup.get(r["batter_id"], r["batter"]),
            "slug": slug_lookup.get(r["batter_id"], ""),
            "country": country_for(r["batter_id"]),
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "tilt_per_match": round(r["tilt_per_match"], 5),
            "total_tilt": round(r["total_tilt"], 5),
            "matches": int(r["matches"]),
            "runs": int(r["runs"]),
        }
        for _, r in top_bat_season.iterrows()
    ]

    # --- Season bowling ---
    bowl_season = bowl_match.groupby(["bowler_id", "bowler", "season"]).agg(
        total_tilt=("tilt", "sum"),
        matches=("match_id", "nunique"),
        wickets=("wickets", "sum"),
        runs_conceded=("runs_conceded", "sum"),
    ).reset_index()
    bowl_season["tilt_per_match"] = bowl_season["total_tilt"] / bowl_season["matches"]
    bowl_season_teams = bowl_match.groupby(["bowler_id", "season"])["bowling_team"].last().reset_index()
    bowl_season = bowl_season.merge(
        bowl_season_teams.rename(columns={"bowler_id": "bowler_id", "bowling_team": "team"}),
        on=["bowler_id", "season"], how="left",
    )
    _bowl_qual = bowl_season[bowl_season["matches"] >= 5]
    top_bowl_season = pd.concat([
        _bowl_qual.nlargest(50, "tilt_per_match"),
        _bowl_qual.nlargest(50, "total_tilt"),
    ]).drop_duplicates(subset=["bowler_id", "season"])
    goat_bowl_season = [
        {
            "player": name_lookup.get(r["bowler_id"], r["bowler"]),
            "slug": slug_lookup.get(r["bowler_id"], ""),
            "country": country_for(r["bowler_id"]),
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "tilt_per_match": round(r["tilt_per_match"], 5),
            "total_tilt": round(r["total_tilt"], 5),
            "matches": int(r["matches"]),
            "wickets": int(r["wickets"]),
            "runs_conceded": int(r["runs_conceded"]),
        }
        for _, r in top_bowl_season.iterrows()
    ]

    # --- Counting-stat leaderboards (issue #66) ---
    # Single-innings runs (batters): runs desc, balls asc as tiebreak
    top_match_runs = bat_match.sort_values(["runs", "balls"], ascending=[False, True]).head(50)
    goat_match_runs = [_bat_match_row(r) for _, r in top_match_runs.iterrows()]

    # Single-innings figures (bowlers): wickets desc, runs_conceded asc as tiebreak
    top_match_wickets = bowl_match.sort_values(
        ["wickets", "runs_conceded"], ascending=[False, True]
    ).head(50)
    goat_match_wickets = [_bowl_match_row(r) for _, r in top_match_wickets.iterrows()]

    # Season runs (batters): min 5 matches to avoid one-off cameos
    top_season_runs = bat_season[bat_season["matches"] >= 5].nlargest(50, "runs")
    goat_season_runs = [
        {
            "player": name_lookup.get(r["batter_id"], r["batter"]),
            "slug": slug_lookup.get(r["batter_id"], ""),
            "country": country_for(r["batter_id"]),
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "runs": int(r["runs"]),
            "matches": int(r["matches"]),
            "tilt_per_match": round(r["tilt_per_match"], 5),
            "total_tilt": round(r["total_tilt"], 5),
        }
        for _, r in top_season_runs.iterrows()
    ]

    # Season wickets (bowlers): min 5 matches
    top_season_wickets = bowl_season[bowl_season["matches"] >= 5].nlargest(50, "wickets")
    goat_season_wickets = [
        {
            "player": name_lookup.get(r["bowler_id"], r["bowler"]),
            "slug": slug_lookup.get(r["bowler_id"], ""),
            "country": country_for(r["bowler_id"]),
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "wickets": int(r["wickets"]),
            "runs_conceded": int(r["runs_conceded"]),
            "matches": int(r["matches"]),
            "tilt_per_match": round(r["tilt_per_match"], 5),
            "total_tilt": round(r["total_tilt"], 5),
        }
        for _, r in top_season_wickets.iterrows()
    ]

    goats = {
        "match_batting": goat_bat_match,
        "match_bowling": goat_bowl_match,
        "match_allround": goat_allround_match,
        "season_batting": goat_bat_season,
        "season_bowling": goat_bowl_season,
        "match_batting_runs": goat_match_runs,
        "match_bowling_wickets": goat_match_wickets,
        "season_batting_runs": goat_season_runs,
        "season_bowling_wickets": goat_season_wickets,
    }

    goats_path = output_dir / "goats.json"
    with open(goats_path, "w") as f:
        json.dump(goats, f, indent=2)

    print(f"  Exported GOAT performances to {goats_path}")
    return goats_path


# %% Team-level counting stats (orthogonal to player-level — different fields)
def _team_batting_counting_stats(slice_df: pd.DataFrame) -> Optional[dict]:
    if len(slice_df) == 0:
        return None
    runs = int(slice_df["runs_batter"].sum())
    balls = int(slice_df["legal_bat"].sum())
    return {
        "runs": runs,
        "balls": balls,
        "sr": round(runs / max(balls, 1) * 100, 2),
        "fours": int((slice_df["runs_batter"] == 4).sum()),
        "sixes": int((slice_df["runs_batter"] == 6).sum()),
        "dots": int((slice_df["runs_batter"] == 0).sum()),
    }


def _team_bowling_counting_stats(slice_df: pd.DataFrame) -> Optional[dict]:
    if len(slice_df) == 0:
        return None
    wickets = int(slice_df["is_bowler_wicket"].sum())
    balls = int(slice_df["legal_bowl"].sum())
    runs_conceded = int(slice_df["runs_total"].sum())
    return {
        "wickets": wickets,
        "balls": balls,
        "runs_conceded": runs_conceded,
        "economy": round(runs_conceded / max(balls, 1) * 6, 2),
    }


# %% Export team details (career + season summary embedded)
def export_team_details(
    deltas_df: pd.DataFrame,
    team_tilt: pd.DataFrame,
    team_season_tilt: pd.DataFrame,
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Write per-team JSON: career + season-by-season summary + all-time roster."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    teams_dir = output_dir / "teams"
    teams_dir.mkdir(parents=True, exist_ok=True)

    deltas_df = _add_legal_flags(deltas_df)
    slug_lookup = _build_player_slug_lookup(player_tilt)
    name_lookup = _build_player_name_lookup(player_tilt)
    match_info = _build_match_info_cache(deltas_df)
    # NRR-aware standings positions, shared with the season page (#146).
    position_map = _season_position_map(team_season_tilt, deltas_df)

    completed_seasons = load_completed_seasons()
    season_champions: dict = {}
    for season_value in deltas_df["season"].unique():
        season_str = str(season_value)
        if season_str not in completed_seasons:
            season_champions[season_str] = None
            continue
        sm = deltas_df[deltas_df["season"] == season_value].sort_values("date")
        if len(sm) == 0:
            season_champions[season_str] = None
            continue
        last_match_id = sm.iloc[-1]["match_id"]
        season_champions[season_str] = match_info.get(last_match_id, {}).get("winner")

    exported = 0
    for canonical, slug in TEAM_SLUG.items():
        team_row = team_tilt[team_tilt["team"] == canonical]
        if len(team_row) == 0:
            continue
        tr = team_row.iloc[0]

        # Career batting/bowling slices
        bat_slice = deltas_df[deltas_df["batting_team"] == canonical]
        bowl_slice = deltas_df[deltas_df["bowling_team"] == canonical]

        # Season-by-season summary
        ts_rows = team_season_tilt[team_season_tilt["team"] == canonical].copy()
        ts_rows["season_year"] = ts_rows["season"].apply(lambda s: int(str(s).split("/")[0]) if "/" in str(s) else int(s))
        ts_rows = ts_rows.sort_values("season_year")
        seasons = []
        for _, r in ts_rows.iterrows():
            season_str = str(r["season"])
            # Match log for this team-season (drives the team-page season dropdown)
            season_bat = bat_slice[bat_slice["season"] == season_str]
            season_bowl = bowl_slice[bowl_slice["season"] == season_str]
            season_match_ids = sorted(set(season_bat["match_id"].unique()) | set(season_bowl["match_id"].unique()))
            season_matches = []
            for mid in season_match_ids:
                mi = match_info.get(mid)
                if mi is None:
                    continue
                opponent = next((t for t in mi["teams"] if t != canonical), None)
                team_tilt_for_match = float(deltas_df[(deltas_df["match_id"] == mid) & (deltas_df["batting_team"] == canonical)]["delta_wp"].sum())
                team_tilt_for_match += float(-deltas_df[(deltas_df["match_id"] == mid) & (deltas_df["bowling_team"] == canonical)]["delta_wp"].sum())
                season_matches.append({
                    "match_id": str(mid),
                    "date": mi["date"],
                    "opponent": opponent,
                    "team_score": mi["scores"].get(canonical),
                    "opponent_score": mi["scores"].get(opponent) if opponent else None,
                    "winner": mi["winner"],
                    "event_stage": mi.get("event_stage"),
                    "team_total_tilt": round(team_tilt_for_match, 5),
                })
            season_matches.sort(key=lambda x: x["date"])
            seasons.append({
                "season": season_str,
                "season_year": int(r["season_year"]),
                "label": season_team_label(canonical, int(r["season_year"])),
                "matches": int(r["matches"]),
                "wins": int(r["wins"]),
                "losses": int(r["losses"]),
                "no_results": int(r.get("no_results", 0)),
                "points": int(r["points"]),
                "win_pct": float(round(r["win_pct"], 4)),
                "nrr": _compute_team_season_nrr(deltas_df, canonical, season_str),
                "team_total_tilt": round(float(r["team_total_tilt"]), 5),
                "team_tilt_per_match": round(float(r["team_tilt_per_match"]), 5),
                "position_est": position_map.get((canonical, season_str), int(r["position_est"])),
                "champion": season_champions.get(season_str) == canonical,
                "match_log": season_matches,
            })

        # All-time roster: each player who appeared (batted or bowled) for this team
        roster_bat = (
            bat_slice.groupby(["batter_id", "batter"])
            .agg(
                matches_for_team=("match_id", "nunique"),
                runs=("runs_batter", "sum"),
                balls=("legal_bat", "sum"),
                bat_total_tilt=("delta_wp", "sum"),
            )
            .reset_index()
            .rename(columns={"batter_id": "player_id", "batter": "player"})
        )
        roster_bowl = (
            bowl_slice.groupby(["bowler_id", "bowler"])
            .agg(
                bowl_matches=("match_id", "nunique"),
                wickets=("is_bowler_wicket", "sum"),
                bowl_balls=("legal_bowl", "sum"),
                bowl_total_tilt=("delta_wp", lambda x: -x.sum()),
            )
            .reset_index()
            .rename(columns={"bowler_id": "player_id", "bowler": "player"})
        )
        roster = pd.merge(roster_bat, roster_bowl, on=["player_id", "player"], how="outer")
        for col in ["matches_for_team", "runs", "balls", "bat_total_tilt", "bowl_matches", "wickets", "bowl_balls", "bowl_total_tilt"]:
            if col in roster.columns:
                roster[col] = roster[col].fillna(0)
        # Total matches across batting and bowling (de-dup by player)
        appearances_bat = bat_slice[["batter_id", "match_id"]].rename(columns={"batter_id": "player_id"})
        appearances_bowl = bowl_slice[["bowler_id", "match_id"]].rename(columns={"bowler_id": "player_id"})
        appearances = pd.concat([appearances_bat, appearances_bowl]).dropna().drop_duplicates()
        total_matches_per_player = appearances.groupby("player_id").size().reset_index(name="total_matches_for_team")
        roster = roster.merge(total_matches_per_player, on="player_id", how="left")
        roster["total_matches_for_team"] = roster["total_matches_for_team"].fillna(0).astype(int)

        roster_records = []
        for _, r in roster.sort_values("total_matches_for_team", ascending=False).iterrows():
            pid = r["player_id"]
            bat_balls = int(r.get("balls") or 0)
            bowl_balls = int(r.get("bowl_balls") or 0)
            n = int(r["total_matches_for_team"])
            bat_tpm = round(float(r.get("bat_total_tilt") or 0) / max(n, 1), 5) if bat_balls >= 50 else None
            bowl_tpm = round(float(r.get("bowl_total_tilt") or 0) / max(n, 1), 5) if bowl_balls >= 50 else None
            total_tilt_value = float(r.get("bat_total_tilt") or 0) + float(r.get("bowl_total_tilt") or 0)
            roster_records.append({
                "player": name_lookup.get(pid, r["player"]),
                "slug": slug_lookup.get(pid, ""),
                "country": country_for(pid),
                "matches": n,
                "runs": int(r.get("runs") or 0),
                "batting_balls": bat_balls,
                "wickets": int(r.get("wickets") or 0),
                "bowling_balls": bowl_balls,
                "batting_tilt_per_match": bat_tpm,
                "bowling_tilt_per_match": bowl_tpm,
                "total_tilt": round(total_tilt_value, 5),
                "total_tilt_per_match": round(total_tilt_value / max(n, 1), 5),
            })

        team_doc = {
            "team": canonical,
            "slug": slug,
            "aliases": TEAM_ALIASES.get(canonical, [canonical]),
            "first_season": int(tr["first_season"]),
            "last_season": int(tr["last_season"]),
            "career": {
                "matches": int(tr["matches"]),
                "wins": int(tr["wins"]),
                "losses": int(tr["losses"]),
                "win_pct": float(round(tr["win_pct"], 4)),
                "team_total_tilt": round(float(tr["team_total_tilt"]), 5),
                "team_tilt_per_match": round(float(tr["team_tilt_per_match"]), 5),
                "batting_tilt_per_match": round(float(tr["batting_tilt_per_match"]), 5),
                "bowling_tilt_per_match": round(float(tr["bowling_tilt_per_match"]), 5),
                "batting_stats": _team_batting_counting_stats(bat_slice),
                "bowling_stats": _team_bowling_counting_stats(bowl_slice),
            },
            "seasons": seasons,
            "all_time_roster": roster_records,
        }
        with open(teams_dir / f"{slug}.json", "w") as f:
            json.dump(team_doc, f, indent=2)
        exported += 1

    print(f"  Exported {exported} team detail files to {teams_dir}")
    return teams_dir


# %% Export team-season details
def export_team_season_details(
    deltas_df: pd.DataFrame,
    team_season_tilt: pd.DataFrame,
    player_season_tilt: pd.DataFrame,
    player_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Write per-(team, season) JSON: header + counting stats + roster + match log."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    ts_dir = output_dir / "team_seasons"
    ts_dir.mkdir(parents=True, exist_ok=True)

    deltas_df = _add_legal_flags(deltas_df)
    slug_lookup = _build_player_slug_lookup(player_tilt)
    name_lookup = _build_player_name_lookup(player_tilt)
    match_info = _build_match_info_cache(deltas_df)
    # NRR-aware standings positions, shared with the season page (#146).
    position_map = _season_position_map(team_season_tilt, deltas_df)

    exported = 0
    for _, ts in team_season_tilt.iterrows():
        canonical = ts["team"]
        season = str(ts["season"])
        slug = TEAM_SLUG.get(canonical)
        if not slug:
            continue
        season_year = int(season.split("/")[0]) if "/" in season else int(season)

        # Season slices for this team
        bat_slice = deltas_df[(deltas_df["batting_team"] == canonical) & (deltas_df["season"] == season)]
        bowl_slice = deltas_df[(deltas_df["bowling_team"] == canonical) & (deltas_df["season"] == season)]

        # Season roster: every player_id appearing in either slice (player-season subset)
        ps_rows = player_season_tilt[
            (player_season_tilt["season"] == season)
            & player_season_tilt["team"].isin([canonical])
        ].copy()

        roster = []
        for _, r in ps_rows.sort_values("total_tilt_per_match", ascending=False).iterrows():
            pid = r["player_id"]
            bat_p_slice = bat_slice[bat_slice["batter_id"] == pid]
            bowl_p_slice = bowl_slice[bowl_slice["bowler_id"] == pid]
            roster.append({
                "player": name_lookup.get(pid, r["player"]),
                "slug": slug_lookup.get(pid, ""),
                "country": country_for(pid),
                "matches": int(r["total_matches"]),
                "batting_stats": _batting_counting_stats(bat_p_slice, pid, r["player"]),
                "bowling_stats": _bowling_counting_stats(bowl_p_slice),
                "batting_tilt_per_match": round(float(r["batting_tilt_per_match"]), 5) if int(r["batting_balls"]) >= 50 else None,
                "bowling_tilt_per_match": round(float(r["bowling_tilt_per_match"]), 5) if int(r["bowling_balls"]) >= 50 else None,
                "shrunk_total_tilt_per_match": round(float(r.get("shrunk_total_tilt_per_match", r["total_tilt_per_match"])), 5),
                "total_tilt": round(float(r["total_tilt"]), 5),
            })

        # Match log for the team-season
        team_match_ids = sorted(set(bat_slice["match_id"].unique()) | set(bowl_slice["match_id"].unique()))
        match_log = []
        for mid in team_match_ids:
            mi = match_info.get(mid)
            if mi is None:
                continue
            # Team's score is the score of canonical in mi.scores
            opponent = next((t for t in mi["teams"] if t != canonical), None)
            team_total_tilt = float(deltas_df[(deltas_df["match_id"] == mid) & (deltas_df["batting_team"] == canonical)]["delta_wp"].sum())
            team_total_tilt += float(-deltas_df[(deltas_df["match_id"] == mid) & (deltas_df["bowling_team"] == canonical)]["delta_wp"].sum())
            match_log.append({
                "match_id": str(mid),
                "date": mi["date"],
                "opponent": opponent,
                "team_score": mi["scores"].get(canonical),
                "opponent_score": mi["scores"].get(opponent) if opponent else None,
                "winner": mi["winner"],
                "event_stage": mi.get("event_stage"),
                "team_total_tilt": round(team_total_tilt, 5),
            })
        match_log.sort(key=lambda x: x["date"])

        doc = {
            "team": canonical,
            "slug": slug,
            "season": season,
            "season_year": season_year,
            "label": season_team_label(canonical, season_year),
            "header": {
                "matches": int(ts["matches"]),
                "wins": int(ts["wins"]),
                "losses": int(ts["losses"]),
                "no_results": int(ts.get("no_results", 0)),
                "points": int(ts["points"]),
                "win_pct": float(round(ts["win_pct"], 4)),
                "nrr": _compute_team_season_nrr(deltas_df, canonical, season),
                "team_total_tilt": round(float(ts["team_total_tilt"]), 5),
                "team_tilt_per_match": round(float(ts["team_tilt_per_match"]), 5),
                "position_est": position_map.get((canonical, season), int(ts["position_est"])),
            },
            "batting_stats": _team_batting_counting_stats(bat_slice),
            "bowling_stats": _team_bowling_counting_stats(bowl_slice),
            "roster": roster,
            "matches": match_log,
        }
        with open(ts_dir / f"{slug}-{season.replace('/', '-')}.json", "w") as f:
            json.dump(doc, f, indent=2)
        exported += 1

    print(f"  Exported {exported} team-season files to {ts_dir}")
    return ts_dir


# %% Helpers for season + leaders exporters
_LEADER_STATS = [
    "runs", "wickets",
    "batting_tilt", "bowling_tilt", "total_tilt",
    "sr", "economy",
    "fifties", "hundreds", "fours", "sixes",
]


def _build_player_season_leaders(
    deltas_df: pd.DataFrame,
    player_season_tilt: pd.DataFrame,
    player_tilt: pd.DataFrame,
    season: str,
    *,
    min_balls: int = 100,
) -> dict:
    """Compute per-stat ranked lists for one season. Returns {stat: [rows]}."""
    slug_lookup = _build_player_slug_lookup(player_tilt)
    name_lookup = _build_player_name_lookup(player_tilt)

    season_balls = deltas_df[deltas_df["season"] == season]

    # Batter-level stats for this season (group on batter_id)
    bat = (
        season_balls.groupby(["batter_id", "batter", "batting_team"])
        .agg(
            runs=("runs_batter", "sum"),
            balls=("legal_bat", "sum"),
            innings=("match_id", "nunique"),
            fours=("runs_batter", lambda x: (x == 4).sum()),
            sixes=("runs_batter", lambda x: (x == 6).sum()),
        )
        .reset_index()
    )
    # Per-match runs for fifties/hundreds + HS
    match_runs = (
        season_balls.groupby(["batter_id", "match_id"])["runs_batter"].sum().reset_index()
    )
    # A "fifty" is 50–99; centuries are counted separately (#156, #176 — this
    # season-leaders aggregation was the parallel path missed by #156's fix).
    fifties = match_runs[(match_runs["runs_batter"] >= 50) & (match_runs["runs_batter"] < 100)].groupby("batter_id").size().reset_index(name="fifties")
    hundreds = match_runs[match_runs["runs_batter"] >= 100].groupby("batter_id").size().reset_index(name="hundreds")
    bat = bat.merge(fifties, on="batter_id", how="left").merge(hundreds, on="batter_id", how="left").fillna({"fifties": 0, "hundreds": 0})
    bat["sr"] = (bat["runs"] / bat["balls"].clip(lower=1) * 100).round(2)
    bat = bat.rename(columns={"batter_id": "player_id"})

    # Bowler-level stats for this season
    bowl = (
        season_balls.groupby(["bowler_id", "bowler", "bowling_team"])
        .agg(
            wickets=("is_bowler_wicket", "sum"),
            runs_conceded=("runs_total", "sum"),
            balls=("legal_bowl", "sum"),
            innings=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["economy"] = (bowl["runs_conceded"] / bowl["balls"].clip(lower=1) * 6).round(2)

    # TILT stats from player_season_tilt
    ps = player_season_tilt[player_season_tilt["season"] == season].copy()

    leaders = {}

    def _row_team_slug(team_name):
        return TEAM_SLUG.get(team_name, "")

    def _bat_row(r, value, ancillary):
        return {
            "player": name_lookup.get(r["player_id"], r.get("batter") or r.get("player", "")),
            "slug": slug_lookup.get(r["player_id"], ""),
            "country": country_for(r["player_id"]),
            "team": r.get("batting_team") or r.get("team", ""),
            "team_slug": _row_team_slug(r.get("batting_team") or r.get("team", "")),
            "value": value,
            **ancillary,
        }

    def _bowl_row(r, value, ancillary):
        return {
            "player": name_lookup.get(r["player_id"], r.get("bowler") or r.get("player", "")),
            "slug": slug_lookup.get(r["player_id"], ""),
            "country": country_for(r["player_id"]),
            "team": r.get("bowling_team") or r.get("team", ""),
            "team_slug": _row_team_slug(r.get("bowling_team") or r.get("team", "")),
            "value": value,
            **ancillary,
        }

    leaders["runs"] = [
        _bat_row(r, int(r["runs"]), {"balls": int(r["balls"]), "sr": float(r["sr"]), "innings": int(r["innings"])})
        for _, r in bat.nlargest(50, "runs").iterrows()
    ]
    leaders["fours"] = [
        _bat_row(r, int(r["fours"]), {"balls": int(r["balls"]), "runs": int(r["runs"])})
        for _, r in bat.nlargest(50, "fours").iterrows()
    ]
    leaders["sixes"] = [
        _bat_row(r, int(r["sixes"]), {"balls": int(r["balls"]), "runs": int(r["runs"])})
        for _, r in bat.nlargest(50, "sixes").iterrows()
    ]
    leaders["fifties"] = [
        _bat_row(r, int(r["fifties"]), {"hundreds": int(r["hundreds"]), "innings": int(r["innings"])})
        for _, r in bat.nlargest(50, "fifties").iterrows()
    ]
    hundreds_qual = bat[bat["hundreds"] >= 1]
    # Tie-break on fifties (more is better), then innings (fewer is better —
    # same hundreds + fifties in fewer innings is the more efficient season).
    hundreds_sorted = hundreds_qual.sort_values(
        by=["hundreds", "fifties", "innings"],
        ascending=[False, False, True],
    ).head(50)
    leaders["hundreds"] = [
        _bat_row(r, int(r["hundreds"]), {"fifties": int(r["fifties"]), "innings": int(r["innings"])})
        for _, r in hundreds_sorted.iterrows()
    ]
    sr_qual = bat[bat["balls"] >= min_balls]
    leaders["sr"] = [
        _bat_row(r, float(r["sr"]), {"runs": int(r["runs"]), "balls": int(r["balls"])})
        for _, r in sr_qual.nlargest(50, "sr").iterrows()
    ]

    leaders["wickets"] = [
        _bowl_row(r, int(r["wickets"]), {"balls": int(r["balls"]), "economy": float(r["economy"]), "innings": int(r["innings"])})
        for _, r in bowl.nlargest(50, "wickets").iterrows()
    ]
    econ_qual = bowl[bowl["balls"] >= min_balls]
    # Lowest economy wins — sort ascending
    leaders["economy"] = [
        _bowl_row(r, float(r["economy"]), {"wickets": int(r["wickets"]), "balls": int(r["balls"])})
        for _, r in econ_qual.nsmallest(50, "economy").iterrows()
    ]

    # TILT-based: rank season leaders by *total* season TILT (not per-match), so
    # the season/leaders pages reward volume of impact over rate. Min-balls
    # qualification still applies. Per-match value is preserved as ancillary.
    if len(ps) > 0:
        ps_bat = ps[ps["batting_balls"] >= min_balls].copy()
        ps_bowl = ps[ps["bowling_balls"] >= min_balls].copy()
        ps_total = ps[(ps["batting_balls"] >= min_balls) | (ps["bowling_balls"] >= min_balls)].copy()

        def _ps_row(r, value, ancillary):
            return {
                "player": name_lookup.get(r["player_id"], r["player"]),
                "slug": slug_lookup.get(r["player_id"], ""),
                "country": country_for(r["player_id"]),
                "team": r.get("team", ""),
                "team_slug": _row_team_slug(r.get("team", "")),
                "value": value,
                **ancillary,
            }

        leaders["batting_tilt"] = [
            _ps_row(r, round(float(r["batting_total_tilt"]), 5),
                    {"per_match": round(float(r["batting_tilt_per_match"]), 5), "matches": int(r["batting_matches"]), "balls": int(r["batting_balls"])})
            for _, r in ps_bat.sort_values("batting_total_tilt", ascending=False).head(50).iterrows()
        ]
        leaders["bowling_tilt"] = [
            _ps_row(r, round(float(r["bowling_total_tilt"]), 5),
                    {"per_match": round(float(r["bowling_tilt_per_match"]), 5), "matches": int(r["bowling_matches"]), "balls": int(r["bowling_balls"])})
            for _, r in ps_bowl.sort_values("bowling_total_tilt", ascending=False).head(50).iterrows()
        ]
        leaders["total_tilt"] = [
            _ps_row(r, round(float(r["total_tilt"]), 5),
                    {"per_match": round(float(r["total_tilt_per_match"]), 5), "matches": int(r["total_matches"])})
            for _, r in ps_total.sort_values("total_tilt", ascending=False).head(50).iterrows()
        ]
    else:
        leaders["batting_tilt"] = []
        leaders["bowling_tilt"] = []
        leaders["total_tilt"] = []

    return leaders


# %% Export season hub pages
def export_seasons(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    player_season_tilt: pd.DataFrame,
    team_season_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Write one JSON per season with team table + top-5 by each stat + match list."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    seasons_dir = output_dir / "seasons"
    seasons_dir.mkdir(parents=True, exist_ok=True)

    deltas_df = _add_legal_flags(deltas_df)
    match_info = _build_match_info_cache(deltas_df)
    completed_seasons = load_completed_seasons()

    exported = 0
    for season in sorted(deltas_df["season"].unique(), key=str):
        season_year = int(season.split("/")[0]) if "/" in season else int(season)

        # Team table for the season — regular-season only (issue #75)
        ts_rows = team_season_tilt[team_season_tilt["season"] == season].copy()
        team_table = []
        for _, r in ts_rows.iterrows():
            canonical = r["team"]
            wins = int(r["wins"])
            losses = int(r["losses"])
            matches = int(r["matches"])
            no_results = int(r.get("no_results", 0))
            team_table.append({
                "team": canonical,
                "slug": TEAM_SLUG.get(canonical, ""),
                "label": season_team_label(canonical, season_year),
                "matches": matches,
                "wins": wins,
                "losses": losses,
                "no_results": no_results,
                "points": int(r.get("points", wins * 2 + no_results)),
                "win_pct": float(round(r["win_pct"], 4)),
                "nrr": _compute_team_season_nrr(deltas_df, canonical, season),
                "team_total_tilt": round(float(r["team_total_tilt"]), 5),
                "team_tilt_per_match": round(float(r["team_tilt_per_match"]), 5),
            })

        # Re-rank by IPL convention: points desc, then wins desc, then NRR
        # desc (issue #102). Wins is the primary tie-breaker on equal points
        # — only matters when teams have different NR counts (e.g. 2015 MI
        # 8W/0NR=16pts vs RCB 7W/2NR=16pts). Genuine ties share rank.
        team_table.sort(key=lambda t: (-t["points"], -t["wins"], -(t["nrr"] if t["nrr"] is not None else -99)))
        prev_key = None
        prev_pos = 0
        for i, t in enumerate(team_table, 1):
            key = (t["points"], t["wins"], t["nrr"])
            if key != prev_key:
                prev_pos = i
                prev_key = key
            t["position_est"] = prev_pos

        # Champion: only populated for seasons listed in config/seasons.yaml.
        # In-progress seasons (where the final hasn't been played yet) get null
        # so the website doesn't crown a winner mid-tournament.
        season_matches_df = deltas_df[deltas_df["season"] == season]
        if season in completed_seasons:
            last_match_id = season_matches_df.sort_values("date").iloc[-1]["match_id"]
            champion = match_info.get(last_match_id, {}).get("winner")
        else:
            champion = None

        # Leaders: top 5 by each stat (full ranked list lives in /leaders/)
        full_leaders = _build_player_season_leaders(deltas_df, player_season_tilt, player_tilt, season)
        leaders_top5 = {stat: rows[:5] for stat, rows in full_leaders.items()}

        # Match list
        season_match_ids = sorted(season_matches_df["match_id"].unique(), key=lambda x: match_info.get(x, {}).get("date", ""))
        matches_list = []
        for mid in season_match_ids:
            mi = match_info.get(mid)
            if mi is None:
                continue
            matches_list.append({
                "match_id": str(mid),
                "date": mi["date"],
                "venue": mi["venue"],
                "teams": mi["teams"],
                "scores": mi.get("scores", {}),
                "winner": mi["winner"],
                "result_margin": mi.get("result_margin"),
                "toss_winner": mi.get("toss_winner"),
                "toss_decision": mi.get("toss_decision"),
                "event_stage": mi.get("event_stage"),
            })

        playoffs = _build_playoffs_block(season_match_ids, match_info)

        doc = {
            "season": season,
            "season_year": season_year,
            "matches": int(season_matches_df["match_id"].nunique()),
            "champion": champion,
            "team_table": team_table,
            "leaders": leaders_top5,
            "matches_list": matches_list,
            "playoffs": playoffs,
        }
        with open(seasons_dir / f"{season.replace('/', '-')}.json", "w") as f:
            json.dump(doc, f, indent=2)
        exported += 1

    print(f"  Exported {exported} season files to {seasons_dir}")
    return seasons_dir


# %% Export leaders (full ranked list per season + stat)
def export_leaders(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    player_season_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """One file per (season, stat). Used by leaders.html for full lists."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    leaders_dir = output_dir / "leaders"
    leaders_dir.mkdir(parents=True, exist_ok=True)

    deltas_df = _add_legal_flags(deltas_df)

    exported = 0
    for season in sorted(deltas_df["season"].unique(), key=str):
        full_leaders = _build_player_season_leaders(deltas_df, player_season_tilt, player_tilt, season)
        season_slug = season.replace("/", "-")
        for stat, rows in full_leaders.items():
            ranked = []
            for i, r in enumerate(rows, 1):
                ranked.append({"rank": i, **r})
            doc = {
                "season": season,
                "season_year": int(season.split("/")[0]) if "/" in season else int(season),
                "stat": stat,
                "leaders": ranked,
            }
            with open(leaders_dir / f"{season_slug}-{stat}.json", "w") as f:
                json.dump(doc, f, indent=2)
            exported += 1

    print(f"  Exported {exported} leader files to {leaders_dir}")
    return leaders_dir


# %% Export team index
def export_team_index(
    team_tilt: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Path:
    """Write a thin index of all teams for navigation / picker components."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.parse_matches import _SEASON_LABELS

    index = []
    for canonical, slug in TEAM_SLUG.items():
        row = team_tilt[team_tilt["team"] == canonical]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        index.append({
            "slug": slug,
            "name": canonical,
            "aliases": TEAM_ALIASES.get(canonical, [canonical]),
            "season_labels": _SEASON_LABELS.get(canonical, []),
            "career_matches": int(r["matches"]),
            "first_season": int(r["first_season"]),
            "last_season": int(r["last_season"]),
            "team_tilt_per_match": round(float(r["team_tilt_per_match"]), 5),
            "win_pct": float(round(r["win_pct"], 4)),
        })
    index.sort(key=lambda x: x["career_matches"], reverse=True)

    index_path = output_dir / "team_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Exported team index ({len(index)} teams) to {index_path}")
    return index_path


# %% Search index
# A few historic team abbreviations don't fall out of standard initials
# (Kings XI Punjab → KXP via initials, but fans wrote it KXIP). Hand-map
# only the irregular ones; standard initials handle DD / PWI / RPS / etc.
_TEAM_ABBREV_OVERRIDES = {
    "Kings XI Punjab": ["kxip"],
}


def _alias_initials(name: str) -> Optional[str]:
    """Initials of each whitespace-separated word, lowercased. Returns None
    for single-word inputs or noise."""
    words = [w for w in name.split() if w]
    if len(words) < 2:
        return None
    init = "".join(w[0] for w in words).lower()
    return init if 2 <= len(init) <= 5 else None


def _build_corpus(sources, *, with_initials: bool = False, extras=()) -> str:
    """Pipe-delimited token set: full lowercased phrase + each individual
    word + (optionally) initials of multi-word phrases + extras. Pipes are
    the token-start sentinel the client-side scorer looks for."""
    tokens = set()
    for s in sources:
        if not s:
            continue
        sl = s.strip().lower()
        if not sl:
            continue
        tokens.add(sl)
        for w in sl.split():
            if w:
                tokens.add(w)
        if with_initials:
            init = _alias_initials(s)
            if init:
                tokens.add(init)
    tokens.update(extras)
    return "|".join(sorted(tk for tk in tokens if tk))


def export_search_index(
    deltas_df: Optional[pd.DataFrame] = None,
    player_tilt: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Build a flat search index for the global nav search box.

    Reads the already-written `tilt_rankings.json` and `team_index.json` so
    role classification and alias flattening don't get re-implemented here.
    Must run after both of those exporters.

    `deltas_df` is used to enumerate distinct teams per player (the rankings
    file's `teams` array only carries one); pass None to skip and fall back
    to the rankings primary team only.

    `player_tilt` is used to also index sub-min_matches players (rookies)
    that have a player page but aren't on the leaderboard (issue #91). Pass
    None to index ranked players only.

    One row per entity:
      t   — 'p' (player) or 'team'
      l   — display label
      s   — slug used to build player.html / team.html URL
      sub — muted secondary line (role/team/matches)
      x   — pipe-delimited lowercase searchable corpus including each
            alias's individual words and initials; pipes act as token-start
            sentinels for the client-side scorer
      b   — numeric tie-break boost
      r   — player rank (only on ranked players; client treats missing as 9999)
    """
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])

    rankings_path = output_dir / config["export"]["rankings_file"]
    team_index_path = output_dir / "team_index.json"
    with open(rankings_path) as f:
        rankings = json.load(f)
    with open(team_index_path) as f:
        teams = json.load(f)

    teams_by_pid: dict = {}
    if deltas_df is not None:
        for pid, grp in deltas_df.groupby("batter_id"):
            teams_by_pid.setdefault(pid, set()).update(grp["batting_team"].dropna().unique())
        for pid, grp in deltas_df.groupby("bowler_id"):
            teams_by_pid.setdefault(pid, set()).update(grp["bowling_team"].dropna().unique())

    rows = []
    ranked_pids: set = set()

    for r in rankings:
        pid = r.get("player_id") or ""
        ranked_pids.add(pid)
        played_for = teams_by_pid.get(pid) or {r["team"]}
        if len(played_for) > 1:
            team_part = f"{len(played_for)} Teams"
        else:
            primary_code = TEAM_SLUG.get(r["team"], "").upper() or r["team"]
            team_part = primary_code
        role_label = "batsman" if r["role"] == "batter" else r["role"]
        sub = f"{role_label} · {team_part} · {r['total_matches']} M"
        rows.append({
            "t": "p",
            "l": r["player"],
            "s": r["slug"],
            "c": r.get("country"),
            "sub": sub,
            "x": _build_corpus([r["player"], r.get("full_name", r["player"])]),
            "b": r["total_tilt_per_match"],
            "r": r["rank"],
        })

    if player_tilt is not None:
        for _, row in player_tilt.iterrows():
            pid = row.get("player_id") or ""
            if pid in ranked_pids:
                continue
            bat_balls = int(row["batting_balls"])
            bowl_balls = int(row["bowling_balls"])
            # Mirrors export_rankings role logic: bowler check overrides all-rounder.
            role = "batter"
            if bowl_balls >= 50 and bat_balls >= 50:
                ratio = bowl_balls / max(bat_balls, 1)
                if 0.3 <= ratio <= 3.0:
                    role = "all-rounder"
            if bowl_balls >= 100 and bowl_balls > bat_balls * 1.5:
                role = "bowler"
            played_for = teams_by_pid.get(pid) or {row["team"]}
            if len(played_for) > 1:
                team_part = f"{len(played_for)} Teams"
            else:
                primary_code = TEAM_SLUG.get(row["team"], "").upper() or row["team"]
                team_part = primary_code
            role_label = "batsman" if role == "batter" else role
            sub = f"{role_label} · {team_part} · {int(row['total_matches'])} M"
            rows.append({
                "t": "p",
                "l": row["player"],
                "s": make_slug(row["player"], pid if pid else None),
                "c": country_for(pid),
                "sub": sub,
                "x": _build_corpus([row["player"], row.get("full_name") or row["player"]]),
                "b": round(float(row["total_tilt_per_match"]), 5),
            })

    for t in teams:
        sources = [t["name"], t["slug"]]
        sources.extend(t.get("aliases", []) or [])
        for rule in t.get("season_labels", []) or []:
            if rule.get("label"):
                sources.append(rule["label"])
        extras = []
        for src in sources:
            if src in _TEAM_ABBREV_OVERRIDES:
                extras.extend(_TEAM_ABBREV_OVERRIDES[src])
        rows.append({
            "t": "team",
            "l": t["name"],
            "s": t["slug"],
            "sub": f"{t['career_matches']} matches",
            "x": _build_corpus(sources, with_initials=True, extras=extras),
            "b": round(t["career_matches"] / 1000.0, 4),
        })

    output_path = output_dir / "search_index.json"
    with open(output_path, "w") as f:
        json.dump(rows, f, separators=(",", ":"))

    print(f"  Exported search index ({len(rows)} entries) to {output_path}")
    return output_path


# %% Main
def export_all(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
    player_season_tilt: Optional[pd.DataFrame] = None,
    team_tilt: Optional[pd.DataFrame] = None,
    team_season_tilt: Optional[pd.DataFrame] = None,
) -> None:
    """Export all JSON files for the website.

    The new aggregated DataFrames (player_season_tilt, team_tilt, team_season_tilt)
    are optional only so legacy callers don't break — `run_pipeline` always passes
    them. If absent, the new exporters are skipped.
    """
    print("\nExporting JSON for website...")
    export_rankings(player_tilt)
    export_player_details(deltas_df, player_tilt)
    export_match_details(deltas_df, player_tilt)
    export_goats(deltas_df, player_tilt)
    export_meta(deltas_df)
    if team_tilt is not None and team_season_tilt is not None and player_season_tilt is not None:
        export_team_details(deltas_df, team_tilt, team_season_tilt, player_tilt)
        export_team_season_details(deltas_df, team_season_tilt, player_season_tilt, player_tilt)
        export_seasons(deltas_df, player_tilt, player_season_tilt, team_season_tilt)
        export_leaders(deltas_df, player_tilt, player_season_tilt)
        export_team_index(team_tilt)
        export_search_index(deltas_df, player_tilt)
    print("  Done!")


if __name__ == "__main__":
    # Load pre-computed data
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    print("Loading deltas and player TILT data...")
    deltas_df = pd.read_parquet(processed_dir / "deltas.parquet")
    player_tilt = pd.read_parquet(processed_dir / "player_tilt.parquet")

    extras = {}
    for name in ["player_season_tilt", "team_tilt", "team_season_tilt"]:
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            extras[name] = pd.read_parquet(path)

    export_all(deltas_df, player_tilt, **extras)
