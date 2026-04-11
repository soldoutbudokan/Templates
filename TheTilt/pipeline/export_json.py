# %% Imports
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from pipeline.compute_tilt import make_slug


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
            "player_id": player_id,
            "slug": make_slug(row["player"], player_id if player_id else None),
            "team": row["team"],
            "teams": row.get("teams", [row["team"]]) if isinstance(row.get("teams"), list) else [row["team"]],
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
    min_matches: int = 10,
) -> Path:
    """Export per-player detail JSON files."""
    config = load_config()
    output_dir = Path(output_dir or config["export"]["output_dir"])
    players_dir = output_dir / config["export"]["players_dir"]
    players_dir.mkdir(parents=True, exist_ok=True)
    min_matches = config["export"].get("min_matches", min_matches)

    qualified = player_tilt[player_tilt["total_matches"] >= min_matches]

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
            .agg(tilt=("delta_wp", "sum"), balls=("delta_wp", "count"), matches=("match_id", "nunique"))
            .reset_index()
        )
        season_batting["tilt_per_match"] = season_batting["tilt"] / season_batting["matches"]

        # Season breakdown (bowling)
        bowl_df_copy = bowl_df.copy()
        bowl_df_copy["bowling_delta"] = -bowl_df_copy["delta_wp"]
        season_bowling = (
            bowl_df_copy.groupby("season")
            .agg(tilt=("bowling_delta", "sum"), balls=("bowling_delta", "count"), matches=("match_id", "nunique"))
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
            merged_seasons.append({
                "season": str(season),
                "batting_tilt_per_match": round(bat_tilt, 5),
                "bowling_tilt_per_match": round(bowl_tilt, 5),
                "batting_matches": bat_matches,
                "bowling_matches": bowl_matches,
                "batting_balls": bat_balls,
                "bowling_balls": bowl_balls,
            })

        # Phase breakdown (batting)
        phase_batting = []
        for phase_name, col in [("powerplay", "is_powerplay"), ("middle", "is_middle"), ("death", "is_death")]:
            phase_df = bat_df[bat_df[col] == 1]
            if len(phase_df) > 0:
                phase_batting.append({
                    "phase": phase_name,
                    "tilt": round(phase_df["delta_wp"].sum(), 5),
                    "balls": len(phase_df),
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
                    "balls": len(phase_df),
                    "avg_delta": round(phase_df["bowling_delta"].mean(), 6),
                })

        # Best/worst match performances (batting)
        match_perf = (
            bat_df.groupby(["match_id", "date", "bowling_team"])
            .agg(tilt=("delta_wp", "sum"), balls=("delta_wp", "count"), runs=("runs_batter", "sum"))
            .reset_index()
            .sort_values("tilt", ascending=False)
        )
        best_matches = match_perf.head(5).to_dict("records")
        worst_matches = match_perf.tail(5).sort_values("tilt").to_dict("records")

        # Best/worst match performances (bowling)
        if len(bowl_df_copy) > 0:
            bowl_match_perf = (
                bowl_df_copy.groupby(["match_id", "date", "batting_team"])
                .agg(tilt=("bowling_delta", "sum"), balls=("bowling_delta", "count"), wickets=("is_wicket", "sum"))
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

        # Build player detail JSON
        bat_balls = int(row["batting_balls"])
        bowl_balls = int(row["bowling_balls"])
        min_role_balls = config["export"].get("min_role_balls", 50)
        bat_qualified = bat_balls >= min_role_balls
        bowl_qualified = bowl_balls >= min_role_balls
        detail = {
            "player": player_name,
            "player_id": player_id,
            "slug": slug,
            "team": row["team"],
            "teams": row.get("teams", [row["team"]]) if isinstance(row.get("teams"), list) else [row["team"]],
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
                    "date": str(r["date"]),
                    "vs": r["bowling_team"],
                    "tilt": round(r["tilt"], 5),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                }
                for r in best_matches
            ],
            "worst_matches": [
                {
                    "date": str(r["date"]),
                    "vs": r["bowling_team"],
                    "tilt": round(r["tilt"], 5),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                }
                for r in worst_matches
            ],
            "bowling_best_matches": [
                {
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
        }

        player_path = players_dir / f"{slug}.json"
        with open(player_path, "w") as f:
            json.dump(detail, f, indent=2)

        exported += 1

    print(f"  Exported {exported} player detail files to {players_dir}")
    return players_dir


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


# %% Main
def export_all(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
) -> None:
    """Export all JSON files for the website."""
    print("\nExporting JSON for website...")
    export_rankings(player_tilt)
    export_player_details(deltas_df, player_tilt)
    export_meta(deltas_df)
    print("  Done!")


if __name__ == "__main__":
    # Load pre-computed data
    import pickle

    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    print("Loading deltas and player TILT data...")
    deltas_df = pd.read_parquet(processed_dir / "deltas.parquet")
    player_tilt = pd.read_parquet(processed_dir / "player_tilt.parquet")

    export_all(deltas_df, player_tilt)
