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
            "full_name": row.get("full_name", row["player"]),
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

            # Per-season team and counting stats
            season_team = None
            season_bat_stats = None
            season_bowl_stats = None

            # Batting counting stats for this season
            bat_season_df = bat_df[bat_df["season"] == season]
            if len(bat_season_df) > 0:
                season_team = bat_season_df["batting_team"].mode().iloc[0]
                s_runs = int(bat_season_df["runs_batter"].sum())
                s_balls = len(bat_season_df)
                s_innings = bat_season_df["match_id"].nunique()
                s_dismissals = int(bat_season_df["player_dismissed_id"].eq(player_id).sum()) if player_id else int(bat_season_df["player_dismissed"].eq(player_name).sum())
                s_match_runs = bat_season_df.groupby("match_id")["runs_batter"].sum()
                season_bat_stats = {
                    "runs": s_runs,
                    "innings": s_innings,
                    "balls": s_balls,
                    "avg": round(s_runs / max(s_dismissals, 1), 2),
                    "sr": round(s_runs / max(s_balls, 1) * 100, 2),
                    "hs": int(s_match_runs.max()),
                    "not_outs": s_innings - s_dismissals,
                    "fifties": int((s_match_runs >= 50).sum()),
                    "hundreds": int((s_match_runs >= 100).sum()),
                }

            # Bowling counting stats for this season
            bowl_season_df = bowl_df[bowl_df["season"] == season]
            if len(bowl_season_df) > 0:
                if season_team is None:
                    season_team = bowl_season_df["bowling_team"].mode().iloc[0]
                s_wickets = int(bowl_season_df["is_wicket"].sum())
                s_bowl_balls = len(bowl_season_df)
                s_runs_conceded = int(bowl_season_df["runs_total"].sum())
                s_bowl_innings = bowl_season_df["match_id"].nunique()
                s_bowl_match = bowl_season_df.groupby("match_id").agg(
                    wickets=("is_wicket", "sum"),
                    runs=("runs_total", "sum"),
                )
                s_best_idx = s_bowl_match["wickets"].idxmax()
                s_best_w = int(s_bowl_match.loc[s_best_idx, "wickets"])
                s_best_r = int(s_bowl_match.loc[s_best_idx, "runs"])
                season_bowl_stats = {
                    "wickets": s_wickets,
                    "innings": s_bowl_innings,
                    "balls": s_bowl_balls,
                    "runs_conceded": s_runs_conceded,
                    "avg": round(s_runs_conceded / max(s_wickets, 1), 2),
                    "economy": round(s_runs_conceded / max(s_bowl_balls, 1) * 6, 2),
                    "best_figures": f"{s_best_w}/{s_best_r}",
                }

            season_entry = {
                "season": str(season),
                "team": str(season_team) if season_team else None,
                "batting_tilt_per_match": round(bat_tilt, 5),
                "bowling_tilt_per_match": round(bowl_tilt, 5),
                "batting_matches": bat_matches,
                "bowling_matches": bowl_matches,
                "batting_balls": bat_balls,
                "bowling_balls": bowl_balls,
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

        # Traditional counting stats (batting)
        batting_stats = None
        if len(bat_df) > 0:
            total_runs = int(bat_df["runs_batter"].sum())
            balls_faced = len(bat_df)
            batting_innings = bat_df["match_id"].nunique()
            # Dismissals: count balls where this player was dismissed
            dismissals = int(bat_df["player_dismissed_id"].eq(player_id).sum()) if player_id else int(bat_df["player_dismissed"].eq(player_name).sum())
            # Per-match runs for HS, 50s, 100s
            match_runs = bat_df.groupby("match_id")["runs_batter"].sum()
            batting_stats = {
                "runs": total_runs,
                "innings": batting_innings,
                "balls": balls_faced,
                "avg": round(total_runs / max(dismissals, 1), 2),
                "sr": round(total_runs / max(balls_faced, 1) * 100, 2),
                "hs": int(match_runs.max()),
                "dismissals": dismissals,
                "not_outs": batting_innings - dismissals,
                "fifties": int((match_runs >= 50).sum()),
                "hundreds": int((match_runs >= 100).sum()),
            }

        # Traditional counting stats (bowling)
        bowling_stats = None
        if len(bowl_df) > 0:
            total_wickets = int(bowl_df["is_wicket"].sum())
            bowling_balls_total = len(bowl_df)
            runs_conceded = int(bowl_df["runs_total"].sum())
            bowling_innings = bowl_df["match_id"].nunique()
            # Best figures per match
            bowl_match = bowl_df.groupby("match_id").agg(
                wickets=("is_wicket", "sum"),
                runs=("runs_total", "sum"),
            )
            best_idx = bowl_match["wickets"].idxmax()
            best_w = int(bowl_match.loc[best_idx, "wickets"])
            best_r = int(bowl_match.loc[best_idx, "runs"])
            bowling_stats = {
                "wickets": total_wickets,
                "innings": bowling_innings,
                "balls": bowling_balls_total,
                "runs_conceded": runs_conceded,
                "avg": round(runs_conceded / max(total_wickets, 1), 2),
                "economy": round(runs_conceded / max(bowling_balls_total, 1) * 6, 2),
                "best_figures": f"{best_w}/{best_r}",
            }

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
                    "match_id": str(r["match_id"]),
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
                    "match_id": str(r["match_id"]),
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

    # Build player slug lookup for linking
    slug_lookup = {}
    if player_tilt is not None:
        for _, row in player_tilt.iterrows():
            pid = row.get("player_id", "")
            if pid:
                slug_lookup[pid] = make_slug(row["player"], pid)

    match_ids = deltas_df["match_id"].unique()
    print(f"  Exporting {len(match_ids)} match detail files...")

    match_index = []

    for match_id in match_ids:
        mdf = deltas_df[deltas_df["match_id"] == match_id].sort_values(["innings", "ball_number"])

        # Match info
        first_row = mdf.iloc[0]
        teams = sorted(mdf["batting_team"].unique().tolist())
        winner = str(first_row["winner"]) if pd.notna(first_row["winner"]) else None

        match_info = {
            "match_id": str(match_id),
            "date": str(first_row["date"]),
            "venue": str(first_row["venue"]),
            "season": str(first_row["season"]),
            "teams": teams,
            "winner": winner,
            "toss_winner": str(first_row.get("toss_winner", "")) if pd.notna(first_row.get("toss_winner")) else None,
            "toss_decision": str(first_row.get("toss_decision", "")) if pd.notna(first_row.get("toss_decision")) else None,
        }

        # Per-innings data
        innings_data = []
        for inn_num in sorted(mdf["innings"].unique()):
            inn_df = mdf[mdf["innings"] == inn_num]
            batting_team = str(inn_df.iloc[0]["batting_team"])
            bowling_team = str(inn_df.iloc[0]["bowling_team"])

            # Batting scorecard — sorted by order of appearance
            bat_card = (
                inn_df.groupby(["batter_id", "batter"])
                .agg(
                    runs=("runs_batter", "sum"),
                    balls=("runs_batter", "count"),
                    tilt=("delta_wp", "sum"),
                    first_ball=("ball_number", "min"),
                )
                .reset_index()
                .sort_values("first_ball")
            )
            batting_scorecard = [
                {
                    "player": r["batter"],
                    "slug": slug_lookup.get(r["batter_id"], ""),
                    "runs": int(r["runs"]),
                    "balls": int(r["balls"]),
                    "sr": round(r["runs"] / max(r["balls"], 1) * 100, 1),
                    "tilt": round(r["tilt"], 5),
                }
                for _, r in bat_card.iterrows()
            ]

            # Bowling scorecard — compute legal deliveries (exclude extras overflow)
            # In Cricsheet, wides/no-balls add extra rows beyond 6 per over
            bowl_card = (
                inn_df.groupby(["bowler_id", "bowler"])
                .agg(
                    runs=("runs_total", "sum"),
                    total_deliveries=("runs_total", "count"),
                    wickets=("is_wicket", "sum"),
                    tilt=("delta_wp", lambda x: -x.sum()),
                    first_ball=("ball_number", "min"),
                )
                .reset_index()
                .sort_values("first_ball")
            )
            # Count legal deliveries per bowler: for each over, min(balls_in_over, 6)
            for idx, row in bowl_card.iterrows():
                bowler_balls = inn_df[inn_df["bowler_id"] == row["bowler_id"]]
                legal = sum(min(len(g), 6) for _, g in bowler_balls.groupby("over"))
                bowl_card.at[idx, "legal_balls"] = legal

            bowling_scorecard = [
                {
                    "player": r["bowler"],
                    "slug": slug_lookup.get(r["bowler_id"], ""),
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

        # Key moments (top 5 by |delta_wp|)
        mdf_sorted = mdf.reindex(mdf["delta_wp"].abs().sort_values(ascending=False).index)
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

    # Build slug lookup
    slug_lookup = {}
    for _, row in player_tilt.iterrows():
        pid = row.get("player_id", "")
        slug_lookup[pid] = make_slug(row["player"], pid if pid else None)

    # Full name lookup
    name_lookup = {}
    for _, row in player_tilt.iterrows():
        pid = row.get("player_id", "")
        name_lookup[pid] = row.get("full_name", row["player"])

    # --- Single-match batting ---
    bat_match = (
        deltas_df.groupby(["batter_id", "batter", "match_id", "date", "season", "batting_team", "innings"])
        .agg(tilt=("delta_wp", "sum"), runs=("runs_batter", "sum"), balls=("delta_wp", "count"))
        .reset_index()
    )

    def _bat_match_row(r):
        return {
            "player": name_lookup.get(r["batter_id"], r["batter"]),
            "slug": slug_lookup.get(r["batter_id"], ""),
            "team": r["batting_team"],
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
            "tilt": round(r["tilt"], 5),
            "runs": int(r["runs"]),
            "balls": int(r["balls"]),
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
            wickets=("is_wicket", "sum"),
            runs_conceded=("runs_total", "sum"),
            balls=("delta_wp", "count"),
        )
        .reset_index()
    )

    def _bowl_match_row(r):
        return {
            "player": name_lookup.get(r["bowler_id"], r["bowler"]),
            "slug": slug_lookup.get(r["bowler_id"], ""),
            "team": r["bowling_team"],
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
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
            "team": r.get("team", ""),
            "season": str(r["season"]),
            "date": str(r["date"]),
            "match_id": str(r["match_id"]),
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
    top_bat_season = bat_season[bat_season["matches"] >= 5].nlargest(50, "tilt_per_match")
    goat_bat_season = [
        {
            "player": name_lookup.get(r["batter_id"], r["batter"]),
            "slug": slug_lookup.get(r["batter_id"], ""),
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
    top_bowl_season = bowl_season[bowl_season["matches"] >= 5].nlargest(50, "tilt_per_match")
    goat_bowl_season = [
        {
            "player": name_lookup.get(r["bowler_id"], r["bowler"]),
            "slug": slug_lookup.get(r["bowler_id"], ""),
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

    goats = {
        "match_batting": goat_bat_match,
        "match_bowling": goat_bowl_match,
        "match_allround": goat_allround_match,
        "season_batting": goat_bat_season,
        "season_bowling": goat_bowl_season,
    }

    goats_path = output_dir / "goats.json"
    with open(goats_path, "w") as f:
        json.dump(goats, f, indent=2)

    print(f"  Exported GOAT performances to {goats_path}")
    return goats_path


# %% Main
def export_all(
    deltas_df: pd.DataFrame,
    player_tilt: pd.DataFrame,
) -> None:
    """Export all JSON files for the website."""
    print("\nExporting JSON for website...")
    export_rankings(player_tilt)
    export_player_details(deltas_df, player_tilt)
    export_match_details(deltas_df, player_tilt)
    export_goats(deltas_df, player_tilt)
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
