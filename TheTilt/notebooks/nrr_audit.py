# %% NRR audit (issue #107)
"""Cross-check our season-final NRRs against Wikipedia's league tables.

Run from `TheTilt/`:
    ./venv/bin/python notebooks/nrr_audit.py

Wikipedia is the de-facto reference for IPL standings. Any divergence above
the tolerance flags a likely bug — usually in the all-out cap or DLS handling
(`pipeline/parse_matches.py:innings_allocation`,
`pipeline/export_json.py:_compute_team_season_nrr`).

Known small residuals: a few rain-affected matches where Wikipedia uses the
DLSR par-score method rather than actual reduced overs (~0.05 NRR units per
team in the affected season).
"""

# %% Imports
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.export_json import _compute_team_season_nrr
from pipeline.parse_matches import normalize_team

# %% Config
TOLERANCE = 0.005  # 3-decimal Wikipedia rounding tolerance
DELTAS_PATH = PROJECT_ROOT / "data" / "processed" / "deltas.parquet"

# Seasons to audit. 2007/08 → 2008 etc per `parse_matches.normalize_season`.
SEASONS = [str(y) for y in range(2008, 2027)]


# %% Wikipedia fetcher
def fetch_wikipedia_table(season: str) -> pd.DataFrame:
    """Pull the league-stage standings from `2{season}_Indian_Premier_League`.

    Returns a DataFrame with columns: Team, M, W, L, NR, Pts, NRR.
    Wikipedia's tables vary in structure across seasons; this uses
    `pandas.read_html` and picks the first table whose columns include both
    `Pts` and `NRR`.
    """
    url = f"https://en.wikipedia.org/wiki/{season}_Indian_Premier_League"
    headers = {"User-Agent": "Mozilla/5.0 (TheTilt-NRR-Audit)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if any("NRR" in c for c in cols) and any("Pts" in c or "Points" in c for c in cols):
            return _normalize_wiki_table(tbl)
    raise ValueError(f"No standings table found for {season}")


def _normalize_wiki_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns and pick the columns we care about."""
    if isinstance(tbl.columns, pd.MultiIndex):
        tbl.columns = [c[-1] for c in tbl.columns]
    rename = {}
    for c in tbl.columns:
        sc = str(c)
        if sc.startswith("Team"):
            rename[c] = "Team"
        elif sc in ("M", "P", "Pld"):
            rename[c] = "M"
        elif sc == "W":
            rename[c] = "W"
        elif sc == "L":
            rename[c] = "L"
        elif sc in ("NR", "N/R"):
            rename[c] = "NR"
        elif "Pt" in sc or "Points" in sc:
            rename[c] = "Pts"
        elif "NRR" in sc:
            rename[c] = "NRR"
    tbl = tbl.rename(columns=rename)
    keep = [c for c in ("Team", "M", "W", "L", "NR", "Pts", "NRR") if c in tbl.columns]
    return tbl[keep].copy()


# %% Audit one season
def audit_season(deltas_df: pd.DataFrame, season: str) -> pd.DataFrame:
    try:
        wiki = fetch_wikipedia_table(season)
    except Exception as exc:
        print(f"  [{season}] skipped: {exc}")
        return pd.DataFrame()

    rows = []
    for _, w in wiki.iterrows():
        raw_team = re.sub(r"\(\w\)", "", str(w["Team"])).strip()
        team = normalize_team(raw_team)
        if not team or team == "unknown":
            continue
        try:
            wiki_nrr = float(str(w["NRR"]).replace("−", "-"))
        except (ValueError, TypeError):
            continue
        ours = _compute_team_season_nrr(deltas_df, team, season)
        if ours is None:
            rows.append({"season": season, "team": team, "wiki_nrr": wiki_nrr,
                         "our_nrr": None, "abs_diff": None, "flag": "missing"})
            continue
        diff = ours - wiki_nrr
        rows.append({
            "season": season, "team": team, "wiki_nrr": wiki_nrr,
            "our_nrr": ours, "abs_diff": abs(diff),
            "flag": "ok" if abs(diff) <= TOLERANCE else "DIVERGE",
        })
    return pd.DataFrame(rows)


# %% Run
if __name__ == "__main__":
    deltas = pd.read_parquet(DELTAS_PATH)
    print(f"Loaded deltas: {len(deltas):,} rows, {deltas['match_id'].nunique()} matches")

    all_rows = []
    for s in SEASONS:
        print(f"  auditing {s}...")
        all_rows.append(audit_season(deltas, s))

    audit = pd.concat([r for r in all_rows if not r.empty], ignore_index=True)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    summary = audit.groupby("flag").size().to_dict()
    print(summary)
    print()
    diverge = audit[audit["flag"] == "DIVERGE"].sort_values("abs_diff", ascending=False)
    print(f"Divergences ({len(diverge)} total):")
    print(diverge.to_string(index=False))

    out = PROJECT_ROOT / "data" / "processed" / "nrr_audit.csv"
    audit.to_csv(out, index=False)
    print(f"\nFull audit saved to {out}")
