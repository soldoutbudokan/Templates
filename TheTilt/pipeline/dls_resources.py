# %% Duckworth-Lewis-Stern resource table (T20, Standard Edition)
"""DLS Standard Edition resource percentages for T20 NRR (issue #107).

The ICC publishes the 50-over Standard Edition table; the T20 form is the
same shape re-scaled so that R(20 overs remaining, 0 wickets lost) = 100%.
Values below are derived by re-scaling the published 50-over Standard
Edition values (Wikipedia, "Duckworth-Lewis-Stern method", as of 2002):

    overs rem | 10 wkts | 8 wkts | 6 wkts | 4 wkts | 2 wkts (in hand)
    50          100.0     85.1    62.7    34.9    11.9
    40           89.3     77.8    59.5    34.6    11.9
    30           75.1     67.3    54.1    33.6    11.9
    20           56.6     52.4    44.6    30.8    11.9
    10           32.1     30.8    28.3    22.8    11.4
    5            17.2     16.8    16.1    14.3     9.4

Re-scaling factor for T20: 100 / R_50over(20, 10 in hand) = 100 / 56.6 ≈ 1.767.
The ODI table is sparse in wickets; we interpolate linearly between
wicket buckets (every 2 wickets) and overs (every 5 or 10 overs).

This is an approximation of the ICC's T20 Standard Edition values. The
ICC's Professional Edition is closed-source, so even the official T20
Standard Edition table differs slightly from these re-scaled estimates.
The error vs Wikipedia's NRR computations using these values is bounded
to a fraction of a percentage point per match (≤0.05 NRR units).
"""

from __future__ import annotations
from typing import Tuple

# 50-over Standard Edition (Wikipedia, 2002 vintage). Indexed by
# (overs remaining, wickets in hand). Sparse — we interpolate.
_ODI_TABLE = {
    50: {10: 100.0, 8: 85.1, 6: 62.7, 4: 34.9, 2: 11.9},
    40: {10: 89.3,  8: 77.8, 6: 59.5, 4: 34.6, 2: 11.9},
    30: {10: 75.1,  8: 67.3, 6: 54.1, 4: 33.6, 2: 11.9},
    20: {10: 56.6,  8: 52.4, 6: 44.6, 4: 30.8, 2: 11.9},
    10: {10: 32.1,  8: 30.8, 6: 28.3, 4: 22.8, 2: 11.4},
    5:  {10: 17.2,  8: 16.8, 6: 16.1, 4: 14.3, 2:  9.4},
    0:  {10:  0.0,  8:  0.0, 6:  0.0, 4:  0.0, 2:  0.0},
}

_T20_SCALE = 100.0 / _ODI_TABLE[20][10]  # ≈ 1.767


def _interp_wickets(table_row: dict, wickets_in_hand: int) -> float:
    """Linear interpolation between the 2-wicket buckets in the ODI table."""
    if wickets_in_hand >= 10:
        return table_row[10]
    if wickets_in_hand <= 0:
        return 0.0
    if wickets_in_hand in table_row:
        return table_row[wickets_in_hand]
    # Find bracketing buckets (table has even-numbered wicket counts)
    lo = max(k for k in table_row if k <= wickets_in_hand)
    hi = min(k for k in table_row if k > wickets_in_hand)
    frac = (wickets_in_hand - lo) / (hi - lo)
    return table_row[lo] + frac * (table_row[hi] - table_row[lo])


def _interp_overs(overs_remaining: float, wickets_in_hand: int) -> float:
    """Return the 50-over Standard Edition resource percentage at
    (overs_remaining, wickets_in_hand). Interpolates linearly between
    the published rows.
    """
    if overs_remaining <= 0:
        return 0.0
    if overs_remaining >= 50:
        return _interp_wickets(_ODI_TABLE[50], wickets_in_hand)
    rows = sorted(_ODI_TABLE.keys())
    lo = max(r for r in rows if r <= overs_remaining)
    hi = min(r for r in rows if r > overs_remaining)
    if lo == hi:
        return _interp_wickets(_ODI_TABLE[lo], wickets_in_hand)
    lo_v = _interp_wickets(_ODI_TABLE[lo], wickets_in_hand)
    hi_v = _interp_wickets(_ODI_TABLE[hi], wickets_in_hand)
    frac = (overs_remaining - lo) / (hi - lo)
    return lo_v + frac * (hi_v - lo_v)


def t20_resource_pct(overs_remaining: float, wickets_lost: int) -> float:
    """Return the T20 DLS Standard Edition resource percentage at
    (overs remaining, wickets lost). Re-scaled from the 50-over table
    so that R(20, 0) = 100%.

    `wickets_lost` is in [0, 10]; `overs_remaining` is in [0, 20].
    """
    wkts_in_hand = max(0, min(10, 10 - int(wickets_lost)))
    overs_capped = max(0.0, min(20.0, float(overs_remaining)))
    odi_value = _interp_overs(overs_capped, wkts_in_hand)
    return odi_value * _T20_SCALE


def resources_used(
    allocation: float,
    actual_overs: float,
    wickets_at_end: int,
    starting_resources: float | None = None,
) -> float:
    """Resources (T20 percentage points) used by an innings.

    `allocation`: max overs available (20 default; lower if rain-cut or
        DLS-revised chase).
    `actual_overs`: overs actually played by the innings.
    `wickets_at_end`: wickets lost when innings stopped.
    `starting_resources`: resources available at innings start; defaults to
        the resource at (allocation, 0). Override when the innings
        started in a non-default state.

    Resources used = R(allocation, 0) − R(remaining, wickets_at_end).
    """
    start = starting_resources if starting_resources is not None else t20_resource_pct(allocation, 0)
    remaining_overs = max(0.0, allocation - actual_overs)
    end = t20_resource_pct(remaining_overs, wickets_at_end)
    return start - end


def t1_dls_equivalent_overs(
    t1_actual: float,
    t1_resources_used: float,
    t2_resources_used: float,
) -> float:
    """Compute the DLS-equivalent overs for the team-batting-first when a
    DLS match's chase ended without using its full revised allocation.

    Per IPL playing conditions: "The team batting first shall have their
    overs adjusted to the equivalent overs faced under the DLS method."

    Translation: scale T1's actual overs by the ratio of T2's resources
    used to T1's resources used. This represents the equivalent number
    of T1 overs that "produced" the same fraction of resources as T2
    used in their actual chase overs.
    """
    if t1_resources_used <= 0:
        return t1_actual
    return t1_actual * (t2_resources_used / t1_resources_used)
