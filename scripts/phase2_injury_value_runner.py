"""Compatibility runner for the Phase 2 injury experiment.

nflreadpy 0.1.5 exposes the injury season-type field as ``game_type`` even
though one nflverse dictionary still documents ``season_type``. This runner
patches only that loader boundary and then calls the main experiment unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import phase2_injury_value as base


def build_team_week_injuries(start_season: int, end_season: int):
    seasons = list(range(start_season, end_season + 1))
    print(f"Loading injuries {start_season}-{end_season}...")
    injuries = base.nfl.load_injuries(seasons).to_pandas()
    players = base.nfl.load_players().to_pandas()
    roles = base.build_snap_role_history(seasons)

    required = {
        "season",
        "team",
        "week",
        "gsis_id",
        "position",
        "report_status",
        "practice_status",
    }
    missing = required.difference(injuries.columns)
    if missing:
        raise RuntimeError(f"Injury data missing required columns: {sorted(missing)}")

    type_col = "game_type" if "game_type" in injuries.columns else "season_type" if "season_type" in injuries.columns else None
    if type_col is None:
        raise RuntimeError(
            "Injury data has neither game_type nor season_type. "
            f"Available columns: {sorted(injuries.columns.tolist())}"
        )

    inj = injuries.loc[injuries[type_col].eq("REG")].copy()
    if "date_modified" in inj.columns:
        inj["date_modified"] = pd.to_datetime(inj["date_modified"], errors="coerce", utc=True)
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")

    mapping = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    inj = inj.merge(mapping, on="gsis_id", how="left")
    inj["order_key"] = inj["season"].astype(int) * 100 + inj["week"].astype(int)
    inj = base.attach_prior_role(inj, roles)
    inj["position_group"] = inj["position"].map(base.pos_group)
    inj["report_bucket"] = inj["report_status"].map(base.report_bucket)
    inj["practice_bucket"] = inj["practice_status"].map(base.practice_bucket)
    inj["role_weight"] = inj["prior_role"].fillna(0.0).clip(0.0, 1.0)

    rows = []
    for (season, week, team), g in inj.groupby(["season", "week", "team"], sort=False):
        row = {
            "season": int(season),
            "week": int(week),
            "team": team,
            "injury_players_listed": int(len(g)),
            "injury_role_coverage": float(g["prior_role"].notna().mean()),
        }
        for status in base.REPORT_BUCKETS:
            mask = g["report_bucket"].eq(status)
            row[f"inj_report_{status}_role_total"] = float(g.loc[mask, "role_weight"].sum())
            row[f"inj_report_{status}_count"] = int(mask.sum())
            for pg in base.POSITION_GROUPS:
                row[f"inj_report_{status}_{pg.lower()}_role"] = float(
                    g.loc[mask & g["position_group"].eq(pg), "role_weight"].sum()
                )
        for status in base.PRACTICE_BUCKETS:
            mask = g["practice_bucket"].eq(status)
            row[f"inj_practice_{status}_role_total"] = float(g.loc[mask, "role_weight"].sum())
        rows.append(row)

    team_week = pd.DataFrame(rows).fillna(0.0)
    diagnostics = {
        "injury_rows": float(len(inj)),
        "pfr_id_match_rate": float(inj["pfr_id"].notna().mean()),
        "prior_role_match_rate": float(inj["prior_role"].notna().mean()),
        "mean_nonzero_role": float(inj.loc[inj["role_weight"] > 0, "role_weight"].mean()) if (inj["role_weight"] > 0).any() else 0.0,
        "max_role": float(inj["role_weight"].max()) if len(inj) else 0.0,
    }
    return team_week, diagnostics


base.build_team_week_injuries = build_team_week_injuries

if __name__ == "__main__":
    base.main()
