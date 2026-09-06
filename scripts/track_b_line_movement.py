"""Build a prospective line-movement panel from immutable live pick snapshots.

This script is analysis-only. It never changes the frozen official 2026 pick rule.
The raw timestamped snapshot archive remains the source of truth; all outputs here
are reproducible derived tables for later prospective research.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--archive-root", type=Path, default=Path("live_archive"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_b_line_movement"))
    return p.parse_args()


def load_snapshots(season_root: Path) -> pd.DataFrame:
    files = sorted(season_root.glob("week_*/snapshots/*/picks.csv"))
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty or "game_id" not in frame.columns:
            continue
        frame["archive_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel["snapshot_utc"] = pd.to_datetime(panel["snapshot_utc"], utc=True, errors="coerce")
    panel["kickoff_utc"] = pd.to_datetime(panel["kickoff_utc"], utc=True, errors="coerce")
    panel = panel.loc[
        panel["snapshot_utc"].notna()
        & panel["kickoff_utc"].notna()
        & (panel["snapshot_utc"] < panel["kickoff_utc"])
    ].copy()
    if panel.empty:
        return panel

    for col in [
        "p_home_market",
        "market_fav_prob",
        "spread_line",
        "total_line",
        "away_moneyline",
        "home_moneyline",
    ]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["lead_minutes"] = (
        (panel["kickoff_utc"] - panel["snapshot_utc"]).dt.total_seconds() / 60.0
    )
    panel = panel.sort_values(["game_id", "snapshot_utc", "archive_path"]).reset_index(drop=True)
    # Exact duplicates can occur during workflow smoke tests. Keep them in the audit
    # archive but collapse them in the research panel so they do not masquerade as
    # independent market observations.
    dedupe_cols = [
        c
        for c in [
            "game_id",
            "snapshot_utc",
            "p_home_market",
            "spread_line",
            "away_moneyline",
            "home_moneyline",
            "final_pick",
            "decision_type",
        ]
        if c in panel.columns
    ]
    panel = panel.drop_duplicates(dedupe_cols, keep="last").copy()
    return panel.sort_values(["game_id", "snapshot_utc"]).reset_index(drop=True)


def add_movement_features(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    out = panel.copy()
    g = out.groupby("game_id", sort=False)
    out["snapshot_sequence"] = g.cumcount() + 1
    out["snapshots_for_game"] = g["game_id"].transform("size")
    out["is_first_snapshot"] = out["snapshot_sequence"].eq(1)
    out["is_latest_pre_kick_snapshot"] = out["snapshot_sequence"].eq(out["snapshots_for_game"])

    for col in ["p_home_market", "spread_line", "total_line", "away_moneyline", "home_moneyline"]:
        if col not in out.columns:
            continue
        first = g[col].transform("first")
        out[f"first_{col}"] = first
        out[f"change_from_first_{col}"] = out[col] - first

    if "p_home_market" in out.columns:
        out["home_market_move_pp"] = 100.0 * out["change_from_first_p_home_market"]
        out["abs_home_market_move_pp"] = out["home_market_move_pp"].abs()
    if "market_pick" in out.columns:
        out["first_market_pick"] = g["market_pick"].transform("first")
        out["market_favorite_flipped"] = out["market_pick"].ne(out["first_market_pick"])
    if "final_pick" in out.columns:
        out["first_final_pick"] = g["final_pick"].transform("first")
        out["model_pick_changed_from_first"] = out["final_pick"].ne(out["first_final_pick"])
    if "decision_type" in out.columns:
        out["first_decision_type"] = g["decision_type"].transform("first")
        out["decision_changed_from_first"] = out["decision_type"].ne(out["first_decision_type"])
    return out


def summarize_games(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for game_id, group in panel.groupby("game_id", sort=False):
        g = group.sort_values("snapshot_utc")
        first = g.iloc[0]
        last = g.iloc[-1]
        home_probs = pd.to_numeric(g.get("p_home_market"), errors="coerce")
        spreads = pd.to_numeric(g.get("spread_line"), errors="coerce")
        rows.append(
            {
                "game_id": game_id,
                "season": int(last["season"]),
                "week": int(last["week"]),
                "away_team": last["away_team"],
                "home_team": last["home_team"],
                "kickoff_utc": last["kickoff_utc"],
                "snapshots": len(g),
                "first_snapshot_utc": first["snapshot_utc"],
                "latest_snapshot_utc": last["snapshot_utc"],
                "first_lead_minutes": float(first["lead_minutes"]),
                "latest_lead_minutes": float(last["lead_minutes"]),
                "first_market_pick": first.get("market_pick"),
                "latest_market_pick": last.get("market_pick"),
                "market_favorite_flip": bool(g.get("market_pick", pd.Series(dtype=str)).nunique(dropna=True) > 1),
                "first_home_market_prob": first.get("p_home_market"),
                "latest_home_market_prob": last.get("p_home_market"),
                "home_market_move_pp": (
                    100.0 * (float(last.get("p_home_market")) - float(first.get("p_home_market")))
                    if pd.notna(first.get("p_home_market")) and pd.notna(last.get("p_home_market"))
                    else np.nan
                ),
                "max_abs_home_market_move_pp": (
                    100.0 * float((home_probs - home_probs.iloc[0]).abs().max())
                    if home_probs.notna().any()
                    else np.nan
                ),
                "first_spread_line": first.get("spread_line"),
                "latest_spread_line": last.get("spread_line"),
                "spread_move": (
                    float(last.get("spread_line")) - float(first.get("spread_line"))
                    if pd.notna(first.get("spread_line")) and pd.notna(last.get("spread_line"))
                    else np.nan
                ),
                "first_model_pick": first.get("final_pick"),
                "latest_model_pick": last.get("final_pick"),
                "model_pick_flip": bool(g.get("final_pick", pd.Series(dtype=str)).nunique(dropna=True) > 1),
                "first_decision_type": first.get("decision_type"),
                "latest_decision_type": last.get("decision_type"),
                "decision_type_changed": bool(g.get("decision_type", pd.Series(dtype=str)).nunique(dropna=True) > 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["week", "kickoff_utc", "game_id"]).reset_index(drop=True)


def render_summary(panel: pd.DataFrame, games: pd.DataFrame, season: int) -> str:
    lines = [f"# Track B: {season} Prospective Line-Movement Collection", ""]
    lines.append("**Status: context-only research. These outputs do not alter the frozen official pick rule.**")
    lines.append("")
    if panel.empty:
        lines.append("No valid pre-kick snapshots are archived yet.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- Games represented: **{len(games)}**",
            f"- Distinct pre-kick snapshot rows: **{len(panel)}**",
            f"- Games with 2+ distinct observations: **{int((games['snapshots'] >= 2).sum())}**",
            f"- Market-favorite flips observed: **{int(games['market_favorite_flip'].sum())}**",
            f"- Official-model pick flips observed: **{int(games['model_pick_flip'].sum())}**",
            "",
        ]
    )

    multi = games.loc[games["snapshots"] >= 2].copy()
    if multi.empty:
        lines.append(
            "The archive is collecting correctly, but no game has multiple distinct market observations yet. "
            "Movement analysis becomes meaningful as later automatic snapshots accumulate."
        )
    else:
        lines.extend(
            [
                "## Largest market-probability moves so far",
                "",
                "| Week | Matchup | Snapshots | Home-prob move | Favorite flip | Model flip |",
                "|---:|---|---:|---:|---|---|",
            ]
        )
        top = multi.sort_values("max_abs_home_market_move_pp", ascending=False).head(12)
        for _, row in top.iterrows():
            matchup = f"{row['away_team']} @ {row['home_team']}"
            lines.append(
                f"| {int(row['week'])} | {matchup} | {int(row['snapshots'])} | "
                f"{row['home_market_move_pp']:+.2f} pp | "
                f"{'YES' if row['market_favorite_flip'] else 'no'} | "
                f"{'YES' if row['model_pick_flip'] else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## Research protocol",
            "",
            "The raw timestamped `live_archive` remains immutable. This builder derives a panel with one row per distinct pre-kick observation and a per-game movement summary. Future tests should predefine movement features and evaluation rules before using completed 2026 outcomes to decide whether any signal can graduate from context-only status.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    season_root = args.archive_root / str(args.season)
    panel = add_movement_features(load_snapshots(season_root))
    games = summarize_games(panel)

    panel.to_csv(args.output_dir / "snapshot_panel.csv", index=False)
    games.to_csv(args.output_dir / "game_summary.csv", index=False)
    summary = render_summary(panel, games, args.season)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
