"""Grade archived 2026 NFL pick'em snapshots without hindsight.

For each game, the official model pick is the latest archived snapshot whose
snapshot timestamp is strictly before that game's kickoff. Later snapshots for
other games do not rewrite already-kicked games. Results are joined only after
the game is complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--archive-root", type=Path, default=Path("live_archive"))
    return p.parse_args()


def collect_official_snapshots(season_root: Path) -> pd.DataFrame:
    files = sorted(season_root.glob("week_*/snapshots/*/picks.csv"))
    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty or "game_id" not in frame.columns:
            continue
        frame["archive_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    all_picks = pd.concat(frames, ignore_index=True)
    all_picks["snapshot_utc"] = pd.to_datetime(
        all_picks["snapshot_utc"], utc=True, errors="coerce"
    )
    all_picks["kickoff_utc"] = pd.to_datetime(
        all_picks["kickoff_utc"], utc=True, errors="coerce"
    )
    all_picks = all_picks.loc[
        all_picks["snapshot_utc"].notna()
        & all_picks["kickoff_utc"].notna()
        & (all_picks["snapshot_utc"] < all_picks["kickoff_utc"])
    ].copy()
    if all_picks.empty:
        return all_picks

    all_picks["lead_minutes"] = (
        (all_picks["kickoff_utc"] - all_picks["snapshot_utc"]).dt.total_seconds()
        / 60.0
    )
    return (
        all_picks.sort_values(["game_id", "snapshot_utc"])
        .drop_duplicates("game_id", keep="last")
        .reset_index(drop=True)
    )


def completed_schedule(season: int) -> pd.DataFrame:
    schedule = nfl.load_schedules(season).to_pandas()
    schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    schedule = schedule.loc[
        schedule["home_score"].notna() & schedule["away_score"].notna()
    ].copy()
    if schedule.empty:
        return schedule

    home_score = pd.to_numeric(schedule["home_score"], errors="coerce")
    away_score = pd.to_numeric(schedule["away_score"], errors="coerce")
    schedule["result_team"] = np.where(
        home_score > away_score,
        schedule["home_team"],
        np.where(home_score < away_score, schedule["away_team"], "TIE"),
    )
    schedule["is_tie"] = home_score.eq(away_score)
    return schedule


def grade(official: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if official.empty or schedule.empty:
        return pd.DataFrame()

    result_cols = [
        "game_id",
        "home_score",
        "away_score",
        "result_team",
        "is_tie",
    ]
    merged = official.merge(schedule[result_cols], on="game_id", how="inner")
    if merged.empty:
        return merged

    merged["pick_correct"] = np.where(
        merged["is_tie"], np.nan, merged["final_pick"].eq(merged["result_team"])
    )
    merged["market_correct"] = np.where(
        merged["is_tie"], np.nan, merged["market_pick"].eq(merged["result_team"])
    )
    merged["net_vs_market"] = (
        pd.to_numeric(merged["pick_correct"], errors="coerce")
        - pd.to_numeric(merged["market_correct"], errors="coerce")
    )
    return merged.sort_values(["week", "kickoff_utc", "game_id"]).reset_index(drop=True)


def record_text(wins: int, losses: int, ties: int = 0) -> str:
    return f"{wins}-{losses}" if ties == 0 else f"{wins}-{losses}-{ties}"


def render_summary(results: pd.DataFrame, season: int) -> str:
    lines = [f"# {season} NFL Pick'em Prospective Scoreboard", ""]
    if results.empty:
        lines.extend(
            [
                "No completed archived games have been graded yet.",
                "",
                "The official pick for each game will be the latest archived snapshot strictly before kickoff.",
            ]
        )
        return "\n".join(lines) + "\n"

    scored = results.loc[~results["is_tie"].astype(bool)].copy()
    ties = int(results["is_tie"].sum())
    model_wins = int(scored["pick_correct"].astype(bool).sum())
    market_wins = int(scored["market_correct"].astype(bool).sum())
    games = len(scored)
    model_losses = games - model_wins
    market_losses = games - market_wins

    lines.extend(
        [
            f"Official scored games: **{games}**" + (f" (+ {ties} tie excluded)" if ties else ""),
            "",
            f"- Model: **{record_text(model_wins, model_losses)} ({model_wins / games:.1%})**" if games else "- Model: no decisions yet",
            f"- Market favorite: **{record_text(market_wins, market_losses)} ({market_wins / games:.1%})**" if games else "- Market: no decisions yet",
            f"- Net correct picks vs market: **{model_wins - market_wins:+d}**",
            "",
            "## By decision type",
            "",
            "| Decision | Games | Record | Accuracy | Net vs market |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for decision, group in scored.groupby("decision_type", dropna=False):
        n = len(group)
        w = int(group["pick_correct"].astype(bool).sum())
        net = int(pd.to_numeric(group["net_vs_market"], errors="coerce").sum())
        lines.append(f"| {decision} | {n} | {w}-{n-w} | {w/n:.1%} | {net:+d} |")

    lines.extend(
        [
            "",
            "## By week",
            "",
            "| Week | Games | Model | Market | Net |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for week, group in scored.groupby("week"):
        n = len(group)
        mw = int(group["pick_correct"].astype(bool).sum())
        vw = int(group["market_correct"].astype(bool).sum())
        lines.append(f"| {int(week)} | {n} | {mw}-{n-mw} | {vw}-{n-vw} | {mw-vw:+d} |")

    lines.extend(
        [
            "",
            "## Audit rule",
            "",
            "For each game, grading uses the latest timestamped snapshot archived before that game's kickoff. "
            "The frozen 2026 model rules are not retuned from these results during the prospective season test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    season_root = args.archive_root / str(args.season)
    season_root.mkdir(parents=True, exist_ok=True)

    official = collect_official_snapshots(season_root)
    schedule = completed_schedule(args.season)
    results = grade(official, schedule)

    results.to_csv(season_root / "season_results.csv", index=False)
    summary = render_summary(results, args.season)
    (season_root / "season_summary.md").write_text(summary, encoding="utf-8")

    if official.empty:
        print("No archived snapshots available yet.")
    elif results.empty:
        print("Archived snapshots exist, but no matching games are complete yet.")
    else:
        scored = results.loc[~results["is_tie"].astype(bool)]
        model = int(scored["pick_correct"].astype(bool).sum())
        market = int(scored["market_correct"].astype(bool).sum())
        print(
            f"Graded {len(scored)} games: model={model}-{len(scored)-model}, "
            f"market={market}-{len(scored)-market}, net={model-market:+d}"
        )
    print(f"Wrote {season_root / 'season_summary.md'}")


if __name__ == "__main__":
    main()
