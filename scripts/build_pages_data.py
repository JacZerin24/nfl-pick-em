"""Build a compact JSON payload for the GitHub Pages pick'em dashboard.

The dashboard intentionally reuses the same immutable snapshot archive as the
prospective grader. For each game it selects the newest archived prediction
strictly before kickoff, so already-started games freeze while later games can
continue to update.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--archive-root", type=Path, default=Path("live_archive"))
    p.add_argument("--output", type=Path, default=Path("site/data/dashboard.json"))
    return p.parse_args()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 1:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_valid_rows(week_dir: Path) -> list[dict[str, str]]:
    rows_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in sorted(week_dir.glob("snapshots/*/picks.csv")):
        for row in read_csv(path):
            game_id = row.get("game_id", "")
            snap = parse_dt(row.get("snapshot_utc"))
            kickoff = parse_dt(row.get("kickoff_utc"))
            if not game_id or snap is None or kickoff is None or snap >= kickoff:
                continue
            row = dict(row)
            row["archive_path"] = str(path)
            rows_by_game[game_id].append(row)

    # Fall back to latest.csv during initial setup if no immutable snapshots exist.
    if not rows_by_game:
        for row in read_csv(week_dir / "latest.csv"):
            game_id = row.get("game_id", "")
            snap = parse_dt(row.get("snapshot_utc"))
            kickoff = parse_dt(row.get("kickoff_utc"))
            if game_id and snap and kickoff and snap < kickoff:
                row = dict(row)
                row["archive_path"] = str(week_dir / "latest.csv")
                rows_by_game[game_id].append(row)

    latest: list[dict[str, str]] = []
    for game_rows in rows_by_game.values():
        game_rows.sort(key=lambda r: parse_dt(r.get("snapshot_utc")) or datetime.min.replace(tzinfo=timezone.utc))
        latest.append(game_rows[-1])
    latest.sort(key=lambda r: parse_dt(r.get("kickoff_utc")) or datetime.max.replace(tzinfo=timezone.utc))
    return latest


def game_payload(row: dict[str, str], result: dict[str, str] | None, now: datetime) -> dict[str, Any]:
    kickoff = parse_dt(row.get("kickoff_utc"))
    snap = parse_dt(row.get("snapshot_utc"))
    frozen = bool(kickoff and now >= kickoff)
    lead_minutes = None
    if kickoff and snap:
        lead_minutes = round((kickoff - snap).total_seconds() / 60.0, 1)

    payload: dict[str, Any] = {
        "game_id": row.get("game_id"),
        "week": as_int(row.get("week")),
        "gameday": row.get("gameday"),
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z") if kickoff else row.get("kickoff_utc"),
        "snapshot_utc": snap.isoformat().replace("+00:00", "Z") if snap else row.get("snapshot_utc"),
        "lead_minutes": lead_minutes,
        "frozen": frozen,
        "away_team": row.get("away_team"),
        "home_team": row.get("home_team"),
        "spread_line": as_float(row.get("spread_line")),
        "total_line": as_float(row.get("total_line")),
        "away_moneyline": as_float(row.get("away_moneyline")),
        "home_moneyline": as_float(row.get("home_moneyline")),
        "market_pick": row.get("market_pick"),
        "market_underdog": row.get("market_underdog"),
        "market_fav_prob": as_float(row.get("market_fav_prob")),
        "p_home_market": as_float(row.get("p_home_market")),
        "p_home_elo": as_float(row.get("p_home_elo")),
        "p_home_logistic": as_float(row.get("p_home_logistic")),
        "p_home_catboost": as_float(row.get("p_home_catboost")),
        "p_home_residual": as_float(row.get("p_home_residual")),
        "p_dog_matchup_logistic": as_float(row.get("p_dog_matchup_logistic")),
        "p_dog_variance_catboost": as_float(row.get("p_dog_variance_catboost")),
        "final_pick": row.get("final_pick"),
        "decision_type": row.get("decision_type"),
        "model_version": row.get("model_version"),
        "training_end_season": as_int(row.get("training_end_season")),
        "pbp_end_season": as_int(row.get("pbp_end_season")),
    }

    if result:
        payload["result"] = {
            "away_score": as_int(result.get("away_score")),
            "home_score": as_int(result.get("home_score")),
            "winner": result.get("result_team"),
            "is_tie": as_bool(result.get("is_tie")),
            "pick_correct": as_bool(result.get("pick_correct")),
            "market_correct": as_bool(result.get("market_correct")),
            "net_vs_market": as_float(result.get("net_vs_market")),
        }
    else:
        payload["result"] = None
    return payload


def scoreboard_payload(results: list[dict[str, str]]) -> dict[str, Any]:
    scored = [r for r in results if not as_bool(r.get("is_tie"))]
    model_wins = sum(1 for r in scored if as_bool(r.get("pick_correct")))
    market_wins = sum(1 for r in scored if as_bool(r.get("market_correct")))
    games = len(scored)

    by_decision: dict[str, dict[str, Any]] = {}
    grouped_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_week: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        grouped_decision[row.get("decision_type") or "UNKNOWN"].append(row)
        week = as_int(row.get("week"))
        if week is not None:
            grouped_week[week].append(row)

    for decision, rows in sorted(grouped_decision.items()):
        wins = sum(1 for r in rows if as_bool(r.get("pick_correct")))
        market = sum(1 for r in rows if as_bool(r.get("market_correct")))
        by_decision[decision] = {
            "games": len(rows),
            "wins": wins,
            "losses": len(rows) - wins,
            "accuracy": wins / len(rows) if rows else None,
            "net_vs_market": wins - market,
        }

    by_week: dict[str, dict[str, Any]] = {}
    for week, rows in sorted(grouped_week.items()):
        wins = sum(1 for r in rows if as_bool(r.get("pick_correct")))
        market = sum(1 for r in rows if as_bool(r.get("market_correct")))
        by_week[str(week)] = {
            "games": len(rows),
            "model_wins": wins,
            "market_wins": market,
            "net_vs_market": wins - market,
        }

    return {
        "games": games,
        "model_wins": model_wins,
        "model_losses": games - model_wins,
        "model_accuracy": model_wins / games if games else None,
        "market_wins": market_wins,
        "market_losses": games - market_wins,
        "market_accuracy": market_wins / games if games else None,
        "net_vs_market": model_wins - market_wins,
        "by_decision": by_decision,
        "by_week": by_week,
    }


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    season_root = args.archive_root / str(args.season)
    results = read_csv(season_root / "season_results.csv")
    result_map = {r.get("game_id", ""): r for r in results if r.get("game_id")}

    weeks: list[dict[str, Any]] = []
    for week_dir in sorted(season_root.glob("week_*")):
        try:
            week_num = int(week_dir.name.split("_")[-1])
        except ValueError:
            continue
        games = [game_payload(row, result_map.get(row.get("game_id", "")), now) for row in latest_valid_rows(week_dir)]
        if games:
            weeks.append({"week": week_num, "games": games})

    current_week = None
    for week in weeks:
        if any(not game["frozen"] for game in week["games"]):
            current_week = week["week"]
            break
    if current_week is None and weeks:
        current_week = weeks[-1]["week"]

    versions = [game.get("model_version") for week in weeks for game in week["games"] if game.get("model_version")]
    latest_snapshots = [
        parse_dt(game.get("snapshot_utc"))
        for week in weeks
        for game in week["games"]
        if game.get("snapshot_utc")
    ]
    latest_snapshots = [dt for dt in latest_snapshots if dt]

    payload = {
        "season": args.season,
        "generated_utc": now.isoformat().replace("+00:00", "Z"),
        "latest_snapshot_utc": max(latest_snapshots).isoformat().replace("+00:00", "Z") if latest_snapshots else None,
        "model_version": versions[-1] if versions else None,
        "current_week": current_week,
        "validated_holdout": {
            "games": 1865,
            "wins": 1254,
            "accuracy": 0.6724,
            "market_wins": 1238,
            "market_accuracy": 0.6638,
            "net_wins": 16,
        },
        "rules": {
            "close_threshold": 0.525,
            "strong_favorite_threshold": 0.80,
            "description": "Market anchor; residual resolves <52.5% tossups; true underdog only when both frozen upset specialists agree; 80%+ favorites stay with market.",
        },
        "scoreboard": scoreboard_payload(results),
        "weeks": weeks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(weeks)} week(s)")


if __name__ == "__main__":
    main()
