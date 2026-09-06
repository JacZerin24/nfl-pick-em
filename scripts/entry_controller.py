"""Plan, wait for, and verify official pre-kick Pick'em entry snapshots.

A scheduled controller can start well before kickoff, then wait inside the job to
T-12. This avoids relying on GitHub cron to fire at an exact near-kick minute.
If the final attempt fails, the same workflow can retry at T-5.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from operational_pickem import add_kickoff_times

OFFICIAL_ROLES = {"FINAL_ENTRY", "FALLBACK_ENTRY"}


def parse_utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def write_github_output(path: str | None, values: dict[str, object]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as f:
        for key, value in values.items():
            text = str(value).lower() if isinstance(value, bool) else str(value)
            f.write(f"{key}={text}\n")


def read_official_game_ids(season_root: Path) -> set[str]:
    official: set[str] = set()
    for path in sorted(season_root.glob("week_*/snapshots/*/picks.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("snapshot_role") or "").upper() in OFFICIAL_ROLES:
                        game_id = row.get("game_id")
                        if game_id:
                            official.add(str(game_id))
        except OSError:
            continue
    return official


def plan(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    schedule = nfl.load_schedules(args.season).to_pandas()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    regular = schedule.loc[schedule["game_type"].eq("REG")].copy()
    regular = add_kickoff_times(regular)
    regular = regular.loc[
        regular["home_score"].isna()
        & regular["kickoff_utc"].notna()
        & (regular["kickoff_utc"] > pd.Timestamp(now))
    ].copy()
    if regular.empty:
        values = {"active": False, "reason": "no_unplayed_games"}
        write_github_output(args.github_output, values)
        print(values)
        return 0

    regular["lead_minutes"] = (
        regular["kickoff_utc"] - pd.Timestamp(now)
    ).dt.total_seconds() / 60.0
    candidates = regular.loc[
        (regular["lead_minutes"] >= args.min_lead_minutes)
        & (regular["lead_minutes"] <= args.max_lead_minutes)
    ].copy()
    if candidates.empty:
        values = {"active": False, "reason": "no_game_in_controller_window"}
        write_github_output(args.github_output, values)
        print(values)
        return 0

    earliest = candidates["kickoff_utc"].min()
    cluster = candidates.loc[candidates["kickoff_utc"].eq(earliest)].copy()
    already = read_official_game_ids(args.archive_root / str(args.season))
    cluster = cluster.loc[~cluster["game_id"].astype(str).isin(already)].copy()
    if cluster.empty:
        values = {"active": False, "reason": "official_entry_already_archived"}
        write_github_output(args.github_output, values)
        print(values)
        return 0

    ids = ",".join(cluster["game_id"].astype(str).tolist())
    kickoff = pd.Timestamp(earliest).to_pydatetime().astimezone(timezone.utc)
    lead = (kickoff - now).total_seconds() / 60.0
    values = {
        "active": True,
        "reason": "entry_cluster_found",
        "game_ids": ids,
        "game_count": len(cluster),
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
        "lead_minutes": f"{lead:.2f}",
    }
    write_github_output(args.github_output, values)
    print(values)
    return 0


def wait_until(args: argparse.Namespace) -> int:
    kickoff = parse_utc(args.kickoff_utc)
    target = kickoff.timestamp() - float(args.target_lead_minutes) * 60.0
    seconds = max(0.0, target - datetime.now(timezone.utc).timestamp())
    if seconds > args.max_sleep_seconds:
        raise SystemExit(
            f"Refusing to sleep {seconds:.0f}s; max is {args.max_sleep_seconds}s"
        )
    print(
        f"Waiting {seconds:.1f}s until T-{args.target_lead_minutes:g} minutes "
        f"for kickoff {kickoff.isoformat()}"
    )
    if seconds > 0:
        time.sleep(seconds)
    return 0


def check(args: argparse.Namespace) -> int:
    requested = {x.strip() for x in args.game_ids.split(",") if x.strip()}
    official = read_official_game_ids(args.archive_root / str(args.season))
    missing = sorted(requested - official)
    if missing:
        print(f"Missing official entry snapshots for: {','.join(missing)}")
        return 1
    print(f"Official entry snapshot exists for all {len(requested)} requested game(s).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--archive-root", type=Path, default=Path("live_archive"))
    p.add_argument("--min-lead-minutes", type=float, default=3.0)
    p.add_argument("--max-lead-minutes", type=float, default=75.0)
    p.add_argument("--github-output", type=str, default=None)
    p.set_defaults(func=plan)

    w = sub.add_parser("wait")
    w.add_argument("--kickoff-utc", required=True)
    w.add_argument("--target-lead-minutes", type=float, required=True)
    w.add_argument("--max-sleep-seconds", type=float, default=5400.0)
    w.set_defaults(func=wait_until)

    c = sub.add_parser("check")
    c.add_argument("--season", type=int, default=2026)
    c.add_argument("--archive-root", type=Path, default=Path("live_archive"))
    c.add_argument("--game-ids", required=True)
    c.set_defaults(func=check)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
