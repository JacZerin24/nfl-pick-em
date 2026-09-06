"""Probe current nflreadpy player-level schemas for Track F research."""
from __future__ import annotations

import nflreadpy as nfl


def show(name: str, frame) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={frame.height} cols={len(frame.columns)}")
    print(frame.columns)
    print(frame.head(2))


def main() -> None:
    season = 2024
    show("player_stats", nfl.load_player_stats([season]))
    show("snap_counts", nfl.load_snap_counts([season]))
    for stat_type in ("pass", "rec", "def"):
        show(f"pfr_{stat_type}", nfl.load_pfr_advstats([season], stat_type=stat_type, summary_level="week"))
    for stat_type in ("passing", "receiving", "rushing"):
        show(f"ngs_{stat_type}", nfl.load_nextgen_stats([season], stat_type=stat_type))


if __name__ == "__main__":
    main()
