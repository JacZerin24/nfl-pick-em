"""Compare the legacy operational runner with the frozen-artifact fast runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXACT_COLUMNS = ["market_pick", "market_underdog", "final_pick", "decision_type"]
NUMERIC_COLUMNS = [
    "market_fav_prob",
    "p_home_market",
    "p_home_elo",
    "p_home_logistic",
    "p_home_catboost",
    "p_home_residual",
    "p_dog_matchup_logistic",
    "p_dog_variance_catboost",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slow-root", type=Path, required=True)
    p.add_argument("--fast-root", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-7)
    p.add_argument("--report", type=Path, default=Path("outputs/fast_live_parity.md"))
    return p.parse_args()


def latest_snapshot(root: Path) -> Path:
    files = sorted(root.glob("**/snapshots/*/picks.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshot picks.csv found under {root}")
    return files[-1]


def main() -> None:
    args = parse_args()
    slow_path = latest_snapshot(args.slow_root)
    fast_path = latest_snapshot(args.fast_root)
    slow = pd.read_csv(slow_path).sort_values("game_id").reset_index(drop=True)
    fast = pd.read_csv(fast_path).sort_values("game_id").reset_index(drop=True)

    errors: list[str] = []
    if slow["game_id"].astype(str).tolist() != fast["game_id"].astype(str).tolist():
        errors.append("game_id sets/order differ")

    exact_rows = []
    for col in EXACT_COLUMNS:
        if col not in slow or col not in fast:
            errors.append(f"missing exact column {col}")
            continue
        same = slow[col].fillna("__NA__").astype(str).eq(fast[col].fillna("__NA__").astype(str))
        mismatches = int((~same).sum())
        exact_rows.append((col, mismatches))
        if mismatches:
            errors.append(f"{col}: {mismatches} mismatch(es)")

    numeric_rows = []
    for col in NUMERIC_COLUMNS:
        if col not in slow or col not in fast:
            errors.append(f"missing numeric column {col}")
            continue
        a = pd.to_numeric(slow[col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(fast[col], errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.abs(a - b)
        diff[both_nan] = 0.0
        max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
        numeric_rows.append((col, max_diff))
        if not np.all((diff <= args.tolerance) | both_nan):
            errors.append(f"{col}: max abs diff {max_diff:.3g} > {args.tolerance:.3g}")

    lines = [
        "# Fast Live Runner Parity",
        "",
        f"- Legacy snapshot: `{slow_path}`",
        f"- Fast snapshot: `{fast_path}`",
        f"- Games: **{len(slow)}**",
        f"- Numeric tolerance: **{args.tolerance:g}**",
        f"- Result: **{'PASS' if not errors else 'FAIL'}**",
        "",
        "## Exact decision parity",
        "",
        "| Field | Mismatches |",
        "|---|---:|",
    ]
    for col, mismatches in exact_rows:
        lines.append(f"| {col} | {mismatches} |")
    lines.extend(["", "## Probability parity", "", "| Field | Max abs difference |", "|---|---:|"])
    for col, max_diff in numeric_rows:
        lines.append(f"| {col} | {max_diff:.3g} |")
    if errors:
        lines.extend(["", "## Failures", ""])
        lines.extend([f"- {error}" for error in errors])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if errors:
        raise SystemExit("Fast runner failed legacy parity")


if __name__ == "__main__":
    main()
