"""Authoritative nested true-upset policy replay.

Uses the complete Phase 1 OOF prediction history for training, including
2016-2018 when testing 2019. Outputs intentionally overwrite the initial upset
files from phase1_pickem_strategy.py so the artifact contains the corrected,
full-history walk-forward result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1_pickem_strategy import upset_policy_summary, walk_forward_upset_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("outputs/phase1/phase1_predictions.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/pickem_strategy"))
    p.add_argument("--first-test-season", type=int, default=2019)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.predictions)
    base = base.loc[base["home_win"].notna()].copy()

    preds, folds, selections = walk_forward_upset_policy(base, args.first_test_season)
    test = base.loc[base["season"] >= args.first_test_season].copy()
    summary = upset_policy_summary(test, preds)

    preds.to_csv(args.output_dir / "true_upset_policy_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "true_upset_policy_folds.csv", index=False)
    selections.to_csv(args.output_dir / "true_upset_policy_selections.csv", index=False)
    summary.to_csv(args.output_dir / "true_upset_policy_summary.csv", index=False)

    print("\nCorrected nested true-upset policy")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
