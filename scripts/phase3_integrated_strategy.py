"""Phase 3 integrated straight-up pick'em decision strategy.

Combines two non-overlapping research signals:
1. market-anchored residual for close/tossup decisions
2. frozen cross-specialist consensus for true upset calls

The consensus pairing must already have been selected using 2016-2018 only and
its predictions are graded on 2019-2025. The market residual is nested and
walk-forward beginning in 2019.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--residual-predictions", type=Path, required=True)
    p.add_argument("--consensus-predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase3_integrated"))
    p.add_argument("--bootstrap-reps", type=int, default=50000)
    return p.parse_args()


def bootstrap_paired(a: np.ndarray, b: np.ndarray, reps: int, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs: list[np.ndarray] = []
    remaining = reps
    while remaining > 0:
        k = min(1000, remaining)
        idx = rng.integers(0, n, size=(k, n))
        diffs.append((a[idx] - b[idx]).mean(axis=1))
        remaining -= k
    d = np.concatenate(diffs)
    return {
        "lift_accuracy_pp": float(100 * np.mean(a-b)),
        "ci95_low_pp": float(100 * np.quantile(d, 0.025)),
        "ci95_high_pp": float(100 * np.quantile(d, 0.975)),
        "bootstrap_prob_positive": float(np.mean(d > 0)),
        "bootstrap_prob_nonnegative": float(np.mean(d >= 0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residual = pd.read_csv(args.residual_predictions)
    consensus = pd.read_csv(args.consensus_predictions)[["game_id", "consensus_upset_call", "selected_pairing"]]
    x = residual.merge(consensus, on="game_id", how="left")
    x["consensus_upset_call"] = x["consensus_upset_call"].fillna(False).astype(bool)
    x["selected_pairing"] = x["selected_pairing"].fillna("")

    y = x["home_win"].astype(int).to_numpy()
    market_home = x["p_home_market"].to_numpy(float) >= 0.5
    residual_home = x["p_home_market_residual"].to_numpy(float) >= 0.5
    dog_home = ~market_home
    consensus_call = x["consensus_upset_call"].to_numpy(bool)
    final_home = np.where(consensus_call, dog_home, residual_home)

    market_correct = market_home == y
    residual_correct = residual_home == y
    final_correct = final_home == y
    close_override = residual_home != market_home

    x["market_pick_home"] = market_home
    x["residual_pick_home"] = residual_home
    x["final_pick_home"] = final_home
    x["market_correct"] = market_correct
    x["residual_correct"] = residual_correct
    x["final_correct"] = final_correct
    x["close_residual_override"] = close_override
    x["decision_type"] = np.select(
        [consensus_call, close_override],
        ["true_upset_consensus", "close_game_residual"],
        default="follow_market",
    )

    strategies = pd.DataFrame(
        [
            {"strategy": "market", "games": len(x), "correct": int(market_correct.sum()), "accuracy": float(market_correct.mean()), "net_vs_market": 0},
            {"strategy": "close_game_residual", "games": len(x), "correct": int(residual_correct.sum()), "accuracy": float(residual_correct.mean()), "net_vs_market": int(residual_correct.sum()-market_correct.sum())},
            {"strategy": "integrated_close_plus_consensus", "games": len(x), "correct": int(final_correct.sum()), "accuracy": float(final_correct.mean()), "net_vs_market": int(final_correct.sum()-market_correct.sum())},
        ]
    )

    decision_rows = []
    for decision in ("close_game_residual", "true_upset_consensus"):
        mask = x["decision_type"].eq(decision).to_numpy()
        n = int(mask.sum())
        decision_rows.append(
            {
                "decision_type": decision,
                "calls": n,
                "final_correct": int(np.sum(final_correct & mask)),
                "market_correct": int(np.sum(market_correct & mask)),
                "call_accuracy": float(np.mean(final_correct[mask])) if n else np.nan,
                "net_correct_vs_market": int(np.sum(final_correct[mask]) - np.sum(market_correct[mask])),
            }
        )
    decisions = pd.DataFrame(decision_rows)

    season_rows = []
    for season, g in x.groupby("season"):
        season_rows.append(
            {
                "season": int(season),
                "games": int(len(g)),
                "market_correct": int(g["market_correct"].sum()),
                "residual_correct": int(g["residual_correct"].sum()),
                "integrated_correct": int(g["final_correct"].sum()),
                "integrated_accuracy": float(g["final_correct"].mean()),
                "net_vs_market": int(g["final_correct"].sum()-g["market_correct"].sum()),
                "close_overrides": int(g["close_residual_override"].sum()),
                "consensus_upsets": int(g["consensus_upset_call"].sum()),
            }
        )
    seasons = pd.DataFrame(season_rows)

    boot = pd.DataFrame([
        {"comparison": "integrated_vs_market", **bootstrap_paired(final_correct.astype(int), market_correct.astype(int), args.bootstrap_reps)},
        {"comparison": "residual_vs_market", **bootstrap_paired(residual_correct.astype(int), market_correct.astype(int), args.bootstrap_reps, 43)},
        {"comparison": "integrated_vs_residual", **bootstrap_paired(final_correct.astype(int), residual_correct.astype(int), args.bootstrap_reps, 44)},
    ])

    overlap = int(np.sum(close_override & consensus_call))
    diagnostics = pd.DataFrame([{
        "games": len(x),
        "close_override_calls": int(close_override.sum()),
        "consensus_upset_calls": int(consensus_call.sum()),
        "overlap_calls": overlap,
    }])

    x.to_csv(args.output_dir / "integrated_predictions.csv", index=False)
    strategies.to_csv(args.output_dir / "integrated_summary.csv", index=False)
    decisions.to_csv(args.output_dir / "integrated_decision_types.csv", index=False)
    seasons.to_csv(args.output_dir / "integrated_by_season.csv", index=False)
    boot.to_csv(args.output_dir / "integrated_bootstrap.csv", index=False)
    diagnostics.to_csv(args.output_dir / "integrated_diagnostics.csv", index=False)

    print("\nIntegrated strategy summary")
    print(strategies.to_string(index=False))
    print("\nDecision-type performance")
    print(decisions.to_string(index=False))
    print("\nSeason performance")
    print(seasons.to_string(index=False))
    print("\nPaired bootstrap")
    print(boot.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
