"""Cross-specialist consensus upset validation.

Four natural pairings are considered between the matchup specialist and the
explosive/variance specialist. The pairing is selected using ONLY development
seasons 2016-2018, then frozen and graded on the untouched 2019-2025 holdout.

An upset call occurs only when both component models independently assign the
market underdog >50% win probability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIRINGS = {
    "matchup_logistic__variance_logistic": ("p_dog_upset_logistic", "p_dog_variance_logistic"),
    "matchup_logistic__variance_catboost": ("p_dog_upset_logistic", "p_dog_variance_catboost"),
    "matchup_catboost__variance_logistic": ("p_dog_upset_catboost", "p_dog_variance_logistic"),
    "matchup_catboost__variance_catboost": ("p_dog_upset_catboost", "p_dog_variance_catboost"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matchup-predictions", type=Path, required=True)
    p.add_argument("--variance-predictions", type=Path, required=True)
    p.add_argument("--development-end-season", type=int, default=2018)
    p.add_argument("--holdout-first-season", type=int, default=2019)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_upset_consensus"))
    return p.parse_args()


def pair_calls(df: pd.DataFrame, cols: tuple[str, str]) -> np.ndarray:
    return (df[cols[0]].to_numpy(float) >= 0.5) & (df[cols[1]].to_numpy(float) >= 0.5)


def evaluate_pair(df: pd.DataFrame, name: str, cols: tuple[str, str]) -> dict[str, float | int | str]:
    calls = pair_calls(df, cols)
    y = df["dog_win"].astype(int).to_numpy()
    n = int(calls.sum())
    wins = int(np.sum(calls & (y == 1)))
    final_correct = np.where(calls, y == 1, y == 0)
    market_correct = y == 0
    return {
        "pairing": name,
        "games": int(len(df)),
        "upset_calls": n,
        "upset_call_wins": wins,
        "upset_call_accuracy": float(wins / n) if n else np.nan,
        "net_on_calls_vs_market": int(2 * wins - n),
        "correct": int(final_correct.sum()),
        "accuracy": float(final_correct.mean()),
        "market_correct": int(market_correct.sum()),
        "net_correct_vs_market": int(final_correct.sum() - market_correct.sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matchup = pd.read_csv(args.matchup_predictions)
    variance = pd.read_csv(args.variance_predictions)
    keep = ["game_id", "p_dog_variance_logistic", "p_dog_variance_catboost"]
    df = matchup.merge(variance[keep], on="game_id", how="inner")

    development = df.loc[df["season"] <= args.development_end_season].copy()
    holdout = df.loc[df["season"] >= args.holdout_first_season].copy()

    dev_rows = [evaluate_pair(development, name, cols) for name, cols in PAIRINGS.items()]
    dev = pd.DataFrame(dev_rows).sort_values(
        ["net_on_calls_vs_market", "upset_call_accuracy", "upset_calls"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = str(dev.iloc[0]["pairing"])
    selected_cols = PAIRINGS[selected]

    hold_rows = [evaluate_pair(holdout, name, cols) for name, cols in PAIRINGS.items()]
    hold = pd.DataFrame(hold_rows)
    hold["selected_from_development"] = hold["pairing"].eq(selected)

    calls = pair_calls(holdout, selected_cols)
    pred = holdout[[
        "game_id", "season", "week", "gameday", "favorite_team", "underdog_team",
        "dog_win", "market_fav_prob", "market_dog_prob",
    ]].copy()
    pred["consensus_upset_call"] = calls
    pred["selected_pairing"] = selected

    season_rows = []
    for season, g in pred.groupby("season"):
        c = g["consensus_upset_call"].astype(bool).to_numpy()
        y = g["dog_win"].astype(int).to_numpy()
        n = int(c.sum()); wins = int(np.sum(c & (y == 1)))
        season_rows.append({
            "season": int(season), "upset_calls": n, "upset_call_wins": wins,
            "upset_call_accuracy": float(wins/n) if n else np.nan,
            "net_correct_vs_market": int(2*wins-n),
        })
    seasons = pd.DataFrame(season_rows)

    dev.to_csv(args.output_dir / "consensus_development_pairings.csv", index=False)
    hold.to_csv(args.output_dir / "consensus_holdout_pairings.csv", index=False)
    pred.to_csv(args.output_dir / "consensus_holdout_predictions.csv", index=False)
    seasons.to_csv(args.output_dir / "consensus_holdout_by_season.csv", index=False)

    print("\nDevelopment pairing selection (2016-2018)")
    print(dev.to_string(index=False))
    print(f"\nSelected pairing: {selected}")
    print("\nHoldout results (2019-2025)")
    print(hold.to_string(index=False))
    print("\nSelected pairing by season")
    print(seasons.to_string(index=False))


if __name__ == "__main__":
    main()
