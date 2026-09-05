"""Nested selector for high-conviction true-upset calls.

Consumes OOF predictions from phase2_upset_specialist.py. For each outer test
season it selects a simple upset-call rule using ONLY earlier seasons, then
applies that frozen rule to the next season. If prior evidence is not strong
enough, the selector abstains and follows the market favorite.

Candidate rule dimensions:
* specialist model (logistic or CatBoost)
* minimum predicted underdog win probability
* minimum specialist edge over market underdog probability
* market-favorite strength band

Selection uses a skeptical Beta prior and minimum call count to reduce the risk
of promoting a tiny lucky pocket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

MODELS = {
    "upset_logistic": "p_dog_upset_logistic",
    "upset_catboost": "p_dog_upset_catboost",
}
MIN_P_GRID = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
MIN_EDGE_GRID = (0.00, 0.025, 0.05, 0.075, 0.10)
MIN_FAV_GRID = (0.525, 0.55, 0.575, 0.60)
MAX_FAV_GRID = (0.575, 0.60, 0.65, 0.70, 0.80)
MIN_TRAIN_CALLS = 15
PRIOR_ALPHA = 10.0
PRIOR_BETA = 10.0
MIN_POSTERIOR_GT_HALF = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/phase2_upset_specialist/upset_specialist_predictions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2_upset_specialist/selector"),
    )
    p.add_argument("--first-test-season", type=int, default=2019)
    return p.parse_args()


def rule_mask(
    df: pd.DataFrame,
    prob_col: str,
    min_p: float,
    min_edge: float,
    min_fav_prob: float,
    max_fav_prob: float,
) -> np.ndarray:
    p = df[prob_col].to_numpy(float)
    market_dog = df["market_dog_prob"].to_numpy(float)
    fav = df["market_fav_prob"].to_numpy(float)
    return (
        (p >= min_p)
        & ((p - market_dog) >= min_edge)
        & (fav >= min_fav_prob)
        & (fav < max_fav_prob)
    )


def candidate_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, prob_col in MODELS.items():
        for min_p in MIN_P_GRID:
            for min_edge in MIN_EDGE_GRID:
                for min_fav in MIN_FAV_GRID:
                    for max_fav in MAX_FAV_GRID:
                        if max_fav <= min_fav:
                            continue
                        mask = rule_mask(train, prob_col, min_p, min_edge, min_fav, max_fav)
                        n = int(mask.sum())
                        if n < MIN_TRAIN_CALLS:
                            continue
                        wins = int(train.loc[mask, "dog_win"].sum())
                        losses = n - wins
                        post_a = PRIOR_ALPHA + wins
                        post_b = PRIOR_BETA + losses
                        post_mean = float(post_a / (post_a + post_b))
                        post_gt_half = float(1.0 - beta.cdf(0.5, post_a, post_b))
                        lower_10 = float(beta.ppf(0.10, post_a, post_b))
                        rows.append(
                            {
                                "model": model,
                                "prob_col": prob_col,
                                "min_p": min_p,
                                "min_edge": min_edge,
                                "min_fav_prob": min_fav,
                                "max_fav_prob": max_fav,
                                "train_calls": n,
                                "train_wins": wins,
                                "train_call_accuracy": wins / n,
                                "train_net_vs_market": 2 * wins - n,
                                "posterior_mean": post_mean,
                                "posterior_prob_gt_half": post_gt_half,
                                "posterior_10pct": lower_10,
                            }
                        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["posterior_10pct", "posterior_prob_gt_half", "train_net_vs_market", "train_calls"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def select_rule(train: pd.DataFrame) -> tuple[dict[str, object] | None, pd.DataFrame]:
    table = candidate_table(train)
    if table.empty:
        return None, table
    best = table.iloc[0].to_dict()
    if (
        int(best["train_net_vs_market"]) <= 0
        or float(best["posterior_prob_gt_half"]) < MIN_POSTERIOR_GT_HALF
    ):
        return None, table
    return best, table


def apply_rule(test: pd.DataFrame, rule: dict[str, object] | None) -> tuple[pd.DataFrame, dict[str, object]]:
    out = test.copy()
    market_correct = (out["dog_win"].astype(int).to_numpy() == 0)
    calls = np.zeros(len(out), dtype=bool)

    if rule is None:
        model = "ABSTAIN"
        min_p = min_edge = min_fav = max_fav = np.nan
    else:
        model = str(rule["model"])
        prob_col = str(rule["prob_col"])
        min_p = float(rule["min_p"])
        min_edge = float(rule["min_edge"])
        min_fav = float(rule["min_fav_prob"])
        max_fav = float(rule["max_fav_prob"])
        calls = rule_mask(out, prob_col, min_p, min_edge, min_fav, max_fav)

    y = out["dog_win"].astype(int).to_numpy()
    final_correct = np.where(calls, y == 1, y == 0)
    call_wins = int(np.sum(calls & (y == 1)))
    n_calls = int(calls.sum())

    out["selector_upset_call"] = calls
    out["selector_pick_underdog"] = calls
    out["selector_model"] = model
    out["selector_min_p"] = min_p
    out["selector_min_edge"] = min_edge
    out["selector_min_fav_prob"] = min_fav
    out["selector_max_fav_prob"] = max_fav

    metrics = {
        "model": model,
        "games": int(len(out)),
        "correct": int(final_correct.sum()),
        "accuracy": float(final_correct.mean()),
        "market_correct": int(market_correct.sum()),
        "market_accuracy": float(market_correct.mean()),
        "net_correct_vs_market": int(final_correct.sum() - market_correct.sum()),
        "upset_calls": n_calls,
        "upset_call_wins": call_wins,
        "upset_call_accuracy": float(call_wins / n_calls) if n_calls else np.nan,
        "min_p": min_p,
        "min_edge": min_edge,
        "min_fav_prob": min_fav,
        "max_fav_prob": max_fav,
    }
    return out, metrics


def run_nested(df: pd.DataFrame, first_test_season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    fold_rows = []
    selection_rows = []

    seasons = sorted(int(s) for s in df["season"].unique() if int(s) >= first_test_season)
    for season in seasons:
        train = df.loc[df["season"] < season].copy()
        test = df.loc[df["season"] == season].copy()
        if train.empty or test.empty:
            continue
        rule, candidates = select_rule(train)
        out, metrics = apply_rule(test, rule)
        metrics["season"] = season
        fold_rows.append(metrics)

        selection = {"season": season, "selected": rule is not None}
        if rule is not None:
            selection.update(rule)
        selection_rows.append(selection)

        pred_rows.append(
            out[
                [
                    "game_id", "season", "week", "gameday", "favorite_team", "underdog_team",
                    "dog_win", "market_fav_prob", "market_dog_prob", "selector_upset_call",
                    "selector_pick_underdog", "selector_model", "selector_min_p", "selector_min_edge",
                    "selector_min_fav_prob", "selector_max_fav_prob",
                ]
            ]
        )
        print(
            f"{season}: model={metrics['model']} calls={metrics['upset_calls']} "
            f"wins={metrics['upset_call_wins']} net={metrics['net_correct_vs_market']:+d}"
        )

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["dog_win"].astype(int).to_numpy()
    calls = pred["selector_upset_call"].astype(bool).to_numpy()
    market_correct = y == 0
    final_correct = np.where(calls, y == 1, y == 0)
    call_wins = int(np.sum(calls & (y == 1)))
    return pd.DataFrame(
        [
            {
                "strategy": "market",
                "games": len(pred),
                "correct": int(market_correct.sum()),
                "accuracy": float(market_correct.mean()),
                "upset_calls": 0,
                "upset_call_wins": 0,
                "upset_call_accuracy": np.nan,
                "net_correct_vs_market": 0,
            },
            {
                "strategy": "nested_selective_upset",
                "games": len(pred),
                "correct": int(final_correct.sum()),
                "accuracy": float(final_correct.mean()),
                "upset_calls": int(calls.sum()),
                "upset_call_wins": call_wins,
                "upset_call_accuracy": float(call_wins / calls.sum()) if calls.any() else np.nan,
                "net_correct_vs_market": int(final_correct.sum() - market_correct.sum()),
            },
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    pred, folds, selections = run_nested(df, args.first_test_season)
    summary = summarize(pred)

    pred.to_csv(args.output_dir / "selective_upset_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "selective_upset_folds.csv", index=False)
    selections.to_csv(args.output_dir / "selective_upset_selections.csv", index=False)
    summary.to_csv(args.output_dir / "selective_upset_summary.csv", index=False)

    print("\nNested selective upset summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
