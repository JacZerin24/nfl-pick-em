"""Nested time-ordered stacker for Phase 1 NFL pick'em predictions.

The base-model predictions consumed by this script must already be out-of-sample
walk-forward predictions from ``phase1_backtest.py``. For each outer test season,
this script:

1. Uses only earlier out-of-sample seasons as meta-training data.
2. Tunes the stacker's regularization with an inner expanding-season replay.
3. Fits a logistic meta-model on logit-transformed base probabilities.
4. Predicts the untouched outer season.

This prevents the common stacking mistake of training a meta-model on in-sample
base predictions or tuning stacker hyperparameters on the season being graded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

BASE_MODELS = ("market", "elo", "logistic", "catboost")
C_GRID = (0.03, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
EPS = 1e-5


@dataclass(frozen=True)
class InnerScore:
    c_value: float
    folds: int
    games: int
    log_loss: float
    brier: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/phase1/phase1_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1/stacker"),
    )
    parser.add_argument(
        "--first-test-season",
        type=int,
        default=2019,
        help="First season graded by the meta-model. Needs multiple prior OOF seasons.",
    )
    return parser.parse_args()


def validate_predictions(df: pd.DataFrame) -> None:
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "home_win",
        *(f"p_home_{m}" for m in BASE_MODELS),
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df["game_id"].duplicated().any():
        raise SystemExit("Predictions contain duplicate game_id rows")


def logit(values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def make_features(df: pd.DataFrame) -> np.ndarray:
    """Build low-dimensional meta-features from OOF base probabilities.

    We intentionally keep the stacker simple. It receives each base model's
    probability on the log-odds scale plus two consensus diagnostics. More
    complex meta-features can be tested later, but complexity has to earn its
    place out of sample.
    """

    probs = np.column_stack([df[f"p_home_{m}"].to_numpy(dtype=float) for m in BASE_MODELS])
    logits = np.column_stack([logit(probs[:, i]) for i in range(probs.shape[1])])
    mean_prob = probs.mean(axis=1, keepdims=True)
    prob_std = probs.std(axis=1, keepdims=True)
    return np.hstack([logits, logit(mean_prob), prob_std])


def fit_meta(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=c_value,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    model.fit(x_scaled, y)
    return scaler, model


def predict_meta(
    scaler: StandardScaler,
    model: LogisticRegression,
    x: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(scaler.transform(x))[:, 1]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prob = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    pick = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pick)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
    }


def tune_c_inner(train_oof: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Choose C by an inner expanding-season replay on prior OOF predictions."""

    seasons = sorted(int(s) for s in train_oof["season"].unique())
    if len(seasons) < 2:
        # Should not happen with the default first outer season, but keep a
        # conservative fixed value rather than tuning on the outer test year.
        return 0.25, pd.DataFrame()

    rows: list[dict[str, float | int]] = []
    for c_value in C_GRID:
        y_all: list[np.ndarray] = []
        p_all: list[np.ndarray] = []
        folds = 0

        for validation_season in seasons[1:]:
            inner_train = train_oof.loc[train_oof["season"] < validation_season]
            inner_valid = train_oof.loc[train_oof["season"] == validation_season]
            if inner_train.empty or inner_valid.empty:
                continue
            if inner_train["home_win"].nunique() < 2:
                continue

            scaler, model = fit_meta(
                make_features(inner_train),
                inner_train["home_win"].astype(int).to_numpy(),
                c_value,
            )
            p = predict_meta(scaler, model, make_features(inner_valid))
            y_all.append(inner_valid["home_win"].astype(int).to_numpy())
            p_all.append(p)
            folds += 1

        if not p_all:
            continue

        y = np.concatenate(y_all)
        p = np.concatenate(p_all)
        score = metrics(y, p)
        rows.append(
            {
                "c_value": c_value,
                "folds": folds,
                "games": len(y),
                **score,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return 0.25, table

    # Probability quality is the primary stacker tuning criterion. Accuracy is
    # used only as a tie-breaker because the final league target is winners.
    table = table.sort_values(
        ["log_loss", "brier", "accuracy", "c_value"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return float(table.iloc[0]["c_value"]), table


def run_nested_stacker(
    predictions: pd.DataFrame,
    first_test_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completed = predictions.loc[predictions["home_win"].notna()].copy()
    completed["season"] = completed["season"].astype(int)
    all_seasons = sorted(int(s) for s in completed["season"].unique())

    if first_test_season not in all_seasons:
        raise SystemExit(
            f"--first-test-season {first_test_season} is not present. Available: {all_seasons}"
        )

    outer_predictions: list[pd.DataFrame] = []
    outer_metrics: list[dict[str, float | int]] = []
    tuning_rows: list[pd.DataFrame] = []

    for season in [s for s in all_seasons if s >= first_test_season]:
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue

        selected_c, tuning = tune_c_inner(train)
        if not tuning.empty:
            tuning = tuning.copy()
            tuning.insert(0, "outer_test_season", season)
            tuning["selected"] = np.isclose(tuning["c_value"], selected_c)
            tuning_rows.append(tuning)

        scaler, meta = fit_meta(
            make_features(train),
            train["home_win"].astype(int).to_numpy(),
            selected_c,
        )
        p_stack = predict_meta(scaler, meta, make_features(test))
        y_test = test["home_win"].astype(int).to_numpy()
        p_market = test["p_home_market"].to_numpy(dtype=float)

        stack_metrics = metrics(y_test, p_stack)
        market_metrics = metrics(y_test, p_market)

        outer_metrics.append(
            {
                "season": season,
                "model": "nested_stacker",
                "games": len(test),
                "selected_c": selected_c,
                **stack_metrics,
                "market_accuracy": market_metrics["accuracy"],
                "market_log_loss": market_metrics["log_loss"],
                "market_brier": market_metrics["brier"],
                "accuracy_lift": stack_metrics["accuracy"] - market_metrics["accuracy"],
                "log_loss_improvement": market_metrics["log_loss"] - stack_metrics["log_loss"],
                "brier_improvement": market_metrics["brier"] - stack_metrics["brier"],
            }
        )

        out = test[
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "away_team",
                "home_team",
                "home_win",
                *(f"p_home_{m}" for m in BASE_MODELS),
            ]
        ].copy()
        out["p_home_nested_stacker"] = p_stack
        out["selected_c"] = selected_c
        outer_predictions.append(out)

        coef_names = [
            *(f"logit_{m}" for m in BASE_MODELS),
            "logit_mean_probability",
            "model_probability_std",
        ]
        coef_text = ", ".join(
            f"{name}={value:+.3f}" for name, value in zip(coef_names, meta.coef_[0], strict=True)
        )
        print(
            f"{season}: stack={stack_metrics['accuracy']:.3f} "
            f"market={market_metrics['accuracy']:.3f} C={selected_c:g} | {coef_text}"
        )

    if not outer_predictions:
        raise SystemExit("No outer stacker folds were produced")

    pred_out = pd.concat(outer_predictions, ignore_index=True)
    metrics_out = pd.DataFrame(outer_metrics)
    tuning_out = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()
    return pred_out, metrics_out, tuning_out


def aggregate_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["home_win"].astype(int).to_numpy()
    rows = []
    for model in ("market", "nested_stacker"):
        col = "p_home_market" if model == "market" else "p_home_nested_stacker"
        score = metrics(y, predictions[col].to_numpy(dtype=float))
        rows.append({"model": model, "games": len(predictions), **score})
    summary = pd.DataFrame(rows)
    market = summary.loc[summary["model"].eq("market")].iloc[0]
    summary["accuracy_lift_vs_market"] = summary["accuracy"] - market["accuracy"]
    summary["log_loss_improvement_vs_market"] = market["log_loss"] - summary["log_loss"]
    summary["brier_improvement_vs_market"] = market["brier"] - summary["brier"]
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    validate_predictions(predictions)

    pred_out, metrics_out, tuning_out = run_nested_stacker(
        predictions,
        first_test_season=args.first_test_season,
    )
    summary = aggregate_summary(pred_out)

    pred_out.to_csv(args.output_dir / "phase1_stacker_predictions.csv", index=False)
    metrics_out.to_csv(args.output_dir / "phase1_stacker_fold_metrics.csv", index=False)
    tuning_out.to_csv(args.output_dir / "phase1_stacker_inner_tuning.csv", index=False)
    summary.to_csv(args.output_dir / "phase1_stacker_summary.csv", index=False)

    print("\nNested stacker summary")
    print(summary.to_string(index=False))
    print(f"\nStacker results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
