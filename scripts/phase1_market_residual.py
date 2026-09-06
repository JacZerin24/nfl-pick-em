"""Nested market-anchored residual correction for NFL pick'em.

Rather than asking a machine-learning model to relearn the entire win
probability, this model treats the de-vigged market probability as the prior and
learns only small, regularized log-odds corrections from disagreements between
football models and the market.

For a game with market probability p_m:

    logit(p_final) = logit(p_m) + intercept
                     + b_elo * (logit(p_elo) - logit(p_m))
                     + b_log * (logit(p_logistic) - logit(p_m))
                     + b_cat * (logit(p_catboost) - logit(p_m))
                     + b_consensus * mean(model-market logit differences)

The coefficient on market log-odds is fixed at 1.0. Correction coefficients are
L2-regularized and the penalty is chosen only from earlier out-of-sample
seasons using an inner expanding replay. This makes the model structurally
conservative: when the extra models add no repeatable information, shrinkage
pushes the forecast back toward the market.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6
LAMBDA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
BASE_MODELS = ("elo", "logistic", "catboost")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/phase1/phase1_predictions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1/market_residual"),
    )
    p.add_argument("--first-test-season", type=int, default=2019)
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    market_logit = logit(df["p_home_market"].to_numpy(dtype=float))
    deltas = []
    for model in BASE_MODELS:
        deltas.append(logit(df[f"p_home_{model}"].to_numpy(dtype=float)) - market_logit)
    delta_matrix = np.column_stack(deltas)
    consensus = delta_matrix.mean(axis=1, keepdims=True)
    x = np.column_stack([np.ones(len(df)), delta_matrix, consensus])
    return market_logit, x


def objective(theta: np.ndarray, market_logit: np.ndarray, x: np.ndarray, y: np.ndarray, penalty: float) -> tuple[float, np.ndarray]:
    eta = market_logit + x @ theta
    p = sigmoid(eta)
    # Sum NLL keeps the penalty scale stable as the training sample grows.
    nll = -np.sum(y * np.log(np.clip(p, EPS, 1.0)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1.0)))
    # Do not penalize the intercept as strongly; shrink football corrections.
    weights = np.ones_like(theta)
    weights[0] = 0.10
    reg = 0.5 * penalty * np.sum(weights * theta * theta)
    grad = x.T @ (p - y) + penalty * weights * theta
    return float(nll + reg), grad


def fit_residual(df: pd.DataFrame, penalty: float) -> np.ndarray:
    market_logit, x = design(df)
    y = df["home_win"].astype(int).to_numpy()
    result = minimize(
        fun=lambda t: objective(t, market_logit, x, y, penalty),
        x0=np.zeros(x.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Residual fit failed: {result.message}")
    return result.x


def predict_residual(df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
    market_logit, x = design(df)
    return sigmoid(market_logit + x @ theta)


def metrics(df: pd.DataFrame, p: np.ndarray) -> dict[str, float | int]:
    y = df["home_win"].astype(int).to_numpy()
    prob = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pick = (prob >= 0.5).astype(int)
    return {
        "games": int(len(df)),
        "correct": int(np.sum(pick == y)),
        "accuracy": float(np.mean(pick == y)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
    }


def tune_penalty(train_oof: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    seasons = sorted(int(s) for s in train_oof["season"].unique())
    rows = []
    if len(seasons) < 2:
        return 100.0, pd.DataFrame()

    for penalty in LAMBDA_GRID:
        probs, ys = [], []
        folds = 0
        for valid_season in seasons[1:]:
            inner_train = train_oof.loc[train_oof["season"] < valid_season]
            inner_valid = train_oof.loc[train_oof["season"] == valid_season]
            if inner_train.empty or inner_valid.empty:
                continue
            theta = fit_residual(inner_train, penalty)
            probs.append(predict_residual(inner_valid, theta))
            ys.append(inner_valid["home_win"].astype(int).to_numpy())
            folds += 1
        if not probs:
            continue
        p = np.concatenate(probs)
        y = np.concatenate(ys)
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "penalty": penalty,
                "folds": folds,
                "games": len(y),
                "accuracy": float(np.mean(pred == y)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "accuracy", "penalty"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return float(table.iloc[0]["penalty"]), table


def run_nested(df: pd.DataFrame, first_test_season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completed = df.loc[df["home_win"].notna()].copy()
    required = {"season", "home_win", "p_home_market", *(f"p_home_{m}" for m in BASE_MODELS)}
    missing = required.difference(completed.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    seasons = sorted(int(s) for s in completed["season"].unique() if int(s) >= first_test_season)
    all_preds = []
    fold_rows = []
    tuning_rows = []

    for season in seasons:
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue

        penalty, tuning = tune_penalty(train)
        if not tuning.empty:
            tuning = tuning.copy()
            tuning.insert(0, "outer_test_season", season)
            tuning["selected"] = np.isclose(tuning["penalty"], penalty)
            tuning_rows.append(tuning)

        theta = fit_residual(train, penalty)
        p_resid = predict_residual(test, theta)
        market_p = test["p_home_market"].to_numpy(dtype=float)
        m_resid = metrics(test, p_resid)
        m_market = metrics(test, market_p)

        row = {
            "season": season,
            "selected_penalty": penalty,
            **{f"residual_{k}": v for k, v in m_resid.items()},
            **{f"market_{k}": v for k, v in m_market.items()},
            "accuracy_lift": m_resid["accuracy"] - m_market["accuracy"],
            "log_loss_improvement": m_market["log_loss"] - m_resid["log_loss"],
            "brier_improvement": m_market["brier"] - m_resid["brier"],
            "theta_intercept": theta[0],
            "theta_elo_delta": theta[1],
            "theta_logistic_delta": theta[2],
            "theta_catboost_delta": theta[3],
            "theta_consensus_delta": theta[4],
        }
        fold_rows.append(row)

        out_cols = [
            c for c in ["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win"]
            if c in test.columns
        ]
        out = test[out_cols].copy()
        out["p_home_market"] = market_p
        out["p_home_market_residual"] = p_resid
        out["selected_penalty"] = penalty
        all_preds.append(out)

        print(
            f"{season}: residual={m_resid['accuracy']:.3f} market={m_market['accuracy']:.3f} "
            f"penalty={penalty:g} logloss_delta={m_market['log_loss'] - m_resid['log_loss']:+.4f}"
        )

    return (
        pd.concat(all_preds, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame(),
    )


def bootstrap_lift(predictions: pd.DataFrame, reps: int = 20000, seed: int = 2026) -> pd.DataFrame:
    y = predictions["home_win"].astype(int).to_numpy()
    market = (predictions["p_home_market"].to_numpy(dtype=float) >= 0.5).astype(int)
    residual = (predictions["p_home_market_residual"].to_numpy(dtype=float) >= 0.5).astype(int)
    diff = (residual == y).astype(float) - (market == y).astype(float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    values = np.empty(reps, dtype=float)
    chunk = 500
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        idx = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = diff[idx].mean(axis=1)
    disagree = residual != market
    return pd.DataFrame(
        [
            {
                "games": n,
                "accuracy_lift": float(diff.mean()),
                "net_additional_correct": int(diff.sum()),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_50": float(np.quantile(values, 0.50)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "prob_lift_gt_0": float(np.mean(values > 0)),
                "disagreements": int(disagree.sum()),
                "residual_accuracy_on_disagreements": float(np.mean(residual[disagree] == y[disagree])) if disagree.any() else np.nan,
                "market_accuracy_on_disagreements": float(np.mean(market[disagree] == y[disagree])) if disagree.any() else np.nan,
            }
        ]
    )


def aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in (
        ("market", "p_home_market"),
        ("market_residual", "p_home_market_residual"),
    ):
        rows.append({"model": name, **metrics(predictions, predictions[col].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    preds, folds, tuning = run_nested(df, args.first_test_season)
    summary = aggregate(preds)
    paired = bootstrap_lift(preds)

    preds.to_csv(args.output_dir / "market_residual_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "market_residual_fold_metrics.csv", index=False)
    tuning.to_csv(args.output_dir / "market_residual_inner_tuning.csv", index=False)
    summary.to_csv(args.output_dir / "market_residual_summary.csv", index=False)
    paired.to_csv(args.output_dir / "market_residual_paired_vs_market.csv", index=False)

    print("\nMarket residual summary")
    print(summary.to_string(index=False))
    print("\nPaired vs market")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
