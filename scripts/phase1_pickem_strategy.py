"""Pick'em decision lab: close games, calibration, and true upset overrides.

This script evaluates the decision problem that matters in straight-up pick'em:
* deciding genuine coin-flip / near-coin-flip games
* identifying when an underdog is worth overriding the market favorite
* abstaining when historical evidence does not support an upset pick

Everything is walk-forward. Base-model probabilities are already OOF from
phase1_backtest.py. Market calibration and upset-policy selection use only prior
seasons before each outer test season.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6
BASE_MODELS = ("logistic", "catboost", "fixed_blend")
EDGE_THRESHOLDS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
UPSET_MAX_FAV_PROBS = (0.55, 0.575, 0.60, 0.65, 0.70)
TRUE_UPSET_MIN_FAV_PROB = 0.525
MIN_POLICY_OVERRIDES = 10
MIN_POSTERIOR_EDGE_PROB = 0.75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("outputs/phase1/phase1_predictions.csv"))
    p.add_argument(
        "--residual-predictions",
        type=Path,
        default=Path("outputs/phase1/market_residual/market_residual_predictions.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/pickem_strategy"))
    p.add_argument("--first-test-season", type=int, default=2019)
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    prob = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pick = prob >= 0.5
    yy = np.asarray(y, dtype=int)
    return {
        "games": int(len(yy)),
        "correct": int(np.sum(pick == yy)),
        "accuracy": float(np.mean(pick == yy)),
        "log_loss": float(log_loss(yy, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(yy, prob)),
    }


def walk_forward_market_calibration(df: pd.DataFrame, first_test_season: int) -> pd.DataFrame:
    """Simple fixed-spec Platt calibration of market log-odds.

    No hyperparameter search is performed. The purpose is diagnostic: determine
    how much of the close-game residual gain can be reproduced from market
    probabilities alone.
    """
    rows = []
    for season in sorted(int(s) for s in df["season"].unique() if int(s) >= first_test_season):
        train = df.loc[df["season"] < season].copy()
        test = df.loc[df["season"] == season].copy()
        if train.empty or test.empty:
            continue
        x_train = logit(train["p_home_market"].to_numpy(float)).reshape(-1, 1)
        x_test = logit(test["p_home_market"].to_numpy(float)).reshape(-1, 1)
        model = LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000)
        model.fit(x_train, train["home_win"].astype(int))
        p = model.predict_proba(x_test)[:, 1]
        out = test[["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win"]].copy()
        out["p_home_market"] = test["p_home_market"].to_numpy(float)
        out["p_home_market_calibrated"] = p
        out["calibration_intercept"] = float(model.intercept_[0])
        out["calibration_slope"] = float(model.coef_[0, 0])
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def build_decision_frame(base: pd.DataFrame, calibrated: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    keep = ["game_id", "p_home_market_calibrated", "calibration_intercept", "calibration_slope"]
    rkeep = ["game_id", "p_home_market_residual"]
    out = base.merge(calibrated[keep], on="game_id", how="inner").merge(residual[rkeep], on="game_id", how="inner")
    out["market_favorite_home"] = out["p_home_market"] >= 0.5
    out["market_fav_prob"] = np.maximum(out["p_home_market"], 1 - out["p_home_market"])
    out["market_dog_prob"] = 1 - out["market_fav_prob"]
    out["market_pick"] = np.where(out["market_favorite_home"], out["home_team"], out["away_team"])
    out["underdog"] = np.where(out["market_favorite_home"], out["away_team"], out["home_team"])
    out["winner"] = np.where(out["home_win"].astype(int).eq(1), out["home_team"], out["away_team"])
    out["actual_upset"] = out["winner"].eq(out["underdog"])
    return out


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80, 1.000001]
    labels = ["50-52.5", "52.5-55", "55-57.5", "57.5-60", "60-65", "65-70", "70-80", "80+"]
    df = df.copy()
    df["market_strength_bucket"] = pd.cut(df["market_fav_prob"], bins=bins, labels=labels, right=False, include_lowest=True)
    strategies = {
        "market": "p_home_market",
        "market_calibrated": "p_home_market_calibrated",
        "market_residual": "p_home_market_residual",
        "logistic": "p_home_logistic",
        "catboost": "p_home_catboost",
        "fixed_blend": "p_home_fixed_blend",
    }
    rows = []
    y = df["home_win"].astype(int).to_numpy()
    for bucket, group in df.groupby("market_strength_bucket", observed=True):
        idx = group.index
        for name, col in strategies.items():
            p = group[col].to_numpy(float)
            pick = p >= 0.5
            yy = group["home_win"].astype(int).to_numpy()
            rows.append(
                {
                    "market_strength_bucket": str(bucket),
                    "strategy": name,
                    "games": int(len(group)),
                    "accuracy": float(np.mean(pick == yy)),
                    "correct": int(np.sum(pick == yy)),
                    "actual_upset_rate": float(group["actual_upset"].mean()),
                }
            )
    return pd.DataFrame(rows)


def overall_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df["home_win"].astype(int).to_numpy()
    for name, col in (
        ("market", "p_home_market"),
        ("market_calibrated", "p_home_market_calibrated"),
        ("market_residual", "p_home_market_residual"),
    ):
        rows.append({"strategy": name, **score(y, df[col].to_numpy(float))})
    return pd.DataFrame(rows).sort_values(["accuracy", "brier"], ascending=[False, True])


def close_game_comparison(df: pd.DataFrame) -> pd.DataFrame:
    close = df.loc[df["market_fav_prob"] < TRUE_UPSET_MIN_FAV_PROB].copy()
    y = close["home_win"].astype(int).to_numpy()
    rows = []
    for name, col in (
        ("market", "p_home_market"),
        ("market_calibrated", "p_home_market_calibrated"),
        ("market_residual", "p_home_market_residual"),
    ):
        p = close[col].to_numpy(float)
        pick = p >= 0.5
        rows.append({"strategy": name, "games": len(close), "correct": int(np.sum(pick == y)), "accuracy": float(np.mean(pick == y))})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def policy_candidate_stats(df: pd.DataFrame, model: str, threshold: float, max_fav_prob: float) -> dict[str, float | int]:
    y = df["home_win"].astype(int).to_numpy()
    market_p = df["p_home_market"].to_numpy(float)
    model_p = df[f"p_home_{model}"].to_numpy(float)
    market_pick = market_p >= 0.5
    model_pick = model_p >= 0.5
    fav_prob = np.maximum(market_p, 1 - market_p)
    edge = np.abs(model_p - 0.5) - np.abs(market_p - 0.5)
    override = (
        (model_pick != market_pick)
        & (edge >= threshold)
        & (fav_prob >= TRUE_UPSET_MIN_FAV_PROB)
        & (fav_prob < max_fav_prob)
    )
    n = int(override.sum())
    wins = int(np.sum((model_pick == y) & override))
    net = int(2 * wins - n)
    posterior_prob = float(1 - beta.cdf(0.5, 5 + wins, 5 + n - wins)) if n else 0.0
    return {"overrides": n, "override_wins": wins, "net_vs_market": net, "posterior_prob_override_beats_market": posterior_prob}


def select_upset_policy(train: pd.DataFrame) -> dict[str, float | int | str] | None:
    candidates = []
    for model in BASE_MODELS:
        for threshold in EDGE_THRESHOLDS:
            for max_fav_prob in UPSET_MAX_FAV_PROBS:
                stats = policy_candidate_stats(train, model, threshold, max_fav_prob)
                if stats["overrides"] < MIN_POLICY_OVERRIDES or stats["net_vs_market"] <= 0:
                    continue
                candidates.append(
                    {
                        "model": model,
                        "threshold": threshold,
                        "max_fav_prob": max_fav_prob,
                        **stats,
                    }
                )
    if not candidates:
        return None
    table = pd.DataFrame(candidates).sort_values(
        ["posterior_prob_override_beats_market", "net_vs_market", "overrides", "threshold"],
        ascending=[False, False, False, False],
    )
    best = table.iloc[0].to_dict()
    if float(best["posterior_prob_override_beats_market"]) < MIN_POSTERIOR_EDGE_PROB:
        return None
    return best


def apply_upset_policy(test: pd.DataFrame, policy: dict[str, float | int | str] | None) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    out = test.copy()
    market_p = out["p_home_market"].to_numpy(float)
    market_pick = market_p >= 0.5
    y = out["home_win"].astype(int).to_numpy()
    override = np.zeros(len(out), dtype=bool)
    model_pick = market_pick.copy()
    model_name = "ABSTAIN"
    threshold = np.nan
    max_fav_prob = np.nan

    if policy is not None:
        model_name = str(policy["model"])
        threshold = float(policy["threshold"])
        max_fav_prob = float(policy["max_fav_prob"])
        model_p = out[f"p_home_{model_name}"].to_numpy(float)
        model_pick = model_p >= 0.5
        fav_prob = np.maximum(market_p, 1 - market_p)
        edge = np.abs(model_p - 0.5) - np.abs(market_p - 0.5)
        override = (
            (model_pick != market_pick)
            & (edge >= threshold)
            & (fav_prob >= TRUE_UPSET_MIN_FAV_PROB)
            & (fav_prob < max_fav_prob)
        )

    final_pick = np.where(override, model_pick, market_pick)
    out["true_upset_override"] = override
    out["true_upset_policy_pick_home"] = final_pick
    out["true_upset_policy_model"] = model_name
    out["true_upset_policy_threshold"] = threshold
    out["true_upset_policy_max_fav_prob"] = max_fav_prob

    wins = int(np.sum((model_pick == y) & override))
    n = int(override.sum())
    metrics = {
        "model": model_name,
        "threshold": threshold,
        "max_fav_prob": max_fav_prob,
        "games": int(len(out)),
        "correct": int(np.sum(final_pick == y)),
        "accuracy": float(np.mean(final_pick == y)),
        "overrides": n,
        "override_wins": wins,
        "override_accuracy": float(wins / n) if n else np.nan,
        "net_correct_vs_market": int(2 * wins - n),
    }
    return out, metrics


def walk_forward_upset_policy(df: pd.DataFrame, first_test_season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows, fold_rows, selection_rows = [], [], []
    seasons = sorted(int(s) for s in df["season"].unique() if int(s) >= first_test_season)
    for season in seasons:
        train = df.loc[df["season"] < season].copy()
        test = df.loc[df["season"] == season].copy()
        if train.empty or test.empty:
            continue
        policy = select_upset_policy(train)
        out, metrics = apply_upset_policy(test, policy)
        metrics["season"] = season
        fold_rows.append(metrics)
        pred_rows.append(out[["game_id", "season", "true_upset_override", "true_upset_policy_pick_home", "true_upset_policy_model", "true_upset_policy_threshold", "true_upset_policy_max_fav_prob"]])
        selection_rows.append({"season": season, "selected_policy": "ABSTAIN" if policy is None else "OVERRIDE", **({} if policy is None else policy)})
        print(f"{season}: upset_policy={metrics['accuracy']:.3f} overrides={metrics['overrides']} net={metrics['net_correct_vs_market']:+d}")
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def upset_policy_summary(df: pd.DataFrame, policy_preds: pd.DataFrame) -> pd.DataFrame:
    x = df.merge(policy_preds[["game_id", "true_upset_policy_pick_home", "true_upset_override"]], on="game_id", how="inner")
    y = x["home_win"].astype(int).to_numpy()
    market_pick = x["p_home_market"].to_numpy(float) >= 0.5
    policy_pick = x["true_upset_policy_pick_home"].astype(bool).to_numpy()
    override = x["true_upset_override"].astype(bool).to_numpy()
    rows = [
        {"strategy": "market", "games": len(x), "correct": int(np.sum(market_pick == y)), "accuracy": float(np.mean(market_pick == y)), "overrides": 0, "net_correct_vs_market": 0},
        {"strategy": "nested_true_upset_policy", "games": len(x), "correct": int(np.sum(policy_pick == y)), "accuracy": float(np.mean(policy_pick == y)), "overrides": int(override.sum()), "net_correct_vs_market": int(np.sum((policy_pick == y).astype(int) - (market_pick == y).astype(int)))},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.predictions)
    residual = pd.read_csv(args.residual_predictions)
    base = base.loc[base["home_win"].notna()].copy()

    calibrated = walk_forward_market_calibration(base, args.first_test_season)
    decisions = build_decision_frame(base, calibrated, residual)
    decisions = decisions.loc[decisions["season"] >= args.first_test_season].copy()

    overall = overall_strategy_summary(decisions)
    regions = region_summary(decisions)
    close = close_game_comparison(decisions)
    upset_preds, upset_folds, upset_selections = walk_forward_upset_policy(decisions, args.first_test_season)
    upset_summary = upset_policy_summary(decisions, upset_preds)

    calibrated.to_csv(args.output_dir / "market_calibration_predictions.csv", index=False)
    decisions.to_csv(args.output_dir / "pickem_decision_frame.csv", index=False)
    overall.to_csv(args.output_dir / "overall_strategy_summary.csv", index=False)
    regions.to_csv(args.output_dir / "market_strength_buckets.csv", index=False)
    close.to_csv(args.output_dir / "close_game_summary.csv", index=False)
    upset_preds.to_csv(args.output_dir / "true_upset_policy_predictions.csv", index=False)
    upset_folds.to_csv(args.output_dir / "true_upset_policy_folds.csv", index=False)
    upset_selections.to_csv(args.output_dir / "true_upset_policy_selections.csv", index=False)
    upset_summary.to_csv(args.output_dir / "true_upset_policy_summary.csv", index=False)

    print("\nOverall strategy summary")
    print(overall.to_string(index=False))
    print("\nClose games (<52.5% market favorite)")
    print(close.to_string(index=False))
    print("\nNested true-upset policy")
    print(upset_summary.to_string(index=False))


if __name__ == "__main__":
    main()
