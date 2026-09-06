"""Analyze Phase 1 walk-forward NFL pick'em predictions.

This script takes the prediction CSV emitted by ``phase1_backtest.py`` and
produces the decision-focused diagnostics we care about for a straight-up
pick'em league:

* aggregate accuracy / Brier / log loss
* season-by-season accuracy
* paired accuracy differences versus the market favorite
* model-vs-market disagreement performance
* accuracy by confidence bucket
* bootstrap confidence intervals for paired accuracy lift

No model is promoted solely because it has the highest point estimate. The
paired confidence interval and the distribution of results across seasons are
reported so we can see whether apparent improvement is stable or noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

MODELS = ("market", "elo", "logistic", "catboost", "fixed_blend")


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
        default=Path("outputs/phase1/analysis"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prob_col(model: str) -> str:
    return f"p_home_{model}"


def validate_predictions(df: pd.DataFrame) -> None:
    required = {
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "home_win",
        *(_prob_col(model) for model in MODELS),
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Prediction file is missing required columns: {sorted(missing)}")
    if df["game_id"].duplicated().any():
        dupes = df.loc[df["game_id"].duplicated(), "game_id"].head().tolist()
        raise SystemExit(f"Prediction file contains duplicate games, e.g. {dupes}")
    if not df["home_win"].dropna().isin([0, 1]).all():
        raise SystemExit("home_win must contain only 0/1 for analyzed games")


def score_model(df: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    y = df["home_win"].astype(int).to_numpy()
    p = np.clip(df[_prob_col(model)].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "model": model,
        "games": int(len(df)),
        "accuracy": float(np.mean(pred == y)),
        "correct": int(np.sum(pred == y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_confidence": float(np.mean(np.maximum(p, 1 - p))),
    }


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([score_model(df, model) for model in MODELS]).sort_values(
        ["accuracy", "brier"], ascending=[False, True]
    )


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for season, group in df.groupby("season", sort=True):
        for model in MODELS:
            row = score_model(group, model)
            row["season"] = int(season)
            rows.append(row)
    return pd.DataFrame(rows)[
        ["season", "model", "games", "correct", "accuracy", "brier", "log_loss", "mean_confidence"]
    ]


def paired_bootstrap_accuracy_lift(
    df: pd.DataFrame,
    model: str,
    benchmark: str = "market",
    reps: int = 10000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    y = df["home_win"].astype(int).to_numpy()
    model_pick = (df[_prob_col(model)].to_numpy(dtype=float) >= 0.5).astype(int)
    bench_pick = (df[_prob_col(benchmark)].to_numpy(dtype=float) >= 0.5).astype(int)
    model_correct = (model_pick == y).astype(float)
    bench_correct = (bench_pick == y).astype(float)
    diff = model_correct - bench_correct

    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        raise ValueError("No games available for bootstrap")

    # Bootstrap in game-sized chunks. A later research phase can add a season-
    # clustered bootstrap as a sensitivity test.
    samples = np.empty(reps, dtype=float)
    chunk = 500
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        idx = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = diff[idx].mean(axis=1)

    return {
        "model": model,
        "benchmark": benchmark,
        "games": n,
        "accuracy_lift": float(diff.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_50": float(np.quantile(samples, 0.50)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "prob_lift_gt_0": float(np.mean(samples > 0)),
    }


def paired_lift_table(df: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, model in enumerate(MODELS):
        if model == "market":
            continue
        rows.append(
            paired_bootstrap_accuracy_lift(
                df,
                model=model,
                benchmark="market",
                reps=reps,
                seed=seed + i,
            )
        )
    return pd.DataFrame(rows).sort_values("accuracy_lift", ascending=False)


def disagreement_table(df: pd.DataFrame) -> pd.DataFrame:
    y = df["home_win"].astype(int).to_numpy()
    market_pick = (df[_prob_col("market")].to_numpy(dtype=float) >= 0.5).astype(int)
    rows = []

    for model in MODELS:
        if model == "market":
            continue
        p = df[_prob_col(model)].to_numpy(dtype=float)
        pick = (p >= 0.5).astype(int)
        disagree = pick != market_pick
        agree = ~disagree

        for label, mask in (("agree", agree), ("disagree", disagree)):
            n = int(mask.sum())
            if n == 0:
                continue
            model_acc = float(np.mean(pick[mask] == y[mask]))
            market_acc = float(np.mean(market_pick[mask] == y[mask]))
            rows.append(
                {
                    "model": model,
                    "subset": label,
                    "games": n,
                    "share_of_games": n / len(df),
                    "model_accuracy": model_acc,
                    "market_accuracy": market_acc,
                    "accuracy_lift": model_acc - market_acc,
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "subset"])


def confidence_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.000001]
    labels = ["50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-90", "90-100"]
    rows = []
    y = df["home_win"].astype(int).to_numpy()

    for model in MODELS:
        p = df[_prob_col(model)].to_numpy(dtype=float)
        pick = (p >= 0.5).astype(int)
        confidence = np.maximum(p, 1 - p)
        bucket = pd.cut(confidence, bins=bins, labels=labels, right=False, include_lowest=True)
        temp = pd.DataFrame(
            {
                "bucket": bucket,
                "correct": (pick == y).astype(int),
                "confidence": confidence,
            }
        )
        for bucket_name, group in temp.groupby("bucket", observed=True):
            rows.append(
                {
                    "model": model,
                    "confidence_bucket": str(bucket_name),
                    "games": int(len(group)),
                    "accuracy": float(group["correct"].mean()),
                    "mean_confidence": float(group["confidence"].mean()),
                    "calibration_gap": float(group["correct"].mean() - group["confidence"].mean()),
                }
            )
    return pd.DataFrame(rows)


def market_override_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate simple disagreement thresholds without declaring a winner.

    These are descriptive diagnostics only. A future nested validation stage
    must select any override threshold using training folds and then test it on
    later untouched folds.
    """

    y = df["home_win"].astype(int).to_numpy()
    market_p = df[_prob_col("market")].to_numpy(dtype=float)
    market_pick = (market_p >= 0.5).astype(int)
    rows = []

    thresholds = (0.00, 0.025, 0.05, 0.075, 0.10, 0.15)
    for model in ("logistic", "catboost", "fixed_blend"):
        p = df[_prob_col(model)].to_numpy(dtype=float)
        model_pick = (p >= 0.5).astype(int)
        disagreement = model_pick != market_pick
        edge = np.abs(p - 0.5) - np.abs(market_p - 0.5)

        for threshold in thresholds:
            override = disagreement & (edge >= threshold)
            hybrid_pick = np.where(override, model_pick, market_pick)
            rows.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "overrides": int(override.sum()),
                    "override_share": float(override.mean()),
                    "hybrid_accuracy": float(np.mean(hybrid_pick == y)),
                    "market_accuracy": float(np.mean(market_pick == y)),
                    "accuracy_lift": float(np.mean(hybrid_pick == y) - np.mean(market_pick == y)),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> None:
    best = overall.iloc[0]
    market = overall.loc[overall["model"].eq("market")].iloc[0]

    lines = [
        "# Phase 1 Backtest Analysis",
        "",
        f"Games evaluated: **{int(best['games'])}**",
        "",
        "## Aggregate scoreboard",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired accuracy lift vs market",
        "",
        paired.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Model/market disagreement",
        "",
        disagreement.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation guardrail",
        "",
        f"The market baseline accuracy in this replay is **{market['accuracy']:.3%}**. ",
        "A challenger should not be promoted from Phase 1 solely because its aggregate ",
        "accuracy is higher. We also require stable season-by-season behavior, good ",
        "probability scoring/calibration, and a paired lift that survives later nested ",
        "time-series validation.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    df = df.loc[df["home_win"].notna()].copy()
    validate_predictions(df)

    overall = overall_summary(df)
    seasons = season_summary(df)
    paired = paired_lift_table(df, reps=args.bootstrap_reps, seed=args.seed)
    disagreements = disagreement_table(df)
    confidence = confidence_table(df)
    override_candidates = market_override_candidates(df)

    overall.to_csv(args.output_dir / "overall_scoreboard.csv", index=False)
    seasons.to_csv(args.output_dir / "season_scoreboard.csv", index=False)
    paired.to_csv(args.output_dir / "paired_lift_vs_market.csv", index=False)
    disagreements.to_csv(args.output_dir / "market_disagreements.csv", index=False)
    confidence.to_csv(args.output_dir / "confidence_buckets.csv", index=False)
    override_candidates.to_csv(args.output_dir / "override_diagnostics.csv", index=False)
    write_markdown_summary(
        args.output_dir / "README.md",
        overall=overall,
        paired=paired,
        disagreement=disagreements,
    )

    print("\nAggregate scoreboard")
    print(overall.to_string(index=False))
    print("\nPaired lift vs market")
    print(paired.to_string(index=False))
    print(f"\nAnalysis written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
