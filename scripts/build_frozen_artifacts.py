"""Build versioned artifacts for the already-frozen 2026 pick'em model.

This is an infrastructure optimization only. It reproduces the exact models that
`operational_pickem.py` currently refits on every live run, then exports them so
live predictions can load the frozen weights instantly.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from frozen_artifacts import (
    export_base_logistic,
    export_simple_logistic,
    sha256_manifest_files,
    write_json,
)
from live_pickem import (
    CLOSE_FAVORITE_MAX,
    HISTORICAL_OOF_FIRST_SEASON,
    UPSET_FAVORITE_MAX,
    actual_team_history,
    base_feature_lists,
    build_variance_training,
    fit_base_models,
    make_matchup_logistic,
    make_variance_catboost,
)
from phase1_backtest import build_model_table, build_team_game_stats, walk_forward_backtest
from phase1_market_residual import fit_residual, tune_penalty
from phase2_upset_specialist import build_upset_table
from phase2_upset_variance import VAR_METRICS, build_variance_game_stats, rolling_variance

MODEL_VERSION = "prospective-v1-frozen-2025"
TRAIN_END_SEASON = 2025
TAIL_GAMES = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--end-season", type=int, default=TRAIN_END_SEASON)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frozen_artifacts") / MODEL_VERSION,
    )
    return p.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tail_history(frame: pd.DataFrame, metrics: list[str] | tuple[str, ...]) -> pd.DataFrame:
    keep = ["game_id", "team", "gameday", *metrics]
    x = frame[keep].copy()
    x["gameday"] = pd.to_datetime(x["gameday"])
    return (
        x.sort_values(["team", "gameday", "game_id"])
        .groupby("team", group_keys=False)
        .tail(TAIL_GAMES)
        .sort_values(["team", "gameday", "game_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    if args.end_season != TRAIN_END_SEASON:
        raise SystemExit(f"Frozen artifact build must end at {TRAIN_END_SEASON}")

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building frozen historical table {args.start_season}-{args.end_season}...")
    historical = build_model_table(args.start_season, args.end_season)
    historical["gameday"] = pd.to_datetime(historical["gameday"])
    frozen_train = historical.loc[historical["home_win"].notna()].copy()
    if frozen_train.empty:
        raise SystemExit("Frozen training table is empty")

    print("Fitting frozen base classifiers...")
    base_logistic, base_cat, base_features, categorical = fit_base_models(frozen_train)
    numeric, expected_categorical = base_feature_lists()
    if categorical != expected_categorical or base_features != [*numeric, *categorical]:
        raise RuntimeError("Base feature ordering changed while building artifacts")
    export_base_logistic(base_logistic, numeric, categorical, out / "base_logistic.json")
    base_cat.save_model(str(out / "base_catboost.json"), format="json")

    print("Fitting frozen residual meta-model...")
    oof, _ = walk_forward_backtest(
        frozen_train,
        HISTORICAL_OOF_FIRST_SEASON,
        TRAIN_END_SEASON,
    )
    penalty, _ = tune_penalty(oof)
    theta = fit_residual(oof, penalty)
    write_json(
        out / "residual.json",
        {
            "kind": "market_residual_v1",
            "penalty": float(penalty),
            "theta": [float(v) for v in theta],
            "oof_first_season": HISTORICAL_OOF_FIRST_SEASON,
            "training_end_season": TRAIN_END_SEASON,
        },
    )

    print("Fitting frozen matchup specialist...")
    matchup_train, matchup_features = build_upset_table(frozen_train)
    matchup_train = matchup_train.loc[
        (matchup_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (matchup_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    matchup_model = make_matchup_logistic()
    matchup_model.fit(matchup_train[matchup_features], matchup_train["dog_win"].astype(int))
    export_simple_logistic(matchup_model, matchup_features, out / "matchup_logistic.json")

    print("Loading historical PBP once for rolling-state tails and variance specialist...")
    seasons = list(range(args.start_season, args.end_season + 1))
    pbp = nfl.load_pbp(seasons)
    schedule = nfl.load_schedules(seasons).to_pandas()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    regular = schedule.loc[schedule["game_type"].eq("REG")].copy()

    base_history = actual_team_history(build_team_game_stats(pbp), regular)
    base_metrics = [c for c in base_history.columns if c not in {"game_id", "team", "gameday", "season", "week"}]
    base_tail = tail_history(base_history, base_metrics)
    base_tail.to_csv(out / "base_history_tail.csv", index=False)

    variance_games = build_variance_game_stats(pbp)
    variance_history = actual_team_history(variance_games, regular)
    variance_tail = tail_history(variance_history, list(VAR_METRICS))
    variance_tail.to_csv(out / "variance_history_tail.csv", index=False)

    variance_rolls = rolling_variance(variance_games, regular)
    variance_train, variance_features = build_variance_training(frozen_train, variance_rolls)
    variance_train = variance_train.loc[
        (variance_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (variance_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    variance_model = make_variance_catboost()
    variance_model.fit(
        variance_train[variance_features], variance_train["dog_win"].astype(int)
    )
    variance_model.save_model(str(out / "variance_catboost.json"), format="json")
    write_json(
        out / "feature_manifest.json",
        {
            "base_features": base_features,
            "base_categorical": categorical,
            "matchup_features": matchup_features,
            "variance_features": variance_features,
            "tail_games": TAIL_GAMES,
        },
    )

    artifact_files = [
        "base_logistic.json",
        "base_catboost.json",
        "residual.json",
        "matchup_logistic.json",
        "variance_catboost.json",
        "base_history_tail.csv",
        "variance_history_tail.csv",
        "feature_manifest.json",
    ]
    hashes = sha256_manifest_files(out, artifact_files)
    manifest = {
        "artifact_schema": 1,
        "model_version": MODEL_VERSION,
        "training_start_season": args.start_season,
        "training_end_season": TRAIN_END_SEASON,
        "frozen_close_favorite_max": CLOSE_FAVORITE_MAX,
        "frozen_upset_favorite_max": UPSET_FAVORITE_MAX,
        "built_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "python": platform.python_version(),
        "packages": {
            "nflreadpy": package_version("nflreadpy"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "catboost": package_version("catboost"),
            "scipy": package_version("scipy"),
        },
        "files": hashes,
    }
    write_json(out / "manifest.json", manifest)

    print(f"Built {MODEL_VERSION} artifacts in {out}")
    print(f"Base history tail rows: {len(base_tail)}")
    print(f"Variance history tail rows: {len(variance_tail)}")
    print(f"Residual penalty: {penalty}")


if __name__ == "__main__":
    main()
