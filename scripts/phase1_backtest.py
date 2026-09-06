"""Phase 1 NFL straight-up winner backtest.

Builds a leak-resistant game table from nflverse data and compares:
  * market favorite
  * Elo favorite
  * elastic-net logistic regression
  * CatBoost
  * a fixed market/model blend (research candidate, not production stack)

The final production ensemble will use nested, time-ordered out-of-fold stacking.
This script intentionally starts simpler so every later feature must prove value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROLL_WINDOWS = (4, 8)
BASE_TEAM_METRICS = (
    "off_epa",
    "off_success",
    "off_pass_epa",
    "off_rush_epa",
    "off_sack_rate",
    "off_turnover_rate",
    "def_epa_allowed",
    "def_success_allowed",
    "def_pass_epa_allowed",
    "def_rush_epa_allowed",
    "def_sack_rate",
    "def_takeaway_rate",
)


def american_implied_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=odds.index, dtype=float)
    neg = odds < 0
    pos = odds > 0
    out.loc[neg] = (-odds.loc[neg]) / ((-odds.loc[neg]) + 100.0)
    out.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
    return out


def add_market_probability(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    home_raw = american_implied_prob(games["home_moneyline"])
    away_raw = american_implied_prob(games["away_moneyline"])
    denom = home_raw + away_raw
    games["market_home_prob"] = home_raw / denom

    # Fallback for rows lacking usable moneyline data. This is deliberately a
    # conservative fixed mapping, not a fitted transform that could leak across folds.
    spread_fallback = 1.0 / (1.0 + np.exp(-pd.to_numeric(games["spread_line"], errors="coerce") / 6.5))
    games["market_home_prob"] = games["market_home_prob"].fillna(spread_fallback).fillna(0.5)
    return games


def build_team_game_stats(pbp: pl.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "posteam",
        "defteam",
        "epa",
        "success",
        "pass",
        "rush",
        "sack",
        "interception",
        "fumble_lost",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise RuntimeError(f"nflverse PBP is missing required columns: {sorted(missing)}")

    plays = (
        pbp.select(sorted(required))
        .filter(
            ((pl.col("pass") == 1) | (pl.col("rush") == 1))
            & pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
        )
        .with_columns(
            pl.col("success").cast(pl.Float64, strict=False),
            pl.col("sack").fill_null(0).cast(pl.Float64),
            pl.col("interception").fill_null(0).cast(pl.Float64),
            pl.col("fumble_lost").fill_null(0).cast(pl.Float64),
            ((pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)) > 0)
            .cast(pl.Float64)
            .alias("turnover"),
        )
    )

    offense = plays.group_by(["game_id", "posteam"]).agg(
        pl.mean("epa").alias("off_epa"),
        pl.mean("success").alias("off_success"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
        pl.mean("sack").alias("off_sack_rate"),
        pl.mean("turnover").alias("off_turnover_rate"),
    ).rename({"posteam": "team"})

    defense = plays.group_by(["game_id", "defteam"]).agg(
        pl.mean("epa").alias("def_epa_allowed"),
        pl.mean("success").alias("def_success_allowed"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_pass_epa_allowed"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_rush_epa_allowed"),
        pl.mean("sack").alias("def_sack_rate"),
        pl.mean("turnover").alias("def_takeaway_rate"),
    ).rename({"defteam": "team"})

    return offense.join(defense, on=["game_id", "team"], how="inner").to_pandas()


def add_rolling_team_features(team_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    context = schedule[["game_id", "gameday", "season", "week"]].drop_duplicates("game_id")
    tg = team_games.merge(context, on="game_id", how="left")
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    for metric in BASE_TEAM_METRICS:
        for window in ROLL_WINDOWS:
            name = f"{metric}_r{window}"
            tg[name] = tg.groupby("team", group_keys=False)[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=2).mean()
            )
    return tg


def add_elo(games: pd.DataFrame) -> pd.DataFrame:
    ordered = games.sort_values(["gameday", "game_id"]).copy()
    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows: list[dict[str, float | str]] = []

    for row in ordered.itertuples(index=False):
        season = int(row.season)
        if current_season is None:
            current_season = season
        elif season != current_season:
            # Regress team strength toward league average between seasons.
            ratings = {team: 1500.0 + 0.75 * (rating - 1500.0) for team, rating in ratings.items()}
            current_season = season

        home = row.home_team
        away = row.away_team
        home_elo = ratings.get(home, 1500.0)
        away_elo = ratings.get(away, 1500.0)
        neutral = str(getattr(row, "location", "")).lower() == "neutral"
        hfa = 0.0 if neutral else 55.0
        elo_diff = home_elo + hfa - away_elo
        home_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        rows.append(
            {
                "game_id": row.game_id,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "elo_home_prob": home_prob,
            }
        )

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        margin = float(row.home_score - row.away_score)
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        mov_multiplier = math.log(abs(margin) + 1.0) * (2.2 / (2.2 + 0.001 * abs(elo_diff)))
        change = 20.0 * mov_multiplier * (actual - home_prob)
        ratings[home] = home_elo + change
        ratings[away] = away_elo - change

    return games.merge(pd.DataFrame(rows), on="game_id", how="left")


def build_model_table(start_season: int, end_season: int) -> pd.DataFrame:
    seasons = list(range(start_season, end_season + 1))
    print(f"Loading schedules {start_season}-{end_season}...")
    schedule = nfl.load_schedules(seasons).to_pandas()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])

    # Phase 1 uses regular season games. Playoffs can be added as a separately
    # validated regime once the regular-season model is stable.
    games = schedule.loc[schedule["game_type"].eq("REG")].copy()
    games = add_market_probability(games)
    games = add_elo(games)

    print(f"Loading play-by-play {start_season}-{end_season}...")
    pbp = nfl.load_pbp(seasons)
    team_games = build_team_game_stats(pbp)
    team_games = add_rolling_team_features(team_games, games)

    rolling_cols = [
        f"{metric}_r{window}" for metric in BASE_TEAM_METRICS for window in ROLL_WINDOWS
    ]
    home = team_games[["game_id", "team", *rolling_cols]].rename(columns={"team": "home_team"})
    home = home.rename(columns={c: f"home_{c}" for c in rolling_cols})
    away = team_games[["game_id", "team", *rolling_cols]].rename(columns={"team": "away_team"})
    away = away.rename(columns={c: f"away_{c}" for c in rolling_cols})

    games = games.merge(home, on=["game_id", "home_team"], how="left")
    games = games.merge(away, on=["game_id", "away_team"], how="left")

    for col in rolling_cols:
        games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]

    games["rest_diff"] = pd.to_numeric(games["home_rest"], errors="coerce") - pd.to_numeric(
        games["away_rest"], errors="coerce"
    )
    games["home_win"] = np.where(
        games["home_score"] > games["away_score"],
        1.0,
        np.where(games["home_score"] < games["away_score"], 0.0, np.nan),
    )
    return games


def score_probabilities(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    pick = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pick)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
    }


def walk_forward_backtest(data: pd.DataFrame, first_test_season: int, end_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    diff_features = [
        f"diff_{metric}_r{window}" for metric in BASE_TEAM_METRICS for window in ROLL_WINDOWS
    ]
    numeric_features = [
        "market_home_prob",
        "spread_line",
        "total_line",
        "rest_diff",
        "elo_diff",
        "elo_home_prob",
        *diff_features,
    ]
    categorical_features = ["roof", "surface", "location", "div_game"]
    feature_cols = numeric_features + categorical_features

    completed = data.loc[data["home_win"].notna()].copy()
    folds: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []

    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue

        x_train = train[feature_cols]
        y_train = train["home_win"].astype(int)
        x_test = test[feature_cols]
        y_test = test["home_win"].astype(int)

        numeric_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ]
        )
        logistic = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.15,
                        C=0.5,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
        logistic.fit(x_train, y_train)
        p_logistic = logistic.predict_proba(x_test)[:, 1]

        cat_train = x_train.copy()
        cat_test = x_test.copy()
        for col in categorical_features:
            cat_train[col] = cat_train[col].astype("string").fillna("__MISSING__")
            cat_test[col] = cat_test[col].astype("string").fillna("__MISSING__")
        cat_indices = [cat_train.columns.get_loc(c) for c in categorical_features]

        cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            random_seed=42,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(cat_train, y_train, cat_features=cat_indices)
        p_cat = cat.predict_proba(cat_test)[:, 1]

        p_market = test["market_home_prob"].to_numpy(dtype=float)
        p_elo = test["elo_home_prob"].to_numpy(dtype=float)

        # Fixed blend is only a challenger for Phase 1. Production stacking will
        # learn weights from time-ordered out-of-fold predictions, not this formula.
        p_blend = 0.50 * p_market + 0.25 * p_logistic + 0.25 * p_cat

        model_probs = {
            "market": p_market,
            "elo": p_elo,
            "logistic": p_logistic,
            "catboost": p_cat,
            "fixed_blend": p_blend,
        }

        fold_pred = test[
            ["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win"]
        ].copy()
        for name, prob in model_probs.items():
            fold_pred[f"p_home_{name}"] = prob
            metrics = score_probabilities(y_test, prob)
            folds.append(
                {
                    "season": season,
                    "model": name,
                    "games": len(test),
                    **metrics,
                }
            )
        predictions.append(fold_pred)
        print(
            f"{season}: market={score_probabilities(y_test, p_market)['accuracy']:.3f} "
            f"logistic={score_probabilities(y_test, p_logistic)['accuracy']:.3f} "
            f"catboost={score_probabilities(y_test, p_cat)['accuracy']:.3f} "
            f"blend={score_probabilities(y_test, p_blend)['accuracy']:.3f}"
        )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def summarize(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = predictions["home_win"].astype(int)
    for model in ["market", "elo", "logistic", "catboost", "fixed_blend"]:
        p = predictions[f"p_home_{model}"].to_numpy(dtype=float)
        rows.append({"model": model, "games": len(predictions), **score_probabilities(y, p)})
    summary = pd.DataFrame(rows).sort_values(["accuracy", "log_loss"], ascending=[False, True])

    print("\nOverall walk-forward results")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSeason-by-season accuracy")
    print(
        folds.pivot(index="season", columns="model", values="accuracy")
        .round(4)
        .to_string()
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--first-test-season", type=int, default=2016)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.first_test_season <= args.start_season:
        raise SystemExit("--first-test-season must be after --start-season")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_model_table(args.start_season, args.end_season)
    predictions, folds = walk_forward_backtest(data, args.first_test_season, args.end_season)
    summary = summarize(folds, predictions)

    predictions.to_csv(args.output_dir / "phase1_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "phase1_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "phase1_summary.csv", index=False)
    print(f"\nSaved Phase 1 results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
