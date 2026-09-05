"""Phase 2 matchup-driven true-upset specialist.

The Phase 1 decision lab showed that generic model-vs-market overrides do not
reliably identify true upsets. This experiment reframes the target around the
market underdog and builds matchup-specific features that ask why THIS underdog
could beat THIS favorite.

Test domain: market favorite probability 52.5% to <80%.
Outside this domain the production concept remains "follow the market" unless a
separate specialist proves otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import BASE_TEAM_METRICS, ROLL_WINDOWS, build_model_table

MIN_FAV_PROB = 0.525
MAX_FAV_PROB = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_upset_specialist"))
    return p.parse_args()


def oriented_side_values(df: pd.DataFrame, stem: str, dog_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    home = pd.to_numeric(df[f"home_{stem}"], errors="coerce").to_numpy(float)
    away = pd.to_numeric(df[f"away_{stem}"], errors="coerce").to_numpy(float)
    dog = np.where(dog_home, home, away)
    fav = np.where(dog_home, away, home)
    return dog, fav


def build_upset_table(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = data.loc[data["home_win"].notna()].copy()
    home_market = x["market_home_prob"].to_numpy(float)
    fav_home = home_market >= 0.5
    dog_home = ~fav_home
    x["market_fav_prob"] = np.maximum(home_market, 1 - home_market)
    x["market_dog_prob"] = 1 - x["market_fav_prob"]
    x["dog_is_home"] = dog_home.astype(int)
    x["dog_win"] = np.where(fav_home, x["home_win"].astype(int).to_numpy() == 0, x["home_win"].astype(int).to_numpy() == 1).astype(int)
    x["favorite_team"] = np.where(fav_home, x["home_team"], x["away_team"])
    x["underdog_team"] = np.where(fav_home, x["away_team"], x["home_team"])

    # Context oriented from underdog perspective.
    x["dog_rest_adv"] = np.where(dog_home, x["rest_diff"], -x["rest_diff"])
    x["dog_elo_adv"] = np.where(dog_home, x["elo_diff"], -x["elo_diff"])
    x["total_line_num"] = pd.to_numeric(x["total_line"], errors="coerce")

    features = ["market_dog_prob", "market_fav_prob", "dog_is_home", "dog_rest_adv", "dog_elo_adv", "total_line_num"]

    # Generic underdog-vs-favorite form differentials plus recent-vs-medium trend.
    for metric in BASE_TEAM_METRICS:
        dog_by_window: dict[int, np.ndarray] = {}
        fav_by_window: dict[int, np.ndarray] = {}
        for window in ROLL_WINDOWS:
            stem = f"{metric}_r{window}"
            dog, fav = oriented_side_values(x, stem, dog_home)
            dog_by_window[window] = dog
            fav_by_window[window] = fav
            name = f"dog_minus_fav_{metric}_r{window}"
            x[name] = dog - fav
            features.append(name)
        if 4 in dog_by_window and 8 in dog_by_window:
            trend = f"trend_diff_{metric}_r4_vs_r8"
            x[trend] = (dog_by_window[4] - dog_by_window[8]) - (fav_by_window[4] - fav_by_window[8])
            features.append(trend)

    # Matchup interactions. EPA/success allowed are worse for the defense when
    # larger, so adding offense quality to opponent allowance is intuitive.
    for window in ROLL_WINDOWS:
        dog_pass_off, fav_pass_off = oriented_side_values(x, f"off_pass_epa_r{window}", dog_home)
        dog_rush_off, fav_rush_off = oriented_side_values(x, f"off_rush_epa_r{window}", dog_home)
        dog_success_off, fav_success_off = oriented_side_values(x, f"off_success_r{window}", dog_home)
        dog_sack_off, fav_sack_off = oriented_side_values(x, f"off_sack_rate_r{window}", dog_home)
        dog_to_off, fav_to_off = oriented_side_values(x, f"off_turnover_rate_r{window}", dog_home)

        dog_pass_def_allow, fav_pass_def_allow = oriented_side_values(x, f"def_pass_epa_allowed_r{window}", dog_home)
        dog_rush_def_allow, fav_rush_def_allow = oriented_side_values(x, f"def_rush_epa_allowed_r{window}", dog_home)
        dog_success_def_allow, fav_success_def_allow = oriented_side_values(x, f"def_success_allowed_r{window}", dog_home)
        dog_sack_def, fav_sack_def = oriented_side_values(x, f"def_sack_rate_r{window}", dog_home)
        dog_takeaway_def, fav_takeaway_def = oriented_side_values(x, f"def_takeaway_rate_r{window}", dog_home)

        matchup_values = {
            f"pass_matchup_edge_r{window}": (dog_pass_off + fav_pass_def_allow) - (fav_pass_off + dog_pass_def_allow),
            f"rush_matchup_edge_r{window}": (dog_rush_off + fav_rush_def_allow) - (fav_rush_off + dog_rush_def_allow),
            f"success_matchup_edge_r{window}": (dog_success_off + fav_success_def_allow) - (fav_success_off + dog_success_def_allow),
            f"pressure_matchup_edge_r{window}": (dog_sack_def + fav_sack_off) - (fav_sack_def + dog_sack_off),
            f"turnover_pressure_edge_r{window}": (dog_takeaway_def + fav_to_off) - (fav_takeaway_def + dog_to_off),
        }
        for name, values in matchup_values.items():
            x[name] = values
            features.append(name)

    x = x.loc[(x["market_fav_prob"] >= MIN_FAV_PROB) & (x["market_fav_prob"] < MAX_FAV_PROB)].copy()
    return x, features


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.20,
                    C=0.25,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    prob = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pick_dog = prob >= 0.5
    yy = np.asarray(y, int)
    return {
        "games": int(len(yy)),
        "upset_calls": int(pick_dog.sum()),
        "correct": int(np.sum(pick_dog == yy)),
        "accuracy": float(np.mean(pick_dog == yy)),
        "upset_call_accuracy": float(np.mean(yy[pick_dog] == 1)) if pick_dog.any() else np.nan,
        "log_loss": float(log_loss(yy, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(yy, prob)),
    }


def run_backtest(table: pd.DataFrame, features: list[str], first_test_season: int, end_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, folds = [], []
    for season in range(first_test_season, end_season + 1):
        train = table.loc[table["season"] < season].copy()
        test = table.loc[table["season"] == season].copy()
        if train.empty or test.empty:
            continue
        y_train = train["dog_win"].astype(int)
        y_test = test["dog_win"].astype(int).to_numpy()

        logit_model = make_logistic()
        logit_model.fit(train[features], y_train)
        p_logit = logit_model.predict_proba(test[features])[:, 1]

        cat = CatBoostClassifier(
            iterations=600,
            depth=5,
            learning_rate=0.025,
            loss_function="Logloss",
            random_seed=42,
            l2_leaf_reg=12.0,
            random_strength=0.7,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(train[features], y_train)
        p_cat = cat.predict_proba(test[features])[:, 1]
        p_market_dog = test["market_dog_prob"].to_numpy(float)

        model_probs = {"market": p_market_dog, "upset_logistic": p_logit, "upset_catboost": p_cat}
        for name, prob in model_probs.items():
            m = probability_metrics(y_test, prob)
            folds.append({"season": season, "model": name, **m})

        out = test[["game_id", "season", "week", "gameday", "favorite_team", "underdog_team", "dog_win", "market_fav_prob", "market_dog_prob"]].copy()
        out["p_dog_upset_logistic"] = p_logit
        out["p_dog_upset_catboost"] = p_cat
        predictions.append(out)
        print(
            f"{season}: market={probability_metrics(y_test, p_market_dog)['accuracy']:.3f} "
            f"logit={probability_metrics(y_test, p_logit)['accuracy']:.3f} "
            f"cat={probability_metrics(y_test, p_cat)['accuracy']:.3f}"
        )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["dog_win"].astype(int).to_numpy()
    rows = []
    for name, col in (
        ("market", "market_dog_prob"),
        ("upset_logistic", "p_dog_upset_logistic"),
        ("upset_catboost", "p_dog_upset_catboost"),
    ):
        rows.append({"model": name, **probability_metrics(y, pred[col].to_numpy(float))})
    market_correct = int(np.sum(y == 0))
    for row in rows:
        row["net_correct_vs_market"] = int(row["correct"] - market_correct)
    return pd.DataFrame(rows).sort_values(["accuracy", "brier"], ascending=[False, True])


def bucket_summary(pred: pd.DataFrame) -> pd.DataFrame:
    bins = [0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80]
    labels = ["52.5-55", "55-57.5", "57.5-60", "60-65", "65-70", "70-80"]
    x = pred.copy()
    x["bucket"] = pd.cut(x["market_fav_prob"], bins=bins, labels=labels, right=False)
    rows = []
    for bucket, g in x.groupby("bucket", observed=True):
        y = g["dog_win"].astype(int).to_numpy()
        for name, col in (
            ("market", "market_dog_prob"),
            ("upset_logistic", "p_dog_upset_logistic"),
            ("upset_catboost", "p_dog_upset_catboost"),
        ):
            m = probability_metrics(y, g[col].to_numpy(float))
            rows.append({"bucket": str(bucket), "model": name, **m, "actual_upset_rate": float(y.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_model_table(args.start_season, args.end_season)
    table, features = build_upset_table(data)
    pred, folds = run_backtest(table, features, args.first_test_season, args.end_season)
    summary = summarize(pred)
    buckets = bucket_summary(pred)

    pred.to_csv(args.output_dir / "upset_specialist_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "upset_specialist_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "upset_specialist_summary.csv", index=False)
    buckets.to_csv(args.output_dir / "upset_specialist_buckets.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.output_dir / "upset_specialist_features.csv", index=False)

    print("\nUpset specialist summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
