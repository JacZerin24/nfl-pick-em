"""Leak-aware Phase 2 quarterback-information ceiling experiment.

The schedule file's historical home_qb_id/away_qb_id is based on the quarterback
who actually started/played in the completed game, so current-game QB identity is
an ORACLE label here, not yet a reproducible historical pregame feed.

Every QB performance feature is shifted and uses only PRIOR weeks. The question
is deliberately narrow: if the correct starter were known at kickoff, does QB
history add stable out-of-sample winner-prediction value beyond market + team
features? If yes, we invest in timestamped expected-starter/injury snapshots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase1_backtest import BASE_TEAM_METRICS, ROLL_WINDOWS, build_model_table, score_probabilities

QB_RECENT_GAMES = 6
QB_FEATURES = [
    "qb_prior_dropbacks",
    "qb_prior_games",
    "qb_career_epa_db",
    "qb_career_cpoe",
    "qb_career_sack_rate",
    "qb_career_int_rate",
    "qb_recent_epa_db",
    "qb_recent_cpoe",
    "qb_recent_sack_rate",
    "qb_recent_int_rate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_qb_ceiling"))
    return p.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_player_history(group: pd.DataFrame) -> pd.DataFrame:
    """Add prior-career and recent-form QB features to one player's rows."""
    g = group.sort_values(["season", "week"]).copy()

    prior_db = g["dropbacks"].cumsum().shift(1).fillna(0.0)
    prior_att = g["attempts"].cumsum().shift(1).fillna(0.0)
    prior_epa = g["passing_epa"].cumsum().shift(1).fillna(0.0)
    prior_sacks = g["sacks_suffered"].cumsum().shift(1).fillna(0.0)
    prior_int = g["passing_interceptions"].cumsum().shift(1).fillna(0.0)
    prior_cpoe_num = g["cpoe_num"].cumsum().shift(1).fillna(0.0)
    prior_cpoe_den = g["cpoe_den"].cumsum().shift(1).fillna(0.0)

    g["qb_prior_dropbacks"] = prior_db
    g["qb_prior_games"] = g["substantial_game"].cumsum().shift(1).fillna(0.0)

    # Fixed neutral priors shrink rookies/tiny samples rather than letting a
    # handful of dropbacks create extreme ratings.
    g["qb_career_epa_db"] = prior_epa / (prior_db + 200.0)
    g["qb_career_cpoe"] = prior_cpoe_num / (prior_cpoe_den + 150.0)
    g["qb_career_sack_rate"] = (prior_sacks + 6.5) / (prior_db + 100.0)
    g["qb_career_int_rate"] = (prior_int + 2.5) / (prior_att + 100.0)

    def recent_sum(col: str) -> pd.Series:
        return g[col].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()

    r_db = recent_sum("dropbacks")
    r_att = recent_sum("attempts")
    g["qb_recent_epa_db"] = recent_sum("passing_epa") / r_db
    g["qb_recent_cpoe"] = recent_sum("cpoe_num") / recent_sum("cpoe_den")
    g["qb_recent_sack_rate"] = recent_sum("sacks_suffered") / r_db
    g["qb_recent_int_rate"] = recent_sum("passing_interceptions") / r_att
    return g


def build_qb_week_table(seasons: list[int]) -> pd.DataFrame:
    print(f"Loading player stats {min(seasons)}-{max(seasons)}...")
    stats = nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    required = {
        "player_id",
        "season",
        "week",
        "season_type",
        "attempts",
        "passing_interceptions",
        "sacks_suffered",
        "passing_epa",
        "passing_cpoe",
    }
    missing = required.difference(stats.columns)
    if missing:
        raise RuntimeError(f"Player stats missing required QB fields: {sorted(missing)}")

    qb = stats.loc[stats["season_type"].eq("REG"), list(required)].copy()
    for col in ("attempts", "passing_interceptions", "sacks_suffered", "passing_epa"):
        qb[col] = numeric(qb[col])
    qb["passing_cpoe"] = pd.to_numeric(qb["passing_cpoe"], errors="coerce")
    qb["dropbacks"] = qb["attempts"] + qb["sacks_suffered"]
    qb = qb.loc[qb["dropbacks"] > 0].copy()
    qb["cpoe_num"] = qb["passing_cpoe"].fillna(0.0) * qb["attempts"]
    qb["cpoe_den"] = np.where(qb["passing_cpoe"].notna(), qb["attempts"], 0.0)

    # Collapse any unusual duplicate player-week rows before history is built.
    qb = (
        qb.groupby(["player_id", "season", "week"], as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            passing_interceptions=("passing_interceptions", "sum"),
            sacks_suffered=("sacks_suffered", "sum"),
            passing_epa=("passing_epa", "sum"),
            dropbacks=("dropbacks", "sum"),
            cpoe_num=("cpoe_num", "sum"),
            cpoe_den=("cpoe_den", "sum"),
        )
        .sort_values(["player_id", "season", "week"])
        .reset_index(drop=True)
    )
    qb["substantial_game"] = (qb["dropbacks"] >= 10).astype(float)

    # Explicit loop avoids pandas groupby.apply behavior changing the group key.
    pieces = []
    for _, group in qb.groupby("player_id", sort=False):
        pieces.append(add_player_history(group))
    qb = pd.concat(pieces, ignore_index=True)
    return qb[["player_id", "season", "week", *QB_FEATURES]]


def add_oracle_qb_features(
    games: pd.DataFrame, qb_week: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    if not {"home_qb_id", "away_qb_id"}.issubset(games.columns):
        raise RuntimeError("Schedule is missing home_qb_id/away_qb_id")

    home = qb_week.rename(columns={"player_id": "home_qb_id"})
    home = home.rename(columns={c: f"home_{c}" for c in QB_FEATURES})
    away = qb_week.rename(columns={"player_id": "away_qb_id"})
    away = away.rename(columns={c: f"away_{c}" for c in QB_FEATURES})

    out = games.merge(home, on=["home_qb_id", "season", "week"], how="left")
    out = out.merge(away, on=["away_qb_id", "season", "week"], how="left")

    diffs = []
    for col in QB_FEATURES:
        name = f"diff_{col}"
        out[name] = out[f"home_{col}"] - out[f"away_{col}"]
        diffs.append(name)

    # Linear-model interaction: starter's prior quality relative to the team's
    # recent passing environment. CatBoost can additionally learn nonlinearities.
    out["diff_qb_vs_team_pass_epa"] = (
        out["home_qb_career_epa_db"] - out["home_off_pass_epa_r8"]
    ) - (
        out["away_qb_career_epa_db"] - out["away_off_pass_epa_r8"]
    )
    diffs.append("diff_qb_vs_team_pass_epa")

    out["both_qb_rows_matched"] = (
        out["home_qb_prior_dropbacks"].notna() & out["away_qb_prior_dropbacks"].notna()
    ).astype(int)
    return out, diffs


def make_logistic(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer([("num", num, numeric_cols), ("cat", cat, categorical_cols)])
    return Pipeline(
        [
            ("preprocess", prep),
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


def run_backtest(
    games: pd.DataFrame,
    qb_diff_cols: list[str],
    first_test_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_diffs = [
        f"diff_{metric}_r{window}" for metric in BASE_TEAM_METRICS for window in ROLL_WINDOWS
    ]
    categorical = ["roof", "surface", "location", "div_game"]
    football = ["rest_diff", "elo_diff", "elo_home_prob", *team_diffs]
    market_football = ["market_home_prob", "spread_line", "total_line", *football]
    full = [*market_football, *qb_diff_cols]

    completed = games.loc[games["home_win"].notna()].copy()
    folds = []
    predictions = []

    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue
        y_train = train["home_win"].astype(int)
        y_test = test["home_win"].astype(int)

        probs: dict[str, np.ndarray] = {
            "market": test["market_home_prob"].to_numpy(dtype=float)
        }
        sets = {
            "football_logistic": football,
            "market_football_logistic": market_football,
            "oracle_qb_logistic": full,
        }
        for name, numeric_cols in sets.items():
            cols = [*numeric_cols, *categorical]
            model = make_logistic(numeric_cols, categorical)
            model.fit(train[cols], y_train)
            probs[name] = model.predict_proba(test[cols])[:, 1]

        cat_cols = [*full, *categorical]
        x_train = train[cat_cols].copy()
        x_test = test[cat_cols].copy()
        for col in categorical:
            x_train[col] = x_train[col].astype("string").fillna("__MISSING__")
            x_test[col] = x_test[col].astype("string").fillna("__MISSING__")
        cat_idx = [x_train.columns.get_loc(c) for c in categorical]
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
        cat.fit(x_train, y_train, cat_features=cat_idx)
        probs["oracle_qb_catboost"] = cat.predict_proba(x_test)[:, 1]

        out_cols = [
            "game_id", "season", "week", "gameday", "away_team", "home_team",
            "home_win", "home_qb_id", "away_qb_id", "both_qb_rows_matched",
        ]
        for name_col in ("home_qb_name", "away_qb_name"):
            if name_col in test.columns:
                out_cols.append(name_col)
        pred = test[out_cols].copy()

        for name, p in probs.items():
            pred[f"p_home_{name}"] = p
            folds.append({"season": season, "model": name, "games": len(test), **score_probabilities(y_test, p)})
        predictions.append(pred)

        print(
            f"{season}: market={score_probabilities(y_test, probs['market'])['accuracy']:.3f} "
            f"base={score_probabilities(y_test, probs['market_football_logistic'])['accuracy']:.3f} "
            f"qb_logit={score_probabilities(y_test, probs['oracle_qb_logistic'])['accuracy']:.3f} "
            f"qb_cat={score_probabilities(y_test, probs['oracle_qb_catboost'])['accuracy']:.3f}"
        )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["home_win"].astype(int)
    models = [
        "market",
        "football_logistic",
        "market_football_logistic",
        "oracle_qb_logistic",
        "oracle_qb_catboost",
    ]
    rows = []
    for model in models:
        p = predictions[f"p_home_{model}"].to_numpy(dtype=float)
        rows.append({"model": model, "games": len(predictions), **score_probabilities(y, p)})
    return pd.DataFrame(rows).sort_values(["accuracy", "brier"], ascending=[False, True])


def paired_qb_vs_base(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["home_win"].astype(int).to_numpy()
    base_pick = (
        predictions["p_home_market_football_logistic"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    base_correct = (base_pick == y).astype(int)
    rows = []
    for model in ("oracle_qb_logistic", "oracle_qb_catboost"):
        pick = (predictions[f"p_home_{model}"].to_numpy(dtype=float) >= 0.5).astype(int)
        correct = (pick == y).astype(int)
        disagree = pick != base_pick
        rows.append(
            {
                "model": model,
                "games": len(y),
                "accuracy_lift_vs_base": float((correct - base_correct).mean()),
                "net_additional_correct": int((correct - base_correct).sum()),
                "disagreements": int(disagree.sum()),
                "model_accuracy_on_disagreements": float(np.mean(pick[disagree] == y[disagree])) if disagree.any() else np.nan,
                "base_accuracy_on_disagreements": float(np.mean(base_pick[disagree] == y[disagree])) if disagree.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.start_season, args.end_season + 1))

    games = build_model_table(args.start_season, args.end_season)
    qb_week = build_qb_week_table(seasons)
    games, qb_diff_cols = add_oracle_qb_features(games, qb_week)
    predictions, folds = run_backtest(
        games, qb_diff_cols, args.first_test_season, args.end_season
    )
    summary = aggregate(predictions)
    paired = paired_qb_vs_base(predictions)

    predictions.to_csv(args.output_dir / "qb_ceiling_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "qb_ceiling_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "qb_ceiling_summary.csv", index=False)
    paired.to_csv(args.output_dir / "qb_ceiling_paired_vs_base.csv", index=False)

    coverage = float(predictions["both_qb_rows_matched"].mean())
    (args.output_dir / "README.md").write_text(
        "# Phase 2 QB Information Ceiling\n\n"
        "Current-game starter identity is an oracle historical label here; all QB "
        "performance metrics are prior-only. This is a value-of-information test, "
        "not yet a production pregame starter feed.\n\n"
        f"Both historical starter IDs matched weekly QB rows in {coverage:.1%} of tested games.\n",
        encoding="utf-8",
    )

    print("\nQB ceiling summary")
    print(summary.to_string(index=False))
    print("\nQB vs market+football base")
    print(paired.to_string(index=False))
    print(f"\nBoth-QB row coverage: {coverage:.1%}")


if __name__ == "__main__":
    main()
