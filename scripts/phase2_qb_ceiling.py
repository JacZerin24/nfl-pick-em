"""Phase 2 quarterback-information ceiling experiment.

Purpose
-------
Measure whether knowing the correct starting quarterback *at kickoff* and using
only that quarterback's PRIOR performance history adds out-of-sample winner
prediction value beyond the Phase 1 market + team-efficiency model.

Important leakage guardrail
---------------------------
The nflverse schedule file's historical home_qb_id / away_qb_id is inferred from
who actually played/started in the completed game. That identity is therefore
NOT a reproducible historical pregame data source by itself.

This script deliberately labels its QB-enhanced models as an ``oracle-starter``
CEILING experiment. All quarterback PERFORMANCE features are shifted and use
only games/weeks before the predicted game, but the current-game starter
identity is treated as if perfectly known at kickoff.

If this ceiling does not add stable value, we should not invest heavily in a
complex live starter/injury pipeline. If it does, the next step is to replace
oracle identity with timestamped expected-starter/depth-chart/injury snapshots.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--first-test-season", type=int, default=2016)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_qb_ceiling"))
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def build_qb_week_table(seasons: list[int]) -> pd.DataFrame:
    """Create one QB-week row and leak-safe prior-history features."""

    print(f"Loading weekly player stats {min(seasons)}-{max(seasons)} for QB history...")
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
        raise RuntimeError(f"Player stats are missing QB columns: {sorted(missing)}")

    qb = stats.loc[stats["season_type"].eq("REG"), list(required)].copy()
    qb["attempts"] = _num(qb["attempts"])
    qb["sacks_suffered"] = _num(qb["sacks_suffered"])
    qb["passing_interceptions"] = _num(qb["passing_interceptions"])
    qb["passing_epa"] = _num(qb["passing_epa"])
    qb["passing_cpoe"] = pd.to_numeric(qb["passing_cpoe"], errors="coerce")
    qb["dropbacks"] = qb["attempts"] + qb["sacks_suffered"]
    qb = qb.loc[qb["dropbacks"] > 0].copy()

    # In the unlikely event a player has multiple rows in a season-week,
    # aggregate them before computing history. CPOE is attempt-weighted.
    qb["cpoe_weighted_num"] = qb["passing_cpoe"].fillna(0.0) * qb["attempts"]
    qb["cpoe_weight"] = np.where(qb["passing_cpoe"].notna(), qb["attempts"], 0.0)

    qb = (
        qb.groupby(["player_id", "season", "week"], as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            sacks_suffered=("sacks_suffered", "sum"),
            passing_interceptions=("passing_interceptions", "sum"),
            passing_epa=("passing_epa", "sum"),
            cpoe_weighted_num=("cpoe_weighted_num", "sum"),
            cpoe_weight=("cpoe_weight", "sum"),
            dropbacks=("dropbacks", "sum"),
        )
        .sort_values(["player_id", "season", "week"])
        .reset_index(drop=True)
    )

    qb["qb_game"] = (qb["dropbacks"] >= 10).astype(float)

    def add_history(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()

        # Career-to-date totals, shifted one appearance so the current game is
        # never used to predict itself.
        prior_db = g["dropbacks"].cumsum().shift(1).fillna(0.0)
        prior_att = g["attempts"].cumsum().shift(1).fillna(0.0)
        prior_epa = g["passing_epa"].cumsum().shift(1).fillna(0.0)
        prior_sacks = g["sacks_suffered"].cumsum().shift(1).fillna(0.0)
        prior_int = g["passing_interceptions"].cumsum().shift(1).fillna(0.0)
        prior_cpoe_num = g["cpoe_weighted_num"].cumsum().shift(1).fillna(0.0)
        prior_cpoe_den = g["cpoe_weight"].cumsum().shift(1).fillna(0.0)

        g["qb_prior_dropbacks"] = prior_db
        g["qb_prior_games"] = g["qb_game"].cumsum().shift(1).fillna(0.0)

        # Shrink low-sample QBs toward neutral league-average priors. These are
        # fixed neutral priors rather than quantities estimated on future data.
        g["qb_career_epa_db"] = prior_epa / (prior_db + 200.0)
        g["qb_career_cpoe"] = prior_cpoe_num / (prior_cpoe_den + 150.0)
        g["qb_career_sack_rate"] = (prior_sacks + 0.065 * 100.0) / (prior_db + 100.0)
        g["qb_career_int_rate"] = (prior_int + 0.025 * 100.0) / (prior_att + 100.0)

        # Recent appearance-level form. Sum numerators/denominators instead of
        # averaging weekly rates so tiny relief appearances cannot dominate.
        shifted_db = g["dropbacks"].shift(1)
        shifted_att = g["attempts"].shift(1)
        recent_db = shifted_db.rolling(QB_RECENT_GAMES, min_periods=2).sum()
        recent_att = shifted_att.rolling(QB_RECENT_GAMES, min_periods=2).sum()
        recent_epa = g["passing_epa"].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()
        recent_sacks = g["sacks_suffered"].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()
        recent_int = (
            g["passing_interceptions"].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()
        )
        recent_cpoe_num = (
            g["cpoe_weighted_num"].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()
        )
        recent_cpoe_den = (
            g["cpoe_weight"].shift(1).rolling(QB_RECENT_GAMES, min_periods=2).sum()
        )

        g["qb_recent_epa_db"] = recent_epa / recent_db
        g["qb_recent_cpoe"] = recent_cpoe_num / recent_cpoe_den
        g["qb_recent_sack_rate"] = recent_sacks / recent_db
        g["qb_recent_int_rate"] = recent_int / recent_att
        return g

    qb = qb.groupby("player_id", group_keys=False).apply(add_history, include_groups=False)
    qb = qb.reset_index(drop=True)

    feature_cols = [
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
    return qb[["player_id", "season", "week", *feature_cols]]


def add_oracle_qb_features(games: pd.DataFrame, qb_week: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    games = games.copy()

    if "home_qb_id" not in games.columns or "away_qb_id" not in games.columns:
        raise RuntimeError("Schedule data does not contain home_qb_id / away_qb_id")

    qb_cols = [c for c in qb_week.columns if c.startswith("qb_")]

    home = qb_week.rename(columns={"player_id": "home_qb_id"})
    home = home.rename(columns={c: f"home_{c}" for c in qb_cols})
    away = qb_week.rename(columns={"player_id": "away_qb_id"})
    away = away.rename(columns={c: f"away_{c}" for c in qb_cols})

    games = games.merge(home, on=["home_qb_id", "season", "week"], how="left")
    games = games.merge(away, on=["away_qb_id", "season", "week"], how="left")

    diff_cols: list[str] = []
    for col in qb_cols:
        name = f"diff_{col}"
        games[name] = games[f"home_{col}"] - games[f"away_{col}"]
        diff_cols.append(name)

    # Explicit interactions for a linear model: how much stronger/weaker is the
    # known current starter's prior passing efficiency than the team's recent
    # passing environment?
    games["home_qb_vs_team_pass_epa"] = (
        games["home_qb_career_epa_db"] - games["home_off_pass_epa_r8"]
    )
    games["away_qb_vs_team_pass_epa"] = (
        games["away_qb_career_epa_db"] - games["away_off_pass_epa_r8"]
    )
    games["diff_qb_vs_team_pass_epa"] = (
        games["home_qb_vs_team_pass_epa"] - games["away_qb_vs_team_pass_epa"]
    )
    diff_cols.append("diff_qb_vs_team_pass_epa")

    games["both_qb_history_available"] = (
        games["home_qb_prior_dropbacks"].notna() & games["away_qb_prior_dropbacks"].notna()
    ).astype(int)
    return games, diff_cols


def logistic_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.15,
                    C=0.5,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_walk_forward(
    games: pd.DataFrame,
    qb_diff_features: list[str],
    first_test_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_diff_features = [
        f"diff_{metric}_r{window}" for metric in BASE_TEAM_METRICS for window in ROLL_WINDOWS
    ]
    categorical = ["roof", "surface", "location", "div_game"]

    football_numeric = ["rest_diff", "elo_diff", "elo_home_prob", *team_diff_features]
    market_football_numeric = [
        "market_home_prob",
        "spread_line",
        "total_line",
        *football_numeric,
    ]
    full_numeric = [*market_football_numeric, *qb_diff_features]

    completed = games.loc[games["home_win"].notna()].copy()
    fold_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue

        y_train = train["home_win"].astype(int)
        y_test = test["home_win"].astype(int)

        candidate_probs: dict[str, np.ndarray] = {
            "market": test["market_home_prob"].to_numpy(dtype=float)
        }

        feature_sets = {
            "football_logistic": football_numeric,
            "market_football_logistic": market_football_numeric,
            "oracle_qb_logistic": full_numeric,
        }
        for model_name, numeric in feature_sets.items():
            cols = [*numeric, *categorical]
            model = logistic_pipeline(numeric, categorical)
            model.fit(train[cols], y_train)
            candidate_probs[model_name] = model.predict_proba(test[cols])[:, 1]

        cat_cols = [*full_numeric, *categorical]
        cat_train = train[cat_cols].copy()
        cat_test = test[cat_cols].copy()
        for col in categorical:
            cat_train[col] = cat_train[col].astype("string").fillna("__MISSING__")
            cat_test[col] = cat_test[col].astype("string").fillna("__MISSING__")
        cat_indices = [cat_train.columns.get_loc(c) for c in categorical]

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
        candidate_probs["oracle_qb_catboost"] = cat.predict_proba(cat_test)[:, 1]

        out_cols = [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "home_win",
            "home_qb_id",
            "away_qb_id",
            "both_qb_history_available",
        ]
        for optional in ("home_qb_name", "away_qb_name"):
            if optional in test.columns:
                out_cols.append(optional)
        fold_pred = test[out_cols].copy()

        for model_name, probability in candidate_probs.items():
            fold_pred[f"p_home_{model_name}"] = probability
            fold_rows.append(
                {
                    "season": season,
                    "model": model_name,
                    "games": len(test),
                    **score_probabilities(y_test, probability),
                }
            )

        prediction_rows.append(fold_pred)
        print(
            f"{season}: market={score_probabilities(y_test, candidate_probs['market'])['accuracy']:.3f} "
            f"base={score_probabilities(y_test, candidate_probs['market_football_logistic'])['accuracy']:.3f} "
            f"qb_logit={score_probabilities(y_test, candidate_probs['oracle_qb_logistic'])['accuracy']:.3f} "
            f"qb_cat={score_probabilities(y_test, candidate_probs['oracle_qb_catboost'])['accuracy']:.3f}"
        )

    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(fold_rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
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


def paired_vs_base(predictions: pd.DataFrame) -> pd.DataFrame:
    """Simple paired descriptive comparison to the same-feature-set base model."""
    y = predictions["home_win"].astype(int).to_numpy()
    base = (predictions["p_home_market_football_logistic"].to_numpy(dtype=float) >= 0.5).astype(int)
    base_correct = (base == y).astype(int)
    rows = []
    for model in ("oracle_qb_logistic", "oracle_qb_catboost"):
        pick = (predictions[f"p_home_{model}"].to_numpy(dtype=float) >= 0.5).astype(int)
        correct = (pick == y).astype(int)
        disagree = pick != base
        rows.append(
            {
                "model": model,
                "games": len(y),
                "accuracy_lift_vs_base": float((correct - base_correct).mean()),
                "net_additional_correct": int((correct - base_correct).sum()),
                "disagreements": int(disagree.sum()),
                "model_accuracy_on_disagreements": (
                    float(np.mean(pick[disagree] == y[disagree])) if disagree.any() else np.nan
                ),
                "base_accuracy_on_disagreements": (
                    float(np.mean(base[disagree] == y[disagree])) if disagree.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.first_test_season <= args.start_season:
        raise SystemExit("--first-test-season must be after --start-season")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = build_model_table(args.start_season, args.end_season)
    seasons = list(range(args.start_season, args.end_season + 1))
    qb_week = build_qb_week_table(seasons)
    games, qb_diff_features = add_oracle_qb_features(games, qb_week)

    predictions, folds = run_walk_forward(
        games,
        qb_diff_features=qb_diff_features,
        first_test_season=args.first_test_season,
        end_season=args.end_season,
    )
    summary = summarize(predictions)
    paired = paired_vs_base(predictions)

    predictions.to_csv(args.output_dir / "qb_ceiling_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "qb_ceiling_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "qb_ceiling_summary.csv", index=False)
    paired.to_csv(args.output_dir / "qb_ceiling_paired_vs_base.csv", index=False)

    coverage = float(predictions["both_qb_history_available"].mean())
    (args.output_dir / "README.md").write_text(
        "# Phase 2 QB Information Ceiling\n\n"
        "This experiment uses the realized historical starter identity as an oracle label, "
        "while every QB performance feature is shifted to prior games only. It measures the "
        "maximum plausible value of adding correct starter identity; it is not yet a valid "
        "historical pregame starter feed.\n\n"
        f"Both starter IDs matched a QB-history row in {coverage:.1%} of evaluated games.\n",
        encoding="utf-8",
    )

    print("\nQB ceiling aggregate results")
    print(summary.to_string(index=False))
    print("\nQB models vs market+football logistic")
    print(paired.to_string(index=False))
    print(f"\nBoth-QB history-row coverage: {coverage:.1%}")
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
