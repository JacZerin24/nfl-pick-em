"""Prospective NFL straight-up pick'em generator.

This is the live implementation of the frozen research architecture:

1. Market is the default anchor.
2. If the market favorite is <52.5%, use the market-anchored residual model.
3. If the market favorite is 52.5%-<80%, call the underdog ONLY when the
   frozen cross-specialist pair independently agrees:
      * matchup elastic-net logistic predicts dog >50%
      * explosive/variance CatBoost predicts dog >50%
4. Otherwise follow the market favorite.

The consensus pairing was selected on 2016-2018 and validated on 2019-2025.
The decision thresholds/pairing are intentionally frozen for prospective 2026
tracking. New context (weather, injuries, starters) may be logged separately but
must not silently alter these picks without a new validated model version.

Important live-feature guardrail
--------------------------------
Historical model tables attach rolling features to completed-game rows. Future
games therefore need an explicit AS-OF builder. This script constructs each
team's r4/r8 state using only regular-season games completed before the target
game date, preventing future-game leakage and avoiding missing rolling values.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from phase1_backtest import (
    BASE_TEAM_METRICS,
    ROLL_WINDOWS,
    add_market_probability,
    build_model_table,
    build_team_game_stats,
    walk_forward_backtest,
)
from phase1_market_residual import fit_residual, predict_residual, tune_penalty
from phase2_upset_specialist import build_upset_table
from phase2_upset_variance import (
    VAR_METRICS,
    build_variance_game_stats,
    rolling_variance,
)

CLOSE_FAVORITE_MAX = 0.525
UPSET_FAVORITE_MAX = 0.80
FROZEN_UPSET_PAIRING = "matchup_logistic__variance_catboost"
HISTORICAL_OOF_FIRST_SEASON = 2016
HISTORICAL_META_END_SEASON = 2025


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/live"))
    return p.parse_args()


def base_feature_lists() -> tuple[list[str], list[str]]:
    diffs = [
        f"diff_{metric}_r{window}"
        for metric in BASE_TEAM_METRICS
        for window in ROLL_WINDOWS
    ]
    numeric = [
        "market_home_prob",
        "spread_line",
        "total_line",
        "rest_diff",
        "elo_diff",
        "elo_home_prob",
        *diffs,
    ]
    categorical = ["roof", "surface", "location", "div_game"]
    return numeric, categorical


def make_base_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )
    return Pipeline(
        [
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


def make_base_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
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


def make_matchup_logistic() -> Pipeline:
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


def make_variance_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=600,
        depth=5,
        learning_rate=0.025,
        loss_function="Logloss",
        random_seed=42,
        l2_leaf_reg=15.0,
        random_strength=0.8,
        verbose=False,
        allow_writing_files=False,
    )


def latest_state(
    history: pd.DataFrame,
    team: str,
    target_date: pd.Timestamp,
    metrics: tuple[str, ...] | list[str],
    windows: tuple[int, ...] | list[int],
) -> dict[str, float]:
    """Return trailing regular-season state strictly before target_date."""
    h = history.loc[
        history["team"].eq(team) & (history["gameday"] < target_date)
    ].sort_values(["gameday", "game_id"])
    out: dict[str, float] = {}
    for metric in metrics:
        for window in windows:
            values = pd.to_numeric(h[metric], errors="coerce").dropna().tail(window)
            out[f"{metric}_r{window}"] = (
                float(values.mean()) if len(values) >= 2 else np.nan
            )
    return out


def actual_team_history(
    team_games: pd.DataFrame,
    regular_schedule: pd.DataFrame,
) -> pd.DataFrame:
    context = regular_schedule[["game_id", "gameday", "season", "week"]].drop_duplicates(
        "game_id"
    )
    h = team_games.merge(context, on="game_id", how="inner")
    h["gameday"] = pd.to_datetime(h["gameday"])
    return h.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def populate_live_base_rolls(
    targets: pd.DataFrame,
    base_history: pd.DataFrame,
) -> pd.DataFrame:
    x = targets.copy()
    roll_cols = [
        f"{metric}_r{window}"
        for metric in BASE_TEAM_METRICS
        for window in ROLL_WINDOWS
    ]
    for col in roll_cols:
        x[f"home_{col}"] = np.nan
        x[f"away_{col}"] = np.nan

    for idx, row in x.iterrows():
        target_date = pd.Timestamp(row["gameday"])
        home_state = latest_state(
            base_history, row["home_team"], target_date, BASE_TEAM_METRICS, ROLL_WINDOWS
        )
        away_state = latest_state(
            base_history, row["away_team"], target_date, BASE_TEAM_METRICS, ROLL_WINDOWS
        )
        for col in roll_cols:
            x.at[idx, f"home_{col}"] = home_state.get(col, np.nan)
            x.at[idx, f"away_{col}"] = away_state.get(col, np.nan)
            x.at[idx, f"diff_{col}"] = x.at[idx, f"home_{col}"] - x.at[idx, f"away_{col}"]
    return x


def add_matchup_live_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mirror phase2_upset_specialist feature construction without requiring outcome."""
    x = data.copy()
    home_market = x["market_home_prob"].to_numpy(float)
    fav_home = home_market >= 0.5
    dog_home = ~fav_home
    x["market_fav_prob"] = np.maximum(home_market, 1 - home_market)
    x["market_dog_prob"] = 1 - x["market_fav_prob"]
    x["dog_is_home"] = dog_home.astype(int)
    x["dog_rest_adv"] = np.where(dog_home, x["rest_diff"], -x["rest_diff"])
    x["dog_elo_adv"] = np.where(dog_home, x["elo_diff"], -x["elo_diff"])
    x["total_line_num"] = pd.to_numeric(x["total_line"], errors="coerce")

    features = [
        "market_dog_prob",
        "market_fav_prob",
        "dog_is_home",
        "dog_rest_adv",
        "dog_elo_adv",
        "total_line_num",
    ]

    def oriented(stem: str) -> tuple[np.ndarray, np.ndarray]:
        home = pd.to_numeric(x[f"home_{stem}"], errors="coerce").to_numpy(float)
        away = pd.to_numeric(x[f"away_{stem}"], errors="coerce").to_numpy(float)
        return np.where(dog_home, home, away), np.where(dog_home, away, home)

    for metric in BASE_TEAM_METRICS:
        dog_by_window: dict[int, np.ndarray] = {}
        fav_by_window: dict[int, np.ndarray] = {}
        for window in ROLL_WINDOWS:
            dog, fav = oriented(f"{metric}_r{window}")
            dog_by_window[window] = dog
            fav_by_window[window] = fav
            name = f"dog_minus_fav_{metric}_r{window}"
            x[name] = dog - fav
            features.append(name)
        trend = f"trend_diff_{metric}_r4_vs_r8"
        x[trend] = (dog_by_window[4] - dog_by_window[8]) - (
            fav_by_window[4] - fav_by_window[8]
        )
        features.append(trend)

    for window in ROLL_WINDOWS:
        dog_pass_off, fav_pass_off = oriented(f"off_pass_epa_r{window}")
        dog_rush_off, fav_rush_off = oriented(f"off_rush_epa_r{window}")
        dog_success_off, fav_success_off = oriented(f"off_success_r{window}")
        dog_sack_off, fav_sack_off = oriented(f"off_sack_rate_r{window}")
        dog_to_off, fav_to_off = oriented(f"off_turnover_rate_r{window}")
        dog_pass_def, fav_pass_def = oriented(f"def_pass_epa_allowed_r{window}")
        dog_rush_def, fav_rush_def = oriented(f"def_rush_epa_allowed_r{window}")
        dog_success_def, fav_success_def = oriented(f"def_success_allowed_r{window}")
        dog_sack_def, fav_sack_def = oriented(f"def_sack_rate_r{window}")
        dog_takeaway, fav_takeaway = oriented(f"def_takeaway_rate_r{window}")

        values = {
            f"pass_matchup_edge_r{window}": (dog_pass_off + fav_pass_def)
            - (fav_pass_off + dog_pass_def),
            f"rush_matchup_edge_r{window}": (dog_rush_off + fav_rush_def)
            - (fav_rush_off + dog_rush_def),
            f"success_matchup_edge_r{window}": (dog_success_off + fav_success_def)
            - (fav_success_off + dog_success_def),
            f"pressure_matchup_edge_r{window}": (dog_sack_def + fav_sack_off)
            - (fav_sack_def + dog_sack_off),
            f"turnover_pressure_edge_r{window}": (dog_takeaway + fav_to_off)
            - (fav_takeaway + dog_to_off),
        }
        for name, value in values.items():
            x[name] = value
            features.append(name)

    return x, features


def build_variance_training(
    base_data: pd.DataFrame,
    variance_rolls: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    cols = [f"{m}_r{w}" for m in VAR_METRICS for w in ROLL_WINDOWS]
    home = variance_rolls[["game_id", "team", *cols]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in cols}}
    )
    away = variance_rolls[["game_id", "team", *cols]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in cols}}
    )
    x = base_data.merge(home, on=["game_id", "home_team"], how="left").merge(
        away, on=["game_id", "away_team"], how="left"
    )
    x = x.loc[x["home_win"].notna()].copy()
    return add_variance_oriented_features(x, require_outcome=True)


def populate_live_variance_rolls(
    targets: pd.DataFrame,
    variance_history: pd.DataFrame,
) -> pd.DataFrame:
    x = targets.copy()
    cols = [f"{m}_r{w}" for m in VAR_METRICS for w in ROLL_WINDOWS]
    for col in cols:
        x[f"home_{col}"] = np.nan
        x[f"away_{col}"] = np.nan

    for idx, row in x.iterrows():
        target_date = pd.Timestamp(row["gameday"])
        home_state = latest_state(
            variance_history, row["home_team"], target_date, VAR_METRICS, ROLL_WINDOWS
        )
        away_state = latest_state(
            variance_history, row["away_team"], target_date, VAR_METRICS, ROLL_WINDOWS
        )
        for col in cols:
            x.at[idx, f"home_{col}"] = home_state.get(col, np.nan)
            x.at[idx, f"away_{col}"] = away_state.get(col, np.nan)
    return x


def add_variance_oriented_features(
    data: pd.DataFrame,
    require_outcome: bool,
) -> tuple[pd.DataFrame, list[str]]:
    x = data.copy()
    hp = x["market_home_prob"].to_numpy(float)
    fav_home = hp >= 0.5
    dog_home = ~fav_home
    x["market_fav_prob"] = np.maximum(hp, 1 - hp)
    x["market_dog_prob"] = 1 - x["market_fav_prob"]
    x["dog_is_home"] = dog_home.astype(int)
    if require_outcome:
        x["dog_win"] = np.where(
            fav_home,
            x["home_win"].astype(int).to_numpy() == 0,
            x["home_win"].astype(int).to_numpy() == 1,
        ).astype(int)

    def orient(stem: str) -> tuple[np.ndarray, np.ndarray]:
        h = pd.to_numeric(x[f"home_{stem}"], errors="coerce").to_numpy(float)
        a = pd.to_numeric(x[f"away_{stem}"], errors="coerce").to_numpy(float)
        return np.where(dog_home, h, a), np.where(dog_home, a, h)

    features = ["market_dog_prob", "market_fav_prob", "dog_is_home"]
    vals: dict[tuple[str, int, str], np.ndarray] = {}
    for metric in VAR_METRICS:
        for window in ROLL_WINDOWS:
            dog, fav = orient(f"{metric}_r{window}")
            vals[(metric, window, "dog")] = dog
            vals[(metric, window, "fav")] = fav
            name = f"dog_minus_fav_{metric}_r{window}"
            x[name] = dog - fav
            features.append(name)
        trend = f"trend_diff_{metric}_r4_vs_r8"
        x[trend] = (vals[(metric, 4, "dog")] - vals[(metric, 8, "dog")]) - (
            vals[(metric, 4, "fav")] - vals[(metric, 8, "fav")]
        )
        features.append(trend)

    for w in ROLL_WINDOWS:
        matchup = {
            f"expl_pass_matchup_r{w}": (
                vals[("off_expl_pass_rate", w, "dog")]
                + vals[("def_expl_pass_allowed", w, "fav")]
            )
            - (
                vals[("off_expl_pass_rate", w, "fav")]
                + vals[("def_expl_pass_allowed", w, "dog")]
            ),
            f"expl_rush_matchup_r{w}": (
                vals[("off_expl_rush_rate", w, "dog")]
                + vals[("def_expl_rush_allowed", w, "fav")]
            )
            - (
                vals[("off_expl_rush_rate", w, "fav")]
                + vals[("def_expl_rush_allowed", w, "dog")]
            ),
            f"big_play_matchup_r{w}": (
                vals[("off_big_epa_rate", w, "dog")]
                + vals[("def_big_epa_allowed", w, "fav")]
            )
            - (
                vals[("off_big_epa_rate", w, "fav")]
                + vals[("def_big_epa_allowed", w, "dog")]
            ),
            f"volatility_matchup_r{w}": (
                vals[("off_epa_std", w, "dog")]
                + vals[("def_epa_std_allowed", w, "fav")]
            )
            - (
                vals[("off_epa_std", w, "fav")]
                + vals[("def_epa_std_allowed", w, "dog")]
            ),
            f"special_teams_edge_r{w}": vals[("st_epa_mean", w, "dog")]
            - vals[("st_epa_mean", w, "fav")],
        }
        for name, value in matchup.items():
            x[name] = value
            features.append(name)
    return x, features


def fit_base_models(train: pd.DataFrame):
    numeric, categorical = base_feature_lists()
    features = [*numeric, *categorical]
    y = train["home_win"].astype(int)

    logistic = make_base_logistic(numeric, categorical)
    logistic.fit(train[features], y)

    cat_train = train[features].copy()
    for col in categorical:
        cat_train[col] = cat_train[col].astype("string").fillna("__MISSING__")
    cat_indices = [cat_train.columns.get_loc(c) for c in categorical]
    cat = make_base_catboost()
    cat.fit(cat_train, y, cat_features=cat_indices)
    return logistic, cat, features, categorical


def base_predict(
    logistic: Pipeline,
    cat: CatBoostClassifier,
    features: list[str],
    categorical: list[str],
    target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    p_log = logistic.predict_proba(target[features])[:, 1]
    cat_target = target[features].copy()
    for col in categorical:
        cat_target[col] = cat_target[col].astype("string").fillna("__MISSING__")
    p_cat = cat.predict_proba(cat_target)[:, 1]
    return p_log, p_cat


def confidence_label(row: pd.Series) -> str:
    if row["decision_type"] == "TRUE_UPSET_CONSENSUS":
        return "UPSET_CONSENSUS"
    if row["market_fav_prob"] < CLOSE_FAVORITE_MAX:
        return "TOSSUP"
    if row["market_fav_prob"] >= 0.70:
        return "HIGH_MARKET"
    if row["market_fav_prob"] >= 0.60:
        return "MEDIUM_MARKET"
    return "LOW_MARKET"


def render_markdown(picks: pd.DataFrame, snapshot_utc: str, season: int, week: int) -> str:
    lines = [
        f"# NFL Pick'em — {season} Week {week}",
        "",
        f"Snapshot UTC: `{snapshot_utc}`",
        "",
        "Frozen prospective rules: market anchor; residual only for <52.5% tossups; "
        "true underdog only when matchup-logistic and variance-CatBoost agree; otherwise favorite.",
        "",
        "| Game | Market | Matchup dog | Variance dog | Pick | Decision |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in picks.itertuples(index=False):
        game = f"{row.away_team} @ {row.home_team}"
        market = f"{row.market_pick} {100*row.market_fav_prob:.1f}%"
        matchup = f"{100*row.p_dog_matchup_logistic:.1f}%"
        variance = f"{100*row.p_dog_variance_catboost:.1f}%"
        lines.append(
            f"| {game} | {market} | {matchup} | {variance} | **{row.final_pick}** | {row.decision_type} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`TRUE_UPSET_CONSENSUS` is the only validated true-dog override. "
            "`CLOSE_RESIDUAL` means the market was essentially a tossup and the residual chose the other side. "
            "All other picks follow the market favorite.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    snapshot = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    seasons = list(range(args.start_season, args.season + 1))

    print(f"Building live model table {args.start_season}-{args.season}...")
    base = build_model_table(args.start_season, args.season)
    base["gameday"] = pd.to_datetime(base["gameday"])

    schedule = nfl.load_schedules(args.season).to_pandas()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    schedule = add_market_probability(schedule)

    week = args.week
    if week is None:
        future = schedule.loc[schedule["home_score"].isna()]
        if future.empty:
            raise SystemExit(f"No unplayed regular-season games found for {args.season}")
        week = int(pd.to_numeric(future["week"], errors="coerce").dropna().min())

    targets = base.loc[
        base["season"].eq(args.season)
        & base["week"].eq(week)
        & base["home_score"].isna()
    ].copy()
    if targets.empty:
        raise SystemExit(f"No unplayed games found for season={args.season} week={week}")

    # Build actual historical team states strictly from regular-season PBP.
    print("Loading PBP for live as-of team states...")
    pbp = nfl.load_pbp(seasons)
    regular_context = nfl.load_schedules(seasons).to_pandas()
    regular_context["gameday"] = pd.to_datetime(regular_context["gameday"])
    regular_context = regular_context.loc[regular_context["game_type"].eq("REG")].copy()

    base_history = actual_team_history(build_team_game_stats(pbp), regular_context)
    targets = populate_live_base_rolls(targets, base_history)

    completed = base.loc[base["home_win"].notna()].copy()
    base_logistic, base_cat, base_features, categorical = fit_base_models(completed)
    p_log, p_cat = base_predict(
        base_logistic, base_cat, base_features, categorical, targets
    )
    targets["p_home_market"] = targets["market_home_prob"].to_numpy(float)
    targets["p_home_elo"] = targets["elo_home_prob"].to_numpy(float)
    targets["p_home_logistic"] = p_log
    targets["p_home_catboost"] = p_cat

    # Freeze residual meta-learning to historical OOF through 2025 for prospective 2026 use.
    historical_meta = base.loc[base["season"] <= HISTORICAL_META_END_SEASON].copy()
    print("Rebuilding historical OOF predictions for frozen residual meta-model...")
    oof, _ = walk_forward_backtest(
        historical_meta,
        HISTORICAL_OOF_FIRST_SEASON,
        HISTORICAL_META_END_SEASON,
    )
    penalty, _ = tune_penalty(oof)
    theta = fit_residual(oof, penalty)
    targets["p_home_residual"] = predict_residual(targets, theta)

    # Frozen matchup specialist. Training may absorb newly completed games, but
    # architecture/hyperparameters/decision threshold remain fixed.
    matchup_train, matchup_features = build_upset_table(base)
    matchup_train = matchup_train.loc[
        (matchup_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (matchup_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    matchup_live, live_matchup_features = add_matchup_live_features(targets)
    if matchup_features != live_matchup_features:
        raise RuntimeError("Live matchup feature order does not match historical specialist")
    matchup_model = make_matchup_logistic()
    matchup_model.fit(matchup_train[matchup_features], matchup_train["dog_win"].astype(int))
    targets["p_dog_matchup_logistic"] = matchup_model.predict_proba(
        matchup_live[matchup_features]
    )[:, 1]

    # Frozen variance specialist.
    variance_games = build_variance_game_stats(pbp)
    variance_history = actual_team_history(variance_games, regular_context)
    variance_rolls = rolling_variance(variance_games, regular_context)
    variance_train, variance_features = build_variance_training(base, variance_rolls)
    variance_train = variance_train.loc[
        (variance_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (variance_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    variance_live = populate_live_variance_rolls(targets, variance_history)
    variance_live, live_variance_features = add_variance_oriented_features(
        variance_live, require_outcome=False
    )
    if variance_features != live_variance_features:
        raise RuntimeError("Live variance feature order does not match historical specialist")
    variance_model = make_variance_catboost()
    variance_model.fit(variance_train[variance_features], variance_train["dog_win"].astype(int))
    targets["p_dog_variance_catboost"] = variance_model.predict_proba(
        variance_live[variance_features]
    )[:, 1]

    p_market = targets["p_home_market"].to_numpy(float)
    market_home = p_market >= 0.5
    fav_prob = np.maximum(p_market, 1 - p_market)
    dog_home = ~market_home
    residual_home = targets["p_home_residual"].to_numpy(float) >= 0.5
    close = fav_prob < CLOSE_FAVORITE_MAX
    consensus = (
        (fav_prob >= CLOSE_FAVORITE_MAX)
        & (fav_prob < UPSET_FAVORITE_MAX)
        & (targets["p_dog_matchup_logistic"].to_numpy(float) >= 0.5)
        & (targets["p_dog_variance_catboost"].to_numpy(float) >= 0.5)
    )

    final_home = market_home.copy()
    final_home[close] = residual_home[close]
    final_home[consensus] = dog_home[consensus]

    targets["market_fav_prob"] = fav_prob
    targets["market_pick"] = np.where(market_home, targets["home_team"], targets["away_team"])
    targets["market_underdog"] = np.where(market_home, targets["away_team"], targets["home_team"])
    targets["final_pick"] = np.where(final_home, targets["home_team"], targets["away_team"])
    targets["decision_type"] = np.select(
        [consensus, close & (residual_home != market_home), close],
        ["TRUE_UPSET_CONSENSUS", "CLOSE_RESIDUAL", "CLOSE_MARKET_ALIGNED"],
        default="FOLLOW_MARKET",
    )
    targets["confidence_label"] = targets.apply(confidence_label, axis=1)
    targets["snapshot_utc"] = snapshot
    targets["model_version"] = "prospective-v1"
    targets["upset_pairing"] = FROZEN_UPSET_PAIRING
    targets["residual_penalty"] = penalty

    output_cols = [
        "snapshot_utc",
        "model_version",
        "game_id",
        "season",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "away_moneyline",
        "home_moneyline",
        "market_pick",
        "market_underdog",
        "market_fav_prob",
        "p_home_market",
        "p_home_elo",
        "p_home_logistic",
        "p_home_catboost",
        "p_home_residual",
        "p_dog_matchup_logistic",
        "p_dog_variance_catboost",
        "final_pick",
        "decision_type",
        "confidence_label",
        "upset_pairing",
        "residual_penalty",
    ]
    output_cols = [c for c in output_cols if c in targets.columns]
    picks = targets[output_cols].sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)

    run_dir = args.output_dir / f"{args.season}_week_{week}"
    run_dir.mkdir(parents=True, exist_ok=True)
    picks.to_csv(run_dir / "picks.csv", index=False)
    (run_dir / "picks.md").write_text(
        render_markdown(picks, snapshot, args.season, week), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "snapshot_utc": snapshot,
                "model_version": "prospective-v1",
                "season": args.season,
                "week": week,
                "training_games": len(completed),
                "historical_meta_end_season": HISTORICAL_META_END_SEASON,
                "residual_penalty": penalty,
                "theta_intercept": theta[0],
                "theta_elo_delta": theta[1],
                "theta_logistic_delta": theta[2],
                "theta_catboost_delta": theta[3],
                "theta_consensus_delta": theta[4],
                "frozen_upset_pairing": FROZEN_UPSET_PAIRING,
                "close_favorite_max": CLOSE_FAVORITE_MAX,
                "upset_favorite_max": UPSET_FAVORITE_MAX,
            }
        ]
    ).to_csv(run_dir / "model_metadata.csv", index=False)

    print("\nProspective picks")
    print(
        picks[
            [
                "away_team",
                "home_team",
                "market_pick",
                "market_fav_prob",
                "p_dog_matchup_logistic",
                "p_dog_variance_catboost",
                "final_pick",
                "decision_type",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved live snapshot to {run_dir}")


if __name__ == "__main__":
    main()
