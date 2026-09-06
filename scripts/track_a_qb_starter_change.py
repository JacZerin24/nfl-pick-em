"""Track A: historical QB starter-change and replacement-value research.

The historical schedule's home_qb_id/away_qb_id identifies who actually started
or played in the completed game. That makes current-game QB identity an ORACLE
label here, not a timestamped pregame expected-starter feed. All QB performance
features are prior-game only. The experiment asks a conservative question:

    Even if we know the eventual starter, do starter-change / replacement-value
    features add stable time-ordered straight-up value beyond the closing market?

If not, future QB work should focus on the timing of late starter news and the
market's reaction rather than generic QB quality. Nothing in this script changes
the frozen 2026 production decision rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import build_model_table, score_probabilities
from phase2_qb_ceiling import QB_FEATURES, add_oracle_qb_features, build_qb_week_table

CHANGE_BASE = [
    "starter_change",
    "change_delta_career_epa_db",
    "change_delta_career_cpoe",
    "change_delta_career_sack_rate",
    "change_delta_career_int_rate",
    "change_current_prior_log_dropbacks",
    "change_current_prior_games",
    "change_inexperienced",
    "change_qb_vs_team_pass_epa",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--bootstrap", type=int, default=30000)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_a_qb_starter_change"))
    return p.parse_args()


def safe_logit(p: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float)
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.20,
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def side_rows(games: pd.DataFrame, side: str) -> pd.DataFrame:
    other = "away" if side == "home" else "home"
    out = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "gameday": games["gameday"],
            "team": games[f"{side}_team"],
            "opponent": games[f"{other}_team"],
            "is_home": int(side == "home"),
            "current_qb_id": games[f"{side}_qb_id"],
            "team_market_prob": games["market_home_prob"] if side == "home" else 1.0 - games["market_home_prob"],
            "team_win": games["home_win"] if side == "home" else 1.0 - games["home_win"],
            "team_off_pass_epa_r8": games[f"{side}_off_pass_epa_r8"],
        }
    )
    for feature in QB_FEATURES:
        out[feature] = games[f"{side}_{feature}"]
    return out


def add_team_change_history(games: pd.DataFrame) -> pd.DataFrame:
    home = side_rows(games, "home")
    away = side_rows(games, "away")
    team = pd.concat([home, away], ignore_index=True)
    team["gameday"] = pd.to_datetime(team["gameday"], errors="coerce")
    team = team.sort_values(["team", "season", "week", "gameday", "game_id"]).reset_index(drop=True)

    by_team_season = team.groupby(["team", "season"], sort=False)
    team["prev_qb_id"] = by_team_season["current_qb_id"].shift(1)
    for feature in QB_FEATURES:
        team[f"prev_{feature}"] = by_team_season[feature].shift(1)

    team["starter_change"] = (
        team["current_qb_id"].notna()
        & team["prev_qb_id"].notna()
        & team["current_qb_id"].ne(team["prev_qb_id"])
    ).astype(int)

    # Replacement deltas compare the incoming starter's pregame history with the
    # outgoing starter's pregame rating from that team's immediately previous game.
    # The outgoing value is therefore safely historical, although slightly stale.
    mappings = {
        "career_epa_db": "qb_career_epa_db",
        "career_cpoe": "qb_career_cpoe",
        "career_sack_rate": "qb_career_sack_rate",
        "career_int_rate": "qb_career_int_rate",
    }
    for short, feature in mappings.items():
        raw_delta = team[feature] - team[f"prev_{feature}"]
        team[f"change_delta_{short}"] = np.where(team["starter_change"].eq(1), raw_delta, 0.0)

    team["change_current_prior_log_dropbacks"] = np.where(
        team["starter_change"].eq(1),
        np.log1p(pd.to_numeric(team["qb_prior_dropbacks"], errors="coerce").clip(lower=0)),
        0.0,
    )
    team["change_current_prior_games"] = np.where(
        team["starter_change"].eq(1),
        pd.to_numeric(team["qb_prior_games"], errors="coerce"),
        0.0,
    )
    team["change_inexperienced"] = (
        team["starter_change"].eq(1)
        & (pd.to_numeric(team["qb_prior_dropbacks"], errors="coerce").fillna(0) < 100)
    ).astype(int)
    team["change_qb_vs_team_pass_epa"] = np.where(
        team["starter_change"].eq(1),
        team["qb_career_epa_db"] - team["team_off_pass_epa_r8"],
        0.0,
    )
    return team


def attach_game_change_features(games: pd.DataFrame, team: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = ["game_id", "is_home", *CHANGE_BASE]
    home = team.loc[team["is_home"].eq(1), cols].drop(columns="is_home").copy()
    away = team.loc[team["is_home"].eq(0), cols].drop(columns="is_home").copy()
    home = home.rename(columns={c: f"home_{c}" for c in CHANGE_BASE})
    away = away.rename(columns={c: f"away_{c}" for c in CHANGE_BASE})
    out = games.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    feature_cols: list[str] = []
    for side in ("home", "away"):
        for base in CHANGE_BASE:
            name = f"{side}_{base}"
            feature_cols.append(name)
            if name not in out.columns:
                out[name] = 0.0

    out["any_starter_change"] = (
        out["home_starter_change"].fillna(0).astype(int)
        | out["away_starter_change"].fillna(0).astype(int)
    ).astype(int)
    out["both_starter_change"] = (
        out["home_starter_change"].fillna(0).astype(int)
        & out["away_starter_change"].fillna(0).astype(int)
    ).astype(int)
    out["qb_change_epa_balance"] = (
        out["home_change_delta_career_epa_db"].fillna(0)
        - out["away_change_delta_career_epa_db"].fillna(0)
    )
    out["qb_change_experience_balance"] = (
        out["home_change_current_prior_log_dropbacks"].fillna(0)
        - out["away_change_current_prior_log_dropbacks"].fillna(0)
    )
    feature_cols.extend(
        [
            "both_starter_change",
            "qb_change_epa_balance",
            "qb_change_experience_balance",
        ]
    )
    out["market_logit"] = safe_logit(out["market_home_prob"])
    return out, ["market_logit", *feature_cols]


def run_backtest(
    games: pd.DataFrame,
    features: list[str],
    first_test_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = games.loc[games["home_win"].notna()].copy()
    predictions: list[pd.DataFrame] = []
    folds: list[dict] = []

    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue

        y_train = train["home_win"].astype(int)
        y_test = test["home_win"].astype(int)
        model_all = make_model()
        model_all.fit(train[features], y_train)
        p_all = model_all.predict_proba(test[features])[:, 1]

        change_train = train.loc[train["any_starter_change"].eq(1)].copy()
        change_test_mask = test["any_starter_change"].eq(1).to_numpy()
        p_change_only = np.full(len(test), np.nan)
        if len(change_train) >= 100 and change_test_mask.any():
            model_change = make_model()
            model_change.fit(change_train[features], change_train["home_win"].astype(int))
            p_change_only[change_test_mask] = model_change.predict_proba(
                test.loc[change_test_mask, features]
            )[:, 1]

        pred_cols = [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "home_win",
            "market_home_prob",
            "home_qb_id",
            "away_qb_id",
            "home_starter_change",
            "away_starter_change",
            "any_starter_change",
            "both_starter_change",
            "home_change_delta_career_epa_db",
            "away_change_delta_career_epa_db",
            "home_change_current_prior_log_dropbacks",
            "away_change_current_prior_log_dropbacks",
        ]
        pred = test[pred_cols].copy()
        pred["p_home_market"] = test["market_home_prob"].to_numpy(dtype=float)
        pred["p_home_change_model"] = p_all
        pred["p_home_change_only_model"] = p_change_only
        predictions.append(pred)

        market_score = score_probabilities(y_test, pred["p_home_market"])
        all_score = score_probabilities(y_test, p_all)
        fold = {
            "season": season,
            "games": len(test),
            "change_games": int(change_test_mask.sum()),
            "market_accuracy": market_score["accuracy"],
            "change_model_accuracy": all_score["accuracy"],
        }
        if change_test_mask.any():
            y_change = y_test.to_numpy()[change_test_mask]
            fold["market_accuracy_change_games"] = score_probabilities(
                y_change, pred.loc[change_test_mask, "p_home_market"]
            )["accuracy"]
            fold["all_model_accuracy_change_games"] = score_probabilities(
                y_change, pred.loc[change_test_mask, "p_home_change_model"]
            )["accuracy"]
            valid = np.isfinite(p_change_only[change_test_mask])
            if valid.any():
                fold["change_only_accuracy_change_games"] = score_probabilities(
                    y_change[valid], p_change_only[change_test_mask][valid]
                )["accuracy"]
        folds.append(fold)
        print(
            f"{season}: games={len(test)} change={int(change_test_mask.sum())} "
            f"market={market_score['accuracy']:.3f} change_model={all_score['accuracy']:.3f}"
        )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def pick_correct(prob: pd.Series, y: pd.Series) -> pd.Series:
    return prob.ge(0.5).astype(int).eq(y.astype(int))


def paired_summary(pred: pd.DataFrame, model_col: str, mask: pd.Series) -> dict:
    frame = pred.loc[mask & pred[model_col].notna()].copy()
    if frame.empty:
        return {"games": 0}
    market_correct = pick_correct(frame["p_home_market"], frame["home_win"]).astype(int)
    model_correct = pick_correct(frame[model_col], frame["home_win"]).astype(int)
    disagree = frame[model_col].ge(0.5).ne(frame["p_home_market"].ge(0.5))
    return {
        "games": len(frame),
        "market_wins": int(market_correct.sum()),
        "model_wins": int(model_correct.sum()),
        "net_correct": int((model_correct - market_correct).sum()),
        "market_accuracy": float(market_correct.mean()),
        "model_accuracy": float(model_correct.mean()),
        "disagreements": int(disagree.sum()),
        "model_wins_on_disagreements": int(model_correct.loc[disagree].sum()),
        "market_wins_on_disagreements": int(market_correct.loc[disagree].sum()),
    }


def bootstrap_lift(pred: pd.DataFrame, model_col: str, mask: pd.Series, n_boot: int) -> dict:
    frame = pred.loc[mask & pred[model_col].notna()].copy()
    if frame.empty:
        return {}
    market_correct = pick_correct(frame["p_home_market"], frame["home_win"]).to_numpy(dtype=float)
    model_correct = pick_correct(frame[model_col], frame["home_win"]).to_numpy(dtype=float)
    diff = model_correct - market_correct
    rng = np.random.default_rng(42)
    chunk = 2000
    means: list[np.ndarray] = []
    for start in range(0, n_boot, chunk):
        n = min(chunk, n_boot - start)
        idx = rng.integers(0, len(diff), size=(n, len(diff)))
        means.append(diff[idx].mean(axis=1))
    boot = np.concatenate(means)
    return {
        "lift": float(diff.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_positive": float(np.mean(boot > 0)),
        "p_nonnegative": float(np.mean(boot >= 0)),
    }


def render_summary(pred: pd.DataFrame, folds: pd.DataFrame, n_boot: int) -> str:
    change = pred["any_starter_change"].eq(1)
    all_stats = paired_summary(pred, "p_home_change_model", pd.Series(True, index=pred.index))
    change_stats = paired_summary(pred, "p_home_change_model", change)
    focused_stats = paired_summary(pred, "p_home_change_only_model", change)
    boot = bootstrap_lift(pred, "p_home_change_only_model", change, n_boot)

    lines = ["# Track A: QB Starter-Change / Replacement-Value Study", ""]
    lines.extend(
        [
            "**Research-only. Current-game historical starter identity is an oracle label; closing market remains the benchmark.**",
            "",
            f"Time-ordered OOS seasons: **{int(pred['season'].min())}-{int(pred['season'].max())}**",
            f"All OOS games: **{len(pred)}**",
            f"Games with at least one team changing starters from its previous game: **{int(change.sum())}** ({change.mean():.1%})",
            "",
            "## All games: market-anchored starter-change model",
            "",
            f"- Market: **{all_stats.get('market_wins', 0)}/{all_stats.get('games', 0)} ({all_stats.get('market_accuracy', float('nan')):.2%})**",
            f"- Change-feature model: **{all_stats.get('model_wins', 0)}/{all_stats.get('games', 0)} ({all_stats.get('model_accuracy', float('nan')):.2%})**",
            f"- Net correct vs market: **{all_stats.get('net_correct', 0):+d}**",
            "",
            "## Starter-change games only",
            "",
            f"- Market: **{focused_stats.get('market_wins', 0)}/{focused_stats.get('games', 0)} ({focused_stats.get('market_accuracy', float('nan')):.2%})**",
            f"- Change-only model: **{focused_stats.get('model_wins', 0)}/{focused_stats.get('games', 0)} ({focused_stats.get('model_accuracy', float('nan')):.2%})**",
            f"- Net correct vs market: **{focused_stats.get('net_correct', 0):+d}**",
            f"- Pick disagreements: **{focused_stats.get('disagreements', 0)}**",
            f"- On disagreements: model **{focused_stats.get('model_wins_on_disagreements', 0)}**, market **{focused_stats.get('market_wins_on_disagreements', 0)}**",
        ]
    )
    if boot:
        lines.extend(
            [
                f"- Paired accuracy lift: **{boot['lift']:+.3%}**",
                f"- {n_boot:,}-sample bootstrap 95% CI: **[{boot['ci_low']:+.3%}, {boot['ci_high']:+.3%}]**",
                f"- P(lift > 0): **{boot['p_positive']:.1%}**",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "This study is a ceiling test for starter-change/replacement-value information against historical closing prices. A positive result would justify building a timestamped expected-starter feed. A null result would not rule out late-news value; it would instead suggest that the exploitable component, if any, is the interval between a surprise change and the market's adjustment. That timing question belongs with the prospective line-movement archive in Track B.",
            "",
            "No result from this experiment is permitted to alter `prospective-v1-frozen-2025` during the 2026 prospective season.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.start_season, args.end_season + 1))

    print("Building historical team/market table...")
    games = build_model_table(args.start_season, args.end_season)
    print("Building prior-only QB history...")
    qb_week = build_qb_week_table(seasons)
    games, _ = add_oracle_qb_features(games, qb_week)
    team = add_team_change_history(games)
    games, features = attach_game_change_features(games, team)

    pred, folds = run_backtest(
        games,
        features,
        args.first_test_season,
        args.end_season,
    )
    pred.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    team.loc[team["starter_change"].eq(1)].to_csv(
        args.output_dir / "starter_change_events.csv", index=False
    )
    summary = render_summary(pred, folds, args.bootstrap)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
