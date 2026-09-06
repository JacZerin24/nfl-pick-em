"""Track G: dynamic team-strength and score-distribution research.

Research-only challenger for the frozen 2026 NFL pick'em system.

Protocol is deliberately staged before the final holdout:
- 2009-2011: warmup/history
- 2012-2013: tune dynamic latent offense/defense state parameters
- 2014-2015: tune the market-anchored score-residual ridge
- 2016-2018: clean development/diagnostic period
- 2019-2025: untouched final holdout

The primary Track G candidate predicts home and away points, then derives win
probability from the implied margin distribution. It never uses the target
result to construct that game's pregame state.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import add_market_probability

START_SEASON = 2009
END_SEASON = 2025
STATE_TUNE_FIRST = 2012
STATE_TUNE_LAST = 2013
RIDGE_TRAIN_END = 2013
RIDGE_VALID_FIRST = 2014
RIDGE_VALID_LAST = 2015
DEV_FIRST = 2016
DEV_LAST = 2018
HOLDOUT_FIRST = 2019
HOLDOUT_LAST = 2025
EPS = 1e-6

Q_GRID = (0.25, 0.75, 1.5, 3.0)
R_GRID = (64.0, 100.0, 144.0, 196.0)
RHO_GRID = (0.55, 0.70, 0.82, 0.90)
HFA_GRID = (1.5, 2.0, 2.5, 3.0)
RIDGE_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LATENT_SIGMA_SCALE_GRID = (0.70, 0.85, 1.00, 1.15, 1.30, 1.50)


@dataclass
class TeamState:
    offense: float = 0.0
    defense: float = 0.0
    var_offense: float = 25.0
    var_defense: float = 25.0


@dataclass(frozen=True)
class StateParams:
    q: float
    r: float
    offseason_rho: float
    home_field_points: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=START_SEASON)
    p.add_argument("--end-season", type=int, default=END_SEASON)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_g_dynamic_score"))
    p.add_argument("--bootstrap-reps", type=int, default=50000)
    return p.parse_args()


def load_games(start_season: int, end_season: int) -> pd.DataFrame:
    games = nfl.load_schedules(list(range(start_season, end_season + 1))).to_pandas()
    games = games.loc[games["game_type"].eq("REG")].copy()
    games["gameday"] = pd.to_datetime(games["gameday"])
    games = add_market_probability(games)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
    games["total_line"] = pd.to_numeric(games["total_line"], errors="coerce")
    games["home_rest_num"] = pd.to_numeric(games.get("home_rest"), errors="coerce")
    games["away_rest_num"] = pd.to_numeric(games.get("away_rest"), errors="coerce")
    games["rest_diff"] = games["home_rest_num"] - games["away_rest_num"]
    games["neutral"] = games.get("location", "").astype(str).str.lower().str.contains("neutral").astype(int)
    games["div_game_num"] = pd.to_numeric(games.get("div_game"), errors="coerce").fillna(0.0)
    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]
    games["home_win"] = np.where(
        games["actual_margin"] > 0,
        1.0,
        np.where(games["actual_margin"] < 0, 0.0, np.nan),
    )
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _state(states: dict[str, TeamState], team: str) -> TeamState:
    if team not in states:
        states[team] = TeamState()
    return states[team]


def run_state_model(games: pd.DataFrame, params: StateParams) -> pd.DataFrame:
    states: dict[str, TeamState] = {}
    current_season: int | None = None
    league_points = 22.5
    league_lr = 0.012
    rows: list[dict[str, float | int | str]] = []

    for row in games.itertuples(index=False):
        season = int(row.season)
        if current_season is None:
            current_season = season
        elif season != current_season:
            for s in states.values():
                s.offense *= params.offseason_rho
                s.defense *= params.offseason_rho
                s.var_offense = min(64.0, s.var_offense + 9.0)
                s.var_defense = min(64.0, s.var_defense + 9.0)
            current_season = season

        hs = _state(states, str(row.home_team))
        as_ = _state(states, str(row.away_team))
        hs.var_offense += params.q
        hs.var_defense += params.q
        as_.var_offense += params.q
        as_.var_defense += params.q

        hfa = 0.0 if int(row.neutral) else params.home_field_points
        pred_home = league_points + 0.5 * hfa + hs.offense - as_.defense
        pred_away = league_points - 0.5 * hfa + as_.offense - hs.defense
        var_home = hs.var_offense + as_.var_defense + params.r
        var_away = as_.var_offense + hs.var_defense + params.r

        rows.append(
            {
                "game_id": row.game_id,
                "latent_home_points": pred_home,
                "latent_away_points": pred_away,
                "latent_margin": pred_home - pred_away,
                "latent_total": pred_home + pred_away,
                "latent_var_home": var_home,
                "latent_var_away": var_away,
                "home_off_state": hs.offense,
                "away_off_state": as_.offense,
                "home_def_state": hs.defense,
                "away_def_state": as_.defense,
                "home_off_sd": math.sqrt(max(hs.var_offense, 1e-9)),
                "away_off_sd": math.sqrt(max(as_.var_offense, 1e-9)),
                "home_def_sd": math.sqrt(max(hs.var_defense, 1e-9)),
                "away_def_sd": math.sqrt(max(as_.var_defense, 1e-9)),
                "league_points_prior": league_points,
            }
        )

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        resid_h = float(row.home_score) - pred_home
        s_h = max(var_home, 1e-9)
        k_oh = hs.var_offense / s_h
        k_da = -as_.var_defense / s_h
        old_voh = hs.var_offense
        old_vda = as_.var_defense
        hs.offense += k_oh * resid_h
        as_.defense += k_da * resid_h
        hs.var_offense = max(0.25, old_voh - old_voh * old_voh / s_h)
        as_.var_defense = max(0.25, old_vda - old_vda * old_vda / s_h)

        resid_a = float(row.away_score) - pred_away
        s_a = max(var_away, 1e-9)
        k_oa = as_.var_offense / s_a
        k_dh = -hs.var_defense / s_a
        old_voa = as_.var_offense
        old_vdh = hs.var_defense
        as_.offense += k_oa * resid_a
        hs.defense += k_dh * resid_a
        as_.var_offense = max(0.25, old_voa - old_voa * old_voa / s_a)
        hs.var_defense = max(0.25, old_vdh - old_vdh * old_vdh / s_a)

        observed_mean = 0.5 * (float(row.home_score) + float(row.away_score))
        league_points = (1.0 - league_lr) * league_points + league_lr * observed_mean

    return games.merge(pd.DataFrame(rows), on="game_id", how="left")


def point_rmse(df: pd.DataFrame) -> float:
    mask = df["home_score"].notna() & df["away_score"].notna()
    h = df.loc[mask, "home_score"].to_numpy(float) - df.loc[mask, "latent_home_points"].to_numpy(float)
    a = df.loc[mask, "away_score"].to_numpy(float) - df.loc[mask, "latent_away_points"].to_numpy(float)
    return float(np.sqrt(np.mean(np.r_[h * h, a * a])))


def tune_state_model(games: pd.DataFrame) -> tuple[StateParams, pd.DataFrame]:
    rows = []
    for q, r, rho, hfa in itertools.product(Q_GRID, R_GRID, RHO_GRID, HFA_GRID):
        params = StateParams(q=q, r=r, offseason_rho=rho, home_field_points=hfa)
        pred = run_state_model(games, params)
        dev = pred.loc[pred["season"].between(STATE_TUNE_FIRST, STATE_TUNE_LAST)].copy()
        rmse = point_rmse(dev)
        margin_rmse = float(np.sqrt(np.nanmean((dev["actual_margin"] - dev["latent_margin"]) ** 2)))
        rows.append({"q": q, "r": r, "offseason_rho": rho, "home_field_points": hfa, "point_rmse": rmse, "margin_rmse": margin_rmse})
    table = pd.DataFrame(rows).sort_values(["point_rmse", "margin_rmse"]).reset_index(drop=True)
    best = table.iloc[0]
    return StateParams(float(best.q), float(best.r), float(best.offseason_rho), float(best.home_field_points)), table


SCORE_FEATURES = [
    "latent_home_gap",
    "latent_away_gap",
    "latent_margin_gap",
    "latent_total_gap",
    "home_off_state",
    "away_off_state",
    "home_def_state",
    "away_def_state",
    "home_off_sd",
    "away_off_sd",
    "home_def_sd",
    "away_def_sd",
    "rest_diff",
    "div_game_num",
    "neutral",
]


def add_market_score_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["market_home_points"] = 0.5 * (x["total_line"] + x["spread_line"])
    x["market_away_points"] = 0.5 * (x["total_line"] - x["spread_line"])
    x["latent_home_gap"] = x["latent_home_points"] - x["market_home_points"]
    x["latent_away_gap"] = x["latent_away_points"] - x["market_away_points"]
    x["latent_margin_gap"] = x["latent_margin"] - x["spread_line"]
    x["latent_total_gap"] = x["latent_total"] - x["total_line"]
    x["home_score_resid_market"] = x["home_score"] - x["market_home_points"]
    x["away_score_resid_market"] = x["away_score"] - x["market_away_points"]
    x["market_score_available"] = x["spread_line"].notna() & x["total_line"].notna()
    return x


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def fit_score_models(train: pd.DataFrame, alpha: float) -> tuple[Pipeline, Pipeline]:
    t = train.loc[train["market_score_available"] & train["home_score"].notna() & train["away_score"].notna()].copy()
    h = make_ridge(alpha)
    a = make_ridge(alpha)
    h.fit(t[SCORE_FEATURES], t["home_score_resid_market"])
    a.fit(t[SCORE_FEATURES], t["away_score_resid_market"])
    return h, a


def predict_score_models(df: pd.DataFrame, home_model: Pipeline, away_model: Pipeline) -> pd.DataFrame:
    x = df.copy()
    available = x["market_score_available"].to_numpy(bool)
    home_resid = np.zeros(len(x), dtype=float)
    away_resid = np.zeros(len(x), dtype=float)
    if available.any():
        home_resid[available] = home_model.predict(x.loc[available, SCORE_FEATURES])
        away_resid[available] = away_model.predict(x.loc[available, SCORE_FEATURES])
    x["score_home_points"] = np.where(
        available,
        x["market_home_points"].to_numpy(float) + home_resid,
        x["latent_home_points"].to_numpy(float),
    )
    x["score_away_points"] = np.where(
        available,
        x["market_away_points"].to_numpy(float) + away_resid,
        x["latent_away_points"].to_numpy(float),
    )
    x["score_margin"] = x["score_home_points"] - x["score_away_points"]
    x["score_total"] = x["score_home_points"] + x["score_away_points"]
    return x


def tune_ridge(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    train = df.loc[df["season"] <= RIDGE_TRAIN_END].copy()
    valid = df.loc[df["season"].between(RIDGE_VALID_FIRST, RIDGE_VALID_LAST)].copy()
    rows = []
    for alpha in RIDGE_ALPHA_GRID:
        hm, am = fit_score_models(train, alpha)
        p = predict_score_models(valid, hm, am)
        mask = p["home_win"].notna()
        margin_rmse = float(np.sqrt(np.nanmean((p.loc[mask, "actual_margin"] - p.loc[mask, "score_margin"]) ** 2)))
        point_rmse_val = float(np.sqrt(np.nanmean(np.r_[
            (p.loc[mask, "home_score"] - p.loc[mask, "score_home_points"]) ** 2,
            (p.loc[mask, "away_score"] - p.loc[mask, "score_away_points"]) ** 2,
        ])))
        accuracy = float(np.mean((p.loc[mask, "score_margin"].to_numpy(float) >= 0) == p.loc[mask, "home_win"].astype(int).to_numpy()))
        rows.append({"alpha": alpha, "margin_rmse": margin_rmse, "point_rmse": point_rmse_val, "winner_accuracy": accuracy})
    table = pd.DataFrame(rows).sort_values(["margin_rmse", "point_rmse", "winner_accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def tune_latent_sigma_scale(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    valid = df.loc[df["season"].between(RIDGE_VALID_FIRST, RIDGE_VALID_LAST) & df["home_win"].notna()].copy()
    y = valid["home_win"].astype(int).to_numpy()
    raw_sd = np.sqrt(np.maximum(valid["latent_var_home"].to_numpy(float) + valid["latent_var_away"].to_numpy(float), 1.0))
    rows = []
    for scale in LATENT_SIGMA_SCALE_GRID:
        prob = norm.cdf(valid["latent_margin"].to_numpy(float) / (raw_sd * scale))
        rows.append({"scale": scale, "log_loss": float(log_loss(y, np.clip(prob, EPS, 1-EPS))), "brier": float(brier_score_loss(y, prob))})
    table = pd.DataFrame(rows).sort_values(["log_loss", "brier"]).reset_index(drop=True)
    return float(table.iloc[0]["scale"]), table


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    prob = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pick = prob >= 0.5
    return {
        "games": int(len(y)),
        "correct": int(np.sum(pick == y)),
        "accuracy": float(np.mean(pick == y)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
    }


def summarize_period(df: pd.DataFrame, first: int, last: int, label: str) -> pd.DataFrame:
    x = df.loc[df["season"].between(first, last) & df["home_win"].notna()].copy()
    y = x["home_win"].astype(int).to_numpy()
    rows = []
    for model, col in [
        ("market", "market_home_prob"),
        ("latent_score_distribution", "p_home_latent_score"),
        ("market_anchored_score_distribution", "p_home_score_residual"),
    ]:
        rows.append({"period": label, "model": model, **probability_metrics(y, x[col].to_numpy(float))})
    return pd.DataFrame(rows)


def bootstrap_paired(a: np.ndarray, b: np.ndarray, reps: int, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    left = reps
    while left:
        k = min(1000, left)
        idx = rng.integers(0, n, size=(k, n))
        vals.append((a[idx] - b[idx]).mean(axis=1))
        left -= k
    d = np.concatenate(vals)
    return {
        "lift_accuracy_pp": float(100 * np.mean(a-b)),
        "ci95_low_pp": float(100 * np.quantile(d, 0.025)),
        "ci95_high_pp": float(100 * np.quantile(d, 0.975)),
        "bootstrap_prob_positive": float(np.mean(d > 0)),
    }


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    games = load_games(args.start_season, args.end_season)

    print("Tuning dynamic offense/defense state model on 2012-2013...")
    best_state, state_tuning = tune_state_model(games)
    print("Selected state params:", best_state)
    state_tuning.to_csv(out / "state_tuning.csv", index=False)

    modeled = run_state_model(games, best_state)
    modeled = add_market_score_features(modeled)

    print("Tuning market-anchored point residual ridge on 2014-2015...")
    best_alpha, ridge_tuning = tune_ridge(modeled)
    ridge_tuning.to_csv(out / "ridge_tuning.csv", index=False)
    print("Selected ridge alpha:", best_alpha)

    latent_scale, latent_scale_tuning = tune_latent_sigma_scale(modeled)
    latent_scale_tuning.to_csv(out / "latent_sigma_tuning.csv", index=False)

    final_train = modeled.loc[modeled["season"] <= RIDGE_VALID_LAST].copy()
    hm, am = fit_score_models(final_train, best_alpha)
    modeled = predict_score_models(modeled, hm, am)

    hm_pre, am_pre = fit_score_models(modeled.loc[modeled["season"] <= RIDGE_TRAIN_END], best_alpha)
    val_for_sigma = predict_score_models(
        modeled.loc[modeled["season"].between(RIDGE_VALID_FIRST, RIDGE_VALID_LAST)].copy(),
        hm_pre,
        am_pre,
    )
    score_sigma = float(np.nanstd(val_for_sigma["actual_margin"] - val_for_sigma["score_margin"], ddof=1))
    score_sigma = max(score_sigma, 6.0)

    latent_sd = np.sqrt(np.maximum(modeled["latent_var_home"].to_numpy(float) + modeled["latent_var_away"].to_numpy(float), 1.0))
    modeled["p_home_latent_score"] = norm.cdf(modeled["latent_margin"].to_numpy(float) / (latent_sd * latent_scale))
    modeled["p_home_score_residual"] = norm.cdf(modeled["score_margin"].to_numpy(float) / score_sigma)

    keep = [
        "game_id", "season", "week", "gameday", "away_team", "home_team",
        "home_score", "away_score", "home_win", "market_home_prob", "spread_line", "total_line",
        "latent_home_points", "latent_away_points", "latent_margin", "latent_total",
        "latent_var_home", "latent_var_away", "home_off_state", "away_off_state", "home_def_state", "away_def_state",
        "score_home_points", "score_away_points", "score_margin", "score_total",
        "p_home_latent_score", "p_home_score_residual", "market_score_available",
    ]
    predictions = modeled.loc[modeled["season"] >= DEV_FIRST, keep].copy()
    predictions.to_csv(out / "track_g_predictions.csv", index=False)

    dev_summary = summarize_period(modeled, DEV_FIRST, DEV_LAST, "development_2016_2018")
    hold_summary = summarize_period(modeled, HOLDOUT_FIRST, HOLDOUT_LAST, "holdout_2019_2025")
    summary = pd.concat([dev_summary, hold_summary], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)

    hold = modeled.loc[modeled["season"].between(HOLDOUT_FIRST, HOLDOUT_LAST) & modeled["home_win"].notna()].copy()
    y = hold["home_win"].astype(int).to_numpy()
    market_correct = (hold["market_home_prob"].to_numpy(float) >= 0.5) == y
    latent_correct = (hold["p_home_latent_score"].to_numpy(float) >= 0.5) == y
    score_correct = (hold["p_home_score_residual"].to_numpy(float) >= 0.5) == y
    boot = pd.DataFrame([
        {"comparison": "latent_score_vs_market", **bootstrap_paired(latent_correct.astype(int), market_correct.astype(int), args.bootstrap_reps, 41)},
        {"comparison": "score_distribution_vs_market", **bootstrap_paired(score_correct.astype(int), market_correct.astype(int), args.bootstrap_reps, 42)},
    ])
    boot.to_csv(out / "holdout_bootstrap.csv", index=False)

    season_rows = []
    for season, g in hold.groupby("season"):
        yy = g["home_win"].astype(int).to_numpy()
        mc = (g["market_home_prob"].to_numpy(float) >= 0.5) == yy
        lc = (g["p_home_latent_score"].to_numpy(float) >= 0.5) == yy
        sc = (g["p_home_score_residual"].to_numpy(float) >= 0.5) == yy
        season_rows.append({
            "season": int(season), "games": int(len(g)),
            "market_correct": int(mc.sum()), "latent_correct": int(lc.sum()), "score_correct": int(sc.sum()),
            "latent_net_vs_market": int(lc.sum()-mc.sum()), "score_net_vs_market": int(sc.sum()-mc.sum()),
        })
    pd.DataFrame(season_rows).to_csv(out / "holdout_by_season.csv", index=False)

    disagree = hold.loc[(hold["p_home_score_residual"] >= 0.5) != (hold["market_home_prob"] >= 0.5), [
        "game_id", "season", "week", "away_team", "home_team", "home_win", "market_home_prob", "spread_line", "total_line",
        "latent_margin", "score_margin", "p_home_score_residual",
    ]].copy()
    disagree["market_correct"] = ((disagree["market_home_prob"] >= 0.5).astype(int) == disagree["home_win"].astype(int))
    disagree["score_correct"] = ((disagree["p_home_score_residual"] >= 0.5).astype(int) == disagree["home_win"].astype(int))
    disagree.to_csv(out / "holdout_disagreements_vs_market.csv", index=False)

    config = pd.DataFrame([{
        "state_q": best_state.q, "state_r": best_state.r, "offseason_rho": best_state.offseason_rho,
        "home_field_points": best_state.home_field_points, "ridge_alpha": best_alpha,
        "latent_sigma_scale": latent_scale, "score_margin_sigma": score_sigma,
        "market_score_coverage_holdout": float(hold["market_score_available"].mean()),
    }])
    config.to_csv(out / "selected_configuration.csv", index=False)

    hold_market = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "market")].iloc[0]
    hold_score = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "market_anchored_score_distribution")].iloc[0]
    hold_latent = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "latent_score_distribution")].iloc[0]
    md = [
        "# Track G dynamic strength + score distribution",
        "",
        f"Selected state parameters: q={best_state.q:g}, r={best_state.r:g}, offseason rho={best_state.offseason_rho:g}, HFA={best_state.home_field_points:g} points.",
        f"Selected score-residual ridge alpha: {best_alpha:g}.",
        f"Holdout market-score coverage: {100*float(hold.market_score_available.mean()):.2f}%.",
        "",
        "## 2019-2025 untouched holdout",
        "",
        f"- Market: {int(hold_market.correct)}/{int(hold_market.games)} = {100*hold_market.accuracy:.2f}%",
        f"- Latent score distribution: {int(hold_latent.correct)}/{int(hold_latent.games)} = {100*hold_latent.accuracy:.2f}% ({int(hold_latent.correct-hold_market.correct):+d} vs market)",
        f"- Market-anchored score distribution: {int(hold_score.correct)}/{int(hold_score.games)} = {100*hold_score.accuracy:.2f}% ({int(hold_score.correct-hold_market.correct):+d} vs market)",
        "",
        "## Guardrail",
        "",
        "Research only. This script does not alter the frozen 2026 operational picker.",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nTrack G summary")
    print(summary.to_string(index=False))
    print("\nPaired bootstrap")
    print(boot.to_string(index=False))
    print("\nBy season")
    print(pd.DataFrame(season_rows).to_string(index=False))
    print("\nSelected configuration")
    print(config.to_string(index=False))


if __name__ == "__main__":
    main()
