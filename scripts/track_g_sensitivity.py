"""Boundary-extension sensitivity for Track G.

Triggered because the first pre-holdout tuning pass selected several grid
boundaries. This is diagnostic and does not erase or replace the first-pass
result. All hyperparameters are still selected without using 2019-2025 outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import track_g_dynamic_score as g

EXT_Q_GRID = (0.05, 0.10, 0.25, 0.75, 1.5, 3.0)
EXT_R_GRID = (64.0, 100.0, 144.0, 196.0)
EXT_RHO_GRID = (0.30, 0.45, 0.55, 0.70, 0.82, 0.90)
EXT_HFA_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
EXT_ALPHA_GRID = (1000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_g_dynamic_score/sensitivity"))
    p.add_argument("--bootstrap-reps", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    g.Q_GRID = EXT_Q_GRID
    g.R_GRID = EXT_R_GRID
    g.RHO_GRID = EXT_RHO_GRID
    g.HFA_GRID = EXT_HFA_GRID
    g.RIDGE_ALPHA_GRID = EXT_ALPHA_GRID

    games = g.load_games(g.START_SEASON, g.END_SEASON)
    state_search_games = games.loc[games["season"] <= g.STATE_TUNE_LAST].copy()
    best_state, state_tuning = g.tune_state_model(state_search_games)
    state_tuning.to_csv(out / "state_tuning_extended.csv", index=False)

    modeled = g.run_state_model(games, best_state)
    modeled = g.add_market_score_features(modeled)
    best_alpha, ridge_tuning = g.tune_ridge(modeled)
    ridge_tuning.to_csv(out / "ridge_tuning_extended.csv", index=False)
    latent_scale, latent_tuning = g.tune_latent_sigma_scale(modeled)
    latent_tuning.to_csv(out / "latent_sigma_tuning.csv", index=False)

    final_train = modeled.loc[modeled["season"] <= g.RIDGE_VALID_LAST].copy()
    hm, am = g.fit_score_models(final_train, best_alpha)
    modeled = g.predict_score_models(modeled, hm, am)

    hm_pre, am_pre = g.fit_score_models(modeled.loc[modeled["season"] <= g.RIDGE_TRAIN_END], best_alpha)
    val_sigma = g.predict_score_models(
        modeled.loc[modeled["season"].between(g.RIDGE_VALID_FIRST, g.RIDGE_VALID_LAST)].copy(), hm_pre, am_pre
    )
    score_sigma = float(np.nanstd(val_sigma["actual_margin"] - val_sigma["score_margin"], ddof=1))
    score_sigma = max(score_sigma, 6.0)
    latent_sd = np.sqrt(np.maximum(modeled["latent_var_home"].to_numpy(float) + modeled["latent_var_away"].to_numpy(float), 1.0))
    modeled["p_home_latent_score"] = norm.cdf(modeled["latent_margin"].to_numpy(float) / (latent_sd * latent_scale))
    modeled["p_home_score_residual"] = norm.cdf(modeled["score_margin"].to_numpy(float) / score_sigma)

    keep = [
        "game_id", "season", "week", "gameday", "away_team", "home_team", "home_score", "away_score", "home_win",
        "market_home_prob", "spread_line", "total_line", "latent_margin", "latent_total", "score_margin", "score_total",
        "p_home_latent_score", "p_home_score_residual", "market_score_available",
    ]
    pred = modeled.loc[modeled["season"] >= g.DEV_FIRST, keep].copy()
    pred.to_csv(out / "track_g_sensitivity_predictions.csv", index=False)

    summary = pd.concat([
        g.summarize_period(modeled, g.DEV_FIRST, g.DEV_LAST, "development_2016_2018"),
        g.summarize_period(modeled, g.HOLDOUT_FIRST, g.HOLDOUT_LAST, "holdout_2019_2025"),
    ], ignore_index=True)
    summary.to_csv(out / "summary_extended.csv", index=False)

    hold = modeled.loc[modeled["season"].between(g.HOLDOUT_FIRST, g.HOLDOUT_LAST) & modeled["home_win"].notna()].copy()
    y = hold["home_win"].astype(int).to_numpy()
    mc = (hold["market_home_prob"].to_numpy(float) >= 0.5) == y
    sc = (hold["p_home_score_residual"].to_numpy(float) >= 0.5) == y
    lc = (hold["p_home_latent_score"].to_numpy(float) >= 0.5) == y
    boot = pd.DataFrame([
        {"comparison": "latent_extended_vs_market", **g.bootstrap_paired(lc.astype(int), mc.astype(int), args.bootstrap_reps, 501)},
        {"comparison": "score_extended_vs_market", **g.bootstrap_paired(sc.astype(int), mc.astype(int), args.bootstrap_reps, 502)},
    ])
    boot.to_csv(out / "bootstrap_extended.csv", index=False)

    by_season = []
    for season, s in hold.groupby("season"):
        yy = s["home_win"].astype(int).to_numpy()
        m = (s["market_home_prob"].to_numpy(float) >= 0.5) == yy
        z = (s["p_home_score_residual"].to_numpy(float) >= 0.5) == yy
        l = (s["p_home_latent_score"].to_numpy(float) >= 0.5) == yy
        by_season.append({"season": int(season), "market_correct": int(m.sum()), "score_correct": int(z.sum()), "latent_correct": int(l.sum()), "score_net_vs_market": int(z.sum()-m.sum()), "latent_net_vs_market": int(l.sum()-m.sum())})
    pd.DataFrame(by_season).to_csv(out / "by_season_extended.csv", index=False)

    config = pd.DataFrame([{
        "state_q": best_state.q, "state_r": best_state.r, "offseason_rho": best_state.offseason_rho,
        "home_field_points": best_state.home_field_points, "ridge_alpha": best_alpha,
        "latent_sigma_scale": latent_scale, "score_margin_sigma": score_sigma,
    }])
    config.to_csv(out / "selected_configuration_extended.csv", index=False)

    market_row = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "market")].iloc[0]
    score_row = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "market_anchored_score_distribution")].iloc[0]
    latent_row = summary.loc[(summary.period == "holdout_2019_2025") & (summary.model == "latent_score_distribution")].iloc[0]
    md = [
        "# Track G boundary-extension sensitivity",
        "",
        f"Selected state: q={best_state.q:g}, r={best_state.r:g}, rho={best_state.offseason_rho:g}, HFA={best_state.home_field_points:g}.",
        f"Selected ridge alpha: {best_alpha:g}.",
        f"Market: {int(market_row.correct)}/{int(market_row.games)} = {100*market_row.accuracy:.2f}%.",
        f"Latent: {int(latent_row.correct)}/{int(latent_row.games)} = {100*latent_row.accuracy:.2f}% ({int(latent_row.correct-market_row.correct):+d}).",
        f"Score residual: {int(score_row.correct)}/{int(score_row.games)} = {100*score_row.accuracy:.2f}% ({int(score_row.correct-market_row.correct):+d}).",
        "",
        "Diagnostic sensitivity only; first-pass Track G results remain part of the research record.",
    ]
    (out / "summary_extended.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nExtended sensitivity summary")
    print(summary.to_string(index=False))
    print("\nSelected extended configuration")
    print(config.to_string(index=False))
    print("\nBootstrap")
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
