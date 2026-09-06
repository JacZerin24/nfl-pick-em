"""Track F final architecture test.

This is the relevant promotion screen: replace ONLY the matchup leg of the
existing matchup-logistic + variance-CatBoost upset consensus on the untouched
2022-2025 Track F holdout. The 2026 production implementation remains unchanged.

The preceding Track F script writes strictly OOS matchup probabilities. This
script rebuilds the already-frozen variance specialist walk-forward and asks
whether player-enhanced matchup probabilities improve final straight-up picks
when consensus with variance is required.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from phase1_backtest import build_model_table
from phase2_upset_variance import build_variance_game_stats, rolling_variance
from live_pickem import build_variance_training, make_variance_catboost

START_SEASON = 2009
END_SEASON = 2025
HOLDOUT = (2022, 2023, 2024, 2025)
MIN_FAV = 0.525
MAX_FAV = 0.80
OUT = Path("outputs/track_f_player_matchups")


def variance_oos() -> pd.DataFrame:
    print("Building variance specialist OOS predictions for Track F holdout...")
    base = build_model_table(START_SEASON, END_SEASON)
    pbp = nfl.load_pbp(list(range(START_SEASON, END_SEASON + 1)))
    vg = build_variance_game_stats(pbp)
    vr = rolling_variance(vg, base)
    table, features = build_variance_training(base, vr)
    table = table.loc[
        (table["market_fav_prob"] >= MIN_FAV)
        & (table["market_fav_prob"] < MAX_FAV)
    ].copy()

    rows: list[pd.DataFrame] = []
    for season in HOLDOUT:
        tr = table.loc[table["season"] < season].copy()
        te = table.loc[table["season"].eq(season)].copy()
        model = make_variance_catboost()
        model.fit(tr[features], tr["dog_win"].astype(int))
        z = te[["game_id", "season", "week", "dog_win"]].copy()
        z["p_variance"] = model.predict_proba(te[features])[:, 1]
        rows.append(z)
        print(f"variance {season}: games={len(te)} dog_calls={(z.p_variance >= .5).sum()}")
    return pd.concat(rows, ignore_index=True)


def final_metrics(y: np.ndarray, matchup_p: np.ndarray, variance_p: np.ndarray) -> dict[str, float | int]:
    yy = np.asarray(y, int)
    call = (np.asarray(matchup_p, float) >= 0.5) & (np.asarray(variance_p, float) >= 0.5)
    correct = np.where(call, yy == 1, yy == 0)
    return {
        "games": int(len(yy)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "upset_calls": int(call.sum()),
        "upset_wins": int(np.sum(yy[call] == 1)),
        "upset_call_accuracy": float(np.mean(yy[call] == 1)) if call.any() else np.nan,
    }


def paired_bootstrap(a_correct: np.ndarray, b_correct: np.ndarray, n: int = 30000) -> dict[str, float]:
    d = np.asarray(a_correct, float) - np.asarray(b_correct, float)
    rng = np.random.default_rng(42)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, len(d), len(d))
        vals[i] = d[idx].mean()
    return {
        "lift_pp": float(100 * d.mean()),
        "ci_low_pp": float(100 * np.quantile(vals, 0.025)),
        "ci_high_pp": float(100 * np.quantile(vals, 0.975)),
        "p_lift_gt_0": float(np.mean(vals > 0)),
    }


def correctness(y: np.ndarray, matchup_p: np.ndarray, variance_p: np.ndarray) -> np.ndarray:
    call = (np.asarray(matchup_p) >= .5) & (np.asarray(variance_p) >= .5)
    return np.where(call, np.asarray(y) == 1, np.asarray(y) == 0)


def main() -> None:
    pred_path = OUT / "holdout_predictions.csv"
    if not pred_path.exists():
        raise SystemExit("Run track_f_player_matchups.py first; holdout_predictions.csv is missing.")
    pred = pd.read_csv(pred_path)
    var = variance_oos()
    x = pred.merge(var[["game_id", "p_variance"]], on="game_id", how="inner", validate="one_to_one")
    if len(x) != len(pred):
        raise RuntimeError(f"Variance merge lost games: matchup={len(pred)} merged={len(x)}")

    y = x["dog_win"].astype(int).to_numpy()
    pvar = x["p_variance"].to_numpy(float)
    market_correct = int(np.sum(y == 0))

    matchup_cols = {
        "current_consensus": "p_current_full",
        "team_window_consensus": "p_team_window",
        "qb_pressure_consensus": "p_qb_pressure",
        "receiving_coverage_consensus": "p_receiving_coverage",
        "continuity_consensus": "p_continuity",
        "all_player_consensus": "p_all_player",
    }
    rows = [{
        "model": "market",
        "games": len(x),
        "correct": market_correct,
        "accuracy": market_correct / len(x),
        "upset_calls": 0,
        "upset_wins": 0,
        "upset_call_accuracy": np.nan,
        "net_correct_vs_market": 0,
    }]
    for name, col in matchup_cols.items():
        m = final_metrics(y, x[col].to_numpy(float), pvar)
        rows.append({"model": name, **m, "net_correct_vs_market": int(m["correct"] - market_correct)})
    summary = pd.DataFrame(rows).sort_values(["correct", "upset_calls"], ascending=[False, True])

    cur = correctness(y, x["p_current_full"].to_numpy(float), pvar)
    allp = correctness(y, x["p_all_player"].to_numpy(float), pvar)
    boot = paired_bootstrap(allp, cur)

    current_call = (x["p_current_full"].to_numpy(float) >= .5) & (pvar >= .5)
    all_call = (x["p_all_player"].to_numpy(float) >= .5) & (pvar >= .5)
    audit = x.loc[current_call != all_call, [
        "game_id", "season", "week", "gameday", "favorite_team", "underdog_team",
        "dog_win", "market_fav_prob", "p_current_full", "p_all_player", "p_variance",
    ]].copy()
    mask = current_call != all_call
    audit["current_calls_upset"] = current_call[mask]
    audit["all_player_calls_upset"] = all_call[mask]
    audit["current_correct"] = cur[mask]
    audit["all_player_correct"] = allp[mask]

    changes = []
    for kind, m in (
        ("added_by_player", (~current_call) & all_call),
        ("removed_by_player", current_call & (~all_call)),
    ):
        games = int(m.sum())
        player_wins = int(np.sum(y[m] == 1)) if games else 0
        changes.append({
            "change": kind,
            "games": games,
            "underdog_wins": player_wins,
            "underdog_win_rate": player_wins / games if games else np.nan,
        })
    change_summary = pd.DataFrame(changes)

    loo = []
    for excluded in HOLDOUT:
        m = x["season"].to_numpy(int) != excluded
        loo.append({
            "excluded_season": excluded,
            "games": int(m.sum()),
            "all_minus_current": int(allp[m].sum() - cur[m].sum()),
            "all_minus_market": int(allp[m].sum() - np.sum(y[m] == 0)),
            "current_minus_market": int(cur[m].sum() - np.sum(y[m] == 0)),
        })
    loo = pd.DataFrame(loo)

    by_season = []
    for season in HOLDOUT:
        m = x["season"].to_numpy(int) == season
        by_season.append({
            "season": season,
            "games": int(m.sum()),
            "market_correct": int(np.sum(y[m] == 0)),
            "current_correct": int(cur[m].sum()),
            "all_player_correct": int(allp[m].sum()),
            "all_minus_current": int(allp[m].sum() - cur[m].sum()),
        })
    by_season = pd.DataFrame(by_season)

    summary.to_csv(OUT / "consensus_summary.csv", index=False)
    audit.to_csv(OUT / "consensus_disagreements.csv", index=False)
    change_summary.to_csv(OUT / "consensus_change_summary.csv", index=False)
    loo.to_csv(OUT / "consensus_leave_one_season_out.csv", index=False)
    by_season.to_csv(OUT / "consensus_by_season.csv", index=False)
    x.to_csv(OUT / "consensus_predictions.csv", index=False)

    cur_row = summary.loc[summary.model.eq("current_consensus")].iloc[0]
    all_row = summary.loc[summary.model.eq("all_player_consensus")].iloc[0]
    lines = [
        "# Track F final consensus architecture test",
        "",
        "**Research only. No production changes.**",
        "",
        "The frozen variance-CatBoost specialist is held fixed. The only experimental change is replacing the current matchup-logistic leg with the pre-specified all-player matchup leg.",
        "",
        "## Headline",
        "",
        f"- Games: **{len(x)}**",
        f"- Market: **{market_correct}/{len(x)} ({100*market_correct/len(x):.2f}%)**",
        f"- Current frozen-style consensus: **{int(cur_row.correct)}/{len(x)} ({100*cur_row.accuracy:.2f}%)**, {int(cur_row.net_correct_vs_market):+d} vs market",
        f"- All-player consensus: **{int(all_row.correct)}/{len(x)} ({100*all_row.accuracy:.2f}%)**, {int(all_row.net_correct_vs_market):+d} vs market",
        f"- All-player minus current: **{int(all_row.correct-cur_row.correct):+d} correct picks**",
        f"- Paired lift: **{boot['lift_pp']:+.3f} pp**, 95% CI **[{boot['ci_low_pp']:+.3f}, {boot['ci_high_pp']:+.3f}]**, P(lift>0) **{100*boot['p_lift_gt_0']:.1f}%**",
        f"- Consensus decisions changed on **{len(audit)}** games.",
        "",
        "## All pre-specified diagnostics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Added/removed upset calls",
        "",
        change_summary.to_markdown(index=False),
        "",
        "## By season",
        "",
        by_season.to_markdown(index=False),
        "",
        "## Leave one holdout season out",
        "",
        loo.to_markdown(index=False),
        "",
        "Promotion requires the pre-specified all-player consensus to improve convincingly and robustly over the current consensus. Diagnostics from individual feature families are not sufficient to promote a post-hoc winner.",
    ]
    (OUT / "consensus_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nConsensus summary")
    print(summary.to_string(index=False))
    print("\nAll-player vs current consensus bootstrap", boot)
    print("\nChange summary")
    print(change_summary.to_string(index=False))
    print("\nBy season")
    print(by_season.to_string(index=False))
    print("\nLeave one season out")
    print(loo.to_string(index=False))
    print(f"\nConsensus changed on {len(audit)} games")


if __name__ == "__main__":
    main()
