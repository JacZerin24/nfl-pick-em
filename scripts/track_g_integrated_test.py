"""Track G integration test against the frozen 2019-2025 decision architecture.

The challenger rules are pre-specified before reading the holdout results:
1. score_close_replace: use Track G score distribution in the <52.5% close zone,
   retain the existing two-specialist true-upset consensus elsewhere.
2. close_blend_50_50: average the existing close residual probability and the
   Track G score probability in the close zone, retain consensus elsewhere.
3. close_agreement_gate: override the market in the close zone only when the
   existing residual and Track G agree on the side, retain consensus elsewhere.
4. triple_upset_confirmation: keep the existing close residual but require the
   Track G score model to also favor the dog before a consensus upset is taken.

No rule is selected using 2019-2025 outcomes; all are reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CLOSE_FAVORITE_MAX = 0.525


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--track-g-predictions", type=Path, required=True)
    p.add_argument("--current-integrated", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_g_dynamic_score/integration"))
    p.add_argument("--bootstrap-reps", type=int, default=50000)
    return p.parse_args()


def bootstrap_paired(a: np.ndarray, b: np.ndarray, reps: int, seed: int) -> dict[str, float]:
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


def compare_correct(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5, alternative="two-sided").pvalue) if n else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "disagreements": n, "mcnemar_exact_p": p}


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(args.track_g_predictions)[[
        "game_id", "p_home_score_residual", "p_home_latent_score", "score_margin", "latent_margin"
    ]]
    cur = pd.read_csv(args.current_integrated)
    x = cur.merge(g, on="game_id", how="inner", validate="one_to_one")
    x = x.loc[x["home_win"].notna()].copy().reset_index(drop=True)

    y = x["home_win"].astype(int).to_numpy()
    p_market = x["p_home_market"].to_numpy(float)
    p_resid = x["p_home_market_residual"].to_numpy(float)
    p_score = x["p_home_score_residual"].to_numpy(float)
    market_home = p_market >= 0.5
    residual_home = p_resid >= 0.5
    score_home = p_score >= 0.5
    dog_home = ~market_home
    fav_prob = np.maximum(p_market, 1-p_market)
    close = fav_prob < CLOSE_FAVORITE_MAX
    consensus = x["consensus_upset_call"].fillna(False).astype(bool).to_numpy()

    current_file = x["final_pick_home"].astype(bool).to_numpy()
    current_reconstructed = market_home.copy()
    current_reconstructed[close] = residual_home[close]
    current_reconstructed[consensus] = dog_home[consensus]
    reconstruction_diffs = int(np.sum(current_file != current_reconstructed))

    score_replace = market_home.copy()
    score_replace[close] = score_home[close]
    score_replace[consensus] = dog_home[consensus]

    blend_prob = 0.5 * p_resid + 0.5 * p_score
    blend_home = blend_prob >= 0.5
    close_blend = market_home.copy()
    close_blend[close] = blend_home[close]
    close_blend[consensus] = dog_home[consensus]

    agree = residual_home == score_home
    agreement_gate = market_home.copy()
    agreement_gate[close & agree] = score_home[close & agree]
    agreement_gate[consensus] = dog_home[consensus]

    triple = consensus & (score_home == dog_home)
    triple_confirm = market_home.copy()
    triple_confirm[close] = residual_home[close]
    triple_confirm[triple] = dog_home[triple]

    strategies = {
        "market": market_home,
        "current_frozen": current_file,
        "score_close_replace": score_replace,
        "close_blend_50_50": close_blend,
        "close_agreement_gate": agreement_gate,
        "triple_upset_confirmation": triple_confirm,
    }
    correct = {name: pred == y for name, pred in strategies.items()}
    rows = []
    market_n = int(correct["market"].sum())
    current_n = int(correct["current_frozen"].sum())
    for name, c in correct.items():
        rows.append({
            "strategy": name,
            "games": len(x),
            "correct": int(c.sum()),
            "accuracy": float(c.mean()),
            "net_vs_market": int(c.sum()-market_n),
            "net_vs_current": int(c.sum()-current_n),
        })
    summary = pd.DataFrame(rows).sort_values(["correct", "strategy"], ascending=[False, True])
    summary.to_csv(out / "integration_summary.csv", index=False)

    boot_rows = []
    for i, name in enumerate(["score_close_replace", "close_blend_50_50", "close_agreement_gate", "triple_upset_confirmation"]):
        boot_rows.append({"comparison": f"{name}_vs_current", **bootstrap_paired(correct[name].astype(int), correct["current_frozen"].astype(int), args.bootstrap_reps, 100+i)})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "integration_bootstrap.csv", index=False)

    pair_rows = []
    for name in ["score_close_replace", "close_blend_50_50", "close_agreement_gate", "triple_upset_confirmation"]:
        pair_rows.append({"challenger": name, **compare_correct(correct[name], correct["current_frozen"])})
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(out / "integration_paired_disagreements.csv", index=False)

    season_rows = []
    for season, gseason in x.groupby("season"):
        idx_arr = gseason.index.to_numpy(dtype=int)
        row = {"season": int(season), "games": int(len(idx_arr))}
        for name, c in correct.items():
            row[f"{name}_correct"] = int(c[idx_arr].sum())
        for name in ["score_close_replace", "close_blend_50_50", "close_agreement_gate", "triple_upset_confirmation"]:
            row[f"{name}_net_vs_current"] = int(correct[name][idx_arr].sum()-correct["current_frozen"][idx_arr].sum())
        season_rows.append(row)
    seasons = pd.DataFrame(season_rows)
    seasons.to_csv(out / "integration_by_season.csv", index=False)

    audit_frames = []
    for name in ["score_close_replace", "close_blend_50_50", "close_agreement_gate", "triple_upset_confirmation"]:
        mask = strategies[name] != current_file
        if not mask.any():
            continue
        a = x.loc[mask, [
            "game_id", "season", "week", "away_team", "home_team", "home_win",
            "p_home_market", "p_home_market_residual", "p_home_score_residual",
            "consensus_upset_call", "decision_type",
        ]].copy()
        a.insert(0, "challenger", name)
        a["current_pick_home"] = current_file[mask]
        a["challenger_pick_home"] = strategies[name][mask]
        a["current_correct"] = correct["current_frozen"][mask]
        a["challenger_correct"] = correct[name][mask]
        audit_frames.append(a)
    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    audit.to_csv(out / "integration_changed_decisions.csv", index=False)

    diagnostics = pd.DataFrame([{
        "games": len(x),
        "close_games": int(close.sum()),
        "consensus_upset_calls": int(consensus.sum()),
        "triple_confirmed_upsets": int(triple.sum()),
        "current_reconstruction_differences": reconstruction_diffs,
        "score_vs_residual_close_disagreements": int(np.sum(close & (score_home != residual_home))),
    }])
    diagnostics.to_csv(out / "integration_diagnostics.csv", index=False)

    best_challenger = summary.loc[~summary["strategy"].isin(["market", "current_frozen"])].iloc[0]
    current_row = summary.loc[summary["strategy"].eq("current_frozen")].iloc[0]
    md = [
        "# Track G integration test",
        "",
        f"Current frozen baseline: {int(current_row.correct)}/{int(current_row.games)} = {100*current_row.accuracy:.2f}%.",
        f"Best reported challenger (not post-hoc promoted): {best_challenger.strategy} = {int(best_challenger.correct)}/{int(best_challenger.games)} = {100*best_challenger.accuracy:.2f}% ({int(best_challenger.net_vs_current):+d} vs current).",
        f"Current architecture reconstruction differences: {reconstruction_diffs}.",
        "",
        "All four challenger architectures were pre-specified and are reported regardless of result. No production code is changed.",
    ]
    (out / "integration_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nIntegration summary")
    print(summary.to_string(index=False))
    print("\nPaired bootstrap vs current")
    print(boot.to_string(index=False))
    print("\nExact paired disagreements")
    print(paired.to_string(index=False))
    print("\nSeason robustness")
    print(seasons.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
