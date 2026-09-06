"""Extended regularization sensitivity for Track E.

The first Track E run selected the largest penalty in its original grid (1000).
That boundary selection is a methodological reason to extend the grid before
accepting the holdout result. This script repeats the pre-declared development
selection with stronger shrinkage values, then reports the untouched 2019-2025
holdout for both main effects and interactions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import track_e_travel_body_clock as e
from phase1_backtest import build_model_table

OUT = Path("outputs/track_e_travel_body_clock")
EXTENDED = (300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


def model_stats(pred: pd.DataFrame, col: str) -> dict[str, float | int]:
    y = pred["home_win"].astype(int).to_numpy()
    market = pred["market_home_prob"].to_numpy(float)
    p = pred[col].to_numpy(float)
    s = e.score(y, p)
    ms = e.score(y, market)
    pick = p >= 0.5; mp = market >= 0.5
    disagree = pick != mp
    boot = e.paired_bootstrap(y, p, market)
    return {
        **s,
        "net_vs_market": int(s["correct"] - ms["correct"]),
        "disagreements": int(disagree.sum()),
        "model_wins_on_disagreements": int(np.sum(disagree & (pick == y))),
        "market_wins_on_disagreements": int(np.sum(disagree & (mp == y))),
        **boot,
    }


def loo(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for season in sorted(pred["season"].unique()):
        sub = pred.loc[pred["season"] != season]
        y = sub["home_win"].astype(int).to_numpy()
        mc = e.score(y, sub["market_home_prob"].to_numpy(float))["correct"]
        cc = e.score(y, sub[col].to_numpy(float))["correct"]
        rows.append({"left_out_season": int(season), "net_vs_market": int(cc - mc)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e.PENALTIES = EXTENDED
    raw = build_model_table(2009, 2025)
    data = e.add_travel_features(raw)
    data = data.loc[data["home_win"].notna() & data["market_home_prob"].notna() & data["kickoff_utc"].notna()].copy()

    main_penalty, main_dev = e.select_penalty(data, e.MAIN_FEATURES)
    int_features = [*e.MAIN_FEATURES, *e.INTERACTIONS]
    int_penalty, int_dev = e.select_penalty(data, int_features)

    main_pred = e.holdout_predictions(data, e.MAIN_FEATURES, main_penalty, "travel_main_ext")
    int_pred = e.holdout_predictions(data, int_features, int_penalty, "travel_interactions_ext")
    pred = main_pred.merge(int_pred[["game_id", "p_home_travel_interactions_ext"]], on="game_id", how="inner")

    main_s = model_stats(pred, "p_home_travel_main_ext")
    int_s = model_stats(pred, "p_home_travel_interactions_ext")
    main_loo = loo(pred, "p_home_travel_main_ext")
    int_loo = loo(pred, "p_home_travel_interactions_ext")

    summary = pd.DataFrame([
        {"model": "travel_main_extended", "selected_penalty": main_penalty, **main_s,
         "loo_min_net": int(main_loo.net_vs_market.min()), "loo_max_net": int(main_loo.net_vs_market.max())},
        {"model": "travel_interactions_extended", "selected_penalty": int_penalty, **int_s,
         "loo_min_net": int(int_loo.net_vs_market.min()), "loo_max_net": int(int_loo.net_vs_market.max())},
    ])
    summary.to_csv(OUT / "extended_sensitivity_summary.csv", index=False)
    main_dev.to_csv(OUT / "extended_development_main.csv", index=False)
    int_dev.to_csv(OUT / "extended_development_interactions.csv", index=False)
    main_loo.assign(model="travel_main_extended").to_csv(OUT / "extended_loo_main.csv", index=False)
    int_loo.assign(model="travel_interactions_extended").to_csv(OUT / "extended_loo_interactions.csv", index=False)
    pred[["game_id", "season", "week", "home_team", "away_team", "home_win", "market_home_prob", "p_home_travel_main_ext", "p_home_travel_interactions_ext"]].to_csv(OUT / "extended_predictions.csv", index=False)

    lines = [
        "# Track E Extended Regularization Sensitivity", "",
        "This rerun was required because the original development search selected the strongest available penalty (1000). The grid was extended before accepting the holdout conclusion.", "",
        f"- Main effects selected penalty: **{main_penalty:g}**; holdout net **{int(main_s['net_vs_market']):+d}** vs market; disagreements **{int(main_s['disagreements'])}**, model/market wins **{int(main_s['model_wins_on_disagreements'])}/{int(main_s['market_wins_on_disagreements'])}**; lift **{main_s['lift_pp']:+.3f} pp**, 95% CI **[{main_s['ci_low_pp']:+.3f}, {main_s['ci_high_pp']:+.3f}]**, P(lift>0) **{100*main_s['p_lift_gt_0']:.1f}%**; LOO range **{int(main_loo.net_vs_market.min()):+d} to {int(main_loo.net_vs_market.max()):+d}**.",
        f"- Interactions selected penalty: **{int_penalty:g}**; holdout net **{int(int_s['net_vs_market']):+d}** vs market; disagreements **{int(int_s['disagreements'])}**, model/market wins **{int(int_s['model_wins_on_disagreements'])}/{int(int_s['market_wins_on_disagreements'])}**; lift **{int_s['lift_pp']:+.3f} pp**, 95% CI **[{int_s['ci_low_pp']:+.3f}, {int_s['ci_high_pp']:+.3f}]**, P(lift>0) **{100*int_s['p_lift_gt_0']:.1f}%**; LOO range **{int(int_loo.net_vs_market.min()):+d} to {int(int_loo.net_vs_market.max()):+d}**.",
    ]
    (OUT / "extended_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nExtended development main")
    print(main_dev.to_string(index=False))
    print("\nExtended development interactions")
    print(int_dev.to_string(index=False))


if __name__ == "__main__":
    main()
