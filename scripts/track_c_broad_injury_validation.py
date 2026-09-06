"""Secondary validation for the pre-specified broad injury ridge in Track C.

Reads Track C OOS predictions and reports paired pick accuracy versus the closing
market. This does not tune or refit the model. It exists because the broad injury
baseline outperformed the richer replacement-aware primary model on the frozen
2019-2024 diagnostic holdout and therefore deserves transparent sensitivity
reporting before the injury track is closed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/track_c_player_value_injuries/predictions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/track_c_player_value_injuries"),
    )
    p.add_argument("--holdout-start", type=int, default=2019)
    p.add_argument("--bootstrap", type=int, default=50000)
    return p.parse_args()


def paired(sub: pd.DataFrame) -> tuple[dict[str, float | int], np.ndarray]:
    y = sub["home_win"].astype(int).to_numpy()
    market = (sub["p_home_market"].to_numpy(float) >= 0.5).astype(int)
    model = (sub["p_home_broad_injury_ridge"].to_numpy(float) >= 0.5).astype(int)
    mc = (market == y).astype(int)
    bc = (model == y).astype(int)
    delta = bc - mc
    disagree = market != model
    return {
        "games": int(len(sub)),
        "market_correct": int(mc.sum()),
        "broad_correct": int(bc.sum()),
        "net_correct_vs_market": int(delta.sum()),
        "accuracy_lift": float(delta.mean()),
        "disagreements": int(disagree.sum()),
        "broad_wins_on_disagreements": int(bc[disagree].sum()) if disagree.any() else 0,
        "market_wins_on_disagreements": int(mc[disagree].sum()) if disagree.any() else 0,
    }, delta.astype(float)


def bootstrap(delta: np.ndarray, n: int, seed: int = 2026) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    chunk = 2000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        idx = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_lift_gt_0": float(np.mean(means > 0)),
        "p_lift_ge_0": float(np.mean(means >= 0)),
    }


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    holdout = pred.loc[pred["season"] >= args.holdout_start].copy()
    material = holdout.loc[holdout["material_value_injury_game"].eq(1)].copy()

    rows = []
    deltas: dict[str, np.ndarray] = {}
    for label, sub in (("holdout_all", holdout), ("holdout_material", material)):
        stats, delta = paired(sub)
        rows.append({"scope": label, **stats, **bootstrap(delta, args.bootstrap)})
        deltas[label] = delta

    # Leave-one-season-out sensitivity checks the extent to which one year drives
    # the observed holdout lift. This is descriptive; it is not a new selection rule.
    loo = []
    for season in sorted(holdout["season"].unique()):
        sub = holdout.loc[holdout["season"] != season]
        stats, _ = paired(sub)
        loo.append({"excluded_season": int(season), **stats})

    results = pd.DataFrame(rows)
    sensitivity = pd.DataFrame(loo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "broad_injury_paired_validation.csv", index=False)
    sensitivity.to_csv(args.output_dir / "broad_injury_leave_one_year_out.csv", index=False)

    a = results.loc[results["scope"].eq("holdout_all")].iloc[0]
    m = results.loc[results["scope"].eq("holdout_material")].iloc[0]
    min_loo = int(sensitivity["net_correct_vs_market"].min())
    max_loo = int(sensitivity["net_correct_vs_market"].max())
    text = f"""# Track C Secondary: Broad Injury Ridge Validation

**Candidate only. This does not change `prospective-v1-frozen-2025`.**

The broad injury ridge was a pre-specified comparison model in Track C. It uses the closing market anchor plus simple injury counts and prior-role-weighted OUT/DOUBTFUL/QUESTIONABLE/DNP/LIMITED burden. It does **not** use the richer core-player/replacement-gap features that failed as the primary Track C hypothesis.

## 2019-2024 holdout

- Games: **{int(a['games'])}**
- Market: **{int(a['market_correct'])}/{int(a['games'])} ({a['market_correct']/a['games']:.2%})**
- Broad injury ridge: **{int(a['broad_correct'])}/{int(a['games'])} ({a['broad_correct']/a['games']:.2%})**
- Net correct vs market: **{int(a['net_correct_vs_market']):+d}**
- Disagreements: **{int(a['disagreements'])}**; broad/market wins: **{int(a['broad_wins_on_disagreements'])}/{int(a['market_wins_on_disagreements'])}**
- Paired lift: **{a['accuracy_lift']:+.3%}**
- 50,000-sample bootstrap 95% CI: **[{a['ci_low']:+.3%}, {a['ci_high']:+.3%}]**
- P(lift > 0): **{a['p_lift_gt_0']:.1%}**

## Material-injury subset

- Games: **{int(m['games'])}**
- Net correct vs market: **{int(m['net_correct_vs_market']):+d}**
- Disagreements: **{int(m['disagreements'])}**; broad/market wins: **{int(m['broad_wins_on_disagreements'])}/{int(m['market_wins_on_disagreements'])}**
- Paired lift: **{m['accuracy_lift']:+.3%}**
- Bootstrap 95% CI: **[{m['ci_low']:+.3%}, {m['ci_high']:+.3%}]**
- P(lift > 0): **{m['p_lift_gt_0']:.1%}**

## Sensitivity

Leaving out any one holdout season leaves the net advantage between **{min_loo:+d} and {max_loo:+d}** correct picks versus market.

## Interpretation

This is more encouraging than the replacement-aware primary model, but it remains a **secondary candidate**, not a production result. The project evaluated multiple injury formulations, the historical feed is final-week injury-report data rather than timestamped late inactive news, and the source ends after 2024. The correct next use is context/research and prospective confirmation, not a silent change to the frozen 2026 decision rule.
"""
    (args.output_dir / "broad_injury_validation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
