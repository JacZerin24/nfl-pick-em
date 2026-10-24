"""Track I: injury burden x opposing-unit strength. Research only.

All team/player-performance inputs are prior-game rolling observations. Injury
labels are final weekly historical reports, NOT auditable as-of news snapshots.
Consequently results are exploratory and may never alter frozen 2026 picks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss

# All hypotheses frozen in code BEFORE examining this track's outcomes.
# Each interaction is asymmetric: the opponent's strength can magnify an
# injured team's vulnerability. Negative means lower values are stronger.
MATCHUPS = (
    ("qb_pressure", "qb", "def_sack_rate", 1),
    ("qb_takeaways", "qb", "def_takeaway_rate", 1),
    ("ol_pressure", "ol", "def_sack_rate", 1),
    ("skill_pass_defense", "skill", "def_pass_epa_allowed", -1),
    ("front_rushing", "front", "off_rush_epa", 1),
    ("front_passing", "front", "off_pass_epa", 1),
    ("db_passing", "db", "off_pass_epa", 1),
)
GROUPS = ("qb", "ol", "skill", "front", "db")


def num(df: pd.DataFrame, name: str, *, missing_zero: bool = False) -> pd.Series:
    if name not in df:
        raise ValueError(f"Missing expected feature column: {name}")
    out = pd.to_numeric(df[name], errors="coerce")
    return out.fillna(0.0) if missing_zero else out


def add_injury_matchup_features(
    games: pd.DataFrame, team_week: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create position main effects and opponent-exposure interaction ablations.

    Difference sign is home advantage; home-vulnerable x away-strong
    contributes negative home edge. Keep market and broad-injury controls fixed.
    """
    required = [f"{g}_severe_backup_gap" for g in GROUPS]
    side = team_week[["season", "week", "team", *required]].copy()
    h = side.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in required}})
    a = side.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in required}})
    out = games.merge(h, on=["season", "week", "home_team"], how="left", validate="many_to_one")
    out = out.merge(a, on=["season", "week", "away_team"], how="left", validate="many_to_one")
    out["home_injury_report_present"] = out[f"home_{required[0]}"].notna().astype(int)
    out["away_injury_report_present"] = out[f"away_{required[0]}"].notna().astype(int)
    out["both_reports_present"] = out["home_injury_report_present"] & out["away_injury_report_present"]
    positions = []
    for g in GROUPS:
        stem = f"{g}_severe_backup_gap"
        out[f"home_{stem}"] = num(out, f"home_{stem}", missing_zero=True)
        out[f"away_{stem}"] = num(out, f"away_{stem}", missing_zero=True)
        col = f"injury_gap_diff_{g}"
        out[col] = out[f"away_{stem}"] - out[f"home_{stem}"]
        positions.append(col)

    interactions = []
    for name, group, stat, direction in MATCHUPS:
        # A positive interaction means the away team's injury exposure is
        # exploited by home strength more than vice versa.
        home_strength = num(out, f"home_{stat}_r8") * direction
        away_strength = num(out, f"away_{stat}_r8") * direction
        home_gap = num(out, f"home_{group}_severe_backup_gap", missing_zero=True)
        away_gap = num(out, f"away_{group}_severe_backup_gap", missing_zero=True)
        col = f"injury_opponent_{name}"
        out[col] = away_gap * home_strength - home_gap * away_strength
        interactions.append(col)
    return out, positions, interactions


def paired_compare(sub: pd.DataFrame, challenger: str, reference: str) -> dict[str, float | int]:
    y = sub["home_win"].to_numpy(dtype=int)
    p = sub[f"p_{challenger}"].to_numpy(float)
    q = sub[f"p_{reference}"].to_numpy(float)
    a = (p >= .5) == y
    b = (q >= .5) == y
    won, lost = int((a & ~b).sum()), int((~a & b).sum())
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(20261008)
    # Cluster by season: accounts more conservatively for shared era effects.
    group = sub.assign(delta=d).groupby("season", sort=True).agg(n=("delta", "size"), gain=("delta", "sum"))
    counts = group["n"].to_numpy()
    gains = group["gain"].to_numpy()
    idx = rng.integers(0, len(group), size=(10000, len(group)))
    lift = 100.0 * gains[idx].sum(axis=1) / counts[idx].sum(axis=1)
    ci = np.quantile(lift, [.025, .975])
    return {
        "games": len(sub), "correct": int(a.sum()), "reference_correct": int(b.sum()),
        "net_correct": won - lost, "lift_pp": 100.0 * float(d.mean()),
        "flip_wins": won, "flip_losses": lost,
        "mcnemar_exact_p": float(binomtest(won, won + lost, .5).pvalue) if won + lost else 1.0,
        "season_bootstrap_lo_pp": float(ci[0]), "season_bootstrap_hi_pp": float(ci[1]),
    }


def run_backtest(
    data: pd.DataFrame, broad: list[str], pos: list[str], interactions: list[str],
    first_test: int, last_test: int,
) -> pd.DataFrame:
    # C is fixed to the prior Track C value, not tuned on the reused holdout.
    from track_c_player_value_injuries import make_ridge, safe_logit

    data = data.loc[data["home_win"].notna()].copy()
    data["market_logit"] = safe_logit(data["market_home_prob"])
    controls = ["market_logit", "home_injury_report_present", "away_injury_report_present", *broad]
    variants = {
        "broad": controls,
        "position": [*controls, *pos],
        "interaction": [*controls, *pos, *interactions],
    }
    out = []
    for year in range(first_test, last_test + 1):
        tr = data.loc[data["season"] < year]
        te = data.loc[data["season"] == year]
        if tr.empty or te.empty:
            continue
        pred = te[["game_id", "season", "week", "home_team", "away_team", "home_win", "both_reports_present", "material_value_injury_game"]].copy()
        pred["p_market"] = te["market_home_prob"].to_numpy(float)
        for name, cols in variants.items():
            model = make_ridge(.08)
            model.fit(tr[cols], tr["home_win"].astype(int))
            pred[f"p_{name}"] = model.predict_proba(te[cols])[:, 1]
        out.append(pred)
        print(f"{year}: {len(te)} games, matched injury reports {int(te['both_reports_present'].sum())}")
    if not out:
        raise RuntimeError("No walk-forward predictions produced")
    return pd.concat(out, ignore_index=True)


def generate_reports(pred: pd.DataFrame, holdout: int, output: Path) -> None:
    scopes = {
        "reused_historical_holdout": pred["season"].ge(holdout),
        "holdout_reported_both": pred["season"].ge(holdout) & pred["both_reports_present"].eq(1),
        "holdout_material_injury": pred["season"].ge(holdout) & pred["material_value_injury_game"].eq(1),
    }
    comparisons = []
    for scope, mask in scopes.items():
        sub = pred.loc[mask]
        if sub.empty:
            continue
        for challenger, reference in (
            ("broad", "market"), ("position", "broad"),
            ("interaction", "position"), ("interaction", "broad"),
            ("interaction", "market"),
        ):
            comparisons.append({"scope": scope, "challenger": challenger, "reference": reference, **paired_compare(sub, challenger, reference)})
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    rows = []
    for year, te in pred.groupby("season", sort=True):
        y = te["home_win"].astype(int).to_numpy()
        for m in ("market", "broad", "position", "interaction"):
            p = te[f"p_{m}"].to_numpy(float)
            rows.append({"season": year, "model": m, "games": len(te),
                         "correct": int(((p >= .5) == y).sum()),
                         "log_loss": float(log_loss(y, p, labels=[0, 1])),
                         "brier": float(brier_score_loss(y, p))})
    pd.DataFrame(rows).to_csv(output / "by_season.csv", index=False)
    pred.to_csv(output / "predictions.csv", index=False)
    summary = pd.DataFrame(comparisons)
    msg = ["# Track I — Exploratory injury × opponent mismatch", "",
           "**DO NOT PROMOTE:** The 2019–2024 evaluation seasons have already been used",
           "during earlier injury research. This is a reused historical diagnostic,",
           "not an untouched confirmatory holdout. Final weekly injury reports and",
           "closing lines are also not independently timestamped decision-time snapshots.", "",
           "All variants are trained in chronological season order; they are compared",
           "on identical games against the market, the prior broad injury model,",
           "and an additive position-gap model. Positive findings require new",
           "prospective injury/inactive snapshots and a separately frozen evaluation.", "",
           "## Paired comparisons", "",
           summary.to_markdown(index=False, floatfmt=".3f"), ""]
    (output / "summary.md").write_text("\n".join(msg), encoding="utf-8")


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--start-season", type=int, default=2012)
    cli.add_argument("--end-season", type=int, default=2024)
    cli.add_argument("--first-test-season", type=int, default=2016)
    cli.add_argument("--holdout-start-season", type=int, default=2019)
    cli.add_argument("--output-dir", type=Path, default=Path("outputs/track_i_injury_matchup"))
    args = cli.parse_args()
    if not (2012 <= args.start_season < args.first_test_season <= args.holdout_start_season <= args.end_season <= 2024):
        raise SystemExit("Require 2012 <= start < test <= holdout <= end <= 2024")
    from phase1_backtest import build_model_table
    from track_c_player_value_injuries import (
        build_team_week_features, load_final_injury_rows, load_snap_history, merge_team_features,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games = build_model_table(args.start_season, args.end_season)
    injury_rows, diag = load_final_injury_rows(args.start_season, args.end_season)
    snap_rows = load_snap_history(list(range(args.start_season, args.end_season + 1)))
    team_week, value_diag, broad_cols, _ = build_team_week_features(injury_rows, snap_rows)
    games, broad_diffs = merge_team_features(games, team_week, broad_cols)
    games, pos, interaction = add_injury_matchup_features(games, team_week)
    pred = run_backtest(games, broad_diffs, pos, interaction, args.first_test_season, args.end_season)
    generate_reports(pred, args.holdout_start_season, args.output_dir)
    pd.DataFrame([{"interaction_hypotheses": len(MATCHUPS), **diag, **value_diag}]).to_csv(
        args.output_dir / "diagnostics.csv", index=False
    )
    (args.output_dir / "features.txt").write_text("\n".join([*broad_diffs, *pos, *interaction]))
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
