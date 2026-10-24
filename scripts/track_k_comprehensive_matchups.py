"""Track K: comprehensive situational efficiency and matchup-ablation research.

Research-only. Uses nflverse historical final/closing market lines; not a
timestamp-accurate early-week replay. Never modifies production pick artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from phase1_backtest import BASE_TEAM_METRICS, ROLL_WINDOWS, build_model_table

# Fixed before examining Track K results. No holdout-specific selection.
RIDGE_PENALTY = 0.1
EXTRA_SITUATIONS = (
    "early_epa", "early_success", "third_epa", "third_success",
    "redzone_epa", "early_pass_rate",
)
ADJUSTED_METRICS = ("off_opponent_relative_epa", "def_opponent_relative_epa")


def aggregate_game_situations(pbp: pl.DataFrame) -> pd.DataFrame:
    required = {
        "game_id", "posteam", "defteam", "epa", "success", "pass", "rush",
        "down", "yardline_100",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")
    plays = (
        pbp.select(sorted(required))
        .filter(
            ((pl.col("pass") == 1) | (pl.col("rush") == 1))
            & pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
        )
        .with_columns(
            pl.col("epa").cast(pl.Float64, strict=False),
            pl.col("success").cast(pl.Float64, strict=False),
            pl.col("pass").fill_null(0).cast(pl.Float64),
            pl.col("down").cast(pl.Int64, strict=False),
            pl.col("yardline_100").cast(pl.Float64, strict=False),
        )
    )
    metrics = {
        "all_epa": pl.col("epa").mean(),
        "early_epa": pl.col("epa").filter(pl.col("down").is_in([1, 2])).mean(),
        "early_success": pl.col("success").filter(pl.col("down").is_in([1, 2])).mean(),
        "third_epa": pl.col("epa").filter(pl.col("down") == 3).mean(),
        "third_success": pl.col("success").filter(pl.col("down") == 3).mean(),
        "redzone_epa": pl.col("epa").filter(
            (pl.col("yardline_100") <= 20) & (pl.col("yardline_100") >= 0)
        ).mean(),
        "early_pass_rate": pl.col("pass").filter(pl.col("down").is_in([1, 2])).mean(),
    }
    off = plays.group_by(["game_id", "posteam"]).agg(
        [expr.alias("off_" + name) for name, expr in metrics.items()]
    ).rename({"posteam": "team"})
    defense = plays.group_by(["game_id", "defteam"]).agg(
        [expr.alias("def_" + name) for name, expr in metrics.items()]
    ).rename({"defteam": "team"})
    return off.join(defense, on=["game_id", "team"], how="inner").to_pandas()


def build_prior_team_features(
    games: pd.DataFrame, per_game: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Use prior opponent's prior-form as a historical expectation.

    At game G: prior opponent strength does not see G. Adjusted team result
    from G enters only the later-game history via an explicit shift(1).
    """
    home = games[[
        "game_id", "gameday", "season", "week", "home_team", "away_team",
        "away_def_epa_allowed_r8", "away_off_epa_r8",
    ]].rename(columns={
        "home_team": "team", "away_team": "opponent",
        "away_def_epa_allowed_r8": "opponent_prior_def_epa_allowed",
        "away_off_epa_r8": "opponent_prior_off_epa",
    })
    away = games[[
        "game_id", "gameday", "season", "week", "home_team", "away_team",
        "home_def_epa_allowed_r8", "home_off_epa_r8",
    ]].rename(columns={
        "away_team": "team", "home_team": "opponent",
        "home_def_epa_allowed_r8": "opponent_prior_def_epa_allowed",
        "home_off_epa_r8": "opponent_prior_off_epa",
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.merge(per_game, on=["game_id", "team"], how="left", validate="one_to_one")
    long["off_opponent_relative_epa"] = (
        long["off_all_epa"] - long["opponent_prior_def_epa_allowed"]
    )
    long["def_opponent_relative_epa"] = (
        long["def_all_epa"] - long["opponent_prior_off_epa"]
    )
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    situations = [f"{side}_{metric}" for side in ("off", "def") for metric in EXTRA_SITUATIONS]
    roll_cols = []
    for source in (*situations, *ADJUSTED_METRICS):
        for window in ROLL_WINDOWS:
            col = f"{source}_r{window}"
            long[col] = long.groupby("team", sort=False)[source].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=2).mean()
            )
            roll_cols.append(col)

    h = long[["game_id", "team", *roll_cols]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in roll_cols}}
    )
    a = long[["game_id", "team", *roll_cols]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in roll_cols}}
    )
    out = games.merge(h, on=["game_id", "home_team"], how="left", validate="one_to_one")
    out = out.merge(a, on=["game_id", "away_team"], how="left", validate="one_to_one")
    situation_diff = []
    adjust_diff = []
    for source in (*situations, *ADJUSTED_METRICS):
        for window in ROLL_WINDOWS:
            name = f"diff_{source}_r{window}"
            out[name] = out[f"home_{source}_r{window}"] - out[f"away_{source}_r{window}"]
            (adjust_diff if source in ADJUSTED_METRICS else situation_diff).append(name)
    return out, situation_diff, adjust_diff


def matchup_interactions(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Products represent opponent-specific exposure not just additive form."""
    x = games.copy()
    pairs = (
        ("pass", "off_pass_epa", "def_pass_epa_allowed", 1),
        ("rush", "off_rush_epa", "def_rush_epa_allowed", 1),
        ("early", "off_early_epa", "def_early_epa", 1),
        ("third", "off_third_epa", "def_third_epa", 1),
        ("redzone", "off_redzone_epa", "def_redzone_epa", 1),
        ("sacks", "off_sack_rate", "def_sack_rate", -1),
    )
    features = []
    for window in ROLL_WINDOWS:
        for name, offensive, defensive, sign in pairs:
            h_off = pd.to_numeric(x[f"home_{offensive}_r{window}"], errors="coerce")
            a_off = pd.to_numeric(x[f"away_{offensive}_r{window}"], errors="coerce")
            h_def = pd.to_numeric(x[f"home_{defensive}_r{window}"], errors="coerce")
            a_def = pd.to_numeric(x[f"away_{defensive}_r{window}"], errors="coerce")
            column = f"matchup_product_{name}_r{window}"
            x[column] = sign * (h_off * a_def - a_off * h_def)
            features.append(column)
    return x, features


def fit_market_residual(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """Market odds logit is an offset; learn only regularized correction."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    scale = StandardScaler()
    x = scale.fit_transform(imp.fit_transform(train[features]))
    z = scale.transform(imp.transform(test[features]))
    y = train["home_win"].to_numpy(dtype=float)
    prior = np.log(
        np.clip(train["market_home_prob"].to_numpy(float), 1e-5, 1 - 1e-5)
        / (1 - np.clip(train["market_home_prob"].to_numpy(float), 1e-5, 1 - 1e-5))
    )
    design = np.column_stack((np.ones(len(x)), x))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = prior + design @ theta
        p = expit(eta)
        loss = float(np.mean(np.logaddexp(0, eta) - y) +
                     RIDGE_PENALTY * np.dot(theta[1:], theta[1:]) / 2)
        grad = design.T @ (p - y) / len(y)
        grad[1:] += RIDGE_PENALTY * theta[1:]
        return loss, grad

    init = np.zeros(design.shape[1])
    if not np.isfinite(design).all() or not np.isfinite(prior).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite input reached penalized residual optimizer")
    result = minimize(objective, init, jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "maxls": 40, "ftol": 1e-10})
    if not result.success:
        # Same objective, alternative numerical solver for collinearity.
        start = result.x if np.isfinite(result.x).all() else init
        alternative = minimize(objective, start, jac=True, method="BFGS",
                               options={"maxiter": 600, "gtol": 1e-6})
        if alternative.success or (
            np.isfinite(alternative.fun) and
            np.max(np.abs(objective(alternative.x)[1])) <= 1e-5
        ):
            result = alternative
        else:
            raise RuntimeError(
                f"Residual optimizer failed: LBFGS={result.message}, "
                f"BFGS={alternative.message}, gradient_max="
                f"{np.max(np.abs(objective(alternative.x)[1])):.4g}"
            )
    market_test = np.clip(test["market_home_prob"].to_numpy(float), 1e-5, 1 - 1e-5)
    prior_test = np.log(market_test / (1 - market_test))
    return expit(prior_test + np.column_stack((np.ones(len(z)), z)) @ result.x)


def compare_paired(df: pd.DataFrame, challenger: str, baseline: str) -> dict:
    y = df["home_win"].to_numpy(int)
    a = (df[f"p_{challenger}"].to_numpy(float) >= 0.5) == y
    b = (df[f"p_{baseline}"].to_numpy(float) >= 0.5) == y
    wins = int((a & ~b).sum())
    losses = int((~a & b).sum())
    rng = np.random.default_rng(20261009)
    gains = df[["season"]].copy()
    gains["adv"] = a.astype(int) - b.astype(int)
    per_year = gains.groupby("season").agg(n=("adv", "size"), delta=("adv", "sum"))
    idx = rng.integers(0, len(per_year), size=(12000, len(per_year)))
    n = per_year["n"].to_numpy()
    delta = per_year["delta"].to_numpy()
    sims = 100 * delta[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {
        "games": len(df), "correct": int(a.sum()), "reference_correct": int(b.sum()),
        "net_correct": wins - losses, "disagreement_wins": wins,
        "disagreement_losses": losses,
        "lift_pp": 100 * float((a.astype(float) - b.astype(float)).mean()),
        "block_bootstrap_lo_pp": float(np.quantile(sims, .025)),
        "block_bootstrap_hi_pp": float(np.quantile(sims, .975)),
        "mcnemar_exact_p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--first-test-season", type=int, default=2016)
    parser.add_argument("--holdout-start-season", type=int, default=2019)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/track_k_comprehensive_matchups"))
    args = parser.parse_args()
    if not (2009 <= args.start_season < args.first_test_season <=
            args.holdout_start_season <= args.end_season <= 2025):
        raise SystemExit("Invalid research seasons")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = build_model_table(args.start_season, args.end_season)
    pbp = nfl.load_pbp(list(range(args.start_season, args.end_season + 1)))
    situations = aggregate_game_situations(pbp)
    games, situ_cols, adj_cols = build_prior_team_features(games, situations)
    games, matchup_cols = matchup_interactions(games)
    core_cols = [
        "spread_line", "total_line", "rest_diff", "elo_diff", "elo_home_prob",
        *[f"diff_{m}_r{w}" for m in BASE_TEAM_METRICS for w in ROLL_WINDOWS],
    ]
    games["neutral_site"] = games["location"].astype(str).str.lower().eq("neutral").astype(int)
    games["divisional_game"] = pd.to_numeric(games["div_game"], errors="coerce")
    core_cols += ["neutral_site", "divisional_game"]
    variations = {
        "core": core_cols,
        "situational": [*core_cols, *situ_cols],
        "opponent_adjusted": [*core_cols, *adj_cols],
        "matchup": [*core_cols, *matchup_cols],
        "integrated": [*core_cols, *situ_cols, *adj_cols, *matchup_cols],
    }
    completed = games.loc[games["home_win"].notna()].copy()
    predicted = []
    for year in range(args.first_test_season, args.end_season + 1):
        train = completed.loc[completed["season"] < year]
        test = completed.loc[completed["season"] == year]
        if train.empty or test.empty:
            continue
        out = test[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_win"]].copy()
        out["p_market"] = test["market_home_prob"].to_numpy(float)
        for name, cols in variations.items():
            out[f"p_{name}"] = fit_market_residual(train, test, cols)
        predicted.append(out)
        print(f"Walk-forward {year}: {len(test)} games", flush=True)
    if not predicted:
        raise RuntimeError("No research predictions generated")
    pred = pd.concat(predicted, ignore_index=True)
    pred.to_csv(args.output_dir / "predictions.csv", index=False)
    metric_rows = []
    for (year, group) in pred.groupby("season"):
        y = group["home_win"].to_numpy(int)
        for m in ("market", *variations):
            p = group[f"p_{m}"].to_numpy(float)
            metric_rows.append({
                "season": year, "model": m, "games": len(group),
                "correct": int(((p >= .5) == y).sum()),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
            })
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "season_metrics.csv", index=False)
    hold = pred.loc[pred["season"].ge(args.holdout_start_season)]
    pairs = [("core", "market"), *[(v, "core") for v in variations if v != "core"],
             ("integrated", "market")]
    comp = pd.DataFrame([
        {"challenger": a, "baseline": b, **compare_paired(hold, a, b)}
        for a, b in pairs
    ])
    comp.to_csv(args.output_dir / "paired_holdout.csv", index=False)
    coverage = [{
        "family": name, "feature_columns": len(cols),
        "mean_missing_fraction": float(games[cols].isna().mean().mean()),
        "all_features_present_fraction": float(games[cols].notna().all(axis=1).mean()),
    } for name, cols in variations.items()]
    pd.DataFrame(coverage).to_csv(args.output_dir / "coverage.csv", index=False)
    pd.DataFrame([(name, f) for name, cols in variations.items() for f in cols],
                 columns=["variant", "feature"]).to_csv(args.output_dir / "feature_registry.csv", index=False)
    report = [
        "# Track K — Expanded situational efficiency / opponent-adjusted matchups", "",
        "**Exploratory historical diagnostics only**: 2019–2025 outcomes have been",
        "reused in previous model-selection and research. These are not untouched",
        "holdout results and cannot establish a new production edge. Likewise,",
        "the nflverse historical market price is closing/near-closing, not a",
        "timestamped earlier live-pick input.", "",
        "Models are regularized logistic corrections to the market probability.",
        f"Ridge penalty fixed at {RIDGE_PENALTY}; expanding-season training.",
        "The 'core' is a new numerical challenger, **not the frozen production model**.",
        "Before promotion, the winning challenger must be tested inside the exact",
        "frozen integrated 2026 architecture and on future pre-kick snapshots.", "",
        "## Reused historical holdout, paired winner accuracy", "",
        comp.to_markdown(index=False, floatfmt=".3f"), "",
        "## Input coverage", "",
        pd.DataFrame(coverage).to_markdown(index=False, floatfmt=".3f"), "",
        "See predictions.csv, season_metrics.csv, and feature_registry.csv.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
