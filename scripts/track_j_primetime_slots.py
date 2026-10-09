"""Track J: Kickoff-slot and historical primetime-style effects (research only).

Labels are schedule/date/time proxies, NOT verified TV-broadcast assignments.
All historical outcome features are explicitly prior-game; market is a
closing-market benchmark, not early-week or final-entry timestamp data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from phase1_backtest import add_market_probability, add_elo

PSEUDO_GAMES = 20.0
PENALTY = 0.1


def slot_labels(games: pd.DataFrame) -> pd.DataFrame:
    x = games.copy()
    day = pd.to_datetime(x["gameday"], errors="coerce")
    clock = x["gametime"].astype(str).str.extract(r"^\s*(\d{1,2}):(\d{2})\s*$")
    h = pd.to_numeric(clock[0], errors="coerce")
    m = pd.to_numeric(clock[1], errors="coerce")
    x["kickoff_minute_et"] = h * 60 + m
    # Invalid times remain OTHER; do not silently classify unknown start times.
    dow = day.dt.dayofweek
    valid = (h.between(0, 23) & m.between(0, 59))
    x["kickoff_minute_et"] = x["kickoff_minute_et"].where(valid)
    pm = x["kickoff_minute_et"].ge(18 * 60 + 30)
    morning = x["kickoff_minute_et"].lt(11 * 60)
    late = x["kickoff_minute_et"].ge(15 * 60)
    x["slot"] = np.select(
        [
            valid & dow.eq(0) & pm,
            valid & dow.eq(3) & pm,
            valid & dow.eq(6) & pm,
            valid & dow.eq(6) & morning,
            valid & dow.eq(6) & late,
            valid & dow.eq(6),
            valid & dow.eq(5),
            valid & dow.eq(4),
        ],
        [
            "MON_NIGHT_PROXY", "THU_NIGHT_PROXY", "SUN_NIGHT_PROXY",
            "SUN_MORNING", "SUN_LATE", "SUN_EARLY", "SATURDAY", "FRIDAY",
        ],
        default="OTHER",
    )
    x["is_primetime_proxy"] = x["slot"].isin(
        ["MON_NIGHT_PROXY", "THU_NIGHT_PROXY", "SUN_NIGHT_PROXY"]
    ).astype(int)
    x["slot_is_missing"] = (~valid).astype(int)
    return x


def add_prior_team_slot_history(games: pd.DataFrame) -> pd.DataFrame:
    """Make strictly prior residuals, across years but within each team/slot.

    Same-day ordering relies on kickoff minute and game ID. Previous completed
    games alone contribute: the current game's surprise is shifted away.
    """
    x = games.copy()
    h = x[["game_id", "season", "week", "gameday", "kickoff_minute_et",
           "slot", "home_team", "home_win", "market_home_prob"]].rename(
               columns={"home_team": "team", "home_win": "team_win",
                        "market_home_prob": "team_market_prob"}
           )
    h["side"] = "home"
    a = x[["game_id", "season", "week", "gameday", "kickoff_minute_et",
           "slot", "away_team", "home_win", "market_home_prob"]].rename(
               columns={"away_team": "team"}
           )
    a["team_win"] = 1 - a["home_win"]
    a["team_market_prob"] = 1 - a["market_home_prob"]
    a = a.drop(columns=["home_win", "market_home_prob"])
    a["side"] = "away"
    long = pd.concat([h, a], ignore_index=True)
    long["surprise"] = long["team_win"] - long["team_market_prob"]
    # A tied or unplayed game cannot be evaluated as a win or loss; require
    # complete score-derived labels before adding it to a cumulative history.
    long["surprise"] = long["surprise"].where(long["team_win"].notna())
    long = long.sort_values(
        ["team", "gameday", "kickoff_minute_et", "game_id"]
    ).reset_index(drop=True)
    # Expanding prior-only surprise sums with missing labels excluded.
    prior_all_sum = long.groupby("team")["surprise"].transform(
        lambda s: s.fillna(0).cumsum().shift(1).fillna(0)
    )
    prior_all_n = long.groupby("team")["surprise"].transform(
        lambda s: s.notna().astype(int).cumsum().shift(1).fillna(0)
    )
    prior_slot_sum = long.groupby(["team", "slot"])["surprise"].transform(
        lambda s: s.fillna(0).cumsum().shift(1).fillna(0)
    )
    prior_slot_n = long.groupby(["team", "slot"])["surprise"].transform(
        lambda s: s.notna().astype(int).cumsum().shift(1).fillna(0)
    )
    long["team_prior_excess_slot_surprise"] = (
        prior_slot_sum / (prior_slot_n + PSEUDO_GAMES)
        - prior_all_sum / (prior_all_n + PSEUDO_GAMES)
    )
    long["team_prior_slot_experience"] = np.log1p(prior_slot_n)
    home = long.loc[long["side"].eq("home"), [
        "game_id", "team", "team_prior_excess_slot_surprise", "team_prior_slot_experience"
    ]].rename(columns={
        "team": "home_team",
        "team_prior_excess_slot_surprise": "home_prior_excess_slot_surprise",
        "team_prior_slot_experience": "home_prior_slot_experience",
    })
    away = long.loc[long["side"].eq("away"), [
        "game_id", "team", "team_prior_excess_slot_surprise", "team_prior_slot_experience"
    ]].rename(columns={
        "team": "away_team",
        "team_prior_excess_slot_surprise": "away_prior_excess_slot_surprise",
        "team_prior_slot_experience": "away_prior_slot_experience",
    })
    x = x.merge(home, on=["game_id", "home_team"], how="left", validate="one_to_one")
    x = x.merge(away, on=["game_id", "away_team"], how="left", validate="one_to_one")
    x["prior_slot_surprise_diff"] = (
        x["home_prior_excess_slot_surprise"] - x["away_prior_excess_slot_surprise"]
    )
    x["prior_slot_experience_diff"] = (
        x["home_prior_slot_experience"] - x["away_prior_slot_experience"]
    )
    return x


def build_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    x = add_prior_team_slot_history(slot_labels(games))
    x["rest_diff"] = pd.to_numeric(x["home_rest"], errors="coerce") - pd.to_numeric(
        x["away_rest"], errors="coerce"
    )
    x["divisional"] = x["div_game"].astype(str).str.lower().isin(["true", "1", "t"]).astype(int)
    for slug, slot in (
        ("tnf", "THU_NIGHT_PROXY"), ("snf", "SUN_NIGHT_PROXY"),
        ("mnf", "MON_NIGHT_PROXY"), ("sun_late", "SUN_LATE"),
        ("sun_morning", "SUN_MORNING"), ("saturday", "SATURDAY"),
        ("friday", "FRIDAY"),
    ):
        x[slug] = x["slot"].eq(slot).astype(int)
    home_rest = pd.to_numeric(x["home_rest"], errors="coerce")
    away_rest = pd.to_numeric(x["away_rest"], errors="coerce")
    x["home_short_rest"] = home_rest.lt(6).astype(int)
    x["away_short_rest"] = away_rest.lt(6).astype(int)
    x["tnf_rest_diff"] = x["tnf"] * x["rest_diff"]
    x["tnf_home_short"] = x["tnf"] * x["home_short_rest"]
    x["tnf_away_short"] = x["tnf"] * x["away_short_rest"]
    x["primetime_prior_surprise_diff"] = x["is_primetime_proxy"] * x["prior_slot_surprise_diff"]

    control = ["elo_diff", "rest_diff", "total_line", "divisional"]
    slots = ["tnf", "snf", "mnf", "sun_late", "sun_morning", "saturday",
             "friday", "slot_is_missing"]
    history = ["prior_slot_surprise_diff", "prior_slot_experience_diff"]
    interacts = ["tnf_rest_diff", "tnf_home_short", "tnf_away_short",
                 "primetime_prior_surprise_diff"]
    variants = {
        "control": control,
        "slots": [*control, *slots],
        "history": [*control, *slots, *history],
        "interactions": [*control, *slots, *interacts],
        "all": [*control, *slots, *history, *interacts],
    }
    return x, variants


def predict_market_offset(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    scale = StandardScaler()
    x = scale.fit_transform(impute.fit_transform(train[features]))
    z = scale.transform(impute.transform(test[features]))
    p = np.clip(train["market_home_prob"].to_numpy(float), 1e-5, 1 - 1e-5)
    prior = np.log(p / (1 - p))
    y = train["home_win"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])

    def f(theta):
        eta = prior + design @ theta
        errors = expit(eta) - y
        loss = float(np.mean(np.logaddexp(0, eta) - y * eta) +
                     PENALTY * (theta[1:] @ theta[1:]) / 2)
        g = design.T @ errors / len(y)
        g[1:] += PENALTY * theta[1:]
        return loss, g

    result = minimize(f, np.zeros(design.shape[1]), jac=True,
                      method="L-BFGS-B", options={"maxiter": 600, "maxls": 40})
    if not result.success:
        other = minimize(f, result.x, jac=True, method="BFGS", options={"maxiter": 600})
        if not (other.success or np.max(np.abs(f(other.x)[1])) < 1e-5):
            raise RuntimeError(f"Slot model fit failed: {result.message} / {other.message}")
        result = other
    q = np.clip(test["market_home_prob"].to_numpy(float), 1e-5, 1 - 1e-5)
    return expit(np.log(q/(1-q)) + np.column_stack([np.ones(len(z)), z]) @ result.x)


def paired(df: pd.DataFrame, name: str, base: str) -> dict:
    y = df["home_win"].to_numpy(int)
    a = (df[f"p_{name}"].to_numpy(float) >= .5) == y
    b = (df[f"p_{base}"].to_numpy(float) >= .5) == y
    w, l = int((a & ~b).sum()), int((~a & b).sum())
    season = df[["season"]].copy()
    season["gain"] = a.astype(int)-b.astype(int)
    s = season.groupby("season").agg(n=("gain", "size"), d=("gain", "sum"))
    rng = np.random.default_rng(20261009)
    indices = rng.integers(len(s), size=(12000, len(s)))
    draws = 100*s["d"].to_numpy()[indices].sum(axis=1)/s["n"].to_numpy()[indices].sum(axis=1)
    return {"challenger":name, "baseline":base, "games":len(df),
            "correct":int(a.sum()), "reference_correct":int(b.sum()),
            "net_correct":w-l, "flip_wins":w, "flip_losses":l,
            "mcnemar_p":float(binomtest(w,w+l).pvalue) if w+l else 1.0,
            "season_boot_lo_pp":float(np.quantile(draws,.025)),
            "season_boot_hi_pp":float(np.quantile(draws,.975))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season",type=int,default=2009)
    parser.add_argument("--end-season",type=int,default=2025)
    parser.add_argument("--first-test-season",type=int,default=2016)
    parser.add_argument("--holdout-season",type=int,default=2019)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/track_j_primetime"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = nfl.load_schedules(list(range(args.start_season, args.end_season+1))).to_pandas()
    games = schedule.loc[schedule["game_type"].eq("REG")].copy()
    games["gameday"] = pd.to_datetime(games["gameday"])
    games = add_elo(add_market_probability(games))
    games["home_win"] = np.where(
        games["home_score"] > games["away_score"], 1.0,
        np.where(games["home_score"] < games["away_score"], 0.0, np.nan)
    )
    games, variants = build_features(games)
    print(games["slot"].value_counts().to_string(), flush=True)
    observed = games.loc[games.home_win.notna()].copy()
    outs = []
    for year in range(args.first_test_season,args.end_season+1):
        train = observed.loc[observed.season.lt(year)]
        test = observed.loc[observed.season.eq(year)]
        if len(train)==0 or len(test)==0: continue
        result = test[["game_id","season","week","slot","home_win"]].copy()
        result["p_market"] = test["market_home_prob"].to_numpy(float)
        for name, cols in variants.items():
            result[f"p_{name}"] = predict_market_offset(train,test,cols)
        outs.append(result)
        print(f"Season {year}: {len(test)} games",flush=True)
    if not outs: raise RuntimeError("No walk-forward games")
    pred = pd.concat(outs,ignore_index=True)
    pred.to_csv(args.output_dir/"predictions.csv",index=False)
    hold = pred.loc[pred.season.ge(args.holdout_season)]
    comps = pd.DataFrame([paired(hold,n,"control") for n in variants if n!="control"]+
                         [paired(hold,"all","market"),paired(hold,"control","market")])
    comps.to_csv(args.output_dir/"paired_holdout.csv",index=False)
    metrics=[]
    for (year,sl),g in pred.groupby(["season","slot"]):
        y=g.home_win.to_numpy(int)
        for name in ("market",*variants):
            p=g[f"p_{name}"].to_numpy(float)
            metrics.append({"season":year,"slot":sl,"model":name,"games":len(g),
                            "correct":int(((p>=.5)==y).sum()),
                            "brier":float(brier_score_loss(y,p)),
                            "log_loss":float(log_loss(y,p,labels=[0,1]))})
    pd.DataFrame(metrics).to_csv(args.output_dir/"by_season_slot.csv",index=False)
    pd.DataFrame([(k,c) for k,cols in variants.items() for c in cols],
                 columns=["variant","feature"]).to_csv(args.output_dir/"features.csv",index=False)
    head=[
        "# Track J — Game-slot and primetime-proxy historical diagnostics","",
        "**Research-only; NO validated production rule.** Labels are derived",
        "from Eastern kickoff day/time, NOT verified TV network/broadcast",
        "assignments. TNF/SNF/MNF proxies and selected-team participation",
        "may produce selection effects. Every team-specific slot residual",
        "uses prior completed games and 20 pseudo-games of shrinkage.",
        "Historical odds are closing/near-closing prices; 2019–2025 outcomes",
        "were previously used for other research and are NOT fresh confirmation.","",
        "## Paired 2019–2025 research results","",
        comps.to_markdown(index=False,floatfmt=".3f"),"",
        "## Observed game-slot counts","",
        games["slot"].value_counts().to_markdown(),"",
        "Compare changes against the exact frozen incumbent separately and",
        "run time-stamped prospective tests before any operational proposal."
    ]
    (args.output_dir/"summary.md").write_text("\n".join(head),encoding="utf-8")
    print((args.output_dir/"summary.md").read_text(),flush=True)


if __name__=="__main__":
    main()
