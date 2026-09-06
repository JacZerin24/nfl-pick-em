"""Track D: market-anchored weather x team-style interaction research.

Research-only. Historical schedule weather is treated as near-kickoff information.
The official 2026 prospective pick system is not modified by this script.

Primary question: do a small set of pre-specified football mechanisms interact
with wind/temperature strongly enough to improve straight-up picks beyond the
closing market and beyond generic weather variables?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from phase1_backtest import build_model_table
from phase2_upset_variance import build_variance_game_stats, rolling_variance

EPS = 1e-6
PENALTY = 300.0
GENERIC = ["wind10", "wind15", "cold40", "freezing32", "heat80", "weather_reported"]
INTERACTIONS = [
    "wind_x_pass_dependence",
    "wind_x_explosive_pass",
    "wind_x_sack_exposure",
    "wind_x_pass_defense",
    "cold_x_pass_dependence",
    "cold_x_rush_efficiency",
    "freezing_x_turnovers",
    "heat_x_off_success",
    "heat_x_def_success_allowed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--holdout-start-season", type=int, default=2019)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_d_weather_interactions"))
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def add_explosive_features(base: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    print(f"Loading PBP explosive-pass context {start}-{end}...")
    pbp = nfl.load_pbp(list(range(start, end + 1)))
    vg = build_variance_game_stats(pbp)
    vr = rolling_variance(vg, base)
    cols = ["off_expl_pass_rate_r8", "def_expl_pass_allowed_r8"]
    home = vr[["game_id", "team", *cols]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in cols}}
    )
    away = vr[["game_id", "team", *cols]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in cols}}
    )
    return base.merge(home, on=["game_id", "home_team"], how="left").merge(
        away, on=["game_id", "away_team"], how="left"
    )


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    temp = pd.to_numeric(x.get("temp"), errors="coerce")
    wind = pd.to_numeric(x.get("wind"), errors="coerce")
    roof = x["roof"].astype("string").str.lower()
    exposed = roof.isin(["outdoors", "open", "retractable"]).astype(float)
    reported = ((temp.notna() | wind.notna()) & exposed.eq(1)).astype(float)

    x["temp_num"] = temp
    x["wind_num"] = wind
    x["weather_exposed"] = exposed
    x["weather_reported"] = reported
    x["wind10"] = (wind - 10.0).clip(lower=0).fillna(0.0) * exposed
    x["wind15"] = (wind - 15.0).clip(lower=0).fillna(0.0) * exposed
    x["cold40"] = (40.0 - temp).clip(lower=0).fillna(0.0) * exposed
    x["freezing32"] = (32.0 - temp).clip(lower=0).fillna(0.0) * exposed
    x["heat80"] = (temp - 80.0).clip(lower=0).fillna(0.0) * exposed

    hp = num(x, "home_off_pass_epa_r8"); hr = num(x, "home_off_rush_epa_r8")
    ap = num(x, "away_off_pass_epa_r8"); ar = num(x, "away_off_rush_epa_r8")
    pass_depend_diff = (hp - hr) - (ap - ar)
    rush_eff_diff = hr - ar
    sack_diff = num(x, "home_off_sack_rate_r8") - num(x, "away_off_sack_rate_r8")
    turnover_diff = num(x, "home_off_turnover_rate_r8") - num(x, "away_off_turnover_rate_r8")
    pass_def_diff = num(x, "home_def_pass_epa_allowed_r8") - num(x, "away_def_pass_epa_allowed_r8")
    off_success_diff = num(x, "home_off_success_r8") - num(x, "away_off_success_r8")
    def_success_diff = num(x, "home_def_success_allowed_r8") - num(x, "away_def_success_allowed_r8")
    expl_diff = num(x, "home_off_expl_pass_rate_r8") - num(x, "away_off_expl_pass_rate_r8")

    x["wind_x_pass_dependence"] = x["wind10"] * pass_depend_diff
    x["wind_x_explosive_pass"] = x["wind10"] * expl_diff
    x["wind_x_sack_exposure"] = x["wind10"] * sack_diff
    x["wind_x_pass_defense"] = x["wind10"] * pass_def_diff
    x["cold_x_pass_dependence"] = x["cold40"] * pass_depend_diff
    x["cold_x_rush_efficiency"] = x["cold40"] * rush_eff_diff
    x["freezing_x_turnovers"] = x["freezing32"] * turnover_diff
    x["heat_x_off_success"] = x["heat80"] * off_success_diff
    x["heat_x_def_success_allowed"] = x["heat80"] * def_success_diff

    x["high_wind_game"] = ((wind >= 15) & exposed.eq(1)).astype(int)
    x["cold_game"] = ((temp <= 40) & exposed.eq(1)).astype(int)
    x["freezing_game"] = ((temp <= 32) & exposed.eq(1)).astype(int)
    x["hot_game"] = ((temp >= 80) & exposed.eq(1)).astype(int)
    return x


def matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = train[cols].apply(pd.to_numeric, errors="coerce").copy()
    te = test[cols].apply(pd.to_numeric, errors="coerce").copy()
    med = tr.median(axis=0).fillna(0.0)
    tr = tr.fillna(med); te = te.fillna(med)
    mean = tr.mean(axis=0); std = tr.std(axis=0).replace(0, 1.0).fillna(1.0)
    a = ((tr - mean) / std).to_numpy(float)
    b = ((te - mean) / std).to_numpy(float)
    return np.column_stack([np.ones(len(a)), a]), np.column_stack([np.ones(len(b)), b])


def fit_offset(train: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, pd.Series, pd.Series]:
    X, _ = matrix(train, train.iloc[:0].copy(), cols)
    med = train[cols].apply(pd.to_numeric, errors="coerce").median(axis=0).fillna(0.0)
    filled = train[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    mean = filled.mean(axis=0); std = filled.std(axis=0).replace(0, 1.0).fillna(1.0)
    X = np.column_stack([np.ones(len(filled)), ((filled - mean) / std).to_numpy(float)])
    y = train["home_win"].astype(int).to_numpy()
    offset = logit(train["market_home_prob"].to_numpy(float))
    weights = np.ones(X.shape[1]); weights[0] = 0.10

    def objective(theta: np.ndarray):
        eta = offset + X @ theta
        p = sigmoid(eta)
        nll = -np.sum(y * np.log(np.clip(p, EPS, 1)) + (1-y) * np.log(np.clip(1-p, EPS, 1)))
        reg = 0.5 * PENALTY * np.sum(weights * theta * theta)
        grad = X.T @ (p-y) + PENALTY * weights * theta
        return float(nll + reg), grad

    res = minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", jac=True,
                   options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(str(res.message))
    return res.x, mean, std


def predict_offset(test: pd.DataFrame, cols: list[str], theta: np.ndarray, mean: pd.Series, std: pd.Series,
                   train_median: pd.Series) -> np.ndarray:
    filled = test[cols].apply(pd.to_numeric, errors="coerce").fillna(train_median)
    X = np.column_stack([np.ones(len(filled)), ((filled - mean) / std).to_numpy(float)])
    return sigmoid(logit(test["market_home_prob"].to_numpy(float)) + X @ theta)


def train_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    raw = train[cols].apply(pd.to_numeric, errors="coerce")
    med = raw.median(axis=0).fillna(0.0)
    filled = raw.fillna(med)
    mean = filled.mean(axis=0); std = filled.std(axis=0).replace(0, 1.0).fillna(1.0)
    X = np.column_stack([np.ones(len(filled)), ((filled - mean) / std).to_numpy(float)])
    y = train["home_win"].astype(int).to_numpy()
    offset = logit(train["market_home_prob"].to_numpy(float))
    weights = np.ones(X.shape[1]); weights[0] = 0.10

    def objective(theta: np.ndarray):
        p = sigmoid(offset + X @ theta)
        nll = -np.sum(y*np.log(np.clip(p, EPS, 1)) + (1-y)*np.log(np.clip(1-p, EPS, 1)))
        reg = 0.5 * PENALTY * np.sum(weights * theta * theta)
        return float(nll + reg), X.T @ (p-y) + PENALTY * weights * theta

    res = minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", jac=True,
                   options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(str(res.message))
    te = test[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    Xte = np.column_stack([np.ones(len(te)), ((te - mean) / std).to_numpy(float)])
    return sigmoid(logit(test["market_home_prob"].to_numpy(float)) + Xte @ res.x)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    yy = np.asarray(y, int); pp = np.clip(np.asarray(p, float), EPS, 1-EPS)
    pick = pp >= 0.5
    return {
        "games": int(len(yy)), "correct": int(np.sum(pick == yy)),
        "accuracy": float(np.mean(pick == yy)),
        "log_loss": float(log_loss(yy, pp, labels=[0,1])),
        "brier": float(brier_score_loss(yy, pp)),
    }


def backtest(data: pd.DataFrame, first: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = data.loc[data["home_win"].notna()].copy()
    rows, folds = [], []
    for season in range(first, end+1):
        tr = completed.loc[completed["season"] < season].copy()
        te = completed.loc[completed["season"] == season].copy()
        if tr.empty or te.empty: continue
        pm = te["market_home_prob"].to_numpy(float)
        pg = train_predict(tr, te, GENERIC)
        pi = train_predict(tr, te, [*GENERIC, *INTERACTIONS])
        y = te["home_win"].astype(int).to_numpy()
        for name, p in (("market", pm), ("generic_weather", pg), ("weather_interactions", pi)):
            folds.append({"season": season, "model": name, **metrics(y,p)})
        out = te[["game_id","season","week","gameday","away_team","home_team","home_win",
                  "market_home_prob","temp_num","wind_num","weather_exposed","weather_reported",
                  "high_wind_game","cold_game","freezing_game","hot_game"]].copy()
        out["p_home_market"] = pm; out["p_home_generic_weather"] = pg; out["p_home_weather_interactions"] = pi
        rows.append(out)
        print(f"{season}: market={metrics(y,pm)['accuracy']:.3f} generic={metrics(y,pg)['accuracy']:.3f} interactions={metrics(y,pi)['accuracy']:.3f}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(folds)


def scope_scores(pred: pd.DataFrame, holdout_start: int) -> pd.DataFrame:
    h = pred.loc[pred["season"] >= holdout_start].copy()
    scopes = {
        "all": np.ones(len(h), dtype=bool),
        "weather_exposed_reported": h["weather_reported"].eq(1).to_numpy(),
        "high_wind_15plus": h["high_wind_game"].eq(1).to_numpy(),
        "cold_40minus": h["cold_game"].eq(1).to_numpy(),
        "freezing_32minus": h["freezing_game"].eq(1).to_numpy(),
        "hot_80plus": h["hot_game"].eq(1).to_numpy(),
    }
    rows=[]
    for scope, mask in scopes.items():
        d = h.loc[mask].copy()
        if d.empty: continue
        y=d["home_win"].astype(int).to_numpy(); mc=metrics(y,d["p_home_market"])["correct"]
        gc=metrics(y,d["p_home_generic_weather"])["correct"]
        for name,col in (("market","p_home_market"),("generic_weather","p_home_generic_weather"),("weather_interactions","p_home_weather_interactions")):
            m=metrics(y,d[col].to_numpy(float))
            rows.append({"scope":scope,"model":name,**m,
                         "net_vs_market":int(m["correct"]-mc),
                         "net_vs_generic":int(m["correct"]-gc)})
    return pd.DataFrame(rows)


def paired_bootstrap(pred: pd.DataFrame, holdout_start: int, scope: str, n: int=50000) -> dict[str,float|int|str]:
    h=pred.loc[pred["season"]>=holdout_start].copy()
    if scope=="weather_exposed_reported": h=h.loc[h["weather_reported"].eq(1)].copy()
    y=h["home_win"].astype(int).to_numpy()
    market=(h["p_home_market"].to_numpy(float)>=.5).astype(int)
    generic=(h["p_home_generic_weather"].to_numpy(float)>=.5).astype(int)
    inter=(h["p_home_weather_interactions"].to_numpy(float)>=.5).astype(int)
    di=(inter==y).astype(float)-(market==y).astype(float)
    dg=(inter==y).astype(float)-(generic==y).astype(float)
    rng=np.random.default_rng(42)
    idx=rng.integers(0,len(y),size=(n,len(y)))
    bi=di[idx].mean(axis=1); bg=dg[idx].mean(axis=1)
    return {
        "scope":scope,"games":len(y),
        "lift_vs_market":float(di.mean()),"ci_market_low":float(np.quantile(bi,.025)),"ci_market_high":float(np.quantile(bi,.975)),"p_gt_market":float(np.mean(bi>0)),
        "lift_vs_generic":float(dg.mean()),"ci_generic_low":float(np.quantile(bg,.025)),"ci_generic_high":float(np.quantile(bg,.975)),"p_gt_generic":float(np.mean(bg>0)),
    }


def main() -> None:
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    base=build_model_table(a.start_season,a.end_season)
    data=add_explosive_features(base,a.start_season,a.end_season)
    data=add_features(data)
    pred,folds=backtest(data,a.first_test_season,a.end_season)
    scopes=scope_scores(pred,a.holdout_start_season)
    boots=pd.DataFrame([
        paired_bootstrap(pred,a.holdout_start_season,"all"),
        paired_bootstrap(pred,a.holdout_start_season,"weather_exposed_reported"),
    ])
    hold=pred.loc[pred["season"]>=a.holdout_start_season]
    diag=pd.DataFrame([{
        "holdout_games":len(hold),
        "weather_exposed_reported":int(hold["weather_reported"].sum()),
        "high_wind_15plus":int(hold["high_wind_game"].sum()),
        "cold_40minus":int(hold["cold_game"].sum()),
        "freezing_32minus":int(hold["freezing_game"].sum()),
        "hot_80plus":int(hold["hot_game"].sum()),
        "penalty":PENALTY,
    }])
    pred.to_csv(a.output_dir/"predictions.csv",index=False)
    folds.to_csv(a.output_dir/"fold_metrics.csv",index=False)
    scopes.to_csv(a.output_dir/"scope_scores.csv",index=False)
    boots.to_csv(a.output_dir/"bootstrap.csv",index=False)
    diag.to_csv(a.output_dir/"diagnostics.csv",index=False)
    pd.DataFrame({"feature":[*GENERIC,*INTERACTIONS]}).to_csv(a.output_dir/"features.csv",index=False)

    def row(scope:str,model:str): return scopes.loc[(scopes.scope==scope)&(scopes.model==model)].iloc[0]
    m=row("all","market"); g=row("all","generic_weather"); w=row("all","weather_interactions")
    e=row("weather_exposed_reported","weather_interactions"); em=row("weather_exposed_reported","market")
    b=boots.loc[boots.scope.eq("all")].iloc[0]; be=boots.loc[boots.scope.eq("weather_exposed_reported")].iloc[0]
    text=f"""# Track D: Weather Interaction Study\n\n**Research-only. No change to `prospective-v1-frozen-2025`.**\n\nHistorical window: **{a.start_season}-{a.end_season}**. Development OOS: **{a.first_test_season}-{a.holdout_start_season-1}**. Official diagnostic holdout: **{a.holdout_start_season}-{a.end_season}**.\n\nThe primary candidate is a fixed-penalty market-anchored ridge using pre-specified wind/temperature interactions with prior team style: pass dependence, explosive passing, sack exposure, pass defense, rushing efficiency, turnover tendency, and success-rate context.\n\n## Holdout: all games\n\n- Market: **{int(m.correct)}/{int(m.games)} ({100*m.accuracy:.2f}%)**\n- Generic weather: **{int(g.correct)}/{int(g.games)} ({100*g.accuracy:.2f}%)**, net **{int(g.net_vs_market):+d}** vs market\n- Weather interactions: **{int(w.correct)}/{int(w.games)} ({100*w.accuracy:.2f}%)**, net **{int(w.net_vs_market):+d}** vs market and **{int(w.net_vs_generic):+d}** vs generic weather\n- Interaction lift vs market: **{100*b.lift_vs_market:+.3f} pp**, 95% CI **[{100*b.ci_market_low:+.3f}, {100*b.ci_market_high:+.3f}]**, P(lift>0) **{100*b.p_gt_market:.1f}%**\n\n## Holdout: outdoor/open games with weather reported\n\n- Games: **{int(e.games)}**\n- Market accuracy: **{100*em.accuracy:.2f}%**\n- Weather-interaction accuracy: **{100*e.accuracy:.2f}%**, net **{int(e.net_vs_market):+d}** vs market\n- Interaction lift vs market: **{100*be.lift_vs_market:+.3f} pp**, 95% CI **[{100*be.ci_market_low:+.3f}, {100*be.ci_market_high:+.3f}]**, P(lift>0) **{100*be.p_gt_market:.1f}%**\n\n## Guardrail\n\nHistorical nflverse schedule weather is near-kickoff/final context, not a timestamped forecast feed. Any positive result must remain research-only until reproduced with pre-kick forecast snapshots.\n"""
    (a.output_dir/"summary.md").write_text(text,encoding="utf-8")
    print(text)
    print("\nScope scores\n",scopes.to_string(index=False))

if __name__=="__main__": main()
