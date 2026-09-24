"""Track I: individualized star-player injury value (research only).

Tests whether strictly pregame, player-specific injury value can improve straight-up
NFL picks beyond the closing market and Track C's simpler broad injury burden.
Development: 2020-2021. Untouched walk-forward holdout: 2022-2024.
Nothing here changes prospective-v1-frozen-2025.
"""
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import build_model_table
from track_c_player_value_injuries import (
    POSITION_GROUPS,
    build_player_history_index,
    build_team_week_features,
    load_final_injury_rows,
    load_snap_history,
    prior_player_role,
    safe_logit,
)

START, END = 2018, 2024
DEV = (2020, 2021)
HOLDOUT = (2022, 2023, 2024)
GRID = (0.01, 0.03, 0.08, 0.20, 0.50)
OUT = Path("outputs/track_i_individual_star_injuries")
SKILL = {"WR", "TE", "RB", "FB", "HB"}
FRONT = {"DE", "DT", "DL", "NT", "LB", "ILB", "OLB", "EDGE"}
DB = {"CB", "DB", "S", "FS", "SS"}


def n(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df else pd.Series(0.0, index=df.index)


def pg(pos):
    p = str(pos or "").upper().strip()
    if p == "QB": return "QB"
    if p in SKILL: return "SKILL"
    if p in FRONT: return "FRONT"
    if p in DB: return "DB"
    if p in {"C", "G", "OG", "T", "OT", "OL"}: return "OL"
    if p in {"K", "P", "LS"}: return "ST"
    return "OTHER"


def roll_prior(g):
    return g.shift(1).rolling(6, min_periods=2).mean()


def quality_histories(seasons):
    """Prior-only QB/SKILL EPA and FRONT/DB playmaking histories."""
    stats = nfl.load_player_stats(seasons).to_pandas()
    stats = stats.loc[stats["season_type"].eq("REG")].copy()
    stats["order"] = stats["season"].astype(int) * 100 + stats["week"].astype(int)
    stats["group"] = stats["position"].map(pg)
    pid = "player_id" if "player_id" in stats else "gsis_id"
    stats["key"] = stats[pid].astype("string")
    dbk = n(stats, "attempts") + n(stats, "sacks_suffered")
    opp = n(stats, "targets") + n(stats, "carries")
    stats["raw"] = np.nan
    q = stats["group"].eq("QB")
    s = stats["group"].eq("SKILL")
    stats.loc[q, "raw"] = (n(stats, "passing_epa")[q] + 0.2*n(stats, "rushing_epa")[q]) / np.maximum(dbk[q] + 0.2*n(stats, "carries")[q], 1)
    stats.loc[s, "raw"] = (n(stats, "receiving_epa")[s] + n(stats, "rushing_epa")[s]) / np.maximum(opp[s], 1)
    off = stats.loc[stats["key"].notna() & stats["group"].isin(["QB","SKILL"]), ["key","order","raw"]].sort_values(["key","order"])
    off["prior"] = off.groupby("key", group_keys=False)["raw"].transform(roll_prior)

    snaps = nfl.load_snap_counts(seasons).to_pandas()
    snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    x = pd.to_numeric(snaps["defense_pct"], errors="coerce").fillna(0)
    if len(x) and x.quantile(.95) > 1.5: x = x / 100
    snaps["dpct"] = x.clip(0,1)
    sp = snaps[["game_id","team","pfr_player_id","position","dpct"]].dropna(subset=["pfr_player_id"])
    sp = sp.sort_values(["game_id","team","pfr_player_id","dpct"]).drop_duplicates(["game_id","team","pfr_player_id"], keep="last")
    d = nfl.load_pfr_advstats(seasons, stat_type="def", summary_level="week").to_pandas()
    d = d.loc[d["game_type"].eq("REG")].merge(sp, on=["game_id","team","pfr_player_id"], how="left")
    d["group"] = d["position"].map(pg)
    d["order"] = d["season"].astype(int)*100 + d["week"].astype(int)
    d["key"] = d["pfr_player_id"].astype("string")
    exp = np.maximum(n(d,"dpct"), .5)
    d["raw"] = np.nan
    f, b = d["group"].eq("FRONT"), d["group"].eq("DB")
    d.loc[f,"raw"] = (n(d,"def_pressures")[f] + .75*n(d,"def_sacks")[f] + .25*n(d,"def_qb_hits")[f]) / exp[f]
    d.loc[b,"raw"] = (2*n(d,"def_ints")[b] + .5*n(d,"def_pass_defended")[b] - .25*n(d,"def_missed_tackles")[b]) / exp[b]
    de = d.loc[d["key"].notna() & d["group"].isin(["FRONT","DB"]), ["key","order","raw"]].sort_values(["key","order"])
    de["prior"] = de.groupby("key", group_keys=False)["raw"].transform(roll_prior)

    out = {}
    for prefix, frame in (("gsis",off),("pfr",de)):
        for key,g in frame.groupby("key", sort=False):
            z = g.loc[g["prior"].notna()]
            if len(z): out[f"{prefix}:{key}"] = (z["order"].to_numpy(int), z["prior"].to_numpy(float))
    return out


def asof(hist, key, order):
    if key not in hist: return np.nan
    orders, vals = hist[key]
    i = np.searchsorted(orders, order, side="left") - 1
    return float(vals[i]) if i >= 0 else np.nan


def impact(group, role, qual):
    role = 0.0 if not np.isfinite(role) else role
    qual = 0.0 if not np.isfinite(qual) else qual
    if group == "QB": mult = 1 + np.clip(qual,-.4,.4)
    elif group == "SKILL": mult = 1 + np.clip(qual,-.35,.35)
    elif group == "FRONT": mult = 1 + .2*np.tanh(qual/4)
    elif group == "DB": mult = 1 + .2*np.tanh(qual/2)
    else: mult = 1
    return max(0.0, role*mult)


def star_table(inj, snaps, hist):
    pidx = build_player_history_index(snaps)
    rows, audit = [], []
    vals = []
    for r in inj.itertuples(index=False):
        role,_ = prior_player_role(pidx, r.pfr_id, int(r.order_key))
        group = str(r.position_group)
        key = f"gsis:{r.gsis_id}" if group in {"QB","SKILL"} else f"pfr:{r.pfr_id}"
        qual = asof(hist,key,int(r.order_key))
        vals.append((role,qual,impact(group,role,qual)))
    x = inj.copy()
    x[["prior_role","prior_quality","star_impact"]] = pd.DataFrame(vals,index=x.index)
    for (season,week,team),g in x.groupby(["season","week","team"],sort=False):
        row={"season":int(season),"week":int(week),"team":team}
        sev=g["report_bucket"].isin(["out","doubtful"])
        v=np.sort(g.loc[sev,"star_impact"].to_numpy(float))[::-1]
        row.update(star_max=float(v[0]) if len(v) else 0, star_top2=float(v[:2].sum()), star_count75=int((v>=.75).sum()))
        for group in POSITION_GROUPS:
            gm=g["position_group"].eq(group)
            for status in ("out","doubtful","questionable"):
                m=gm & g["report_bucket"].eq(status)
                vv=g.loc[m,"star_impact"]
                row[f"{group.lower()}_{status}_max"]=float(vv.max()) if len(vv) else 0
                row[f"{group.lower()}_{status}_sum"]=float(vv.sum())
            sv=np.sort(g.loc[gm & sev,"star_impact"].to_numpy(float))[::-1]
            row[f"{group.lower()}_severe_max"]=float(sv[0]) if len(sv) else 0
            row[f"{group.lower()}_severe_top2"]=float(sv[:2].sum())
        rows.append(row); audit.append(g)
    t=pd.DataFrame(rows).fillna(0)
    return t,pd.concat(audit,ignore_index=True),[c for c in t if c not in {"season","week","team"}]


def merge_diffs(games, team, cols):
    h=team[["season","week","team",*cols]].rename(columns={"team":"home_team",**{c:f"home_{c}" for c in cols}})
    a=team[["season","week","team",*cols]].rename(columns={"team":"away_team",**{c:f"away_{c}" for c in cols}})
    z=games.merge(h,on=["season","week","home_team"],how="left").merge(a,on=["season","week","away_team"],how="left")
    diffs=[]
    for c in cols:
        hc,ac=f"home_{c}",f"away_{c}"
        z[hc]=pd.to_numeric(z[hc],errors="coerce").fillna(0); z[ac]=pd.to_numeric(z[ac],errors="coerce").fillna(0)
        d=f"diff_{c}"; z[d]=z[hc]-z[ac]; diffs.append(d)
    return z,diffs


def ridge(C):
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(C=C,solver="lbfgs",max_iter=4000,random_state=42))])


def tune(g,features):
    rows=[]
    for C in GRID:
        yy=[]; pp=[]
        for s in DEV:
            tr=g[(g.season>=START)&(g.season<s)]; te=g[g.season==s]
            m=ridge(C).fit(tr[features],tr.home_win.astype(int)); p=m.predict_proba(te[features])[:,1]
            yy.append(te.home_win.astype(int).to_numpy()); pp.append(p)
        y=np.concatenate(yy); p=np.concatenate(pp)
        rows.append({"C":C,"accuracy":np.mean((p>=.5)==y),"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p)})
    t=pd.DataFrame(rows).sort_values(["log_loss","brier","C"])
    return float(t.iloc[0].C),t


def walk(g,features,C,name):
    out=[]
    for s in HOLDOUT:
        tr=g[(g.season>=START)&(g.season<s)]; te=g[g.season==s]
        m=ridge(C).fit(tr[features],tr.home_win.astype(int))
        z=te[["game_id","season","week","home_win"]].copy(); z[f"p_{name}"]=m.predict_proba(te[features])[:,1]; out.append(z)
    return pd.concat(out,ignore_index=True)


def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    return {"games":len(y),"correct":int(((p>=.5)==y).sum()),"accuracy":float(np.mean((p>=.5)==y)),"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p)}


def bootstrap(y,p,mkt,nboot=30000):
    y=np.asarray(y,int); d=((np.asarray(p)>=.5)==y).astype(float)-((np.asarray(mkt)>=.5)==y).astype(float)
    rng=np.random.default_rng(42); vals=np.empty(nboot)
    for i in range(0,nboot,2000):
        j=min(i+2000,nboot); idx=rng.integers(0,len(d),size=(j-i,len(d))); vals[i:j]=d[idx].mean(1)
    return {"lift":d.mean(),"low":np.quantile(vals,.025),"high":np.quantile(vals,.975),"p_positive":np.mean(vals>0)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    games=build_model_table(START,END); games=games[games.home_win.notna()].copy(); games["market_logit"]=safe_logit(games.market_home_prob)
    inj,_=load_final_injury_rows(START,END); snaps=load_snap_history(list(range(START,END+1)))
    team,diag,broad,_=build_team_week_features(inj,snaps)
    games,bd=merge_diffs(games,team,broad)
    hist=quality_histories(list(range(START,END+1))); stars,audit,sc=star_table(inj,snaps,hist)
    games,sd=merge_diffs(games,stars,sc)
    variants={"broad":["market_logit",*bd],"star":["market_logit",*sd],"broad_star":["market_logit",*bd,*sd]}
    selected={}; tuning=[]
    for name,features in variants.items():
        C,t=tune(games,features); t["model"]=name; selected[name]=C; tuning.append(t)
    pred=games[games.season.isin(HOLDOUT)][["game_id","season","week","gameday","away_team","home_team","home_win","market_home_prob"]].copy().rename(columns={"market_home_prob":"p_market"})
    for name,features in variants.items():
        p=walk(games,features,selected[name],name); pred=pred.merge(p[["game_id",f"p_{name}"]],on="game_id")
    models={"market":"p_market","broad":"p_broad","star":"p_star","broad_star":"p_broad_star"}
    summary=[]; season=[]
    for name,col in models.items():
        summary.append({"model":name,**metrics(pred.home_win,pred[col])})
        for s in HOLDOUT:
            z=pred[pred.season==s]; mm=metrics(z.home_win,z[col]); season.append({"model":name,"season":s,**mm,"hit_70pct":mm["accuracy"]>=.70})
    summary=pd.DataFrame(summary); season=pd.DataFrame(season)
    boot=bootstrap(pred.home_win,pred.p_broad_star,pred.p_market)
    primary=summary[summary.model=="broad_star"].iloc[0]; market=summary[summary.model=="market"].iloc[0]
    sy=season[season.model=="broad_star"]
    pred.to_csv(OUT/"holdout_predictions.csv",index=False); summary.to_csv(OUT/"holdout_summary.csv",index=False); season.to_csv(OUT/"holdout_by_season.csv",index=False)
    pd.concat(tuning).to_csv(OUT/"development_tuning.csv",index=False); audit.to_csv(OUT/"player_injury_audit.csv",index=False)
    pd.DataFrame([diag]).to_csv(OUT/"data_diagnostics.csv",index=False)
    text=[
      "# Track I: Individualized Star-Player Injury Study","",
      "**Research only. No change to `prospective-v1-frozen-2025`.**","",
      "Player-specific value uses strictly prior snap role plus position-appropriate prior production. Hyperparameters use 2020-2021 only; 2022-2024 is untouched walk-forward holdout.","",
      f"- Market: **{int(market.correct)}/{int(market.games)} ({market.accuracy:.2%})**",
      f"- Primary broad + individualized-star model: **{int(primary.correct)}/{int(primary.games)} ({primary.accuracy:.2%})**",
      f"- Paired lift vs market: **{boot['lift']:+.3%}**, 95% CI **[{boot['low']:+.3%}, {boot['high']:+.3%}]**, P(lift>0) **{boot['p_positive']:.1%}**","",
      "## 70% target by season","",sy[["season","games","correct","accuracy","hit_70pct"]].to_markdown(index=False),"",
      "## All models","",summary.to_markdown(index=False),"",
      "A positive result here would still require a live timestamped injury/inactive feed and prospective challenger validation before any production change."
    ]
    (OUT/"summary.md").write_text("\n".join(text),encoding="utf-8")
    print("\n".join(text))


if __name__=="__main__":
    main()
