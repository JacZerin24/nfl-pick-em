"""Track M: coverage/pressure and roster-participation matchup research.

Participation data: NFL Next Gen Stats via nflverse (<=2022), FTN Data
via nflverse (2023+, CC-BY-SA 4.0). Post-2023 participation releases
after playoffs, so NEVER presume this is 2026 in-season live data.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import build_model_table
from phase2_upset_specialist import build_upset_table, make_logistic, oriented_side_values
from phase2_upset_variance import build_variance_game_stats, rolling_variance
from live_pickem import build_variance_training, make_variance_catboost

OUT = Path("outputs/track_m_participation")
CHART_YEARS = list(range(2016, 2026))
TEST_YEARS = (2023, 2024, 2025)
WINDOWS = (4, 8)
MIN_SAMPLE = 3
COVERAGE_METRICS = (
    "off_vs_man_epa", "off_vs_zone_epa", "off_under_pressure_epa",
    "off_deep_route_share", "def_man_rate", "def_zone_rate",
    "def_man_epa_allowed", "def_zone_epa_allowed",
    "def_pressure_rate", "def_2deep_rate",
)
MATCHUPS = (
    ("against_man", "off_vs_man_epa", "def_man_rate"),
    ("against_zone", "off_vs_zone_epa", "def_zone_rate"),
    ("against_pressure", "off_under_pressure_epa", "def_pressure_rate"),
    ("deep_vs_two_deep", "off_deep_route_share", "def_2deep_rate"),
)


def join_participation(part: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_p = {
        "nflverse_game_id", "play_id", "defense_man_zone_type",
        "defense_coverage_type", "was_pressure", "route",
        "offense_players", "defense_players",
    }
    required_b = {"game_id", "play_id", "posteam", "defteam", "epa", "pass", "rush"}
    if required_p - set(part) or required_b - set(pbp):
        raise ValueError(
            f"Missing participation fields: {sorted(required_p-set(part))}; "
            f"missing PBP fields: {sorted(required_b-set(pbp))}"
        )
    p = part[list(sorted(required_p))].copy().rename(columns={"nflverse_game_id": "game_id"})
    b = pbp[list(sorted(required_b))].copy()
    for df in (p, b):
        df["play_id"] = pd.to_numeric(df["play_id"], errors="coerce").astype("Int64")
        df.dropna(subset=["game_id", "play_id"], inplace=True)
    p.drop_duplicates(inplace=True)
    b.drop_duplicates(inplace=True)
    if p.duplicated(["game_id", "play_id"]).any() or b.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Conflicting multiple participation/PBP records for one play ID")
    joined = p.merge(b, on=["game_id", "play_id"], how="left", validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    coverage = {"source_rows": len(p), "joined_rows": int(matched.sum()),
                "join_share": float(matched.mean()) if len(p) else 0.}
    if coverage["join_share"] < .88:
        raise ValueError(f"Poor participation-to-PBP join: {coverage}")
    q = joined.loc[matched].drop(columns="_merge").copy()
    q = q.loc[pd.to_numeric(q["pass"], errors="coerce").eq(1) &
              q["posteam"].notna() & q["defteam"].notna() & q["epa"].notna()].copy()
    typ = q["defense_man_zone_type"].astype(str).str.upper().str.strip()
    q["man"] = typ.eq("MAN")
    q["zone"] = typ.eq("ZONE")
    q["valid_coverage"] = q["man"] | q["zone"]
    cp = q["defense_coverage_type"].astype(str).str.upper().str.strip()
    q["two_deep"] = cp.isin(["COVER_2", "2_MAN", "COVER_4", "COVER_6"])
    q["valid_shell"] = cp.isin([
        "COVER_0", "COVER_1", "COVER_2", "2_MAN", "COVER_3",
        "COVER_4", "COVER_6", "COVER_9", "COMBO", "BLOWN",
    ])
    q["pressure_known"] = q["was_pressure"].notna()
    q["pressured"] = q["was_pressure"].astype("boolean").fillna(False).astype(bool)
    q["deep_route"] = q["route"].astype(str).str.upper().str.strip().isin(
        ["GO", "POST", "CORNER", "WHEEL"]
    )
    q["route_known"] = q["route"].notna()
    q["epa"] = pd.to_numeric(q["epa"], errors="coerce")
    return q, coverage


def conditional_mean(g: pd.DataFrame, mask: pd.Series) -> float:
    v = g.loc[mask, "epa"].dropna()
    return float(v.mean()) if len(v) >= MIN_SAMPLE else np.nan


def game_unit(g: pd.DataFrame, defense: bool) -> dict:
    man = g["man"]
    zone = g["zone"]
    if defense:
        covered = g["valid_coverage"]
        pressured = g["pressure_known"]
        shelled = g["valid_shell"]
        return {
            "def_man_rate": float(man.sum()/covered.sum()) if covered.any() else np.nan,
            "def_zone_rate": float(zone.sum()/covered.sum()) if covered.any() else np.nan,
            "def_man_epa_allowed": conditional_mean(g, man),
            "def_zone_epa_allowed": conditional_mean(g, zone),
            "def_pressure_rate": float(g.loc[pressured,"pressured"].mean()) if pressured.any() else np.nan,
            "def_2deep_rate": float(g.loc[shelled,"two_deep"].mean()) if shelled.any() else np.nan,
            "def_coverage_charted_n": int(covered.sum()),
            "def_pressure_charted_n": int(pressured.sum()),
        }
    known_route = g["route_known"]
    return {
        "off_vs_man_epa": conditional_mean(g, man),
        "off_vs_zone_epa": conditional_mean(g, zone),
        "off_under_pressure_epa": conditional_mean(g,g["pressured"] & g["pressure_known"]),
        "off_deep_route_share": float(g.loc[known_route,"deep_route"].mean()) if known_route.any() else np.nan,
        "off_route_charted_n":int(known_route.sum()),
        "off_coverage_charted_n":int(g["valid_coverage"].sum()),
    }


def game_summary(rows: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for defensive, key in [(False, "posteam"), (True, "defteam")]:
        for (gid, team), group in rows.groupby(["game_id",key],sort=False):
            record=mapping.setdefault((gid,team), {"game_id":gid, "team":team})
            record.update(game_unit(group,defensive))
    return pd.DataFrame(mapping.values())


def roll_teams(base: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    sch=base[["game_id","gameday","home_team","away_team"]].copy()
    home=sch.rename(columns={"home_team":"team"})[["game_id","gameday","team"]]
    away=sch.rename(columns={"away_team":"team"})[["game_id","gameday","team"]]
    long=pd.concat([home,away],ignore_index=True).merge(
        perf,on=["game_id","team"],how="left",validate="one_to_one")
    long["gameday"]=pd.to_datetime(long["gameday"])
    long=long.sort_values(["team","gameday","game_id"]).reset_index(drop=True)
    names=[]
    for metric in COVERAGE_METRICS:
        for window in WINDOWS:
            stem=f"{metric}_r{window}"
            long[stem]=long.groupby("team",sort=False)[metric].transform(
                lambda z,w=window:z.shift(1).rolling(w,min_periods=2).mean())
            names.append(stem)
    h=long[["game_id","team",*names]].rename(
        columns={"team":"home_team",**{n:"home_"+n for n in names}})
    a=long[["game_id","team",*names]].rename(
        columns={"team":"away_team",**{n:"away_"+n for n in names}})
    return base.merge(h,on=["game_id","home_team"],how="left",validate="one_to_one").merge(
        a,on=["game_id","away_team"],how="left",validate="one_to_one")


def add_coverage_matchups(x: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,list[str]]]:
    x=x.copy()
    dog_home=x["dog_is_home"].astype(bool).to_numpy()
    names=[]
    for metric in COVERAGE_METRICS:
        for win in WINDOWS:
            d,f=oriented_side_values(x,f"{metric}_r{win}",dog_home)
            name=f"participation_diff_{metric}_r{win}"
            x[name]=d-f
            names.append(name)
    products=[]
    for stem,off,defense in MATCHUPS:
        for w in WINDOWS:
            do,fo=oriented_side_values(x,f"{off}_r{w}",dog_home)
            dd,fd=oriented_side_values(x,f"{defense}_r{w}",dog_home)
            name=f"coverage_interaction_{stem}_r{w}"
            x[name]=do*fd-fo*dd
            products.append(name)
    return x, {"main":names,"interactions":products}


def model():
    return Pipeline([
        ("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
        ("scale",StandardScaler()),
        ("logistic",LogisticRegression(C=.02,solver="lbfgs",max_iter=2500,random_state=42)),
    ])


def compare_consensus(df:pd.DataFrame, challenger:str, baseline:str)->dict:
    y=df["dog_win"].to_numpy(int)
    variance=df["p_variance"].to_numpy(float)>=.5
    a=(df[f"p_{challenger}"].to_numpy(float)>=.5)&variance
    b=(df[f"p_{baseline}"].to_numpy(float)>=.5)&variance
    ca=a==y
    cb=b==y
    wins=int((ca&~cb).sum())
    losses=int((~ca&cb).sum())
    return {"challenger":challenger,"baseline":baseline,"games":len(y),
            "correct":int(ca.sum()),"baseline_correct":int(cb.sum()),
            "net_correct":wins-losses,"flip_wins":wins,"flip_losses":losses,
            "exact_p":float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=OUT)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print("Loading baseline game table...",flush=True)
    base=build_model_table(2009,2025)
    print("Loading NFL Next Gen Stats/FTN participation since 2016...",flush=True)
    part=nfl.load_participation(CHART_YEARS).to_pandas()
    pbp=nfl.load_pbp(CHART_YEARS)
    use_pbp=["game_id","play_id","posteam","defteam","epa","pass","rush"]
    plays,join_stats=join_participation(part,pbp.select(use_pbp).to_pandas())
    summary=game_summary(plays)
    # Uncharted plays remain missing, not zeros or invented assignments.
    season_map=base[["game_id","season"]].drop_duplicates("game_id")
    scored=summary.merge(season_map,on="game_id",how="left",validate="many_to_one")
    source_year=scored.groupby("season",dropna=False).agg(
        team_games=("game_id","size"),
        coverage_plays=("off_coverage_charted_n","sum"),
        pressure_plays=("def_pressure_charted_n","sum"),
        route_plays=("off_route_charted_n","sum"),
    ).reset_index()
    source_year.to_csv(args.output_dir/"source_coverage_by_year.csv",index=False)
    base=roll_teams(base,summary)
    upset,features=build_upset_table(base)
    upset,additional=add_coverage_matchups(upset)
    print("Rebuilding frozen variance specialist...",flush=True)
    from_pbp=nfl.load_pbp(list(range(2009,2026)))
    vg=build_variance_game_stats(from_pbp)
    vr=rolling_variance(vg,base)
    vt,vf=build_variance_training(base,vr)
    vt=vt.loc[vt["market_fav_prob"].ge(.525)&vt["market_fav_prob"].lt(.8)].copy()
    variants={
        "same_era_team":features,
        "coverage_main":[*features,*additional["main"]],
        "coverage_interaction":[*features,*additional["main"],*additional["interactions"]],
    }
    predictions=[]
    for year in TEST_YEARS:
        train=upset.loc[upset.season.lt(year)]
        era=train.loc[train.season.ge(2016)]
        test=upset.loc[upset.season.eq(year)]
        vrtr=vt.loc[vt.season.lt(year)]
        vrte=vt.loc[vt.season.eq(year)]
        if not len(era) or not len(test) or not len(vrte):
            raise RuntimeError(f"Missing historical training or forecast data {year}")
        baseline=make_logistic()
        baseline.fit(train[features],train.dog_win.astype(int))
        vmodel=make_variance_catboost()
        vmodel.fit(vrtr[vf],vrtr.dog_win.astype(int))
        out=test[["game_id","season","week","dog_win","market_fav_prob"]].copy()
        out["p_team_full_history"]=baseline.predict_proba(test[features])[:,1]
        for name,cols in variants.items():
            fitted=model()
            fitted.fit(era[cols],era.dog_win.astype(int))
            out[f"p_{name}"]=fitted.predict_proba(test[cols])[:,1]
        vv=vrte[["game_id"]].copy()
        vv["p_variance"]=vmodel.predict_proba(vrte[vf])[:,1]
        out=out.merge(vv,on="game_id",how="left",validate="one_to_one")
        if len(out)!=len(test) or out["p_variance"].isna().any():
            raise RuntimeError("Variance OOS merge incomplete")
        print(f"Compared coverage models for {year}: {len(out)} games",flush=True)
        predictions.append(out)
    x=pd.concat(predictions,ignore_index=True)
    x.to_csv(args.output_dir/"predictions.csv",index=False)
    y=x.dog_win.to_numpy(int)
    scores=[]
    for name in ["market","team_full_history",*variants]:
        call=np.zeros(len(x),bool) if name=="market" else (
            (x[f"p_{name}"].to_numpy(float)>=.5)&(x.p_variance.to_numpy(float)>=.5))
        scores.append({"model":name,"games":len(x),"correct":int((call==y).sum()),
                       "upset_calls":int(call.sum()),"correct_upsets":int(np.sum(y[call]))})
    score_df=pd.DataFrame(scores)
    pairs=pd.DataFrame([compare_consensus(x,k,"team_full_history") for k in variants])
    score_df.to_csv(args.output_dir/"consensus_scores.csv",index=False)
    pairs.to_csv(args.output_dir/"paired_tests.csv",index=False)
    cohort=upset.loc[upset.season.isin(TEST_YEARS)]
    cov=pd.DataFrame([{"feature_group":k,"n_features":len(v),
                       "missing_share":float(cohort[v].isna().mean().mean())}
                       for k,v in additional.items()])
    cov.to_csv(args.output_dir/"feature_coverage.csv",index=False)
    pd.DataFrame([join_stats]).to_csv(args.output_dir/"join_coverage.csv",index=False)
    report=["# Track M — Participation man/zone/pressure matchups","",
        "**Historical exploratory experiment, NOT available in-season in 2026**",
        "for 2023+ participation charting. NFL NGS via nflverse <=2022;",
        "FTN Data via nflverse 2023+ (CC-BY-SA 4.0). All charted game",
        "features enter *later* games only through shifted histories.",
        "Source season-by-season coverage audit is mandatory: do not assume",
        "all coverage and route fields are populated uniformly.","",
        "## Reused 2023–2025 historical upset-domain consensus results","",
        score_df.to_markdown(index=False),"",
        "## Paired changes vs existing matchup+variance consensus","",
        pairs.to_markdown(index=False,floatfmt=".4f"),"",
        "## Historical field coverage by source season","",
        source_year.to_markdown(index=False),"",
        "## Held-out feature missingness","",
        cov.to_markdown(index=False,floatfmt=".4f"),"",
        "Participation players-on-field are descriptive after the game;",
        "they do NOT identify a precise man-to-man assignment or guarantee",
        "who will start at a future kickoff. This data must not be treated",
        "as a 2026 live availability feed. Reused historical seasons require",
        "a new prospective frozen challenger before any production promotion."]
    (args.output_dir/"summary.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print((args.output_dir/"summary.md").read_text(),flush=True)


if __name__=="__main__":
    main()
