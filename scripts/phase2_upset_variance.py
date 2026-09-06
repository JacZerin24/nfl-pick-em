"""Phase 2 explosive-play / variance upset experiment.

Underdogs often win through asymmetric outcomes rather than steady efficiency:
explosive passes/runs, a few huge EPA plays, pressure/turnovers, or special-teams
swings. Phase 1 already models mean EPA/success/turnovers; this experiment adds
play-level tail and volatility features and evaluates them strictly walk-forward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import build_model_table

ROLL_WINDOWS = (4, 8)
VAR_METRICS = (
    "off_expl_pass_rate",
    "off_expl_rush_rate",
    "off_big_epa_rate",
    "off_bust_epa_rate",
    "off_epa_std",
    "def_expl_pass_allowed",
    "def_expl_rush_allowed",
    "def_big_epa_allowed",
    "def_havoc_epa_rate",
    "def_epa_std_allowed",
    "st_epa_mean",
    "st_epa_std",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_upset_variance"))
    return p.parse_args()


def build_variance_game_stats(pbp: pl.DataFrame) -> pd.DataFrame:
    required = {
        "game_id", "posteam", "defteam", "epa", "pass", "rush", "yards_gained",
        "special_teams_play",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise RuntimeError(f"PBP missing variance columns: {sorted(missing)}")

    core = (
        pbp.select(sorted(required))
        .filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
        .with_columns(
            pl.col("pass").fill_null(0).cast(pl.Int8),
            pl.col("rush").fill_null(0).cast(pl.Int8),
            pl.col("special_teams_play").fill_null(0).cast(pl.Int8),
            pl.col("yards_gained").cast(pl.Float64, strict=False),
        )
    )

    scrim = (
        core.filter((pl.col("pass") == 1) | (pl.col("rush") == 1))
        .with_columns(
            ((pl.col("pass") == 1) & (pl.col("yards_gained") >= 20)).cast(pl.Float64).alias("expl_pass"),
            ((pl.col("rush") == 1) & (pl.col("yards_gained") >= 10)).cast(pl.Float64).alias("expl_rush"),
            (pl.col("epa") >= 1.5).cast(pl.Float64).alias("big_epa"),
            (pl.col("epa") <= -1.5).cast(pl.Float64).alias("bust_epa"),
        )
    )

    offense = scrim.group_by(["game_id", "posteam"]).agg(
        pl.col("expl_pass").filter(pl.col("pass") == 1).mean().alias("off_expl_pass_rate"),
        pl.col("expl_rush").filter(pl.col("rush") == 1).mean().alias("off_expl_rush_rate"),
        pl.mean("big_epa").alias("off_big_epa_rate"),
        pl.mean("bust_epa").alias("off_bust_epa_rate"),
        pl.std("epa").alias("off_epa_std"),
    ).rename({"posteam": "team"})

    defense = scrim.group_by(["game_id", "defteam"]).agg(
        pl.col("expl_pass").filter(pl.col("pass") == 1).mean().alias("def_expl_pass_allowed"),
        pl.col("expl_rush").filter(pl.col("rush") == 1).mean().alias("def_expl_rush_allowed"),
        pl.mean("big_epa").alias("def_big_epa_allowed"),
        pl.mean("bust_epa").alias("def_havoc_epa_rate"),
        pl.std("epa").alias("def_epa_std_allowed"),
    ).rename({"defteam": "team"})

    st = core.filter(pl.col("special_teams_play") == 1).group_by(["game_id", "posteam"]).agg(
        pl.mean("epa").alias("st_epa_mean"),
        pl.std("epa").alias("st_epa_std"),
    ).rename({"posteam": "team"})

    out = offense.join(defense, on=["game_id", "team"], how="inner")
    out = out.join(st, on=["game_id", "team"], how="left")
    return out.to_pandas()


def rolling_variance(team_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    context = schedule[["game_id", "gameday", "season", "week"]].drop_duplicates("game_id")
    x = team_games.merge(context, on="game_id", how="left")
    x["gameday"] = pd.to_datetime(x["gameday"])
    x = x.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    for metric in VAR_METRICS:
        for window in ROLL_WINDOWS:
            x[f"{metric}_r{window}"] = x.groupby("team", group_keys=False)[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=2).mean()
            )
    return x


def orient(df: pd.DataFrame, stem: str, dog_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = pd.to_numeric(df[f"home_{stem}"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df[f"away_{stem}"], errors="coerce").to_numpy(float)
    return np.where(dog_home, h, a), np.where(dog_home, a, h)


def build_table(start: int, end: int) -> tuple[pd.DataFrame, list[str]]:
    base = build_model_table(start, end)
    seasons = list(range(start, end + 1))
    print(f"Loading PBP variance features {start}-{end}...")
    pbp = nfl.load_pbp(seasons)
    vg = build_variance_game_stats(pbp)
    vr = rolling_variance(vg, base)
    cols = [f"{m}_r{w}" for m in VAR_METRICS for w in ROLL_WINDOWS]

    home = vr[["game_id", "team", *cols]].rename(columns={"team": "home_team", **{c: f"home_{c}" for c in cols}})
    away = vr[["game_id", "team", *cols]].rename(columns={"team": "away_team", **{c: f"away_{c}" for c in cols}})
    x = base.merge(home, on=["game_id", "home_team"], how="left").merge(away, on=["game_id", "away_team"], how="left")
    x = x.loc[x["home_win"].notna()].copy()

    hp = x["market_home_prob"].to_numpy(float)
    fav_home = hp >= 0.5
    dog_home = ~fav_home
    x["market_fav_prob"] = np.maximum(hp, 1-hp)
    x["market_dog_prob"] = 1-x["market_fav_prob"]
    x["dog_win"] = np.where(fav_home, x["home_win"].astype(int).to_numpy() == 0, x["home_win"].astype(int).to_numpy() == 1).astype(int)
    x["dog_is_home"] = dog_home.astype(int)

    features = ["market_dog_prob", "market_fav_prob", "dog_is_home"]
    vals: dict[tuple[str,int,str], np.ndarray] = {}
    for metric in VAR_METRICS:
        for window in ROLL_WINDOWS:
            stem = f"{metric}_r{window}"
            dog, fav = orient(x, stem, dog_home)
            vals[(metric,window,"dog")] = dog
            vals[(metric,window,"fav")] = fav
            name = f"dog_minus_fav_{metric}_r{window}"
            x[name] = dog-fav
            features.append(name)
        trend = f"trend_diff_{metric}_r4_vs_r8"
        x[trend] = (vals[(metric,4,"dog")]-vals[(metric,8,"dog")]) - (vals[(metric,4,"fav")]-vals[(metric,8,"fav")])
        features.append(trend)

    for w in ROLL_WINDOWS:
        matchup = {
            f"expl_pass_matchup_r{w}":
                (vals[("off_expl_pass_rate",w,"dog")] + vals[("def_expl_pass_allowed",w,"fav")])
                - (vals[("off_expl_pass_rate",w,"fav")] + vals[("def_expl_pass_allowed",w,"dog")]),
            f"expl_rush_matchup_r{w}":
                (vals[("off_expl_rush_rate",w,"dog")] + vals[("def_expl_rush_allowed",w,"fav")])
                - (vals[("off_expl_rush_rate",w,"fav")] + vals[("def_expl_rush_allowed",w,"dog")]),
            f"big_play_matchup_r{w}":
                (vals[("off_big_epa_rate",w,"dog")] + vals[("def_big_epa_allowed",w,"fav")])
                - (vals[("off_big_epa_rate",w,"fav")] + vals[("def_big_epa_allowed",w,"dog")]),
            f"volatility_matchup_r{w}":
                (vals[("off_epa_std",w,"dog")] + vals[("def_epa_std_allowed",w,"fav")])
                - (vals[("off_epa_std",w,"fav")] + vals[("def_epa_std_allowed",w,"dog")]),
            f"special_teams_edge_r{w}": vals[("st_epa_mean",w,"dog")] - vals[("st_epa_mean",w,"fav")],
        }
        for name, value in matchup.items():
            x[name] = value
            features.append(name)

    x = x.loc[(x["market_fav_prob"] >= 0.525) & (x["market_fav_prob"] < 0.80)].copy()
    return x, features


def logit_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.25, C=0.20,
            max_iter=5000, random_state=42,
        )),
    ])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str,float|int]:
    yy=np.asarray(y,int); prob=np.clip(np.asarray(p,float),1e-6,1-1e-6); call=prob>=0.5
    return {
        "games":len(yy), "upset_calls":int(call.sum()), "correct":int(np.sum(call==yy)),
        "accuracy":float(np.mean(call==yy)),
        "upset_call_accuracy":float(np.mean(yy[call]==1)) if call.any() else np.nan,
        "log_loss":float(log_loss(yy,prob,labels=[0,1])), "brier":float(brier_score_loss(yy,prob)),
    }


def backtest(x: pd.DataFrame, features: list[str], first: int, end: int):
    preds=[]; folds=[]
    for season in range(first,end+1):
        tr=x[x.season<season]; te=x[x.season==season]
        if tr.empty or te.empty: continue
        ytr=tr.dog_win.astype(int); y=te.dog_win.astype(int).to_numpy()
        lm=logit_model(); lm.fit(tr[features],ytr); plog=lm.predict_proba(te[features])[:,1]
        cat=CatBoostClassifier(iterations=600,depth=5,learning_rate=.025,loss_function="Logloss",random_seed=42,l2_leaf_reg=15.0,random_strength=.8,verbose=False,allow_writing_files=False)
        cat.fit(tr[features],ytr); pcat=cat.predict_proba(te[features])[:,1]
        pm=te.market_dog_prob.to_numpy(float)
        for name,p in (("market",pm),("variance_logistic",plog),("variance_catboost",pcat)):
            folds.append({"season":season,"model":name,**metrics(y,p)})
        out=te[["game_id","season","week","gameday","home_team","away_team","dog_win","market_fav_prob","market_dog_prob"]].copy()
        out["p_dog_variance_logistic"]=plog; out["p_dog_variance_catboost"]=pcat
        preds.append(out)
        print(f"{season}: market={metrics(y,pm)['accuracy']:.3f} logit={metrics(y,plog)['accuracy']:.3f} cat={metrics(y,pcat)['accuracy']:.3f}")
    return pd.concat(preds,ignore_index=True),pd.DataFrame(folds)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    y=pred.dog_win.astype(int).to_numpy(); market_correct=int(np.sum(y==0)); rows=[]
    for name,col in (("market","market_dog_prob"),("variance_logistic","p_dog_variance_logistic"),("variance_catboost","p_dog_variance_catboost")):
        row={"model":name,**metrics(y,pred[col].to_numpy(float))}; row["net_correct_vs_market"]=row["correct"]-market_correct; rows.append(row)
    return pd.DataFrame(rows).sort_values(["accuracy","brier"],ascending=[False,True])


def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    x,features=build_table(args.start_season,args.end_season)
    pred,fold=backtest(x,features,args.first_test_season,args.end_season)
    summary=summarize(pred)
    pred.to_csv(args.output_dir/"variance_upset_predictions.csv",index=False)
    fold.to_csv(args.output_dir/"variance_upset_folds.csv",index=False)
    summary.to_csv(args.output_dir/"variance_upset_summary.csv",index=False)
    pd.DataFrame({"feature":features}).to_csv(args.output_dir/"variance_upset_features.csv",index=False)
    print("\nVariance upset summary")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
