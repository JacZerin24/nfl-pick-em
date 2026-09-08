"""Archive/reference helpers for Track H."""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd


def live_reference(root: Path, season: int) -> pd.DataFrame:
    wanted=["game_id","season","week","kickoff_utc","away_team","home_team","p_home_market","market_pick","final_pick"]
    frames=[]
    for path in sorted(root.glob(f"{season}/week_*/latest.csv")):
        try: f=pd.read_csv(path)
        except Exception: continue
        if set(wanted).issubset(f.columns): frames.append(f[wanted].copy())
    if not frames: return pd.DataFrame(columns=wanted)
    x=pd.concat(frames,ignore_index=True); x["kickoff_utc"]=pd.to_datetime(x["kickoff_utc"],utc=True,errors="coerce")
    return x.sort_values("kickoff_utc").drop_duplicates("game_id",keep="last")

def near_games(ref: pd.DataFrame, snapshot: pd.Timestamp, max_minutes: float) -> pd.DataFrame:
    if ref.empty: return ref
    x=ref.copy(); x["lead_minutes"]=(x["kickoff_utc"]-snapshot).dt.total_seconds()/60.0
    return x.loc[(x["lead_minutes"]>0)&(x["lead_minutes"]<=max_minutes)].copy()

def build_season_panels(root: Path, season: int, outdir: Path) -> None:
    snaproot=root/str(season)/"snapshots"; bp=[]; cp=[]
    for p in sorted(snaproot.glob("*/bookmakers.csv")):
        try: bp.append(pd.read_csv(p))
        except Exception: pass
    for p in sorted(snaproot.glob("*/consensus.csv")):
        try: cp.append(pd.read_csv(p))
        except Exception: pass
    if bp:
        b=pd.concat(bp,ignore_index=True).drop_duplicates(["snapshot_utc","game_id","bookmaker_key"],keep="last")
        b.to_csv(outdir/"bookmaker_snapshot_panel.csv",index=False)
    if not cp: return
    c=pd.concat(cp,ignore_index=True).drop_duplicates(["snapshot_utc","game_id"],keep="last")
    c["_snap"]=pd.to_datetime(c["snapshot_utc"],utc=True,errors="coerce"); c=c.sort_values(["game_id","_snap"]).reset_index(drop=True)
    c.drop(columns="_snap").to_csv(outdir/"game_consensus_snapshot_panel.csv",index=False)
    rows=[]
    for game_id,g in c.groupby("game_id",sort=False):
        first,last=g.iloc[0],g.iloc[-1]; disp=pd.to_numeric(g["consensus_home_range"],errors="coerce")
        fp,lp=pd.to_numeric(pd.Series([first.get("consensus_home_median"),last.get("consensus_home_median")]),errors="coerce")
        rows.append({"game_id":game_id,"away_team":last.get("away_team"),"home_team":last.get("home_team"),"kickoff_utc":last.get("kickoff_utc"),
            "snapshots":len(g),"first_snapshot_utc":first.get("snapshot_utc"),"latest_snapshot_utc":last.get("snapshot_utc"),
            "first_consensus_home_prob":fp,"latest_consensus_home_prob":lp,"consensus_home_move_pp":100*(lp-fp) if pd.notna(fp) and pd.notna(lp) else math.nan,
            "first_consensus_pick":first.get("consensus_market_pick"),"latest_consensus_pick":last.get("consensus_market_pick"),
            "consensus_favorite_flip":bool(first.get("consensus_market_pick")!=last.get("consensus_market_pick")),"latest_book_count":last.get("h2h_book_count"),
            "max_cross_book_range_pp":100*float(disp.max()) if disp.notna().any() else math.nan,
            "latest_cross_book_range_pp":100*float(disp.iloc[-1]) if len(disp) and pd.notna(disp.iloc[-1]) else math.nan,
            "latest_pinnacle_minus_retail_pp":last.get("pinnacle_minus_retail_pp"),
            "ever_consensus_vs_production_flip":bool(g["consensus_vs_production_market_flip"].astype(str).str.lower().eq("true").any())})
    pd.DataFrame(rows).to_csv(outdir/"game_movement_summary.csv",index=False)
