"""Core normalization and cross-book consensus helpers for Track H."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_BOOK = "pinnacle"
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

def iso_z(ts: pd.Timestamp) -> str:
    if pd.isna(ts): return ""
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def american_prob(price: Any) -> float:
    try: x = float(price)
    except (TypeError, ValueError): return math.nan
    if x < 0: return (-x) / ((-x) + 100.0)
    if x > 0: return 100.0 / (x + 100.0)
    return math.nan

def _market(book: dict[str, Any], key: str) -> dict[str, Any] | None:
    return next((m for m in book.get("markets", []) if m.get("key") == key), None)

def _outcomes(m: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not m: return {}
    return {str(o.get("name")): o for o in m.get("outcomes", []) if o.get("name") is not None}

def _match_game(ref: pd.DataFrame, away: str, home: str, kickoff: pd.Timestamp) -> dict[str, Any]:
    if ref.empty: return {}
    x = ref.loc[ref["away_team"].eq(away) & ref["home_team"].eq(home)].copy()
    if x.empty: return {}
    x["delta_seconds"] = (x["kickoff_utc"] - kickoff).abs().dt.total_seconds()
    row = x.sort_values("delta_seconds").iloc[0]
    return row.to_dict() if float(row["delta_seconds"]) <= 21600 else {}

def normalize(payload: list[dict[str, Any]], ref: pd.DataFrame, snapshot: pd.Timestamp, role: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in payload:
        home_name, away_name = str(event.get("home_team", "")), str(event.get("away_team", ""))
        home, away = TEAM_NAME_TO_ABBR.get(home_name, home_name), TEAM_NAME_TO_ABBR.get(away_name, away_name)
        kickoff = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(kickoff): continue
        gr = _match_game(ref, away, home, kickoff)
        game_id = gr.get("game_id") or f"odds_{event.get('id', 'unknown')}"
        for book in event.get("bookmakers", []):
            h2h, spr, tot = _outcomes(_market(book,"h2h")), _outcomes(_market(book,"spreads")), _outcomes(_market(book,"totals"))
            hp, ap = h2h.get(home_name,{}).get("price",math.nan), h2h.get(away_name,{}).get("price",math.nan)
            hraw, araw = american_prob(hp), american_prob(ap); denom = hraw + araw
            novig = hraw/denom if np.isfinite(denom) and denom > 0 else math.nan
            updated = pd.to_datetime(book.get("last_update"), utc=True, errors="coerce")
            rows.append({
                "snapshot_utc":iso_z(snapshot),"snapshot_role":role,"api_event_id":event.get("id"),"game_id":game_id,
                "season":gr.get("season"),"week":gr.get("week"),"kickoff_utc":iso_z(kickoff),
                "lead_minutes":(kickoff-snapshot).total_seconds()/60.0,"away_team":away,"home_team":home,
                "bookmaker_key":book.get("key"),"bookmaker_title":book.get("title"),"bookmaker_last_update":iso_z(updated),
                "book_age_seconds":(snapshot-updated).total_seconds() if not pd.isna(updated) else math.nan,
                "h2h_away_price":ap,"h2h_home_price":hp,"h2h_away_raw_prob":araw,"h2h_home_raw_prob":hraw,
                "h2h_hold":denom-1.0 if np.isfinite(denom) else math.nan,"h2h_home_no_vig_prob":novig,
                "book_market_pick":home if np.isfinite(novig) and novig>=.5 else away if np.isfinite(novig) else None,
                "spread_away_point":spr.get(away_name,{}).get("point",math.nan),"spread_away_price":spr.get(away_name,{}).get("price",math.nan),
                "spread_home_point":spr.get(home_name,{}).get("point",math.nan),"spread_home_price":spr.get(home_name,{}).get("price",math.nan),
                "total_point":tot.get("Over",{}).get("point",tot.get("Under",{}).get("point",math.nan)),
                "total_over_price":tot.get("Over",{}).get("price",math.nan),"total_under_price":tot.get("Under",{}).get("price",math.nan),
            })
    return pd.DataFrame(rows)

def _stat(s: pd.Series, kind: str) -> float:
    x = pd.to_numeric(s,errors="coerce").dropna()
    if x.empty: return math.nan
    return {"mean":x.mean(),"median":x.median(),"std":x.std(ddof=0),"iqr":x.quantile(.75)-x.quantile(.25),"range":x.max()-x.min()}[kind]

def build_consensus(b: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    if b.empty: return pd.DataFrame()
    groups=["snapshot_utc","snapshot_role","api_event_id","game_id","season","week","kickoff_utc","lead_minutes","away_team","home_team"]
    rows=[]
    for keys,g in b.groupby(groups,dropna=False,sort=False):
        m=dict(zip(groups,keys)); hp=pd.to_numeric(g["h2h_home_no_vig_prob"],errors="coerce").dropna()
        med=float(hp.median()) if not hp.empty else math.nan; home,away=str(m["home_team"]),str(m["away_team"])
        pick=home if np.isfinite(med) and med>=.5 else away if np.isfinite(med) else None
        hb,ab=int((hp>.5).sum()),int((hp<.5).sum()); ages=pd.to_numeric(g["book_age_seconds"],errors="coerce").dropna()
        pin=pd.to_numeric(g.loc[g["bookmaker_key"].eq(REFERENCE_BOOK),"h2h_home_no_vig_prob"],errors="coerce").dropna()
        retail=pd.to_numeric(g.loc[~g["bookmaker_key"].eq(REFERENCE_BOOK),"h2h_home_no_vig_prob"],errors="coerce").dropna()
        pp=float(pin.median()) if not pin.empty else math.nan; rm=float(retail.median()) if not retail.empty else math.nan
        prod=ref.loc[ref["game_id"].eq(m["game_id"])] if not ref.empty else pd.DataFrame(); pr=prod.iloc[-1].to_dict() if not prod.empty else {}
        pprod=pd.to_numeric(pd.Series([pr.get("p_home_market")]),errors="coerce").iloc[0]; prod_pick=pr.get("market_pick")
        rows.append({**m,"book_count":int(g["bookmaker_key"].nunique()),"h2h_book_count":int(hp.size),
            "consensus_home_mean":_stat(g["h2h_home_no_vig_prob"],"mean"),"consensus_home_median":med,
            "consensus_home_std":_stat(g["h2h_home_no_vig_prob"],"std"),"consensus_home_iqr":_stat(g["h2h_home_no_vig_prob"],"iqr"),
            "consensus_home_range":_stat(g["h2h_home_no_vig_prob"],"range"),"consensus_market_pick":pick,
            "consensus_fav_prob":max(med,1-med) if np.isfinite(med) else math.nan,"home_fav_books":hb,"away_fav_books":ab,
            "favorite_split":f"{hb}-{ab}" if hp.size else None,"majority_margin":abs(hb-ab)/hp.size if hp.size else math.nan,
            "median_book_age_seconds":float(ages.median()) if not ages.empty else math.nan,"oldest_book_age_seconds":float(ages.max()) if not ages.empty else math.nan,
            "stale_book_count_5m":int((ages>300).sum()) if not ages.empty else 0,"pinnacle_home_prob":pp,"retail_home_median":rm,
            "pinnacle_minus_retail_pp":100*(pp-rm) if np.isfinite(pp) and np.isfinite(rm) else math.nan,
            "spread_home_median":_stat(g["spread_home_point"],"median"),"spread_home_range":_stat(g["spread_home_point"],"range"),
            "total_median":_stat(g["total_point"],"median"),"total_range":_stat(g["total_point"],"range"),
            "production_p_home_market":float(pprod) if pd.notna(pprod) else math.nan,"production_market_pick":prod_pick,
            "production_final_pick":pr.get("final_pick"),"consensus_minus_production_pp":100*(med-float(pprod)) if np.isfinite(med) and pd.notna(pprod) else math.nan,
            "consensus_vs_production_market_flip":bool(pick and prod_pick and pick!=prod_pick)})
    return pd.DataFrame(rows)
