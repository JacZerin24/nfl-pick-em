#!/usr/bin/env python3
"""Track H prospective multi-book NFL market microstructure collector (research-only)."""
from __future__ import annotations
import argparse,json,os,shutil,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import pandas as pd
from track_h_market_archive import build_season_panels,live_reference,near_games
from track_h_market_core import build_consensus,iso_z,normalize

SPORT_KEY="americanfootball_nfl"
DEFAULT_BOOKMAKERS="pinnacle,draftkings,fanduel,betmgm,betrivers,betonlineag,bovada,betus,lowvig,mybookieag"
DEFAULT_MARKETS="h2h,spreads,totals"

def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2026); p.add_argument("--markets",default=DEFAULT_MARKETS)
    p.add_argument("--bookmakers",default=DEFAULT_BOOKMAKERS); p.add_argument("--api-key"); p.add_argument("--fixture",type=Path)
    p.add_argument("--archive-root",type=Path,default=Path("market_archive")); p.add_argument("--live-archive-root",type=Path,default=Path("live_archive"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/track_h_market_microstructure")); p.add_argument("--snapshot-role",default="SCHEDULED_FULL")
    p.add_argument("--near-kickoff-only",action="store_true"); p.add_argument("--near-max-minutes",type=float,default=90.0)
    p.add_argument("--now-utc"); p.add_argument("--no-archive",action="store_true"); return p.parse_args()

def now(value: str|None) -> pd.Timestamp:
    if value:
        ts=pd.Timestamp(value); return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return pd.Timestamp(datetime.now(timezone.utc))

def fetch(api_key: str, markets: str, bookmakers: str) -> tuple[list[dict[str,Any]],dict[str,str]]:
    q=urllib.parse.urlencode({"apiKey":api_key,"bookmakers":bookmakers,"markets":markets,"oddsFormat":"american","dateFormat":"iso"})
    req=urllib.request.Request(f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds?{q}",headers={"User-Agent":"nfl-pick-em-track-h/1.0"})
    with urllib.request.urlopen(req,timeout=30) as r:
        payload=json.loads(r.read().decode("utf-8")); h={"quota_remaining":r.headers.get("x-requests-remaining",""),"quota_used":r.headers.get("x-requests-used",""),"quota_last":r.headers.get("x-requests-last","")}
    if not isinstance(payload,list): raise RuntimeError("Unexpected The Odds API response")
    return payload,h

def summary(outdir: Path,b: pd.DataFrame,c: pd.DataFrame,m: dict[str,Any]) -> None:
    lines=["# Track H Market Microstructure","",f"- Snapshot: `{m['snapshot_utc']}`",f"- Role: `{m['snapshot_role']}`",f"- Markets: `{m['markets']}`",
        f"- Events: **{c['game_id'].nunique() if not c.empty else 0}**",f"- Bookmaker rows: **{len(b)}**",
        f"- Quota last / used / remaining: **{m.get('quota_last','?')} / {m.get('quota_used','?')} / {m.get('quota_remaining','?')}**",""]
    if not c.empty:
        lines += ["| Game | Books | Consensus | Range | Pinnacle-retail | vs production |","|---|---:|---|---:|---:|---:|"]
        for r in c.itertuples(index=False):
            pick=f"{r.consensus_market_pick} {100*r.consensus_fav_prob:.1f}%" if pd.notna(r.consensus_fav_prob) else "n/a"
            rng=f"{100*r.consensus_home_range:.2f} pp" if pd.notna(r.consensus_home_range) else "n/a"; pr=f"{r.pinnacle_minus_retail_pp:+.2f} pp" if pd.notna(r.pinnacle_minus_retail_pp) else "n/a"
            vp=f"{r.consensus_minus_production_pp:+.2f} pp" if pd.notna(r.consensus_minus_production_pp) else "n/a"; lines.append(f"| {r.away_team} @ {r.home_team} | {r.h2h_book_count} | {pick} | {rng} | {pr} | {vp} |")
    lines += ["","Research-only: no Track H field changes the frozen production pick.",""]; (outdir/"summary.md").write_text("\n".join(lines),encoding="utf-8")

def main() -> None:
    a=args(); snap=now(a.now_utc); a.output_dir.mkdir(parents=True,exist_ok=True); ref=live_reference(a.live_archive_root,a.season)
    if a.near_kickoff_only:
        near=near_games(ref,snap,a.near_max_minutes)
        if near.empty:
            (a.output_dir/"summary.md").write_text("# Track H Market Microstructure\n\nNo game is inside the near-kickoff window; no API credits used.\n",encoding="utf-8"); print("No game inside near-kickoff window; skipping API call."); return
        print("Near-kickoff games:",", ".join(near["game_id"].astype(str)))
    headers={}
    if a.fixture: payload=json.loads(a.fixture.read_text(encoding="utf-8"))
    else:
        key=a.api_key or os.environ.get("THE_ODDS_API_KEY")
        if not key: raise SystemExit("THE_ODDS_API_KEY is required for live collection")
        payload,headers=fetch(key,a.markets,a.bookmakers)
    b=normalize(payload,ref,snap,a.snapshot_role)
    if a.near_kickoff_only and not b.empty:
        ids=set(near["game_id"].astype(str)); b=b.loc[b["game_id"].astype(str).isin(ids)].copy()
    c=build_consensus(b,ref); b.to_csv(a.output_dir/"current_bookmakers.csv",index=False); c.to_csv(a.output_dir/"current_consensus.csv",index=False)
    meta={"snapshot_utc":iso_z(snap),"snapshot_role":a.snapshot_role,"sport_key":SPORT_KEY,"markets":a.markets,"bookmakers_requested":a.bookmakers,
        "events_returned":len(payload),"bookmaker_rows_written":len(b),"quota_remaining":headers.get("quota_remaining","fixture"),"quota_used":headers.get("quota_used","fixture"),
        "quota_last":headers.get("quota_last","fixture"),"source":"fixture" if a.fixture else "the-odds-api-v4","research_only":True}
    (a.output_dir/"current_metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8"); summary(a.output_dir,b,c,meta)
    if not a.no_archive and not a.fixture and not b.empty:
        stamp=snap.strftime("%Y%m%dT%H%M%SZ"); d=a.archive_root/str(a.season)/"snapshots"/stamp; d.mkdir(parents=True,exist_ok=False)
        (d/"raw.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); b.to_csv(d/"bookmakers.csv",index=False); c.to_csv(d/"consensus.csv",index=False)
        (d/"metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8"); root=a.archive_root/str(a.season)
        for src,dst in [(d/"bookmakers.csv",root/"latest_bookmakers.csv"),(d/"consensus.csv",root/"latest_consensus.csv"),(d/"metadata.json",root/"latest_metadata.json")]: shutil.copy2(src,dst)
        build_season_panels(a.archive_root,a.season,a.output_dir)
    print((a.output_dir/"summary.md").read_text(encoding="utf-8"))

if __name__=="__main__": main()
