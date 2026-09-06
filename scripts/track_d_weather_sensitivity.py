"""Secondary reproducible diagnostics for Track D weather interactions."""
from pathlib import Path
import pandas as pd

OUT = Path("outputs/track_d_weather_interactions")
p = pd.read_csv(OUT / "predictions.csv")
h = p.loc[p["season"] >= 2019].copy()
h["y"] = h["home_win"].astype(int)
for stem in ("market", "generic_weather", "weather_interactions"):
    h[f"pick_{stem}"] = (h[f"p_home_{stem}"] >= 0.5).astype(int)

rows=[]
for season,g in h.groupby("season"):
    mc=int((g.pick_market==g.y).sum())
    gc=int((g.pick_generic_weather==g.y).sum())
    wc=int((g.pick_weather_interactions==g.y).sum())
    rows.append({"season":int(season),"games":len(g),"market_correct":mc,"generic_correct":gc,
                 "interaction_correct":wc,"net_vs_market":wc-mc,"net_vs_generic":wc-gc,
                 "disagreements_vs_market":int((g.pick_weather_interactions!=g.pick_market).sum()),
                 "disagreements_vs_generic":int((g.pick_weather_interactions!=g.pick_generic_weather).sum())})
yearly=pd.DataFrame(rows)
yearly.to_csv(OUT/"season_sensitivity.csv",index=False)

dis=h.pick_weather_interactions!=h.pick_market
inter_wins=int((h.loc[dis].pick_weather_interactions==h.loc[dis].y).sum())
market_wins=int((h.loc[dis].pick_market==h.loc[dis].y).sum())

disg=h.pick_weather_interactions!=h.pick_generic_weather
inter_g_wins=int((h.loc[disg].pick_weather_interactions==h.loc[disg].y).sum())
generic_wins=int((h.loc[disg].pick_generic_weather==h.loc[disg].y).sum())

loo=[]
for season in sorted(h.season.unique()):
    g=h.loc[h.season!=season]
    net=int(((g.pick_weather_interactions==g.y).astype(int)-(g.pick_market==g.y).astype(int)).sum())
    loo.append({"left_out_season":int(season),"net_vs_market":net})
loo=pd.DataFrame(loo)
loo.to_csv(OUT/"leave_one_season_out.csv",index=False)

text=f"""# Track D Sensitivity\n\n- Holdout disagreements vs market: **{int(dis.sum())}**; interaction/market wins: **{inter_wins}/{market_wins}**.\n- Disagreements vs generic weather: **{int(disg.sum())}**; interaction/generic wins: **{inter_g_wins}/{generic_wins}**.\n- Leave-one-season-out net advantage vs market ranges from **{int(loo.net_vs_market.min()):+d} to {int(loo.net_vs_market.max()):+d}** correct picks.\n\n## By season\n\n{yearly.to_markdown(index=False)}\n"""
(OUT/"sensitivity.md").write_text(text,encoding="utf-8")
print(text)
