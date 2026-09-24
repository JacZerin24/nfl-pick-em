"""Track I stage 2: position-group and specific-player diagnostics.

Post-primary diagnostic. Uses the same pregame-safe feature construction as
track_i_individual_star_injuries.py. This is exploratory and cannot turn the
already-touched 2022-2024 holdout into a new production validation set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_backtest import build_model_table
from track_c_player_value_injuries import (
    build_team_week_features,
    load_final_injury_rows,
    load_snap_history,
    safe_logit,
)
from track_i_individual_star_injuries import (
    START, END, DEV, HOLDOUT, bootstrap, merge_diffs, quality_histories,
    star_table, tune, walk, metrics,
)

OUT = Path("outputs/track_i_star_position_diagnostics")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = build_model_table(START, END)
    games = games[games.home_win.notna()].copy()
    games["market_logit"] = safe_logit(games.market_home_prob)

    inj, _ = load_final_injury_rows(START, END)
    snaps = load_snap_history(list(range(START, END + 1)))
    team, _, broad_cols, _ = build_team_week_features(inj, snaps)
    games, bd = merge_diffs(games, team, broad_cols)

    hist = quality_histories(list(range(START, END + 1)))
    stars, audit, star_cols = star_table(inj, snaps, hist)
    games, sd = merge_diffs(games, stars, star_cols)

    def group_features(group):
        return [x for x in sd if x.startswith(f"diff_{group}_")]

    severe = [
        x for x in sd
        if x in {"diff_star_max", "diff_star_top2", "diff_star_count75"}
        or "_severe_" in x
    ]
    out_only = [x for x in sd if "_out_" in x]
    doubtful_only = [x for x in sd if "_doubtful_" in x]
    questionable_only = [x for x in sd if "_questionable_" in x]

    variants = {
        "broad": ["market_logit", *bd],
        "broad_qb": ["market_logit", *bd, *group_features("qb")],
        "broad_skill": ["market_logit", *bd, *group_features("skill")],
        "broad_ol": ["market_logit", *bd, *group_features("ol")],
        "broad_front": ["market_logit", *bd, *group_features("front")],
        "broad_db": ["market_logit", *bd, *group_features("db")],
        "broad_offense": ["market_logit", *bd, *group_features("qb"), *group_features("skill"), *group_features("ol")],
        "broad_defense": ["market_logit", *bd, *group_features("front"), *group_features("db")],
        "broad_severe": ["market_logit", *bd, *severe],
        "broad_out": ["market_logit", *bd, *out_only],
        "broad_doubtful": ["market_logit", *bd, *doubtful_only],
        "broad_questionable": ["market_logit", *bd, *questionable_only],
    }

    pred = games[games.season.isin(HOLDOUT)][
        ["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win", "market_home_prob"]
    ].copy().rename(columns={"market_home_prob": "p_market"})

    tuning, selected = [], {}
    for name, feats in variants.items():
        C, t = tune(games, feats)
        t["model"] = name
        tuning.append(t)
        selected[name] = C
        p = walk(games, feats, C, name)
        pred = pred.merge(p[["game_id", f"p_{name}"]], on="game_id")

    summary, by_season = [], []
    y = pred.home_win.astype(int).to_numpy()
    for name in ["market", *variants.keys()]:
        col = "p_market" if name == "market" else f"p_{name}"
        mm = metrics(y, pred[col])
        row = {"model": name, **mm}
        if name != "market":
            b = bootstrap(y, pred[col], pred.p_market, nboot=30000)
            row.update(lift_vs_market=b["lift"], ci_low=b["low"], ci_high=b["high"], p_positive=b["p_positive"])
        summary.append(row)
        for s in HOLDOUT:
            z = pred[pred.season == s]
            sm = metrics(z.home_win, z[col])
            by_season.append({"model": name, "season": s, **sm, "hit_70pct": sm["accuracy"] >= .70})

    summary = pd.DataFrame(summary).sort_values(["accuracy", "log_loss"], ascending=[False, True])
    by_season = pd.DataFrame(by_season)

    h = pred[["game_id","season","week","home_team","home_win","p_market"]].rename(columns={"home_team":"team"})
    h["team_win"] = h.home_win
    h["p_team_market"] = h.p_market
    a = pred[["game_id","season","week","away_team","home_win","p_market"]].rename(columns={"away_team":"team"})
    a["team_win"] = 1 - a.home_win
    a["p_team_market"] = 1 - a.p_market
    long = pd.concat([h, a], ignore_index=True)
    au = audit.merge(long[["game_id","season","week","team","team_win","p_team_market"]], on=["season","week","team"], how="inner")
    sev = au[
        au.season.isin(HOLDOUT)
        & au.report_bucket.isin(["out","doubtful"])
        & (au.star_impact >= .75)
    ].copy()
    sev["market_residual"] = sev.team_win - sev.p_team_market

    player = (
        sev.groupby(["full_name","position_group"], dropna=False)
        .agg(
            severe_games=("game_id","nunique"),
            mean_star_impact=("star_impact","mean"),
            actual_win_rate=("team_win","mean"),
            market_expected_win_rate=("p_team_market","mean"),
            mean_market_residual=("market_residual","mean"),
        )
        .reset_index()
    )
    player = player[player.severe_games >= 3].sort_values(["severe_games","mean_star_impact"], ascending=[False,False])

    pos_team_game = (
        sev.sort_values("star_impact", ascending=False)
        .drop_duplicates(["game_id","team","position_group"])
    )
    position = (
        pos_team_game.groupby("position_group")
        .agg(
            team_games=("game_id","count"),
            actual_win_rate=("team_win","mean"),
            market_expected_win_rate=("p_team_market","mean"),
            mean_market_residual=("market_residual","mean"),
            mean_star_impact=("star_impact","mean"),
        )
        .reset_index()
        .sort_values("team_games", ascending=False)
    )

    pred.to_csv(OUT/"holdout_predictions.csv", index=False)
    summary.to_csv(OUT/"holdout_summary.csv", index=False)
    by_season.to_csv(OUT/"holdout_by_season.csv", index=False)
    pd.concat(tuning).to_csv(OUT/"development_tuning.csv", index=False)
    player.to_csv(OUT/"specific_player_absence_descriptive.csv", index=False)
    position.to_csv(OUT/"position_star_absence_market_residual.csv", index=False)

    best = summary.iloc[0]
    text = [
        "# Track I Stage 2: Position / Specific-Star Diagnostics", "",
        "**Exploratory post-primary diagnostics; not a fresh production validation.**", "",
        "## Position-group model comparison", "",
        summary.to_markdown(index=False), "",
        "## Best diagnostic model by season", "",
        by_season[by_season.model.eq(best.model)][["season","games","correct","accuracy","hit_70pct"]].to_markdown(index=False), "",
        "## High-impact OUT/DOUBTFUL absences by position", "",
        position.to_markdown(index=False), "",
        "## Repeated specific-player absences (>=3 holdout games)", "",
        player.head(30).to_markdown(index=False), "",
        "Specific-player rows are descriptive only: samples are small and the closing market already incorporates injury news."
    ]
    (OUT/"summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
