"""Track L — FTN play-charted scheme vs opposing-unit matchup research.

FTN Data / nflverse CC-BY-SA 4.0; attribution: FTN Data via nflverse.
Research-only, no modification to frozen official 2026 picks.
Feature snapshots are lagged; historically collected closing odds are NOT
timestamp-equivalent to the league's final-entry cutoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

from phase1_backtest import build_model_table
from phase2_upset_specialist import build_upset_table, make_logistic, oriented_side_values
from phase2_upset_variance import build_variance_game_stats, rolling_variance
from live_pickem import build_variance_training, make_variance_catboost

SOURCE_START = 2022
FROZEN_TRAIN_START = 2009
TEST_YEARS = (2024, 2025)
ROLLS = (4, 8)
MIN_CONDITIONAL_PLAYS = 3
OUT = Path("outputs/track_l_scheme_matchups")
RATE_NAMES = (
    "off_motion_rate", "off_play_action_rate", "off_screen_rate",
    "off_rpo_rate", "off_shotgun_rate", "def_blitz_rate",
)
EFFECT_NAMES = (
    "off_motion_epa", "off_play_action_epa", "off_screen_epa",
    "off_rpo_epa", "off_vs_blitz_epa", "off_vs_no_blitz_epa",
    "def_motion_epa_allowed", "def_play_action_epa_allowed",
    "def_screen_epa_allowed", "def_rpo_epa_allowed",
    "def_vs_blitz_epa_allowed",
)
ALL_NAMES = RATE_NAMES + EFFECT_NAMES

# Fixed a priori. These are broad schematic tendencies, not individual
# player-to-player, man-zone, or route-level physical matchups.
PRODUCTS = (
    ("blitz_susceptibility", "off_vs_blitz_epa", "def_blitz_rate"),
    ("play_action_exploitation", "off_play_action_epa", "def_play_action_epa_allowed"),
    ("screen_exploitation", "off_screen_epa", "def_screen_epa_allowed"),
    ("rpo_exploitation", "off_rpo_epa", "def_rpo_epa_allowed"),
    ("motion_exploitation", "off_motion_epa", "def_motion_epa_allowed"),
)


def join_ftn_pbp(ftn: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    needed_ftn = {
        "nflverse_game_id", "nflverse_play_id", "is_motion", "is_play_action",
        "is_screen_pass", "is_rpo", "qb_location", "n_blitzers",
    }
    needed_pbp = {"game_id", "play_id", "posteam", "defteam", "epa", "pass", "rush"}
    if needed_ftn - set(ftn.columns) or needed_pbp - set(pbp.columns):
        raise ValueError(
            f"Missing FTN {sorted(needed_ftn-set(ftn))}; PBP {sorted(needed_pbp-set(pbp))}"
        )
    f = ftn[list(sorted(needed_ftn))].rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}
    ).copy()
    p = pbp[list(sorted(needed_pbp))].copy()
    f["play_id"] = pd.to_numeric(f["play_id"], errors="coerce").astype("Int64")
    p["play_id"] = pd.to_numeric(p["play_id"], errors="coerce").astype("Int64")
    f = f.dropna(subset=["game_id", "play_id"])
    p = p.dropna(subset=["game_id", "play_id"])
    f = f.drop_duplicates()
    if f.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Conflicting duplicate FTN chart rows for the same play")
    if p.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Conflicting duplicate play-by-play IDs")
    combined = f.merge(p, on=["game_id", "play_id"], how="left",
                       validate="one_to_one", indicator=True)
    matched = combined["_merge"].eq("both")
    coverage = {
        "ftn_rows": len(f), "matched_rows": int(matched.sum()),
        "matched_share": float(matched.mean()) if len(f) else 0.0,
        "unmatched_rows": int((~matched).sum()),
    }
    if coverage["matched_share"] < 0.90:
        raise ValueError(f"FTN-to-PBP join coverage too low: {coverage}")
    c = combined.loc[matched].drop(columns="_merge")
    c = c.loc[(pd.to_numeric(c["pass"], errors="coerce").eq(1) |
               pd.to_numeric(c["rush"], errors="coerce").eq(1)) &
              c["epa"].notna() & c["posteam"].notna() & c["defteam"].notna()].copy()
    for name in ("is_motion", "is_play_action", "is_screen_pass", "is_rpo"):
        c[name] = c[name].astype("boolean").fillna(False).astype(bool)
    c["pass_play"] = pd.to_numeric(c["pass"], errors="coerce").eq(1)
    c["blitz"] = pd.to_numeric(c["n_blitzers"], errors="coerce").gt(0)
    c["shotgun"] = c["qb_location"].astype(str).eq("S")
    c["epa"] = pd.to_numeric(c["epa"], errors="coerce")
    return c, coverage


def mean_if(g: pd.DataFrame, condition: pd.Series) -> float:
    x = g.loc[condition, "epa"].dropna()
    return float(x.mean()) if len(x) >= MIN_CONDITIONAL_PLAYS else np.nan


def summarize_unit(g: pd.DataFrame, *, defense: bool) -> dict[str, float]:
    n = len(g)
    p = g["pass_play"]
    motion = g["is_motion"]
    pa = g["is_play_action"] & p
    screen = g["is_screen_pass"] & p
    rpo = g["is_rpo"]
    blitz = g["blitz"] & p
    if defense:
        return {
            "def_blitz_rate": float(g.loc[p, "blitz"].mean()) if p.any() else np.nan,
            "def_motion_epa_allowed": mean_if(g, motion),
            "def_play_action_epa_allowed": mean_if(g, pa),
            "def_screen_epa_allowed": mean_if(g, screen),
            "def_rpo_epa_allowed": mean_if(g, rpo),
            "def_vs_blitz_epa_allowed": mean_if(g, blitz),
        }
    return {
        "off_motion_rate": float(motion.mean()) if n else np.nan,
        "off_play_action_rate": float(pa.sum()/p.sum()) if p.any() else np.nan,
        "off_screen_rate": float(screen.sum()/p.sum()) if p.any() else np.nan,
        "off_rpo_rate": float(rpo.mean()) if n else np.nan,
        "off_shotgun_rate": float(g["shotgun"].mean()) if n else np.nan,
        "off_motion_epa": mean_if(g, motion),
        "off_play_action_epa": mean_if(g, pa),
        "off_screen_epa": mean_if(g, screen),
        "off_rpo_epa": mean_if(g, rpo),
        "off_vs_blitz_epa": mean_if(g, blitz),
        "off_vs_no_blitz_epa": mean_if(g, p & ~g["blitz"]),
    }


def team_game_scheme(c: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for side, col in ((False, "posteam"), (True, "defteam")):
        for (game_id, team), group in c.groupby(["game_id", col], sort=False):
            item = rows.setdefault((game_id, team), {"game_id": game_id, "team": team})
            item.update(summarize_unit(group, defense=side))
            item["def_charted_plays" if side else "off_charted_plays"] = len(group)
    return pd.DataFrame(rows.values())


def rolling_scheme(base: pd.DataFrame, per_game: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    schedule = base[["game_id", "gameday", "season", "week", "home_team", "away_team"]]
    home = schedule.rename(columns={"home_team": "team"})[
        ["game_id", "gameday", "season", "week", "team"]]
    away = schedule.rename(columns={"away_team": "team"})[
        ["game_id", "gameday", "season", "week", "team"]]
    full = pd.concat([home, away], ignore_index=True)
    full = full.merge(per_game, on=["game_id", "team"], how="left", validate="one_to_one")
    full["gameday"] = pd.to_datetime(full["gameday"])
    full = full.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    # Current game can only affect *later* games. Noncharted weeks remain
    # missing—not falsely encoded as a team having zero motion or blitz.
    for col in ALL_NAMES:
        for w in ROLLS:
            full[f"{col}_r{w}"] = full.groupby("team", sort=False)[col].transform(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=2).mean()
            )
    rolled = [f"{c}_r{w}" for c in ALL_NAMES for w in ROLLS]
    h = full[["game_id", "team", *rolled]].rename(
        columns={"team": "home_team", **{s: "home_"+s for s in rolled}}
    )
    a = full[["game_id", "team", *rolled]].rename(
        columns={"team": "away_team", **{s: "away_"+s for s in rolled}}
    )
    result = base.merge(h, on=["game_id", "home_team"], how="left", validate="one_to_one")
    result = result.merge(a, on=["game_id", "away_team"], how="left", validate="one_to_one")
    return result, {
        "source_team_games": len(per_game),
        "charted_team_game_fraction": float(full["off_charted_plays"].notna().mean()),
        "source_early_start": SOURCE_START,
    }


def add_underdog_scheme(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    x = table.copy()
    dog_home = x["dog_is_home"].astype(bool).to_numpy()
    rate_features, effect_features, prod_features = [], [], []
    for metric in ALL_NAMES:
        for w in ROLLS:
            dog, fav = oriented_side_values(x, f"{metric}_r{w}", dog_home)
            name = f"scheme_dog_minus_fav_{metric}_r{w}"
            x[name] = dog - fav
            (rate_features if metric in RATE_NAMES else effect_features).append(name)
    for name, off, defense in PRODUCTS:
        for w in ROLLS:
            dog_o, fav_o = oriented_side_values(x, f"{off}_r{w}", dog_home)
            dog_d, fav_d = oriented_side_values(x, f"{defense}_r{w}", dog_home)
            col = f"scheme_matchup_{name}_r{w}"
            x[col] = dog_o * fav_d - fav_o * dog_d
            prod_features.append(col)
    return x, {
        "rates": rate_features,
        "effects": effect_features,
        "matchups": prod_features,
    }


def scheme_logistic():
    """Fixed high shrinkage for limited 2022+ charting era, no grid search."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.015, solver="lbfgs", max_iter=2000, random_state=42
        )),
    ])


def paired_pick_test(x: pd.DataFrame, challenger: str, incumbent: str) -> dict:
    y = x["dog_win"].to_numpy(int)
    var = x["p_variance"].to_numpy(float) >= .5
    a = (x[f"p_{challenger}"].to_numpy(float) >= .5) & var
    b = (x[f"p_{incumbent}"].to_numpy(float) >= .5) & var
    ca = (a == y)
    cb = (b == y)
    wins = int((ca & ~cb).sum())
    losses = int((~ca & cb).sum())
    return {
        "challenger": challenger, "incumbent": incumbent,
        "games": len(x), "correct": int(ca.sum()),
        "incumbent_correct": int(cb.sum()), "net_correct": wins-losses,
        "upset_calls": int(a.sum()), "wins_on_disagreements": wins,
        "losses_on_disagreements": losses,
        "mcnemar_exact_p": float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=list(TEST_YEARS))
    args = parser.parse_args()
    if not set(args.test_seasons).issubset({2024, 2025}):
        raise SystemExit("Only predeclared 2024-2025 diagnostic seasons are permitted")
    args.out.mkdir(parents=True, exist_ok=True)
    print("FTN Data via nflverse (CC-BY-SA 4.0). Loading historical game model...", flush=True)
    base = build_model_table(FROZEN_TRAIN_START, 2025)
    print("Loading FTN charting and matching 2022-2025 play-by-play...", flush=True)
    ftn = nfl.load_ftn_charting(list(range(SOURCE_START, 2026))).to_pandas()
    pbp_22 = nfl.load_pbp(list(range(SOURCE_START, 2026)))
    cols = ["game_id", "play_id", "posteam", "defteam", "epa", "pass", "rush"]
    joined, join_coverage = join_ftn_pbp(ftn, pbp_22.select(cols).to_pandas())
    tg = team_game_scheme(joined)
    base, roll_coverage = rolling_scheme(base, tg)
    upset, existing_features = build_upset_table(base)
    upset, scheme = add_underdog_scheme(upset)
    upset = upset.loc[
        (upset["market_fav_prob"] >= .525) &
        (upset["market_fav_prob"] < .80)
    ].copy()
    print("Loading frozen-style variance specialist, using full historical PBP...", flush=True)
    pbp_full = nfl.load_pbp(list(range(FROZEN_TRAIN_START, 2026)))
    variance_games = build_variance_game_stats(pbp_full)
    variance_rolls = rolling_variance(variance_games, base)
    var_table, var_features = build_variance_training(base, variance_rolls)
    var_table = var_table.loc[
        var_table["market_fav_prob"].ge(.525) & var_table["market_fav_prob"].lt(.80)
    ].copy()

    variants = {
        "same_era_team": existing_features,
        "scheme_rates": [*existing_features, *scheme["rates"]],
        "scheme_effects": [*existing_features, *scheme["rates"], *scheme["effects"]],
        "scheme_all": [*existing_features, *scheme["rates"],
                       *scheme["effects"], *scheme["matchups"]],
    }
    frames = []
    for year in args.test_seasons:
        training = upset.loc[upset["season"].lt(year)]
        training_era = training.loc[training["season"].ge(SOURCE_START)]
        te = upset.loc[upset["season"].eq(year)]
        vrtr = var_table.loc[var_table["season"].lt(year)]
        vrte = var_table.loc[var_table["season"].eq(year)]
        if training_era.empty or te.empty or vrte.empty:
            raise RuntimeError(f"Missing 2022+ model training or variance data for {year}")
        baseline = make_logistic()
        baseline.fit(training[existing_features], training["dog_win"].astype(int))
        var_model = make_variance_catboost()
        var_model.fit(vrtr[var_features], vrtr["dog_win"].astype(int))
        out = te[["game_id", "season", "week", "dog_win", "market_fav_prob",
                  "favorite_team", "underdog_team"]].copy()
        out["p_team_full_history"] = baseline.predict_proba(te[existing_features])[:, 1]
        for variant, cols in variants.items():
            model = scheme_logistic()
            model.fit(training_era[cols], training_era["dog_win"].astype(int))
            out[f"p_{variant}"] = model.predict_proba(te[cols])[:, 1]
        vv = vrte[["game_id"]].copy()
        vv["p_variance"] = var_model.predict_proba(vrte[var_features])[:,1]
        out = out.merge(vv, on="game_id", how="left", validate="one_to_one")
        if out["p_variance"].isna().any() or len(out) != len(te):
            raise RuntimeError(f"Variance predictions could not align for {year}")
        frames.append(out)
        print(f"Finished {year}: {len(out)} eligible games", flush=True)

    x = pd.concat(frames, ignore_index=True)
    x.to_csv(args.out/"predictions.csv", index=False)
    rows = []
    y = x["dog_win"].to_numpy(int)
    for name in ["market", "team_full_history", *variants]:
        call = (x[f"p_{name}"].to_numpy(float) >= .5) if name != "market" else np.zeros(len(x), bool)
        cons = call & (x["p_variance"].to_numpy(float) >= .5)
        row = {"model":name, "games":len(x), "correct":int(np.sum(cons==y)),
               "upset_calls":int(cons.sum()), "correct_upsets":int(np.sum(y[cons]==1))}
        if name != "market":
            p = x[f"p_{name}"].to_numpy(float)
            row.update({
                "brier_dog":float(brier_score_loss(y,p)),
                "log_loss_dog":float(log_loss(y,p,labels=[0,1])),
            })
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out/"consensus_scores.csv",index=False)
    comparisons = pd.DataFrame([
        paired_pick_test(x, name, "team_full_history") for name in variants
    ])
    comparisons.to_csv(args.out/"paired_against_incumbent.csv",index=False)
    side_missing = upset.loc[upset["season"].isin(args.test_seasons)]
    coverage_rows = []
    for group, cols in scheme.items():
        coverage_rows.append({"family":group, "features":len(cols),
                              "raw_missing_rate":float(side_missing[cols].isna().mean().mean())})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.out/"coverage.csv",index=False)
    pd.DataFrame([{"matched_play_rows":len(joined), **join_coverage, **roll_coverage}]
                 ).to_csv(args.out/"source_coverage.csv",index=False)
    lines = [
        "# Track L: FTN-charted schematic usage/efficiency and opponent interactions",
        "", "Attribution: **FTN Data via nflverse, CC-BY-SA 4.0.**",
        "**Exploratory only.** 2024–2025 outcomes have been inspected during",
        "Track F and earlier research. No source here establishes validated",
        "as-of exact pre-kickoff bookmaker odds or current live personnel.",
        "",
        "## Frozen-style upset consensus on 52.5%-<80% favorite domain",
        "",
        scores.to_markdown(index=False,floatfmt=".3f"),"",
        "## Net changes versus full-history matchup + variance consensus",
        "",
        comparisons.to_markdown(index=False,floatfmt=".3f"),"",
        "## Input coverage", "", coverage.to_markdown(index=False,floatfmt=".3f"),"",
        "The full-history incumbent trains the matchup model on 2009–2025",
        "prior seasons; the scheme models train with the available 2022+",
        "FTN history only, so training sample length remains a confound.",
        "No test can be promoted from these reused outcomes.",
        "This test measures scheme types, not individual named OL/EDGE or",
        "receiver/DB matchups; coverage labels and confirmed current",
        "player availability are absent from this public FTN subset.",
    ]
    (args.out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print((args.out/"summary.md").read_text(),flush=True)


if __name__ == "__main__":
    main()
