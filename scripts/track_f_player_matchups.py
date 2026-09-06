"""Track F: player-level matchup enhancements for NFL straight-up pick'em.

Research-only experiment. Nothing here changes prospective-v1-frozen-2025.

Question
--------
Can compact, pregame-safe player/unit matchup features improve the existing
52.5%-to-80% market-underdog matchup specialist?

Method
------
* Historical player feeds begin in 2018 for the PFR advanced fields used here.
* All player/unit state is shifted before rolling, so the target game never
  contributes to its own features.
* 2020-2021 are development seasons used only to choose regularization.
* 2022-2025 are untouched diagnostic holdout seasons.
* The current production matchup logistic is also re-created with its full
  2009-prior-season training history for a fair operational comparison.
* Player variants train only on 2018+ because those are the seasons with the
  required player feeds.

If a player variant looks promising here, a separate follow-up must test it in
combination with the frozen variance-CatBoost specialist before any production
promotion is considered.
"""
from __future__ import annotations

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
from phase2_upset_specialist import build_upset_table, make_logistic as make_current_matchup

START_SEASON = 2009
PLAYER_START_SEASON = 2018
END_SEASON = 2025
DEV_SEASONS = (2020, 2021)
HOLDOUT_SEASONS = (2022, 2023, 2024, 2025)
ROLL_WINDOW = 4
OUT = Path("outputs/track_f_player_matchups")
EPS = 1e-9

FRONT_POS = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "EDGE"}
DB_POS = {"CB", "DB", "S", "FS", "SS"}
OL_POS = {"C", "G", "OG", "T", "OT", "OL"}
SKILL_POS = {"WR", "TE", "RB", "FB"}

QB_RAW = (
    "qb_epa_per_db",
    "qb_cpoe",
    "qb_pressure_pct",
    "qb_bad_throw_pct",
)
FRONT_RAW = (
    "front_pressure_rate",
    "front_pressure_top1_share",
    "front_sack_hit_rate",
)
SKILL_RAW = (
    "skill_rec_epa_per_target",
    "skill_yac_per_reception",
    "skill_explosive_rate",
    "skill_top1_target_share",
    "skill_top3_target_share",
)
DB_RAW = (
    "db_yards_per_target",
    "db_passer_rating_allowed",
    "db_ball_play_rate",
    "db_missed_tackle_rate",
)
CONT_RAW = (
    "ol_continuity",
    "skill_continuity",
    "front_continuity",
    "db_continuity",
)
RAW_METRICS = QB_RAW + FRONT_RAW + SKILL_RAW + DB_RAW + CONT_RAW

GRID = (
    (0.01, 0.20),
    (0.03, 0.20),
    (0.10, 0.20),
    (0.25, 0.20),
    (0.03, 0.50),
    (0.10, 0.50),
)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_ratio(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> np.ndarray:
    n = np.asarray(numer, dtype=float)
    d = np.asarray(denom, dtype=float)
    return np.where(np.isfinite(d) & (d > 0), n / d, np.nan)


def normalize_pct(s: pd.Series) -> pd.Series:
    x = num(s).fillna(0.0)
    if len(x) and x.quantile(0.95) > 1.5:
        x = x / 100.0
    return x.clip(0.0, 1.0)


def unit_name(position: object) -> str:
    p = str(position or "").upper().strip()
    if p in OL_POS:
        return "ol"
    if p in SKILL_POS:
        return "skill"
    if p in FRONT_POS:
        return "front"
    if p in DB_POS:
        return "db"
    return "other"


def load_player_inputs(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Loading player stats {min(seasons)}-{max(seasons)}...")
    stats = nfl.load_player_stats(seasons).to_pandas()
    stats = stats.loc[stats["season_type"].eq("REG")].copy()

    print(f"Loading snap counts {min(seasons)}-{max(seasons)}...")
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    snaps["offense_pct"] = normalize_pct(snaps["offense_pct"])
    snaps["defense_pct"] = normalize_pct(snaps["defense_pct"])
    snaps["unit"] = snaps["position"].map(unit_name)

    print("Loading PFR advanced passing...")
    pfr_pass = nfl.load_pfr_advstats(seasons, stat_type="pass", summary_level="week").to_pandas()
    pfr_pass = pfr_pass.loc[pfr_pass["game_type"].eq("REG")].copy()

    print("Loading PFR advanced defense...")
    pfr_def = nfl.load_pfr_advstats(seasons, stat_type="def", summary_level="week").to_pandas()
    pfr_def = pfr_def.loc[pfr_def["game_type"].eq("REG")].copy()
    return stats, snaps, pfr_pass, pfr_def


def primary_qb_features(stats: pd.DataFrame, pfr_pass: pd.DataFrame) -> pd.DataFrame:
    qb = stats.loc[stats["position"].eq("QB")].copy()
    qb["attempts_n"] = num(qb["attempts"]).fillna(0.0)
    qb["sacks_n"] = num(qb["sacks_suffered"]).fillna(0.0)
    qb["dropbacks"] = qb["attempts_n"] + qb["sacks_n"]
    qb = qb.sort_values(["game_id", "team", "dropbacks", "attempts_n"])
    qb = qb.groupby(["game_id", "team"], as_index=False).tail(1).copy()
    qb["qb_epa_per_db"] = safe_ratio(num(qb["passing_epa"]), qb["dropbacks"])
    qb["qb_cpoe"] = num(qb["passing_cpoe"])
    out = qb[["game_id", "team", "qb_epa_per_db", "qb_cpoe"]].copy()

    pp = pfr_pass.copy()
    workload = (
        num(pp["times_pressured"]).fillna(0.0)
        + num(pp["times_sacked"]).fillna(0.0)
        + num(pp["times_blitzed"]).fillna(0.0)
    )
    pp["workload"] = workload
    pp = pp.sort_values(["game_id", "team", "workload"])
    pp = pp.groupby(["game_id", "team"], as_index=False).tail(1).copy()
    pp["qb_pressure_pct"] = num(pp["times_pressured_pct"])
    pp["qb_bad_throw_pct"] = num(pp["passing_bad_throw_pct"])
    return out.merge(
        pp[["game_id", "team", "qb_pressure_pct", "qb_bad_throw_pct"]],
        on=["game_id", "team"],
        how="outer",
    )


def skill_features(stats: pd.DataFrame) -> pd.DataFrame:
    x = stats.loc[stats["position"].isin(SKILL_POS)].copy()
    x["targets_n"] = num(x["targets"]).fillna(0.0)
    x["receptions_n"] = num(x["receptions"]).fillna(0.0)
    x["rec_epa_n"] = num(x["receiving_epa"]).fillna(0.0)
    x["yac_n"] = num(x["receiving_yards_after_catch"]).fillna(0.0)
    x["rec20_n"] = num(x["receiving_20"]).fillna(0.0)
    x["target_share_n"] = num(x["target_share"]).fillna(0.0)

    rows: list[dict[str, float | str]] = []
    for (game_id, team), g in x.groupby(["game_id", "team"], sort=False):
        targets = float(g["targets_n"].sum())
        receptions = float(g["receptions_n"].sum())
        shares = np.sort(g["target_share_n"].to_numpy(float))[::-1]
        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "skill_rec_epa_per_target": float(g["rec_epa_n"].sum() / targets) if targets > 0 else np.nan,
                "skill_yac_per_reception": float(g["yac_n"].sum() / receptions) if receptions > 0 else np.nan,
                "skill_explosive_rate": float(g["rec20_n"].sum() / targets) if targets > 0 else np.nan,
                "skill_top1_target_share": float(shares[0]) if len(shares) else np.nan,
                "skill_top3_target_share": float(shares[:3].sum()) if len(shares) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def defense_features(stats: pd.DataFrame, snaps: pd.DataFrame, pfr_def: pd.DataFrame) -> pd.DataFrame:
    team_snaps = snaps.groupby(["game_id", "team"], as_index=False).agg(
        team_def_snaps=("defense_snaps", "max")
    )

    snap_pos = (
        snaps.sort_values(["game_id", "team", "pfr_player_id", "defense_pct"])
        .drop_duplicates(["game_id", "team", "pfr_player_id"], keep="last")
        [["game_id", "team", "pfr_player_id", "position", "unit"]]
    )
    d = pfr_def.merge(snap_pos, on=["game_id", "team", "pfr_player_id"], how="left")
    d = d.merge(team_snaps, on=["game_id", "team"], how="left")
    for c in (
        "def_pressures",
        "def_targets",
        "def_yards_allowed",
        "def_passer_rating_allowed",
        "def_ints",
        "def_missed_tackles",
        "def_tackles_combined",
    ):
        d[c] = num(d[c]).fillna(0.0)

    rows: list[dict[str, float | str]] = []
    for (game_id, team), g in d.groupby(["game_id", "team"], sort=False):
        den = float(num(g["team_def_snaps"]).dropna().max()) if g["team_def_snaps"].notna().any() else np.nan
        front = g.loc[g["unit"].eq("front")]
        db = g.loc[g["unit"].eq("db")]
        front_press = float(front["def_pressures"].sum())
        db_targets = float(db["def_targets"].sum())
        db_tackle_denom = float((db["def_missed_tackles"] + db["def_tackles_combined"]).sum())
        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "front_pressure_rate": front_press / den if np.isfinite(den) and den > 0 else np.nan,
                "front_pressure_top1_share": float(front["def_pressures"].max() / front_press) if front_press > 0 and len(front) else 0.0,
                "db_yards_per_target": float(db["def_yards_allowed"].sum() / db_targets) if db_targets > 0 else np.nan,
                "db_passer_rating_allowed": float(np.average(db["def_passer_rating_allowed"], weights=db["def_targets"])) if db_targets > 0 else np.nan,
                "db_missed_tackle_rate": float(db["def_missed_tackles"].sum() / db_tackle_denom) if db_tackle_denom > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)

    ps = stats.copy()
    ps["unit"] = ps["position"].map(unit_name)
    for c in ("def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended"):
        ps[c] = num(ps[c]).fillna(0.0)
    ps = ps.merge(team_snaps, on=["game_id", "team"], how="left")
    rows2: list[dict[str, float | str]] = []
    for (game_id, team), g in ps.groupby(["game_id", "team"], sort=False):
        den = float(num(g["team_def_snaps"]).dropna().max()) if g["team_def_snaps"].notna().any() else np.nan
        front = g.loc[g["unit"].eq("front")]
        db = g.loc[g["unit"].eq("db")]
        rows2.append(
            {
                "game_id": game_id,
                "team": team,
                "front_sack_hit_rate": float((front["def_sacks"] + front["def_qb_hits"]).sum() / den) if np.isfinite(den) and den > 0 else np.nan,
                "db_ball_play_rate": float((db["def_interceptions"] + db["def_pass_defended"]).sum() / den) if np.isfinite(den) and den > 0 else np.nan,
            }
        )
    return out.merge(pd.DataFrame(rows2), on=["game_id", "team"], how="outer")


def continuity_features(snaps: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    unit_spec = {
        "ol": ("offense_pct", 5),
        "skill": ("offense_pct", 4),
        "front": ("defense_pct", 5),
        "db": ("defense_pct", 5),
    }
    base = context[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id")
    rows: list[dict[str, float | int | str | set[str]]] = []
    for unit, (pct_col, n_top) in unit_spec.items():
        u = snaps.loc[snaps["unit"].eq(unit)].copy()
        u = u.loc[u["pfr_player_id"].notna()].copy()
        u = u.sort_values(["game_id", "team", pct_col], ascending=[True, True, False])
        top = u.groupby(["game_id", "team"], sort=False).head(n_top)
        for (game_id, team), g in top.groupby(["game_id", "team"], sort=False):
            rows.append({"game_id": game_id, "team": team, "unit": unit, "players": set(g["pfr_player_id"].astype(str))})
    sets = pd.DataFrame(rows).merge(base, on="game_id", how="left")
    sets = sets.sort_values(["team", "unit", "gameday", "game_id"])

    out_rows: list[dict[str, float | str]] = []
    prev: dict[tuple[str, str], tuple[int, set[str]]] = {}
    for r in sets.itertuples(index=False):
        key = (str(r.team), str(r.unit))
        current = set(r.players)
        value = np.nan
        if key in prev and prev[key][0] == int(r.season):
            prior = prev[key][1]
            denom = max(1, min(len(current), len(prior)))
            value = len(current.intersection(prior)) / denom
        prev[key] = (int(r.season), current)
        out_rows.append({"game_id": r.game_id, "team": r.team, f"{r.unit}_continuity": value})

    long = pd.DataFrame(out_rows)
    if long.empty:
        return pd.DataFrame(columns=["game_id", "team", *CONT_RAW])
    return long.groupby(["game_id", "team"], as_index=False).first()


def long_schedule(base: pd.DataFrame) -> pd.DataFrame:
    g = base[["game_id", "season", "week", "gameday", "home_team", "away_team"]].drop_duplicates("game_id")
    h = g[["game_id", "season", "week", "gameday", "home_team"]].rename(columns={"home_team": "team"})
    a = g[["game_id", "season", "week", "gameday", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([h, a], ignore_index=True)


def build_player_rolls(base: pd.DataFrame) -> pd.DataFrame:
    seasons = list(range(PLAYER_START_SEASON, END_SEASON + 1))
    stats, snaps, pfr_pass, pfr_def = load_player_inputs(seasons)
    q = primary_qb_features(stats, pfr_pass)
    sk = skill_features(stats)
    de = defense_features(stats, snaps, pfr_def)
    co = continuity_features(snaps, base)

    x = long_schedule(base)
    for f in (q, sk, de, co):
        x = x.merge(f, on=["game_id", "team"], how="left")
    x = x.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    for metric in RAW_METRICS:
        if metric in CONT_RAW:
            x[f"{metric}_r4"] = x.groupby(["team", "season"], group_keys=False)[metric].transform(
                lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean()
            )
        else:
            x[f"{metric}_r4"] = x.groupby("team", group_keys=False)[metric].transform(
                lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean()
            )
    return x


def orient_values(x: pd.DataFrame, stem: str, dog_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = num(x[f"home_{stem}"]).to_numpy(float)
    a = num(x[f"away_{stem}"]).to_numpy(float)
    return np.where(dog_home, h, a), np.where(dog_home, a, h)


def add_player_matchup_features(team_table: pd.DataFrame, rolls: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cols = [f"{m}_r4" for m in RAW_METRICS]
    home = rolls[["game_id", "team", *cols]].rename(columns={"team": "home_team", **{c: f"home_{c}" for c in cols}})
    away = rolls[["game_id", "team", *cols]].rename(columns={"team": "away_team", **{c: f"away_{c}" for c in cols}})
    x = team_table.merge(home, on=["game_id", "home_team"], how="left").merge(
        away, on=["game_id", "away_team"], how="left"
    )
    dog_home = x["dog_is_home"].astype(bool).to_numpy()
    oriented: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diff_names: dict[str, str] = {}
    for metric in RAW_METRICS:
        dog, fav = orient_values(x, f"{metric}_r4", dog_home)
        oriented[metric] = (dog, fav)
        name = f"player_diff_{metric}_r4"
        x[name] = dog - fav
        diff_names[metric] = name

    dog_qb_press, fav_qb_press = oriented["qb_pressure_pct"]
    dog_front_press, fav_front_press = oriented["front_pressure_rate"]
    x["player_pressure_matchup_r4"] = (fav_qb_press + dog_front_press) - (dog_qb_press + fav_front_press)

    dog_skill_epa, fav_skill_epa = oriented["skill_rec_epa_per_target"]
    dog_db_ypt, fav_db_ypt = oriented["db_yards_per_target"]
    x["player_skill_coverage_matchup_r4"] = (dog_skill_epa + fav_db_ypt) - (fav_skill_epa + dog_db_ypt)

    dog_expl, fav_expl = oriented["skill_explosive_rate"]
    dog_db_miss, fav_db_miss = oriented["db_missed_tackle_rate"]
    x["player_explosive_tackle_matchup_r4"] = (dog_expl + fav_db_miss) - (fav_expl + dog_db_miss)

    dog_ol, fav_ol = oriented["ol_continuity"]
    dog_front_hit, fav_front_hit = oriented["front_sack_hit_rate"]
    x["player_protection_front_matchup_r4"] = (dog_ol - fav_front_hit) - (fav_ol - dog_front_hit)

    continuity_dog = np.nanmean(np.vstack([oriented[m][0] for m in CONT_RAW]), axis=0)
    continuity_fav = np.nanmean(np.vstack([oriented[m][1] for m in CONT_RAW]), axis=0)
    x["player_continuity_edge_r4"] = continuity_dog - continuity_fav

    qb_pressure = [diff_names[m] for m in (*QB_RAW, *FRONT_RAW)] + [
        "player_pressure_matchup_r4",
        "player_protection_front_matchup_r4",
    ]
    receiving_coverage = [diff_names[m] for m in (*SKILL_RAW, *DB_RAW)] + [
        "player_skill_coverage_matchup_r4",
        "player_explosive_tackle_matchup_r4",
    ]
    continuity = [diff_names[m] for m in CONT_RAW] + ["player_continuity_edge_r4"]
    groups = {
        "qb_pressure": qb_pressure,
        "receiving_coverage": receiving_coverage,
        "continuity": continuity,
        "all_player": list(dict.fromkeys(qb_pressure + receiving_coverage + continuity)),
    }
    return x, groups


def make_variant_model(C: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=7000,
                    random_state=42,
                ),
            ),
        ]
    )


def tune_variant(x: pd.DataFrame, features: list[str], name: str) -> tuple[tuple[float, float], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for C, l1 in GRID:
        all_y: list[np.ndarray] = []
        all_p: list[np.ndarray] = []
        for season in DEV_SEASONS:
            tr = x.loc[(x["season"] >= PLAYER_START_SEASON) & (x["season"] < season)].copy()
            te = x.loc[x["season"].eq(season)].copy()
            if tr.empty or te.empty:
                continue
            model = make_variant_model(C, l1)
            model.fit(tr[features], tr["dog_win"].astype(int))
            p = model.predict_proba(te[features])[:, 1]
            all_y.append(te["dog_win"].astype(int).to_numpy())
            all_p.append(p)
        y = np.concatenate(all_y)
        p = np.concatenate(all_p)
        rows.append(
            {
                "variant": name,
                "C": C,
                "l1_ratio": l1,
                "games": len(y),
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "accuracy": float(np.mean((p >= 0.5) == y)),
                "upset_calls": int((p >= 0.5).sum()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["log_loss", "brier", "C"]).reset_index(drop=True)
    best = table.iloc[0]
    print(f"{name}: selected C={best.C} l1={best.l1_ratio} dev_logloss={best.log_loss:.5f}")
    return (float(best.C), float(best.l1_ratio)), table


def walk_holdout(x: pd.DataFrame, features: list[str], params: tuple[float, float], name: str) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    C, l1 = params
    for season in HOLDOUT_SEASONS:
        tr = x.loc[(x["season"] >= PLAYER_START_SEASON) & (x["season"] < season)].copy()
        te = x.loc[x["season"].eq(season)].copy()
        model = make_variant_model(C, l1)
        model.fit(tr[features], tr["dog_win"].astype(int))
        z = te[["game_id", "season", "week", "dog_win"]].copy()
        z[f"p_{name}"] = model.predict_proba(te[features])[:, 1]
        out.append(z)
    return pd.concat(out, ignore_index=True)


def current_full_history_predictions(team_table: pd.DataFrame, team_features: list[str]) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for season in HOLDOUT_SEASONS:
        tr = team_table.loc[team_table["season"] < season].copy()
        te = team_table.loc[team_table["season"].eq(season)].copy()
        model = make_current_matchup()
        model.fit(tr[team_features], tr["dog_win"].astype(int))
        z = te[["game_id", "season", "week", "dog_win"]].copy()
        z["p_current_full"] = model.predict_proba(te[team_features])[:, 1]
        out.append(z)
    return pd.concat(out, ignore_index=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    yy = np.asarray(y, int)
    prob = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    call = prob >= 0.5
    return {
        "games": int(len(yy)),
        "correct": int(np.sum(call == yy)),
        "accuracy": float(np.mean(call == yy)),
        "upset_calls": int(call.sum()),
        "upset_call_wins": int(np.sum(yy[call] == 1)),
        "upset_call_accuracy": float(np.mean(yy[call] == 1)) if call.any() else np.nan,
        "log_loss": float(log_loss(yy, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(yy, prob)),
    }


def paired_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 20000) -> dict[str, float]:
    yy = np.asarray(y, int)
    ca = ((np.asarray(pa) >= 0.5) == yy).astype(float)
    cb = ((np.asarray(pb) >= 0.5) == yy).astype(float)
    d = ca - cb
    rng = np.random.default_rng(42)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, len(d), len(d))
        vals[i] = d[idx].mean()
    return {
        "lift_pp": float(100 * d.mean()),
        "ci_low_pp": float(100 * np.quantile(vals, 0.025)),
        "ci_high_pp": float(100 * np.quantile(vals, 0.975)),
        "p_lift_gt_0": float(np.mean(vals > 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building existing team matchup table...")
    base = build_model_table(START_SEASON, END_SEASON)
    team_table, team_features = build_upset_table(base)

    print("Building strictly prior-only player/unit state...")
    rolls = build_player_rolls(base)
    enhanced, groups = add_player_matchup_features(team_table, rolls)
    enhanced = enhanced.loc[enhanced["season"] >= PLAYER_START_SEASON].copy()

    variants = {
        "team_window": team_features,
        "qb_pressure": team_features + groups["qb_pressure"],
        "receiving_coverage": team_features + groups["receiving_coverage"],
        "continuity": team_features + groups["continuity"],
        "all_player": team_features + groups["all_player"],
    }

    tuning_tables: list[pd.DataFrame] = []
    selected: dict[str, tuple[float, float]] = {}
    for name, features in variants.items():
        selected[name], t = tune_variant(enhanced, features, name)
        tuning_tables.append(t)
    tuning = pd.concat(tuning_tables, ignore_index=True)

    pred = enhanced.loc[enhanced["season"].isin(HOLDOUT_SEASONS), [
        "game_id", "season", "week", "gameday", "favorite_team", "underdog_team",
        "dog_win", "market_fav_prob", "market_dog_prob",
    ]].copy()
    current = current_full_history_predictions(team_table, team_features)
    pred = pred.merge(current[["game_id", "p_current_full"]], on="game_id", how="left")
    for name, features in variants.items():
        p = walk_holdout(enhanced, features, selected[name], name)
        pred = pred.merge(p[["game_id", f"p_{name}"]], on="game_id", how="left")

    y = pred["dog_win"].astype(int).to_numpy()
    market_correct = int(np.sum(y == 0))
    current_correct = metrics(y, pred["p_current_full"].to_numpy(float))["correct"]
    rows: list[dict[str, float | int | str]] = []
    models = [("market", "market_dog_prob"), ("current_full", "p_current_full")]
    models += [(name, f"p_{name}") for name in variants]
    for name, col in models:
        m = metrics(y, pred[col].to_numpy(float))
        rows.append(
            {
                "model": name,
                **m,
                "net_correct_vs_market": int(m["correct"] - market_correct),
                "net_correct_vs_current": int(m["correct"] - current_correct),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["correct", "log_loss"], ascending=[False, True])

    boot_current = paired_bootstrap(
        y,
        pred["p_all_player"].to_numpy(float),
        pred["p_current_full"].to_numpy(float),
    )
    boot_market = paired_bootstrap(
        y,
        pred["p_all_player"].to_numpy(float),
        pred["market_dog_prob"].to_numpy(float),
    )

    current_call = pred["p_current_full"].to_numpy(float) >= 0.5
    all_call = pred["p_all_player"].to_numpy(float) >= 0.5
    audit = pred.loc[current_call != all_call].copy()
    audit["current_calls_dog"] = current_call[current_call != all_call]
    audit["all_player_calls_dog"] = all_call[current_call != all_call]
    audit["current_correct"] = (audit["current_calls_dog"].to_numpy() == audit["dog_win"].to_numpy())
    audit["all_player_correct"] = (audit["all_player_calls_dog"].to_numpy() == audit["dog_win"].to_numpy())

    loo_rows = []
    for excluded in HOLDOUT_SEASONS:
        g = pred.loc[pred["season"] != excluded]
        yy = g["dog_win"].astype(int).to_numpy()
        all_c = int(np.sum((g["p_all_player"].to_numpy(float) >= 0.5) == yy))
        cur_c = int(np.sum((g["p_current_full"].to_numpy(float) >= 0.5) == yy))
        mkt_c = int(np.sum(yy == 0))
        loo_rows.append({
            "excluded_season": excluded,
            "games": len(g),
            "all_minus_current": all_c - cur_c,
            "all_minus_market": all_c - mkt_c,
        })
    loo = pd.DataFrame(loo_rows)

    coverage_rows = []
    hold = enhanced.loc[enhanced["season"].isin(HOLDOUT_SEASONS)]
    for metric in RAW_METRICS:
        hcol = f"home_{metric}_r4"
        acol = f"away_{metric}_r4"
        coverage_rows.append({
            "metric": metric,
            "home_nonnull_rate": float(hold[hcol].notna().mean()),
            "away_nonnull_rate": float(hold[acol].notna().mean()),
        })
    coverage = pd.DataFrame(coverage_rows)

    pred.to_csv(OUT / "holdout_predictions.csv", index=False)
    summary.to_csv(OUT / "holdout_summary.csv", index=False)
    tuning.to_csv(OUT / "development_tuning.csv", index=False)
    audit.to_csv(OUT / "disagreements_vs_current.csv", index=False)
    loo.to_csv(OUT / "leave_one_season_out.csv", index=False)
    coverage.to_csv(OUT / "player_feature_coverage.csv", index=False)
    pd.DataFrame(
        [{"variant": k, "C": v[0], "l1_ratio": v[1], "n_features": len(variants[k])} for k, v in selected.items()]
    ).to_csv(OUT / "selected_models.csv", index=False)
    pd.DataFrame(
        [{"group": g, "feature": f} for g, fs in groups.items() for f in fs]
    ).to_csv(OUT / "player_features.csv", index=False)

    all_row = summary.loc[summary["model"].eq("all_player")].iloc[0]
    current_row = summary.loc[summary["model"].eq("current_full")].iloc[0]
    market_row = summary.loc[summary["model"].eq("market")].iloc[0]
    lines = [
        "# Track F player-level matchup study",
        "",
        "**Research only. No production model changes.**",
        "",
        "Protocol: player/PFR data from 2018 onward; 2020-2021 development for regularization; 2022-2025 untouched holdout.",
        "",
        "## Holdout headline",
        "",
        f"- Games in 52.5%-<80% favorite domain: **{int(all_row.games):,}**",
        f"- Market: **{int(market_row.correct)}/{int(market_row.games)} ({100*market_row.accuracy:.2f}%)**",
        f"- Current full-history matchup logistic: **{int(current_row.correct)}/{int(current_row.games)} ({100*current_row.accuracy:.2f}%)**, {int(current_row.net_correct_vs_market):+d} vs market",
        f"- All-player enhanced matchup: **{int(all_row.correct)}/{int(all_row.games)} ({100*all_row.accuracy:.2f}%)**, {int(all_row.net_correct_vs_market):+d} vs market, {int(all_row.net_correct_vs_current):+d} vs current matchup",
        f"- All-player upset calls: **{int(all_row.upset_calls)}**, wins: **{int(all_row.upset_call_wins)}** ({100*all_row.upset_call_accuracy:.1f}% when called)",
        "",
        "## Paired bootstrap",
        "",
        f"- All-player vs current matchup lift: **{boot_current['lift_pp']:+.3f} pp**, 95% CI **[{boot_current['ci_low_pp']:+.3f}, {boot_current['ci_high_pp']:+.3f}]**, P(lift>0) **{100*boot_current['p_lift_gt_0']:.1f}%**",
        f"- All-player vs market lift: **{boot_market['lift_pp']:+.3f} pp**, 95% CI **[{boot_market['ci_low_pp']:+.3f}, {boot_market['ci_high_pp']:+.3f}]**, P(lift>0) **{100*boot_market['p_lift_gt_0']:.1f}%**",
        "",
        "## Model comparison",
        "",
        summary.to_markdown(index=False),
        "",
        "## Robustness",
        "",
        loo.to_markdown(index=False),
        "",
        f"The enhanced and current matchup models disagree on **{len(audit)}** holdout games.",
        "",
        "A positive result here is only a screening result. Promotion requires a second test replacing the matchup leg inside the frozen matchup-logistic + variance-CatBoost consensus, with the 2026 system left untouched until that combined validation is complete.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nHoldout summary")
    print(summary.to_string(index=False))
    print("\nAll-player vs current bootstrap", boot_current)
    print("All-player vs market bootstrap", boot_market)
    print(f"Disagreements all-player vs current: {len(audit)}")


if __name__ == "__main__":
    main()
