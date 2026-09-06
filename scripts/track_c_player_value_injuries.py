"""Track C: replacement-aware player-value injury research.

This experiment asks a narrower question than the original broad injury model:
can we improve straight-up winner picks by distinguishing *which* injured players
matter and how difficult they were to replace?

Pregame-safety rules
--------------------
* Injury status comes from the final historical weekly injury report.
* Every player-role and depth feature uses snap counts strictly BEFORE the target
  week. Current-game snap participation is never used.
* Historical betting data is the closing market, so this is a demanding
  value-of-information test after the market had time to process injury news.
* The primary model is a strongly regularized market-anchored ridge logistic.
  CatBoost is reported only as a secondary nonlinear diagnostic.
* Nothing in this file is allowed to alter prospective-v1-frozen-2025.

The public nflverse injury feed currently ends after 2024, so a positive result
would still require a timestamped current-season injury/inactive source before
live use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1_backtest import build_model_table, score_probabilities
from phase2_injury_value import pos_group, practice_bucket, report_bucket

POSITION_GROUPS = ("QB", "OL", "SKILL", "FRONT", "DB", "ST", "OTHER")
REPORT_BUCKETS = ("out", "doubtful", "questionable")
LOOKBACK_TEAM_GAMES = 4
CORE_ROLE_THRESHOLD = 0.65
MATERIAL_GAP_THRESHOLD = 0.35
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2012)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--holdout-start-season", type=int, default=2019)
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/track_c_player_value_injuries")
    )
    return p.parse_args()


def safe_logit(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def normalize_snap_pct(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if x.quantile(0.95) > 1.5:
        x = x / 100.0
    return x.clip(0.0, 1.0)


def load_snap_history(seasons: list[int]) -> pd.DataFrame:
    print(f"Loading snap counts {min(seasons)}-{max(seasons)}...")
    raw = nfl.load_snap_counts(seasons).to_pandas()
    needed = {
        "pfr_player_id",
        "season",
        "week",
        "game_type",
        "team",
        "position",
        "offense_pct",
        "defense_pct",
    }
    missing = needed.difference(raw.columns)
    if missing:
        raise RuntimeError(f"Snap counts missing required columns: {sorted(missing)}")

    keep = list(needed)
    if "st_pct" in raw.columns:
        keep.append("st_pct")
    s = raw.loc[raw["game_type"].eq("REG"), keep].copy()
    s["offense_pct"] = normalize_snap_pct(s["offense_pct"])
    s["defense_pct"] = normalize_snap_pct(s["defense_pct"])
    if "st_pct" in s.columns:
        s["st_pct"] = normalize_snap_pct(s["st_pct"])
    else:
        s["st_pct"] = 0.0
    s["usage"] = s[["offense_pct", "defense_pct", "st_pct"]].max(axis=1)
    s["position_group"] = s["position"].map(pos_group)
    s["order_key"] = s["season"].astype(int) * 100 + s["week"].astype(int)
    return s.sort_values(["team", "position_group", "order_key", "pfr_player_id"]).reset_index(drop=True)


def load_final_injury_rows(start_season: int, end_season: int) -> tuple[pd.DataFrame, dict[str, float]]:
    seasons = list(range(start_season, end_season + 1))
    print(f"Loading injury reports {start_season}-{end_season}...")
    raw = nfl.load_injuries(seasons).to_pandas()
    players = nfl.load_players().to_pandas()

    needed = {
        "season",
        "week",
        "team",
        "gsis_id",
        "position",
        "report_status",
        "practice_status",
    }
    missing = needed.difference(raw.columns)
    if missing:
        raise RuntimeError(f"Injury data missing required columns: {sorted(missing)}")

    type_col = "game_type" if "game_type" in raw.columns else "season_type" if "season_type" in raw.columns else None
    if type_col is None:
        raise RuntimeError("Injury feed has neither game_type nor season_type")

    inj = raw.loc[raw[type_col].eq("REG")].copy()
    if "date_modified" in inj.columns:
        inj["date_modified"] = pd.to_datetime(inj["date_modified"], errors="coerce", utc=True)
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")

    mapping = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    inj = inj.merge(mapping, on="gsis_id", how="left")
    inj["position_group"] = inj["position"].map(pos_group)
    inj["report_bucket"] = inj["report_status"].map(report_bucket)
    inj["practice_bucket"] = inj["practice_status"].map(practice_bucket)
    inj["order_key"] = inj["season"].astype(int) * 100 + inj["week"].astype(int)

    diagnostics = {
        "injury_rows": float(len(inj)),
        "pfr_id_match_rate": float(inj["pfr_id"].notna().mean()),
    }
    return inj, diagnostics


def build_player_history_index(snaps: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    valid = snaps.loc[snaps["pfr_player_id"].notna(), ["pfr_player_id", "order_key", "usage"]]
    for pid, g in valid.groupby("pfr_player_id", sort=False):
        gg = g.sort_values("order_key")
        index[str(pid)] = (
            gg["order_key"].to_numpy(dtype=int),
            gg["usage"].to_numpy(dtype=float),
        )
    return index


def prior_player_role(
    index: dict[str, tuple[np.ndarray, np.ndarray]], pid: object, target_order: int
) -> tuple[float, int]:
    if pd.isna(pid) or str(pid) not in index:
        return 0.0, 0
    orders, usage = index[str(pid)]
    end = int(np.searchsorted(orders, target_order, side="left"))
    if end <= 0:
        return 0.0, 0
    start = max(0, end - LOOKBACK_TEAM_GAMES)
    return float(np.mean(usage[start:end])), end


def build_unit_state_index(
    snaps: pd.DataFrame, targets: pd.DataFrame
) -> dict[tuple[str, str, int], dict[str, object]]:
    """Build prior-only empirical depth states for injury team/position/week keys."""
    by_unit = {
        (str(team), str(pg)): g.copy()
        for (team, pg), g in snaps.groupby(["team", "position_group"], sort=False)
    }
    states: dict[tuple[str, str, int], dict[str, object]] = {}
    target_keys = targets[["team", "position_group", "order_key"]].drop_duplicates()

    for row in target_keys.itertuples(index=False):
        key = (str(row.team), str(row.position_group))
        target_order = int(row.order_key)
        g = by_unit.get(key)
        if g is None or g.empty:
            states[(key[0], key[1], target_order)] = {
                "roles": pd.Series(dtype=float), "slots": 1.0, "core_ids": set()
            }
            continue

        prior = g.loc[g["order_key"] < target_order]
        if prior.empty:
            states[(key[0], key[1], target_order)] = {
                "roles": pd.Series(dtype=float), "slots": 1.0, "core_ids": set()
            }
            continue

        game_keys = np.sort(prior["order_key"].unique())[-LOOKBACK_TEAM_GAMES:]
        recent = prior.loc[prior["order_key"].isin(game_keys)]
        roles = (
            recent.groupby("pfr_player_id", dropna=True)["usage"]
            .mean()
            .sort_values(ascending=False)
        )
        game_slots = recent.groupby("order_key")["usage"].sum()
        slots = float(game_slots.mean()) if len(game_slots) else 1.0
        n_core = max(1, int(np.rint(slots)))
        core_ids = set(str(x) for x in roles.head(n_core).index)
        states[(key[0], key[1], target_order)] = {
            "roles": roles,
            "slots": max(slots, 1.0),
            "core_ids": core_ids,
        }
    return states


def backup_gap(
    state: dict[str, object], injured: pd.DataFrame, status_set: set[str]
) -> tuple[float, float, int, float]:
    """Return injured core role, empirical backup capacity, core count, uncovered gap."""
    roles: pd.Series = state["roles"]  # type: ignore[assignment]
    core_ids: set[str] = state["core_ids"]  # type: ignore[assignment]
    if roles.empty or injured.empty:
        return 0.0, 0.0, 0, 0.0

    selected = injured.loc[injured["report_bucket"].isin(status_set)].copy()
    if selected.empty:
        return 0.0, 0.0, 0, 0.0

    selected["pid_str"] = selected["pfr_id"].astype("string")
    selected["is_core"] = selected["pid_str"].isin(core_ids) | (
        selected["prior_role"] >= CORE_ROLE_THRESHOLD
    )
    core = selected.loc[selected["is_core"]]
    core_count = int(len(core))
    injured_core_role = float(core["prior_role"].sum())
    if core_count == 0:
        return 0.0, 0.0, 0, 0.0

    n_core_slots = max(1, len(core_ids))
    unavailable = set(selected["pid_str"].dropna().astype(str))
    backup_roles = []
    for rank, (pid, role) in enumerate(roles.items(), start=1):
        pid_str = str(pid)
        if rank <= n_core_slots or pid_str in unavailable:
            continue
        backup_roles.append(float(role))
    backup_roles.sort(reverse=True)
    capacity = float(sum(backup_roles[:core_count]))
    gap = max(0.0, injured_core_role - capacity)
    return injured_core_role, capacity, core_count, gap


def build_team_week_features(
    injuries: pd.DataFrame, snaps: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float], list[str], list[str]]:
    player_index = build_player_history_index(snaps)
    inj = injuries.copy()
    roles = []
    prior_games = []
    for row in inj.itertuples(index=False):
        role, games = prior_player_role(player_index, row.pfr_id, int(row.order_key))
        roles.append(role)
        prior_games.append(games)
    inj["prior_role"] = np.asarray(roles, dtype=float)
    inj["prior_games"] = np.asarray(prior_games, dtype=int)
    inj["high_role"] = inj["prior_role"] >= CORE_ROLE_THRESHOLD

    states = build_unit_state_index(snaps, inj)
    rows: list[dict[str, float | int | str]] = []

    for (season, week, team), team_inj in inj.groupby(["season", "week", "team"], sort=False):
        row: dict[str, float | int | str] = {
            "season": int(season),
            "week": int(week),
            "team": str(team),
            "injury_players_listed": int(len(team_inj)),
            "role_match_rate": float((team_inj["prior_games"] > 0).mean()),
            "out_count": int(team_inj["report_bucket"].eq("out").sum()),
            "doubtful_count": int(team_inj["report_bucket"].eq("doubtful").sum()),
            "questionable_count": int(team_inj["report_bucket"].eq("questionable").sum()),
            "dnp_count": int(team_inj["practice_bucket"].eq("dnp").sum()),
            "limited_count": int(team_inj["practice_bucket"].eq("limited").sum()),
        }
        for status in REPORT_BUCKETS:
            m = team_inj["report_bucket"].eq(status)
            vals = team_inj.loc[m, "prior_role"]
            row[f"{status}_role_total"] = float(vals.sum())
            row[f"{status}_max_role"] = float(vals.max()) if len(vals) else 0.0
            row[f"{status}_high_role_count"] = int((vals >= CORE_ROLE_THRESHOLD).sum())
        for practice in ("dnp", "limited"):
            m = team_inj["practice_bucket"].eq(practice)
            row[f"{practice}_role_total"] = float(team_inj.loc[m, "prior_role"].sum())

        global_out_core = global_severe_core = 0.0
        global_out_capacity = global_severe_capacity = 0.0
        global_out_gap = global_severe_gap = 0.0
        global_out_core_count = global_severe_core_count = 0
        multi_core_units = 0

        for pg in POSITION_GROUPS:
            pg_inj = team_inj.loc[team_inj["position_group"].eq(pg)].copy()
            state = states.get((str(team), pg, int(season) * 100 + int(week)), {
                "roles": pd.Series(dtype=float), "slots": 1.0, "core_ids": set()
            })
            out_core, out_cap, out_n, out_gap = backup_gap(state, pg_inj, {"out"})
            sev_core, sev_cap, sev_n, sev_gap = backup_gap(
                state, pg_inj, {"out", "doubtful"}
            )
            q_role = float(
                pg_inj.loc[pg_inj["report_bucket"].eq("questionable"), "prior_role"].sum()
            )
            row[f"{pg.lower()}_out_core_role"] = out_core
            row[f"{pg.lower()}_severe_core_role"] = sev_core
            row[f"{pg.lower()}_questionable_role"] = q_role
            row[f"{pg.lower()}_out_core_count"] = out_n
            row[f"{pg.lower()}_severe_core_count"] = sev_n
            row[f"{pg.lower()}_out_backup_gap"] = out_gap
            row[f"{pg.lower()}_severe_backup_gap"] = sev_gap
            row[f"{pg.lower()}_severe_backup_capacity"] = sev_cap
            row[f"{pg.lower()}_unit_slots"] = float(state["slots"])
            row[f"{pg.lower()}_multi_core_loss"] = int(sev_n >= 2)

            global_out_core += out_core
            global_severe_core += sev_core
            global_out_capacity += out_cap
            global_severe_capacity += sev_cap
            global_out_gap += out_gap
            global_severe_gap += sev_gap
            global_out_core_count += out_n
            global_severe_core_count += sev_n
            multi_core_units += int(sev_n >= 2)

        row["out_core_role_total"] = global_out_core
        row["severe_core_role_total"] = global_severe_core
        row["out_backup_capacity"] = global_out_capacity
        row["severe_backup_capacity"] = global_severe_capacity
        row["out_backup_gap_total"] = global_out_gap
        row["severe_backup_gap_total"] = global_severe_gap
        row["out_core_count"] = global_out_core_count
        row["severe_core_count"] = global_severe_core_count
        row["multi_core_units"] = multi_core_units
        row["out_gap_ratio"] = global_out_gap / max(global_out_core, EPS)
        row["severe_gap_ratio"] = global_severe_gap / max(global_severe_core, EPS)
        row["material_value_injury"] = int(
            global_severe_gap >= MATERIAL_GAP_THRESHOLD or global_out_core >= CORE_ROLE_THRESHOLD
        )
        rows.append(row)

    team_week = pd.DataFrame(rows).fillna(0.0)
    diagnostics = {
        "injury_rows": float(len(inj)),
        "prior_role_match_rate": float((inj["prior_games"] > 0).mean()),
        "high_role_injury_share": float(inj["high_role"].mean()),
        "team_weeks": float(len(team_week)),
        "material_team_week_share": float(team_week["material_value_injury"].mean()),
    }

    broad = [
        "injury_players_listed",
        "out_count",
        "doubtful_count",
        "questionable_count",
        "dnp_count",
        "limited_count",
        "out_role_total",
        "doubtful_role_total",
        "questionable_role_total",
        "dnp_role_total",
        "limited_role_total",
    ]
    exclude = {"season", "week", "team", "role_match_rate", "material_value_injury", *broad}
    player_value = [c for c in team_week.columns if c not in exclude]
    return team_week, diagnostics, broad, player_value


def merge_team_features(
    games: pd.DataFrame, team_week: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    home = team_week[["season", "week", "team", *feature_cols, "material_value_injury"]].copy()
    away = home.copy()
    home = home.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}, "material_value_injury": "home_material_value_injury"}
    )
    away = away.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}, "material_value_injury": "away_material_value_injury"}
    )
    out = games.merge(home, on=["season", "week", "home_team"], how="left")
    out = out.merge(away, on=["season", "week", "away_team"], how="left")
    diffs = []
    for c in feature_cols:
        h, a = f"home_{c}", f"away_{c}"
        out[h] = pd.to_numeric(out[h], errors="coerce").fillna(0.0)
        out[a] = pd.to_numeric(out[a], errors="coerce").fillna(0.0)
        name = f"diff_{c}"
        out[name] = out[h] - out[a]
        diffs.append(name)
    out["home_material_value_injury"] = pd.to_numeric(
        out["home_material_value_injury"], errors="coerce"
    ).fillna(0).astype(int)
    out["away_material_value_injury"] = pd.to_numeric(
        out["away_material_value_injury"], errors="coerce"
    ).fillna(0).astype(int)
    out["material_value_injury_game"] = (
        (out["home_material_value_injury"] == 1) | (out["away_material_value_injury"] == 1)
    ).astype(int)
    return out, diffs


def make_ridge(c_value: float = 0.08) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    C=c_value,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_backtest(
    games: pd.DataFrame,
    broad_diffs: list[str],
    player_diffs: list[str],
    first_test_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = games.loc[games["home_win"].notna()].copy()
    completed["market_logit"] = safe_logit(completed["market_home_prob"])
    pred_rows: list[pd.DataFrame] = []
    folds: list[dict[str, float | int | str]] = []

    calibration_cols = ["market_logit"]
    broad_cols = ["market_logit", *broad_diffs]
    player_cols = ["market_logit", *broad_diffs, *player_diffs]

    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue
        y_train = train["home_win"].astype(int)
        y_test = test["home_win"].astype(int)
        probs: dict[str, np.ndarray] = {
            "market": test["market_home_prob"].to_numpy(dtype=float)
        }

        for name, cols, c_val in (
            ("market_calibration", calibration_cols, 1.0),
            ("broad_injury_ridge", broad_cols, 0.08),
            ("player_value_ridge", player_cols, 0.08),
        ):
            model = make_ridge(c_val)
            model.fit(train[cols], y_train)
            probs[name] = model.predict_proba(test[cols])[:, 1]

        cat_cols = ["market_home_prob", *broad_diffs, *player_diffs]
        cat = CatBoostClassifier(
            iterations=350,
            depth=4,
            learning_rate=0.025,
            loss_function="Logloss",
            random_seed=42,
            l2_leaf_reg=12.0,
            random_strength=0.8,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(train[cat_cols], y_train)
        probs["player_value_catboost"] = cat.predict_proba(test[cat_cols])[:, 1]

        keep = [
            "game_id", "season", "week", "gameday", "away_team", "home_team", "home_win",
            "market_home_prob", "material_value_injury_game",
        ]
        pred = test[keep].copy()
        for name, p in probs.items():
            pred[f"p_home_{name}"] = p
            folds.append(
                {"season": season, "model": name, "games": len(test), **score_probabilities(y_test, p)}
            )
        pred_rows.append(pred)
        print(
            f"{season}: market={score_probabilities(y_test, probs['market'])['accuracy']:.3f} "
            f"broad={score_probabilities(y_test, probs['broad_injury_ridge'])['accuracy']:.3f} "
            f"value={score_probabilities(y_test, probs['player_value_ridge'])['accuracy']:.3f} "
            f"cat={score_probabilities(y_test, probs['player_value_catboost'])['accuracy']:.3f} "
            f"material={int(test['material_value_injury_game'].sum())}"
        )

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(folds)


def model_scores(pred: pd.DataFrame, mask: np.ndarray | pd.Series, label: str) -> pd.DataFrame:
    sub = pred.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    y = sub["home_win"].astype(int)
    rows = []
    for model in (
        "market",
        "market_calibration",
        "broad_injury_ridge",
        "player_value_ridge",
        "player_value_catboost",
    ):
        p = sub[f"p_home_{model}"].to_numpy(float)
        rows.append({"scope": label, "model": model, "games": len(sub), **score_probabilities(y, p)})
    return pd.DataFrame(rows)


def paired_stats(pred: pd.DataFrame, model: str, mask: np.ndarray | pd.Series) -> dict[str, float | int]:
    sub = pred.loc[mask].copy()
    y = sub["home_win"].astype(int).to_numpy()
    market_pick = (sub["p_home_market"].to_numpy(float) >= 0.5).astype(int)
    model_pick = (sub[f"p_home_{model}"].to_numpy(float) >= 0.5).astype(int)
    market_correct = (market_pick == y).astype(int)
    model_correct = (model_pick == y).astype(int)
    disagree = model_pick != market_pick
    return {
        "games": int(len(sub)),
        "net_correct_vs_market": int((model_correct - market_correct).sum()),
        "accuracy_lift": float((model_correct - market_correct).mean()) if len(sub) else np.nan,
        "disagreements": int(disagree.sum()),
        "model_wins_on_disagreements": int((model_correct[disagree]).sum()) if disagree.any() else 0,
        "market_wins_on_disagreements": int((market_correct[disagree]).sum()) if disagree.any() else 0,
    }


def paired_bootstrap(
    pred: pd.DataFrame,
    model: str,
    mask: np.ndarray | pd.Series,
    n: int = 30000,
    seed: int = 42,
) -> dict[str, float]:
    sub = pred.loc[mask].copy()
    y = sub["home_win"].astype(int).to_numpy()
    market = (sub["p_home_market"].to_numpy(float) >= 0.5).astype(int)
    model_pick = (sub[f"p_home_{model}"].to_numpy(float) >= 0.5).astype(int)
    delta = (model_pick == y).astype(float) - (market == y).astype(float)
    if len(delta) == 0:
        return {"lift": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_gt_0": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    chunk = 2000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        idx = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[idx].mean(axis=1)
    return {
        "lift": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_gt_0": float(np.mean(means > 0)),
    }


def write_summary(
    path: Path,
    scores: pd.DataFrame,
    paired: pd.DataFrame,
    boot: pd.DataFrame,
    diagnostics: dict[str, float],
    holdout_start: int,
    end_season: int,
) -> None:
    def row(scope: str, model: str) -> pd.Series:
        return scores.loc[(scores["scope"] == scope) & (scores["model"] == model)].iloc[0]

    h_market = row("holdout_all", "market")
    h_value = row("holdout_all", "player_value_ridge")
    m_market = row("holdout_material", "market") if (scores["scope"] == "holdout_material").any() else None
    m_value = row("holdout_material", "player_value_ridge") if m_market is not None else None
    p_all = paired.loc[paired["scope"].eq("holdout_all")].iloc[0]
    p_mat = paired.loc[paired["scope"].eq("holdout_material")].iloc[0] if (paired["scope"] == "holdout_material").any() else None
    b_all = boot.loc[boot["scope"].eq("holdout_all")].iloc[0]
    b_mat = boot.loc[boot["scope"].eq("holdout_material")].iloc[0] if (boot["scope"] == "holdout_material").any() else None

    lines = [
        "# Track C: Player-Value Injury Study",
        "",
        "**Research-only. No change to `prospective-v1-frozen-2025`.**",
        "",
        "The primary model is a pre-specified strongly regularized market-anchored ridge model. Player value is represented by prior snap role, empirically inferred core-unit status, multiple core losses, and prior-only backup capacity/replacement gap.",
        "",
        f"Historical injury/snap window: **2012-{end_season}**. Official diagnostic holdout: **{holdout_start}-{end_season}**.",
        f"Prior-role match rate: **{diagnostics['prior_role_match_rate']:.1%}**.",
        f"Material player-value injury team-weeks: **{diagnostics['material_team_week_share']:.1%}**.",
        "",
        "## Holdout: all games",
        "",
        f"- Market: **{int(round(h_market['accuracy'] * h_market['games']))}/{int(h_market['games'])} ({h_market['accuracy']:.2%})**",
        f"- Player-value ridge: **{int(round(h_value['accuracy'] * h_value['games']))}/{int(h_value['games'])} ({h_value['accuracy']:.2%})**",
        f"- Net correct vs market: **{int(p_all['net_correct_vs_market']):+d}**",
        f"- Disagreements: **{int(p_all['disagreements'])}**; model/market wins on disagreements: **{int(p_all['model_wins_on_disagreements'])}/{int(p_all['market_wins_on_disagreements'])}**",
        f"- Paired lift: **{b_all['lift']:+.3%}**; bootstrap 95% CI **[{b_all['ci_low']:+.3%}, {b_all['ci_high']:+.3%}]**; P(lift>0) **{b_all['p_gt_0']:.1%}**",
    ]
    if m_market is not None and m_value is not None and p_mat is not None and b_mat is not None:
        lines += [
            "",
            "## Holdout: material injury games",
            "",
            f"- Games: **{int(m_market['games'])}**",
            f"- Market accuracy: **{m_market['accuracy']:.2%}**",
            f"- Player-value ridge accuracy: **{m_value['accuracy']:.2%}**",
            f"- Net correct vs market: **{int(p_mat['net_correct_vs_market']):+d}**",
            f"- Disagreements: **{int(p_mat['disagreements'])}**; model/market wins: **{int(p_mat['model_wins_on_disagreements'])}/{int(p_mat['market_wins_on_disagreements'])}**",
            f"- Paired lift: **{b_mat['lift']:+.3%}**; bootstrap 95% CI **[{b_mat['ci_low']:+.3%}, {b_mat['ci_high']:+.3%}]**; P(lift>0) **{b_mat['p_gt_0']:.1%}**",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "Because the benchmark is the historical closing market, a null result does not mean injuries are unimportant. It means final-report player-value information was already priced well enough that this model could not reliably improve straight-up picks. A positive result would justify a stricter follow-up on timestamped late inactives before any live promotion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.start_season < 2012:
        raise SystemExit("Snap-count data begins in 2012")
    if args.end_season > 2024:
        raise SystemExit("Public nflverse historical injury feed currently ends after 2024")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seasons = list(range(args.start_season, args.end_season + 1))
    games = build_model_table(args.start_season, args.end_season)
    injuries, injury_diag = load_final_injury_rows(args.start_season, args.end_season)
    snaps = load_snap_history(seasons)
    team_week, value_diag, broad_cols, player_cols = build_team_week_features(injuries, snaps)
    diagnostics = {**injury_diag, **value_diag}

    games, broad_diffs = merge_team_features(games, team_week, broad_cols)
    # Add the player-value features onto the already merged table without duplicating material flags.
    player_only = team_week[["season", "week", "team", *player_cols]].copy()
    home = player_only.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in player_cols}})
    away = player_only.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in player_cols}})
    games = games.merge(home, on=["season", "week", "home_team"], how="left")
    games = games.merge(away, on=["season", "week", "away_team"], how="left")
    player_diffs = []
    for c in player_cols:
        h, a = f"home_{c}", f"away_{c}"
        games[h] = pd.to_numeric(games[h], errors="coerce").fillna(0.0)
        games[a] = pd.to_numeric(games[a], errors="coerce").fillna(0.0)
        name = f"diff_{c}"
        games[name] = games[h] - games[a]
        player_diffs.append(name)

    predictions, folds = run_backtest(
        games, broad_diffs, player_diffs, args.first_test_season, args.end_season
    )

    holdout = predictions["season"] >= args.holdout_start_season
    material = predictions["material_value_injury_game"].eq(1)
    scopes = {
        "oos_all": np.ones(len(predictions), dtype=bool),
        "oos_material": material,
        "holdout_all": holdout,
        "holdout_material": holdout & material,
    }
    score_tables = [model_scores(predictions, mask, label) for label, mask in scopes.items()]
    scores = pd.concat([x for x in score_tables if not x.empty], ignore_index=True)

    paired_rows = []
    boot_rows = []
    for label, mask in scopes.items():
        stats = paired_stats(predictions, "player_value_ridge", mask)
        paired_rows.append({"scope": label, **stats})
        bs = paired_bootstrap(predictions, "player_value_ridge", mask)
        boot_rows.append({"scope": label, **bs})
    paired = pd.DataFrame(paired_rows)
    boot = pd.DataFrame(boot_rows)

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    scores.to_csv(args.output_dir / "scope_scores.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_market.csv", index=False)
    boot.to_csv(args.output_dir / "bootstrap.csv", index=False)
    team_week.to_csv(args.output_dir / "team_week_player_value_features.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(args.output_dir / "data_diagnostics.csv", index=False)
    write_summary(
        args.output_dir / "summary.md",
        scores,
        paired,
        boot,
        diagnostics,
        args.holdout_start_season,
        args.end_season,
    )

    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
