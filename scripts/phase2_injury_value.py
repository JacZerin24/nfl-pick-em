"""Phase 2 snap-weighted injury availability experiment.

Uses historical final weekly injury/practice reports (2012-2024 testable with
snap-count role weights) and only PRIOR game snap participation to measure player
importance. It does not treat every listed injury equally.

This is a historical value-of-information experiment. nflverse's public injury
feed ends after 2024, so any feature promoted from this study must later be fed
by a current timestamped injury/inactive source for live picks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase1_backtest import BASE_TEAM_METRICS, ROLL_WINDOWS, build_model_table, score_probabilities

POSITION_GROUPS = ("QB", "OL", "SKILL", "FRONT", "DB", "ST", "OTHER")
REPORT_BUCKETS = ("out", "doubtful", "questionable", "probable")
PRACTICE_BUCKETS = ("dnp", "limited", "full")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2012)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_injury_value"))
    return p.parse_args()


def pos_group(position: object) -> str:
    pos = str(position or "").upper().strip()
    if pos == "QB":
        return "QB"
    if pos in {"C", "G", "T", "OT", "OG", "OL"}:
        return "OL"
    if pos in {"WR", "TE", "RB", "FB", "HB"}:
        return "SKILL"
    if pos in {"DE", "DT", "DL", "NT", "LB", "ILB", "OLB", "EDGE"}:
        return "FRONT"
    if pos in {"CB", "DB", "S", "FS", "SS"}:
        return "DB"
    if pos in {"K", "P", "LS"}:
        return "ST"
    return "OTHER"


def report_bucket(value: object) -> str:
    text = str(value or "").strip().lower()
    if "out" in text:
        return "out"
    if "doubt" in text:
        return "doubtful"
    if "question" in text:
        return "questionable"
    if "probab" in text:
        return "probable"
    return "none"


def practice_bucket(value: object) -> str:
    text = str(value or "").strip().lower()
    if "did not" in text or text in {"dnp", "did_not_participate"}:
        return "dnp"
    if "limit" in text:
        return "limited"
    if "full" in text:
        return "full"
    return "none"


def build_snap_role_history(seasons: list[int]) -> pd.DataFrame:
    """Return per-PFR-player appearance rows with recent PRIOR-role proxy.

    The row's ``recent_role`` includes that appearance. Injury week W will later
    look up only an appearance strictly before W, making the role estimate
    pregame-safe.
    """
    print(f"Loading snap counts {min(seasons)}-{max(seasons)}...")
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    needed = {
        "pfr_player_id", "season", "week", "game_type", "offense_pct", "defense_pct"
    }
    missing = needed.difference(snaps.columns)
    if missing:
        raise RuntimeError(f"Snap counts missing required columns: {sorted(missing)}")

    s = snaps.loc[snaps["game_type"].eq("REG"), list(needed)].copy()
    s["offense_pct"] = pd.to_numeric(s["offense_pct"], errors="coerce").fillna(0.0)
    s["defense_pct"] = pd.to_numeric(s["defense_pct"], errors="coerce").fillna(0.0)
    s["usage"] = s[["offense_pct", "defense_pct"]].max(axis=1)

    # Accommodate either 0-1 or 0-100 source representation.
    if s["usage"].quantile(0.95) > 1.5:
        s["usage"] = s["usage"] / 100.0
    s["usage"] = s["usage"].clip(0.0, 1.0)
    s["order_key"] = s["season"].astype(int) * 100 + s["week"].astype(int)
    s = s.sort_values(["pfr_player_id", "order_key"]).reset_index(drop=True)

    pieces = []
    for _, g in s.groupby("pfr_player_id", sort=False):
        gg = g.copy()
        gg["recent_role"] = gg["usage"].rolling(4, min_periods=1).mean()
        pieces.append(gg[["pfr_player_id", "order_key", "recent_role"]])
    return pd.concat(pieces, ignore_index=True)


def attach_prior_role(injuries: pd.DataFrame, role_history: pd.DataFrame) -> pd.DataFrame:
    """As-of lookup: latest snap-role row strictly before each injury week."""
    out = injuries.copy()
    out["prior_role"] = np.nan

    histories = {
        pid: (
            g["order_key"].to_numpy(dtype=int),
            g["recent_role"].to_numpy(dtype=float),
        )
        for pid, g in role_history.groupby("pfr_player_id", sort=False)
        if pd.notna(pid)
    }

    for pid, idx in out.groupby("pfr_id", sort=False).groups.items():
        if pd.isna(pid) or pid not in histories:
            continue
        orders, roles = histories[pid]
        target = out.loc[idx, "order_key"].to_numpy(dtype=int)
        loc = np.searchsorted(orders, target, side="left") - 1
        valid = loc >= 0
        if valid.any():
            selected_idx = np.asarray(list(idx))[valid]
            out.loc[selected_idx, "prior_role"] = roles[loc[valid]]

    return out


def build_team_week_injuries(start_season: int, end_season: int) -> tuple[pd.DataFrame, dict[str, float]]:
    seasons = list(range(start_season, end_season + 1))
    print(f"Loading injuries {start_season}-{end_season}...")
    injuries = nfl.load_injuries(seasons).to_pandas()
    players = nfl.load_players().to_pandas()
    roles = build_snap_role_history(seasons)

    required = {
        "season", "team", "week", "gsis_id", "position", "report_status",
        "practice_status", "season_type",
    }
    missing = required.difference(injuries.columns)
    if missing:
        raise RuntimeError(f"Injury data missing required columns: {sorted(missing)}")

    inj = injuries.loc[injuries["season_type"].eq("REG")].copy()
    if "date_modified" in inj.columns:
        inj["date_modified"] = pd.to_datetime(inj["date_modified"], errors="coerce", utc=True)
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")

    mapping = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    inj = inj.merge(mapping, on="gsis_id", how="left")
    inj["order_key"] = inj["season"].astype(int) * 100 + inj["week"].astype(int)
    inj = attach_prior_role(inj, roles)
    inj["position_group"] = inj["position"].map(pos_group)
    inj["report_bucket"] = inj["report_status"].map(report_bucket)
    inj["practice_bucket"] = inj["practice_status"].map(practice_bucket)

    # Missing prior snaps means uncertain player role. Keep a separate coverage
    # diagnostic and use zero contribution rather than inventing a large role.
    inj["role_weight"] = inj["prior_role"].fillna(0.0).clip(0.0, 1.0)

    rows = []
    for (season, week, team), g in inj.groupby(["season", "week", "team"], sort=False):
        row: dict[str, float | int | str] = {
            "season": int(season), "week": int(week), "team": team,
            "injury_players_listed": int(len(g)),
            "injury_role_coverage": float(g["prior_role"].notna().mean()),
        }
        for status in REPORT_BUCKETS:
            mask = g["report_bucket"].eq(status)
            row[f"inj_report_{status}_role_total"] = float(g.loc[mask, "role_weight"].sum())
            row[f"inj_report_{status}_count"] = int(mask.sum())
            for pg in POSITION_GROUPS:
                row[f"inj_report_{status}_{pg.lower()}_role"] = float(
                    g.loc[mask & g["position_group"].eq(pg), "role_weight"].sum()
                )
        for status in PRACTICE_BUCKETS:
            mask = g["practice_bucket"].eq(status)
            row[f"inj_practice_{status}_role_total"] = float(g.loc[mask, "role_weight"].sum())
        rows.append(row)

    team_week = pd.DataFrame(rows).fillna(0.0)
    diagnostics = {
        "injury_rows": float(len(inj)),
        "pfr_id_match_rate": float(inj["pfr_id"].notna().mean()),
        "prior_role_match_rate": float(inj["prior_role"].notna().mean()),
    }
    return team_week, diagnostics


def merge_injury_features(games: pd.DataFrame, team_week: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in team_week.columns if c not in {"season", "week", "team"}]
    home = team_week.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}})
    away = team_week.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}})
    out = games.merge(home, on=["season", "week", "home_team"], how="left")
    out = out.merge(away, on=["season", "week", "away_team"], how="left")

    diffs = []
    for col in feature_cols:
        h, a = f"home_{col}", f"away_{col}"
        out[h] = pd.to_numeric(out[h], errors="coerce").fillna(0.0)
        out[a] = pd.to_numeric(out[a], errors="coerce").fillna(0.0)
        name = f"diff_{col}"
        out[name] = out[h] - out[a]
        diffs.append(name)
    return out, diffs


def make_logistic(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([("num", num, numeric_cols), ("cat", cat, categorical_cols)])
    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.15, C=0.5,
            max_iter=5000, random_state=42,
        )),
    ])


def run_backtest(
    games: pd.DataFrame, injury_diff_cols: list[str], first_test_season: int, end_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_diffs = [
        f"diff_{metric}_r{window}" for metric in BASE_TEAM_METRICS for window in ROLL_WINDOWS
    ]
    categorical = ["roof", "surface", "location", "div_game"]
    base_numeric = [
        "market_home_prob", "spread_line", "total_line", "rest_diff", "elo_diff",
        "elo_home_prob", *team_diffs,
    ]
    full_numeric = [*base_numeric, *injury_diff_cols]
    completed = games.loc[games["home_win"].notna()].copy()

    pred_rows, fold_rows = [], []
    for season in range(first_test_season, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue
        y_train, y_test = train["home_win"].astype(int), test["home_win"].astype(int)
        probs: dict[str, np.ndarray] = {"market": test["market_home_prob"].to_numpy(float)}

        for name, nums in {
            "market_football_logistic": base_numeric,
            "injury_logistic": full_numeric,
        }.items():
            cols = [*nums, *categorical]
            model = make_logistic(nums, categorical)
            model.fit(train[cols], y_train)
            probs[name] = model.predict_proba(test[cols])[:, 1]

        cols = [*full_numeric, *categorical]
        xtr, xte = train[cols].copy(), test[cols].copy()
        for c in categorical:
            xtr[c] = xtr[c].astype("string").fillna("__MISSING__")
            xte[c] = xte[c].astype("string").fillna("__MISSING__")
        cat_idx = [xtr.columns.get_loc(c) for c in categorical]
        cat = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03, loss_function="Logloss",
            random_seed=42, l2_leaf_reg=8.0, random_strength=0.5,
            verbose=False, allow_writing_files=False,
        )
        cat.fit(xtr, y_train, cat_features=cat_idx)
        probs["injury_catboost"] = cat.predict_proba(xte)[:, 1]

        pred = test[["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win"]].copy()
        for name, p in probs.items():
            pred[f"p_home_{name}"] = p
            fold_rows.append({"season": season, "model": name, "games": len(test), **score_probabilities(y_test, p)})
        pred_rows.append(pred)
        print(
            f"{season}: market={score_probabilities(y_test, probs['market'])['accuracy']:.3f} "
            f"base={score_probabilities(y_test, probs['market_football_logistic'])['accuracy']:.3f} "
            f"inj_logit={score_probabilities(y_test, probs['injury_logistic'])['accuracy']:.3f} "
            f"inj_cat={score_probabilities(y_test, probs['injury_catboost'])['accuracy']:.3f}"
        )

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["home_win"].astype(int)
    rows = []
    for model in ("market", "market_football_logistic", "injury_logistic", "injury_catboost"):
        p = predictions[f"p_home_{model}"].to_numpy(float)
        rows.append({"model": model, "games": len(predictions), **score_probabilities(y, p)})
    return pd.DataFrame(rows).sort_values(["accuracy", "brier"], ascending=[False, True])


def paired_vs_base(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["home_win"].astype(int).to_numpy()
    base = (predictions["p_home_market_football_logistic"].to_numpy(float) >= 0.5).astype(int)
    base_correct = (base == y).astype(int)
    rows = []
    for model in ("injury_logistic", "injury_catboost"):
        pick = (predictions[f"p_home_{model}"].to_numpy(float) >= 0.5).astype(int)
        correct = (pick == y).astype(int)
        disagree = pick != base
        rows.append({
            "model": model,
            "games": len(y),
            "accuracy_lift_vs_base": float((correct - base_correct).mean()),
            "net_additional_correct": int((correct - base_correct).sum()),
            "disagreements": int(disagree.sum()),
            "model_accuracy_on_disagreements": float(np.mean(pick[disagree] == y[disagree])) if disagree.any() else np.nan,
            "base_accuracy_on_disagreements": float(np.mean(base[disagree] == y[disagree])) if disagree.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.start_season < 2012:
        raise SystemExit("Snap-count weighting is available from 2012 onward")
    if args.end_season > 2024:
        raise SystemExit("nflverse historical injury feed currently ends after 2024")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = build_model_table(args.start_season, args.end_season)
    team_week, diagnostics = build_team_week_injuries(args.start_season, args.end_season)
    games, injury_diff_cols = merge_injury_features(games, team_week)
    predictions, folds = run_backtest(
        games, injury_diff_cols, args.first_test_season, args.end_season
    )
    summary = summarize(predictions)
    paired = paired_vs_base(predictions)

    predictions.to_csv(args.output_dir / "injury_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "injury_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "injury_summary.csv", index=False)
    paired.to_csv(args.output_dir / "injury_paired_vs_base.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(args.output_dir / "injury_data_diagnostics.csv", index=False)

    print("\nInjury value summary")
    print(summary.to_string(index=False))
    print("\nInjury models vs market+football base")
    print(paired.to_string(index=False))
    print("\nData diagnostics")
    print(pd.DataFrame([diagnostics]).to_string(index=False))


if __name__ == "__main__":
    main()
