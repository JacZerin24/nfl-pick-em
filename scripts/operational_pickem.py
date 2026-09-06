"""Operational 2026 NFL pick'em runner.

This is the prospective production wrapper around the validated research system.
Model architecture and training labels are frozen through the end of 2025 for
all of 2026. New 2026 games may update team-state inputs, but they do not
retrain classifier weights or meta-model coefficients midseason.

The runner is kickoff-aware. nflverse schedule `gametime` values are interpreted
in America/New_York, matching NFL published schedule times. Only games that have
not kicked off are eligible, and scheduled runs can use a short horizon so the
expensive model build occurs only shortly before an actual kickoff window.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import numpy as np
import pandas as pd

from phase1_backtest import add_market_probability, build_model_table, build_team_game_stats, walk_forward_backtest
from phase1_market_residual import fit_residual, predict_residual, tune_penalty
from phase2_upset_specialist import build_upset_table
from phase2_upset_variance import build_variance_game_stats, rolling_variance
from live_pickem import (
    CLOSE_FAVORITE_MAX,
    FROZEN_UPSET_PAIRING,
    HISTORICAL_OOF_FIRST_SEASON,
    UPSET_FAVORITE_MAX,
    actual_team_history,
    add_matchup_live_features,
    add_variance_oriented_features,
    base_predict,
    build_variance_training,
    fit_base_models,
    make_matchup_logistic,
    make_variance_catboost,
    populate_live_base_rolls,
    populate_live_variance_rolls,
)

TRAIN_END_SEASON = 2025
MODEL_VERSION = "prospective-v1-frozen-2025"
EASTERN = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--horizon-hours", type=float, default=None)
    p.add_argument("--as-of-utc", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("live_archive"))
    return p.parse_args()


def parse_snapshot(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def kickoff_utc(gameday: object, gametime: object) -> pd.Timestamp:
    if pd.isna(gameday) or pd.isna(gametime):
        return pd.NaT
    day = pd.Timestamp(gameday).strftime("%Y-%m-%d")
    text = str(gametime).strip()
    try:
        local = pd.Timestamp(f"{day} {text}").tz_localize(EASTERN)
    except Exception:
        return pd.NaT
    return local.tz_convert("UTC")


def add_kickoff_times(schedule: pd.DataFrame) -> pd.DataFrame:
    x = schedule.copy()
    x["kickoff_utc"] = [kickoff_utc(d, t) for d, t in zip(x["gameday"], x["gametime"])]
    return x


def select_eligible_games(
    schedule: pd.DataFrame,
    snapshot: datetime,
    week: int | None,
    horizon_hours: float | None,
) -> tuple[int, pd.DataFrame]:
    now = pd.Timestamp(snapshot)
    unplayed = schedule.loc[schedule["home_score"].isna()].copy()
    unplayed = unplayed.loc[unplayed["kickoff_utc"].notna() & (unplayed["kickoff_utc"] > now)]
    if unplayed.empty:
        raise SystemExit("No unplayed regular-season games remain.")

    resolved_week = week
    if resolved_week is None:
        resolved_week = int(pd.to_numeric(unplayed["week"], errors="coerce").dropna().min())

    eligible = unplayed.loc[unplayed["week"].eq(resolved_week)].copy()
    if horizon_hours is not None:
        cutoff = now + pd.Timedelta(hours=float(horizon_hours))
        eligible = eligible.loc[eligible["kickoff_utc"] <= cutoff].copy()

    eligible["hours_to_kickoff"] = (
        (eligible["kickoff_utc"] - now).dt.total_seconds() / 3600.0
    )
    return resolved_week, eligible.sort_values(["kickoff_utc", "game_id"])


def render_markdown(picks: pd.DataFrame, snapshot: str, season: int, week: int) -> str:
    lines = [
        f"# NFL Pick'em — {season} Week {week}",
        "",
        f"Snapshot UTC: `{snapshot}`",
        f"Model: `{MODEL_VERSION}`",
        "",
        "Rules are frozen for the 2026 prospective test: residual decides only true tossups; "
        "a real underdog is selected only when the frozen matchup-logistic and variance-CatBoost specialists agree.",
        "",
        "| Kickoff UTC | Game | Market | Matchup dog | Variance dog | Pick | Decision |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in picks.itertuples(index=False):
        game = f"{r.away_team} @ {r.home_team}"
        market = f"{r.market_pick} {100*r.market_fav_prob:.1f}%"
        kickoff = pd.Timestamp(r.kickoff_utc).strftime("%Y-%m-%d %H:%MZ")
        lines.append(
            f"| {kickoff} | {game} | {market} | {100*r.p_dog_matchup_logistic:.1f}% | "
            f"{100*r.p_dog_variance_catboost:.1f}% | **{r.final_pick}** | {r.decision_type} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    snapshot_dt = parse_snapshot(args.as_of_utc)
    snapshot = snapshot_dt.isoformat().replace("+00:00", "Z")

    # Cheap schedule gate first. Scheduled CI runs exit before the expensive
    # historical rebuild when no game is close enough to kickoff.
    schedule = nfl.load_schedules(args.season).to_pandas()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    schedule = add_market_probability(schedule)
    schedule = add_kickoff_times(schedule)
    week, eligible = select_eligible_games(
        schedule, snapshot_dt, args.week, args.horizon_hours
    )
    if eligible.empty:
        print(
            f"No eligible games within horizon for season={args.season} week={week} "
            f"as of {snapshot}. Exiting without a model rebuild."
        )
        return

    print("Eligible pre-kickoff games:")
    print(
        eligible[["game_id", "away_team", "home_team", "kickoff_utc", "hours_to_kickoff"]]
        .to_string(index=False)
    )

    seasons = list(range(args.start_season, args.season + 1))
    print(f"Building model table {args.start_season}-{args.season}...")
    base = build_model_table(args.start_season, args.season)
    base["gameday"] = pd.to_datetime(base["gameday"])

    targets = base.loc[base["game_id"].isin(eligible["game_id"])].copy()
    if targets.empty:
        raise SystemExit("Eligible schedule games were not found in the model table.")
    targets = targets.merge(
        eligible[["game_id", "kickoff_utc", "hours_to_kickoff"]],
        on="game_id",
        how="left",
    )

    # Current-season completed games are allowed to update state inputs only.
    # All learned weights remain frozen to completed <=2025 games.
    frozen_train = base.loc[
        base["home_win"].notna() & (base["season"] <= TRAIN_END_SEASON)
    ].copy()
    if frozen_train.empty:
        raise SystemExit("Frozen historical training set is empty.")

    print("Loading PBP for current as-of team states...")
    pbp = nfl.load_pbp(seasons)
    regular_context = nfl.load_schedules(seasons).to_pandas()
    regular_context["gameday"] = pd.to_datetime(regular_context["gameday"])
    regular_context = regular_context.loc[regular_context["game_type"].eq("REG")].copy()
    base_history = actual_team_history(build_team_game_stats(pbp), regular_context)
    targets = populate_live_base_rolls(targets, base_history)

    base_logistic, base_cat, base_features, categorical = fit_base_models(frozen_train)
    p_log, p_cat = base_predict(base_logistic, base_cat, base_features, categorical, targets)
    targets["p_home_market"] = targets["market_home_prob"].to_numpy(float)
    targets["p_home_elo"] = targets["elo_home_prob"].to_numpy(float)
    targets["p_home_logistic"] = p_log
    targets["p_home_catboost"] = p_cat

    # Frozen residual fit uses historical OOF predictions through 2025 only.
    print("Rebuilding frozen historical OOF residual model...")
    oof, _ = walk_forward_backtest(
        frozen_train,
        HISTORICAL_OOF_FIRST_SEASON,
        TRAIN_END_SEASON,
    )
    penalty, _ = tune_penalty(oof)
    theta = fit_residual(oof, penalty)
    targets["p_home_residual"] = predict_residual(targets, theta)

    # Frozen upset specialist 1: matchup elastic-net logistic.
    matchup_train, matchup_features = build_upset_table(frozen_train)
    matchup_train = matchup_train.loc[
        (matchup_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (matchup_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    matchup_live, live_matchup_features = add_matchup_live_features(targets)
    if matchup_features != live_matchup_features:
        raise RuntimeError("Live matchup feature order differs from historical specialist.")
    matchup_model = make_matchup_logistic()
    matchup_model.fit(matchup_train[matchup_features], matchup_train["dog_win"].astype(int))
    targets["p_dog_matchup_logistic"] = matchup_model.predict_proba(
        matchup_live[matchup_features]
    )[:, 1]

    # Frozen upset specialist 2: explosive/variance CatBoost.
    variance_games = build_variance_game_stats(pbp)
    variance_history = actual_team_history(variance_games, regular_context)
    variance_rolls = rolling_variance(variance_games, regular_context)
    variance_train, variance_features = build_variance_training(frozen_train, variance_rolls)
    variance_train = variance_train.loc[
        (variance_train["market_fav_prob"] >= CLOSE_FAVORITE_MAX)
        & (variance_train["market_fav_prob"] < UPSET_FAVORITE_MAX)
    ].copy()
    variance_live = populate_live_variance_rolls(targets, variance_history)
    variance_live, live_variance_features = add_variance_oriented_features(
        variance_live, require_outcome=False
    )
    if variance_features != live_variance_features:
        raise RuntimeError("Live variance feature order differs from historical specialist.")
    variance_model = make_variance_catboost()
    variance_model.fit(variance_train[variance_features], variance_train["dog_win"].astype(int))
    targets["p_dog_variance_catboost"] = variance_model.predict_proba(
        variance_live[variance_features]
    )[:, 1]

    p_market = targets["p_home_market"].to_numpy(float)
    market_home = p_market >= 0.5
    fav_prob = np.maximum(p_market, 1 - p_market)
    dog_home = ~market_home
    residual_home = targets["p_home_residual"].to_numpy(float) >= 0.5
    close = fav_prob < CLOSE_FAVORITE_MAX
    consensus = (
        (fav_prob >= CLOSE_FAVORITE_MAX)
        & (fav_prob < UPSET_FAVORITE_MAX)
        & (targets["p_dog_matchup_logistic"].to_numpy(float) >= 0.5)
        & (targets["p_dog_variance_catboost"].to_numpy(float) >= 0.5)
    )

    final_home = market_home.copy()
    final_home[close] = residual_home[close]
    final_home[consensus] = dog_home[consensus]

    targets["market_fav_prob"] = fav_prob
    targets["market_pick"] = np.where(market_home, targets["home_team"], targets["away_team"])
    targets["market_underdog"] = np.where(market_home, targets["away_team"], targets["home_team"])
    targets["final_pick"] = np.where(final_home, targets["home_team"], targets["away_team"])
    targets["decision_type"] = np.select(
        [consensus, close & (residual_home != market_home), close],
        ["TRUE_UPSET_CONSENSUS", "CLOSE_RESIDUAL", "CLOSE_MARKET_ALIGNED"],
        default="FOLLOW_MARKET",
    )
    targets["snapshot_utc"] = snapshot
    targets["model_version"] = MODEL_VERSION
    targets["upset_pairing"] = FROZEN_UPSET_PAIRING
    targets["residual_penalty"] = penalty
    targets["training_end_season"] = TRAIN_END_SEASON

    output_cols = [
        "snapshot_utc",
        "model_version",
        "game_id",
        "season",
        "week",
        "gameday",
        "gametime",
        "kickoff_utc",
        "hours_to_kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "away_moneyline",
        "home_moneyline",
        "market_pick",
        "market_underdog",
        "market_fav_prob",
        "p_home_market",
        "p_home_elo",
        "p_home_logistic",
        "p_home_catboost",
        "p_home_residual",
        "p_dog_matchup_logistic",
        "p_dog_variance_catboost",
        "final_pick",
        "decision_type",
        "upset_pairing",
        "residual_penalty",
        "training_end_season",
    ]
    picks = targets[[c for c in output_cols if c in targets.columns]].copy()
    picks = picks.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    stamp = snapshot_dt.strftime("%Y%m%dT%H%M%SZ")
    week_dir = args.output_dir / str(args.season) / f"week_{week:02d}"
    snap_dir = week_dir / "snapshots" / stamp
    snap_dir.mkdir(parents=True, exist_ok=True)
    picks.to_csv(snap_dir / "picks.csv", index=False)
    md = render_markdown(picks, snapshot, args.season, week)
    (snap_dir / "picks.md").write_text(md, encoding="utf-8")
    picks.to_csv(week_dir / "latest.csv", index=False)
    (week_dir / "latest.md").write_text(md, encoding="utf-8")

    metadata = pd.DataFrame(
        [
            {
                "snapshot_utc": snapshot,
                "model_version": MODEL_VERSION,
                "season": args.season,
                "week": week,
                "eligible_games": len(picks),
                "training_end_season": TRAIN_END_SEASON,
                "frozen_upset_pairing": FROZEN_UPSET_PAIRING,
                "close_favorite_max": CLOSE_FAVORITE_MAX,
                "upset_favorite_max": UPSET_FAVORITE_MAX,
                "residual_penalty": penalty,
                "theta_intercept": theta[0],
                "theta_elo_delta": theta[1],
                "theta_logistic_delta": theta[2],
                "theta_catboost_delta": theta[3],
                "theta_consensus_delta": theta[4],
            }
        ]
    )
    metadata.to_csv(snap_dir / "model_metadata.csv", index=False)
    metadata.to_csv(week_dir / "latest_model_metadata.csv", index=False)

    print("\nOperational picks")
    print(
        picks[[
            "away_team",
            "home_team",
            "kickoff_utc",
            "market_pick",
            "market_fav_prob",
            "p_dog_matchup_logistic",
            "p_dog_variance_catboost",
            "final_pick",
            "decision_type",
        ]].to_string(index=False)
    )
    print(f"\nArchived snapshot: {snap_dir}")


if __name__ == "__main__":
    main()
