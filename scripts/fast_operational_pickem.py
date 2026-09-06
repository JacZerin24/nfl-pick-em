"""Fast operational runner for the frozen 2026 NFL pick'em model.

This runner is prediction-equivalent to `operational_pickem.py`, but it loads
versioned frozen model artifacts instead of rebuilding/refitting 2009-2025 on
every live run. Only current schedule/market inputs and current-season rolling
team state are refreshed.
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from frozen_artifacts import (
    artifact_manifest_hash,
    load_catboost,
    predict_base_logistic,
    predict_simple_logistic,
    read_json,
    sha256_file,
)
from live_pickem import (
    CLOSE_FAVORITE_MAX,
    FROZEN_UPSET_PAIRING,
    UPSET_FAVORITE_MAX,
    actual_team_history,
    add_matchup_live_features,
    add_variance_oriented_features,
    populate_live_base_rolls,
    populate_live_variance_rolls,
)
from operational_pickem import (
    MODEL_VERSION,
    TRAIN_END_SEASON,
    add_kickoff_times,
    build_live_schedule,
    parse_snapshot,
    select_eligible_games,
)
from phase1_backtest import add_market_probability, build_team_game_stats
from phase1_market_residual import predict_residual
from phase2_upset_variance import build_variance_game_stats

DEFAULT_ARTIFACT_DIR = Path("frozen_artifacts") / MODEL_VERSION
OFFICIAL_ROLES = {"FINAL_ENTRY", "FALLBACK_ENTRY"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--horizon-hours", type=float, default=None)
    p.add_argument("--as-of-utc", type=str, default=None)
    p.add_argument("--game-ids", type=str, default="")
    p.add_argument("--snapshot-role", type=str, default="UPDATE")
    p.add_argument("--output-dir", type=Path, default=Path("live_archive"))
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return p.parse_args()


def verify_artifacts(root: Path) -> tuple[dict, dict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen artifact manifest not found at {manifest_path}. "
            "Run scripts/build_frozen_artifacts.py first."
        )
    manifest = read_json(manifest_path)
    if manifest.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            f"Artifact model version {manifest.get('model_version')} != {MODEL_VERSION}"
        )
    if int(manifest.get("training_end_season", -1)) != TRAIN_END_SEASON:
        raise RuntimeError("Artifact training end season does not match frozen production model")
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Artifact hash mismatch for {name}: {actual} != {expected}")
    features = read_json(root / "feature_manifest.json")
    return manifest, features


def load_current_pbp(season: int):
    try:
        pbp = nfl.load_pbp([season])
    except (ValueError, FileNotFoundError) as exc:
        print(f"Current-season PBP unavailable ({exc}); using frozen history tails only.")
        return None
    try:
        if len(pbp) == 0:
            return None
    except TypeError:
        pass
    return pbp


def combine_history(frozen_path: Path, current: pd.DataFrame | None) -> pd.DataFrame:
    frozen = pd.read_csv(frozen_path)
    frozen["gameday"] = pd.to_datetime(frozen["gameday"])
    if current is None or current.empty:
        return frozen.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    current = current.copy()
    current["gameday"] = pd.to_datetime(current["gameday"])
    cols = [c for c in frozen.columns if c in current.columns]
    merged = pd.concat([frozen[cols], current[cols]], ignore_index=True)
    return merged.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def render_markdown(picks: pd.DataFrame, snapshot: str, season: int, week: int, role: str) -> str:
    official = role in OFFICIAL_ROLES
    lines = [
        f"# NFL Pick'em — {season} Week {week}",
        "",
        f"Snapshot UTC: `{snapshot}`",
        f"Model: `{MODEL_VERSION}`",
        f"Snapshot role: **{role}**" + (" — official entry snapshot" if official else ""),
        "",
        "Frozen 2026 rules: market anchor; residual only for <52.5% tossups; "
        "true underdog only when matchup-logistic and variance-CatBoost agree; "
        "80%+ favorite remains the market pick.",
        "",
        "| Kickoff UTC | Game | Market | Matchup dog | Variance dog | Pick | Decision |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in picks.itertuples(index=False):
        game = f"{row.away_team} @ {row.home_team}"
        market = f"{row.market_pick} {100 * row.market_fav_prob:.1f}%"
        kickoff = pd.Timestamp(row.kickoff_utc).strftime("%Y-%m-%d %H:%MZ")
        lines.append(
            f"| {kickoff} | {game} | {market} | "
            f"{100 * row.p_dog_matchup_logistic:.1f}% | "
            f"{100 * row.p_dog_variance_catboost:.1f}% | "
            f"**{row.final_pick}** | {row.decision_type} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    role = args.snapshot_role.strip().upper().replace("-", "_") or "UPDATE"
    snapshot_dt = parse_snapshot(args.as_of_utc)
    snapshot = snapshot_dt.isoformat().replace("+00:00", "Z")

    manifest, feature_manifest = verify_artifacts(args.artifact_dir)
    manifest_hash = artifact_manifest_hash(args.artifact_dir / "manifest.json")

    current_schedule = nfl.load_schedules(args.season).to_pandas()
    current_schedule["gameday"] = pd.to_datetime(current_schedule["gameday"])
    current_schedule = current_schedule.loc[current_schedule["game_type"].eq("REG")].copy()
    current_schedule = add_market_probability(current_schedule)
    current_schedule = add_kickoff_times(current_schedule)
    week, eligible = select_eligible_games(
        current_schedule, snapshot_dt, args.week, args.horizon_hours
    )
    requested_ids = {x.strip() for x in args.game_ids.split(",") if x.strip()}
    if requested_ids:
        eligible = eligible.loc[eligible["game_id"].astype(str).isin(requested_ids)].copy()
    if eligible.empty:
        print(
            f"No eligible games for season={args.season} week={week} as of {snapshot}; "
            "exiting before live feature refresh."
        )
        return

    live_schedule = build_live_schedule(args.start_season, args.season)
    targets = live_schedule.loc[live_schedule["game_id"].isin(eligible["game_id"])].copy()
    targets = targets.merge(
        eligible[["game_id", "kickoff_utc", "hours_to_kickoff"]], on="game_id", how="left"
    )
    if targets.empty:
        raise SystemExit("Eligible games were not found in live schedule table")

    current_pbp = load_current_pbp(args.season)
    pbp_end_season = TRAIN_END_SEASON
    current_base_history = None
    current_variance_history = None
    if current_pbp is not None:
        current_context = live_schedule.loc[live_schedule["season"].eq(args.season)].copy()
        current_base_games = build_team_game_stats(current_pbp)
        current_base_history = actual_team_history(current_base_games, current_context)
        current_variance_games = build_variance_game_stats(current_pbp)
        current_variance_history = actual_team_history(current_variance_games, current_context)
        if not current_base_history.empty:
            pbp_end_season = args.season

    base_history = combine_history(
        args.artifact_dir / "base_history_tail.csv", current_base_history
    )
    targets = populate_live_base_rolls(targets, base_history)

    base_logistic_spec = read_json(args.artifact_dir / "base_logistic.json")
    base_features = list(feature_manifest["base_features"])
    categorical = list(feature_manifest["base_categorical"])
    p_log = predict_base_logistic(base_logistic_spec, targets)
    base_cat = load_catboost(args.artifact_dir / "base_catboost.json")
    cat_target = targets[base_features].copy()
    for col in categorical:
        cat_target[col] = cat_target[col].astype("string").fillna("__MISSING__")
    p_cat = base_cat.predict_proba(cat_target)[:, 1]

    targets["p_home_market"] = targets["market_home_prob"].to_numpy(float)
    targets["p_home_elo"] = targets["elo_home_prob"].to_numpy(float)
    targets["p_home_logistic"] = p_log
    targets["p_home_catboost"] = p_cat

    residual = read_json(args.artifact_dir / "residual.json")
    theta = np.asarray(residual["theta"], dtype=float)
    penalty = float(residual["penalty"])
    targets["p_home_residual"] = predict_residual(targets, theta)

    matchup_live, matchup_features = add_matchup_live_features(targets)
    expected_matchup = list(feature_manifest["matchup_features"])
    if matchup_features != expected_matchup:
        raise RuntimeError("Live matchup feature order differs from frozen artifact")
    matchup_spec = read_json(args.artifact_dir / "matchup_logistic.json")
    targets["p_dog_matchup_logistic"] = predict_simple_logistic(matchup_spec, matchup_live)

    variance_history = combine_history(
        args.artifact_dir / "variance_history_tail.csv", current_variance_history
    )
    variance_live = populate_live_variance_rolls(targets, variance_history)
    variance_live, variance_features = add_variance_oriented_features(
        variance_live, require_outcome=False
    )
    expected_variance = list(feature_manifest["variance_features"])
    if variance_features != expected_variance:
        raise RuntimeError("Live variance feature order differs from frozen artifact")
    variance_model = load_catboost(args.artifact_dir / "variance_catboost.json")
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
    targets["market_underdog"] = np.where(
        market_home, targets["away_team"], targets["home_team"]
    )
    targets["final_pick"] = np.where(final_home, targets["home_team"], targets["away_team"])
    targets["decision_type"] = np.select(
        [consensus, close & (residual_home != market_home), close],
        ["TRUE_UPSET_CONSENSUS", "CLOSE_RESIDUAL", "CLOSE_MARKET_ALIGNED"],
        default="FOLLOW_MARKET",
    )
    targets["snapshot_utc"] = snapshot
    targets["snapshot_role"] = role
    targets["official_entry_snapshot"] = role in OFFICIAL_ROLES
    targets["runner_mode"] = "fast_frozen_artifacts"
    targets["artifact_manifest_sha256"] = manifest_hash
    targets["model_version"] = MODEL_VERSION
    targets["upset_pairing"] = FROZEN_UPSET_PAIRING
    targets["residual_penalty"] = penalty
    targets["training_end_season"] = TRAIN_END_SEASON
    targets["pbp_end_season"] = pbp_end_season

    output_cols = [
        "snapshot_utc", "snapshot_role", "official_entry_snapshot", "runner_mode",
        "artifact_manifest_sha256", "model_version", "game_id", "season", "week",
        "gameday", "gametime", "kickoff_utc", "hours_to_kickoff", "away_team",
        "home_team", "spread_line", "total_line", "away_moneyline", "home_moneyline",
        "market_pick", "market_underdog", "market_fav_prob", "p_home_market",
        "p_home_elo", "p_home_logistic", "p_home_catboost", "p_home_residual",
        "p_dog_matchup_logistic", "p_dog_variance_catboost", "final_pick",
        "decision_type", "upset_pairing", "residual_penalty", "training_end_season",
        "pbp_end_season",
    ]
    picks = targets[[c for c in output_cols if c in targets.columns]].copy()
    picks = picks.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    stamp = snapshot_dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    week_dir = args.output_dir / str(args.season) / f"week_{week:02d}"
    snap_dir = week_dir / "snapshots" / stamp
    snap_dir.mkdir(parents=True, exist_ok=True)
    picks.to_csv(snap_dir / "picks.csv", index=False)
    markdown = render_markdown(picks, snapshot, args.season, week, role)
    (snap_dir / "picks.md").write_text(markdown, encoding="utf-8")
    picks.to_csv(week_dir / "latest.csv", index=False)
    (week_dir / "latest.md").write_text(markdown, encoding="utf-8")

    metadata = pd.DataFrame([
        {
            "snapshot_utc": snapshot,
            "snapshot_role": role,
            "official_entry_snapshot": role in OFFICIAL_ROLES,
            "runner_mode": "fast_frozen_artifacts",
            "artifact_manifest_sha256": manifest_hash,
            "model_version": MODEL_VERSION,
            "season": args.season,
            "week": week,
            "eligible_games": len(picks),
            "training_end_season": TRAIN_END_SEASON,
            "pbp_end_season": pbp_end_season,
            "frozen_upset_pairing": FROZEN_UPSET_PAIRING,
            "close_favorite_max": CLOSE_FAVORITE_MAX,
            "upset_favorite_max": UPSET_FAVORITE_MAX,
            "artifact_built_utc": manifest.get("built_utc"),
        }
    ])
    metadata.to_csv(snap_dir / "metadata.csv", index=False)

    print(markdown)
    print(f"Saved immutable {role} snapshot to {snap_dir}")


if __name__ == "__main__":
    main()
