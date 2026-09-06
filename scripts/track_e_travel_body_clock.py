"""Track E: travel and body-clock effects for NFL straight-up pick'em.

Research-only. The frozen 2026 production model is not modified.

Protocol
--------
* Build pregame-safe schedule/team context for 2009-2025.
* Use the closing market as the probability offset/anchor.
* Use 2016-2018 OOS folds only to select ridge strength.
* Freeze that strength, then evaluate 2019-2025 as the diagnostic holdout.
* Compare market vs travel main effects vs pre-specified travel/body-clock
  interactions.

The feature family is deliberately small: distance, time zones crossed,
east/west direction, kickoff hour on each team's home body clock, short rest,
and prior road sequencing. Neutral/international games are flagged separately.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from phase1_backtest import build_model_table

EPS = 1e-6
EASTERN = ZoneInfo("America/New_York")
PENALTIES = (10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float
    tz: str
    international: bool = False


# Approximate home-stadium coordinates are sufficient for travel distance and
# time-zone research. Relocated historical teams keep separate abbreviations.
TEAM_BASES: dict[str, Place] = {
    "ARI": Place(33.5276, -112.2626, "America/Phoenix"),
    "ATL": Place(33.7554, -84.4010, "America/New_York"),
    "BAL": Place(39.2780, -76.6227, "America/New_York"),
    "BUF": Place(42.7738, -78.7868, "America/New_York"),
    "CAR": Place(35.2258, -80.8528, "America/New_York"),
    "CHI": Place(41.8623, -87.6167, "America/Chicago"),
    "CIN": Place(39.0954, -84.5160, "America/New_York"),
    "CLE": Place(41.5061, -81.6995, "America/New_York"),
    "DAL": Place(32.7473, -97.0945, "America/Chicago"),
    "DEN": Place(39.7439, -105.0201, "America/Denver"),
    "DET": Place(42.3400, -83.0456, "America/New_York"),
    "GB": Place(44.5013, -88.0622, "America/Chicago"),
    "HOU": Place(29.6847, -95.4107, "America/Chicago"),
    "IND": Place(39.7601, -86.1639, "America/New_York"),
    "JAX": Place(30.3239, -81.6373, "America/New_York"),
    "JAC": Place(30.3239, -81.6373, "America/New_York"),
    "KC": Place(39.0489, -94.4839, "America/Chicago"),
    "LV": Place(36.0909, -115.1833, "America/Los_Angeles"),
    "OAK": Place(37.7516, -122.2005, "America/Los_Angeles"),
    "LAC": Place(33.9535, -118.3392, "America/Los_Angeles"),
    "SD": Place(32.7831, -117.1227, "America/Los_Angeles"),
    "LA": Place(33.9535, -118.3392, "America/Los_Angeles"),
    "LAR": Place(33.9535, -118.3392, "America/Los_Angeles"),
    "STL": Place(38.6328, -90.1886, "America/Chicago"),
    "MIA": Place(25.9580, -80.2389, "America/New_York"),
    "MIN": Place(44.9736, -93.2575, "America/Chicago"),
    "NE": Place(42.0909, -71.2643, "America/New_York"),
    "NO": Place(29.9511, -90.0812, "America/Chicago"),
    "NYG": Place(40.8135, -74.0745, "America/New_York"),
    "NYJ": Place(40.8135, -74.0745, "America/New_York"),
    "PHI": Place(39.9008, -75.1675, "America/New_York"),
    "PIT": Place(40.4468, -80.0158, "America/New_York"),
    "SEA": Place(47.5952, -122.3316, "America/Los_Angeles"),
    "SF": Place(37.4030, -121.9700, "America/Los_Angeles"),
    "TB": Place(27.9759, -82.5033, "America/New_York"),
    "TEN": Place(36.1665, -86.7713, "America/Chicago"),
    "WAS": Place(38.9078, -76.8645, "America/New_York"),
    "WSH": Place(38.9078, -76.8645, "America/New_York"),
}

# Known regular-season neutral/international venue keywords. Unknown neutral
# venues are retained with a separate flag rather than assigned fake distance.
NEUTRAL_VENUES: list[tuple[tuple[str, ...], Place]] = [
    (("wembley", "tottenham", "twickenham", "london"), Place(51.5560, -0.2796, "Europe/London", True)),
    (("allianz", "munich"), Place(48.2188, 11.6247, "Europe/Berlin", True)),
    (("frankfurt", "deutsche bank"), Place(50.0686, 8.6455, "Europe/Berlin", True)),
    (("berlin", "olympiastadion"), Place(52.5147, 13.2395, "Europe/Berlin", True)),
    (("azteca", "mexico"), Place(19.3029, -99.1505, "America/Mexico_City", True)),
    (("corinthians", "sao paulo", "são paulo"), Place(-23.5453, -46.4742, "America/Sao_Paulo", True)),
    (("bernabeu", "bernabéu", "madrid"), Place(40.4531, -3.6883, "Europe/Madrid", True)),
    (("croke", "dublin"), Place(53.3607, -6.2511, "Europe/Dublin", True)),
    (("rogers centre", "toronto"), Place(43.6414, -79.3894, "America/Toronto", True)),
]

MAIN_FEATURES = [
    "neutral_game", "international_game", "neutral_unknown",
    "away_travel_1000", "home_travel_1000",
    "away_tz_cross", "home_tz_cross",
    "away_eastward", "away_westward", "home_eastward", "home_westward",
    "away_early_deficit", "home_early_deficit",
    "away_prior_road_streak", "home_prior_road_streak",
    "away_short_rest_deficit", "home_short_rest_deficit",
]

INTERACTIONS = [
    "away_east_x_early", "home_east_x_early",
    "away_tz_x_early", "home_tz_x_early",
    "away_travel_x_short", "home_travel_x_short",
    "away_travel_x_road", "home_travel_x_road",
    "away_east_x_short", "home_east_x_short",
    "away_international_x_early", "home_international_x_early",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/track_e_travel_body_clock"))
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def kickoff_utc(day: object, clock: object) -> pd.Timestamp:
    if pd.isna(day) or pd.isna(clock):
        return pd.NaT
    try:
        local = pd.Timestamp(f"{pd.Timestamp(day).strftime('%Y-%m-%d')} {str(clock).strip()}").tz_localize(EASTERN)
        return local.tz_convert("UTC")
    except Exception:
        return pd.NaT


def haversine_miles(a: Place, b: Place) -> float:
    r = 3958.7613
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def team_place(team: str) -> Place:
    key = str(team).strip().upper()
    if key not in TEAM_BASES:
        raise KeyError(f"No Track E team base for {key}")
    return TEAM_BASES[key]


def neutral_place(stadium: object) -> Place | None:
    s = str(stadium or "").lower()
    for keys, place in NEUTRAL_VENUES:
        if any(k in s for k in keys):
            return place
    return None


def utc_offset_hours(ts: pd.Timestamp, zone: str) -> float:
    local = ts.tz_convert(ZoneInfo(zone))
    return float(local.utcoffset().total_seconds() / 3600.0)


def body_hour(ts: pd.Timestamp, zone: str) -> float:
    local = ts.tz_convert(ZoneInfo(zone))
    return local.hour + local.minute / 60.0


def add_road_sequence(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games[["game_id", "gameday", "home_team", "away_team", "location"]].itertuples(index=False):
        neutral = str(r.location).lower() == "neutral"
        rows.append((r.game_id, r.gameday, r.home_team, 1 if neutral else 0))
        rows.append((r.game_id, r.gameday, r.away_team, 1))
    long = pd.DataFrame(rows, columns=["game_id", "gameday", "team", "road_like"])
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    prior = np.zeros(len(long), dtype=float)
    for _, idx in long.groupby("team", sort=False).groups.items():
        streak = 0
        for i in idx:
            prior[i] = streak
            streak = streak + 1 if int(long.at[i, "road_like"]) == 1 else 0
    long["prior_road_streak"] = prior
    home = long.rename(columns={"team": "home_team", "prior_road_streak": "home_prior_road_streak"})[["game_id", "home_team", "home_prior_road_streak"]]
    away = long.rename(columns={"team": "away_team", "prior_road_streak": "away_prior_road_streak"})[["game_id", "away_team", "away_prior_road_streak"]]
    return games.merge(home, on=["game_id", "home_team"], how="left").merge(away, on=["game_id", "away_team"], how="left")


def add_travel_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    x["kickoff_utc"] = [kickoff_utc(d, t) for d, t in zip(x["gameday"], x["gametime"])]
    x = add_road_sequence(x)

    records: list[dict[str, float | int]] = []
    unknown_teams: set[str] = set()
    for r in x.itertuples(index=False):
        try:
            hp = team_place(r.home_team)
            ap = team_place(r.away_team)
        except KeyError as exc:
            unknown_teams.add(str(exc))
            records.append({})
            continue

        neutral = str(getattr(r, "location", "")).lower() == "neutral"
        venue = neutral_place(getattr(r, "stadium", "")) if neutral else hp
        neutral_unknown = int(neutral and venue is None)
        if venue is None:
            venue = hp

        ts = r.kickoff_utc
        if pd.isna(ts):
            records.append({})
            continue
        ts = pd.Timestamp(ts)

        home_travel = haversine_miles(hp, venue) if neutral else 0.0
        away_travel = haversine_miles(ap, venue)

        h_team_off = utc_offset_hours(ts, hp.tz)
        a_team_off = utc_offset_hours(ts, ap.tz)
        venue_off = utc_offset_hours(ts, venue.tz)
        h_shift = venue_off - h_team_off
        a_shift = venue_off - a_team_off

        h_hour = body_hour(ts, hp.tz)
        a_hour = body_hour(ts, ap.tz)
        h_early = max(0.0, 12.0 - h_hour)
        a_early = max(0.0, 12.0 - a_hour)

        h_rest = pd.to_numeric(pd.Series([getattr(r, "home_rest", np.nan)]), errors="coerce").iloc[0]
        a_rest = pd.to_numeric(pd.Series([getattr(r, "away_rest", np.nan)]), errors="coerce").iloc[0]
        h_short = max(0.0, 7.0 - float(h_rest)) if pd.notna(h_rest) else 0.0
        a_short = max(0.0, 7.0 - float(a_rest)) if pd.notna(a_rest) else 0.0

        international = int(neutral and venue.international)
        row = {
            "neutral_game": int(neutral),
            "international_game": international,
            "neutral_unknown": neutral_unknown,
            "away_travel_1000": away_travel / 1000.0,
            "home_travel_1000": home_travel / 1000.0,
            "away_tz_cross": abs(a_shift),
            "home_tz_cross": abs(h_shift),
            "away_eastward": max(0.0, a_shift),
            "away_westward": max(0.0, -a_shift),
            "home_eastward": max(0.0, h_shift),
            "home_westward": max(0.0, -h_shift),
            "away_body_clock_hour": a_hour,
            "home_body_clock_hour": h_hour,
            "away_early_deficit": a_early,
            "home_early_deficit": h_early,
            "away_short_rest_deficit": a_short,
            "home_short_rest_deficit": h_short,
        }
        row.update({
            "away_east_x_early": row["away_eastward"] * a_early,
            "home_east_x_early": row["home_eastward"] * h_early,
            "away_tz_x_early": row["away_tz_cross"] * a_early,
            "home_tz_x_early": row["home_tz_cross"] * h_early,
            "away_travel_x_short": row["away_travel_1000"] * a_short,
            "home_travel_x_short": row["home_travel_1000"] * h_short,
            "away_travel_x_road": row["away_travel_1000"] * float(getattr(r, "away_prior_road_streak", 0) or 0),
            "home_travel_x_road": row["home_travel_1000"] * float(getattr(r, "home_prior_road_streak", 0) or 0),
            "away_east_x_short": row["away_eastward"] * a_short,
            "home_east_x_short": row["home_eastward"] * h_short,
            "away_international_x_early": international * a_early,
            "home_international_x_early": international * h_early,
        })
        records.append(row)

    if unknown_teams:
        raise RuntimeError(f"Track E missing team mappings: {sorted(unknown_teams)}")

    feat = pd.DataFrame(records, index=x.index)
    for c in feat.columns:
        x[c] = feat[c]
    x["away_prior_road_streak"] = pd.to_numeric(x["away_prior_road_streak"], errors="coerce").fillna(0.0)
    x["home_prior_road_streak"] = pd.to_numeric(x["home_prior_road_streak"], errors="coerce").fillna(0.0)
    return x


@dataclass
class OffsetFit:
    theta: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    medians: np.ndarray


def matrix(df: pd.DataFrame, features: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if medians is None:
        medians = np.nanmedian(arr, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = np.take(medians, np.where(bad)[1])
    return arr, medians


def fit_offset(df: pd.DataFrame, features: list[str], penalty: float) -> OffsetFit:
    raw_x, meds = matrix(df, features)
    mean = raw_x.mean(axis=0)
    scale = raw_x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (raw_x - mean) / scale
    X = np.column_stack([np.ones(len(z)), z])
    y = df["home_win"].astype(int).to_numpy()
    base = logit(df["market_home_prob"].to_numpy(float))
    weights = np.ones(X.shape[1]); weights[0] = 0.10

    def objective(theta: np.ndarray):
        p = sigmoid(base + X @ theta)
        nll = -np.sum(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
        reg = 0.5 * penalty * np.sum(weights * theta * theta)
        grad = X.T @ (p - y) + penalty * weights * theta
        return float(nll + reg), grad

    res = minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", jac=True, options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(res.message)
    return OffsetFit(res.x, mean, scale, meds)


def predict_offset(df: pd.DataFrame, features: list[str], fit: OffsetFit) -> np.ndarray:
    raw_x, _ = matrix(df, features, fit.medians)
    z = (raw_x - fit.mean) / fit.scale
    X = np.column_stack([np.ones(len(z)), z])
    return sigmoid(logit(df["market_home_prob"].to_numpy(float)) + X @ fit.theta)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    yy = np.asarray(y, int); pp = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    pick = pp >= 0.5
    return {
        "games": int(len(yy)),
        "correct": int(np.sum(pick == yy)),
        "accuracy": float(np.mean(pick == yy)),
        "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
        "brier": float(brier_score_loss(yy, pp)),
    }


def select_penalty(data: pd.DataFrame, features: list[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for penalty in PENALTIES:
        probs, ys = [], []
        for season in (2016, 2017, 2018):
            tr = data.loc[data["season"] < season]
            te = data.loc[data["season"] == season]
            if tr.empty or te.empty:
                continue
            fit = fit_offset(tr, features, penalty)
            probs.append(predict_offset(te, features, fit))
            ys.append(te["home_win"].astype(int).to_numpy())
        p = np.concatenate(probs); y = np.concatenate(ys)
        rows.append({"penalty": penalty, **score(y, p)})
    table = pd.DataFrame(rows).sort_values(["log_loss", "brier", "penalty"])
    return float(table.iloc[0]["penalty"]), table


def holdout_predictions(data: pd.DataFrame, features: list[str], penalty: float, name: str) -> pd.DataFrame:
    outs = []
    for season in range(2019, 2026):
        tr = data.loc[data["season"] < season]
        te = data.loc[data["season"] == season].copy()
        fit = fit_offset(tr, features, penalty)
        te[f"p_home_{name}"] = predict_offset(te, features, fit)
        outs.append(te)
    return pd.concat(outs, ignore_index=True)


def paired_bootstrap(y: np.ndarray, p: np.ndarray, market: np.ndarray, n: int = 50000) -> dict[str, float]:
    yy = np.asarray(y, int)
    a = ((np.asarray(p) >= 0.5) == yy).astype(float)
    b = ((np.asarray(market) >= 0.5) == yy).astype(float)
    d = a - b
    rng = np.random.default_rng(42)
    # Chunked multinomial-equivalent resampling avoids a huge n x games array.
    vals = np.empty(n, dtype=float)
    chunk = 1000
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        idx = rng.integers(0, len(d), size=(m, len(d)))
        vals[start:start+m] = d[idx].mean(axis=1)
    return {
        "lift_pp": float(d.mean() * 100),
        "ci_low_pp": float(np.quantile(vals, 0.025) * 100),
        "ci_high_pp": float(np.quantile(vals, 0.975) * 100),
        "p_lift_gt_0": float(np.mean(vals > 0)),
    }


def scope_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "all": return pd.Series(True, index=df.index)
    if name == "early_body_clock": return df["away_body_clock_hour"] < 11.0
    if name == "cross_2plus": return df["away_tz_cross"] >= 2.0
    if name == "eastward_2plus": return df["away_eastward"] >= 2.0
    if name == "westward_2plus": return df["away_westward"] >= 2.0
    if name == "long_travel_1500": return df["away_travel_1000"] >= 1.5
    if name == "short_rest_long_travel": return (df["away_short_rest_deficit"] > 0) & (df["away_travel_1000"] >= 1.0)
    if name == "consecutive_road": return df["away_prior_road_streak"] >= 1.0
    if name == "international": return df["international_game"] == 1
    raise KeyError(name)


def summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = ["all", "early_body_clock", "cross_2plus", "eastward_2plus", "westward_2plus", "long_travel_1500", "short_rest_long_travel", "consecutive_road", "international"]
    rows = []
    for scope in scopes:
        sub = pred.loc[scope_mask(pred, scope)].copy()
        if sub.empty: continue
        y = sub["home_win"].astype(int).to_numpy()
        market = sub["market_home_prob"].to_numpy(float)
        m_correct = score(y, market)["correct"]
        for name, col in (("market", "market_home_prob"), ("travel_main", "p_home_travel_main"), ("travel_interactions", "p_home_travel_interactions")):
            s = score(y, sub[col].to_numpy(float))
            rows.append({"scope": scope, "model": name, **s, "net_vs_market": int(s["correct"] - m_correct)})
    scores = pd.DataFrame(rows)

    year_rows = []
    for season, sub in pred.groupby("season"):
        y = sub["home_win"].astype(int).to_numpy(); market = sub["market_home_prob"].to_numpy(float)
        mc = score(y, market)["correct"]
        mainc = score(y, sub["p_home_travel_main"].to_numpy(float))["correct"]
        intc = score(y, sub["p_home_travel_interactions"].to_numpy(float))["correct"]
        year_rows.append({"season": int(season), "games": len(sub), "market_correct": mc, "main_correct": mainc, "interaction_correct": intc, "interaction_net_vs_market": intc-mc})
    years = pd.DataFrame(year_rows)

    y = pred["home_win"].astype(int).to_numpy(); market = pred["market_home_prob"].to_numpy(float); pi = pred["p_home_travel_interactions"].to_numpy(float)
    pm = pred["p_home_travel_main"].to_numpy(float)
    mi = (pi >= .5); mm = (market >= .5); mg = (pm >= .5)
    disagreement = pd.DataFrame([{
        "interactions_vs_market_disagreements": int(np.sum(mi != mm)),
        "interaction_wins_when_disagree": int(np.sum((mi != mm) & (mi == y))),
        "market_wins_when_disagree": int(np.sum((mi != mm) & (mm == y))),
        "interactions_vs_main_disagreements": int(np.sum(mi != mg)),
        "interaction_wins_vs_main_when_disagree": int(np.sum((mi != mg) & (mi == y))),
        "main_wins_when_disagree": int(np.sum((mi != mg) & (mg == y))),
    }])
    return scores, years, disagreement


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model_table(args.start_season, args.end_season)
    data = add_travel_features(raw)
    data = data.loc[data["home_win"].notna() & data["market_home_prob"].notna() & data["kickoff_utc"].notna()].copy()

    main_penalty, main_dev = select_penalty(data, MAIN_FEATURES)
    int_features = [*MAIN_FEATURES, *INTERACTIONS]
    int_penalty, int_dev = select_penalty(data, int_features)
    print(f"Selected development penalties: main={main_penalty:g}, interactions={int_penalty:g}")

    main_pred = holdout_predictions(data, MAIN_FEATURES, main_penalty, "travel_main")
    int_pred = holdout_predictions(data, int_features, int_penalty, "travel_interactions")
    keep = ["game_id", "p_home_travel_interactions"]
    pred = main_pred.merge(int_pred[keep], on="game_id", how="inner")

    scores, years, disagreement = summarize(pred)
    y = pred["home_win"].astype(int).to_numpy(); market = pred["market_home_prob"].to_numpy(float)
    boot = paired_bootstrap(y, pred["p_home_travel_interactions"].to_numpy(float), market)

    loo = []
    for season in sorted(pred["season"].unique()):
        sub = pred.loc[pred["season"] != season]
        yy = sub["home_win"].astype(int).to_numpy()
        mc = score(yy, sub["market_home_prob"].to_numpy(float))["correct"]
        ic = score(yy, sub["p_home_travel_interactions"].to_numpy(float))["correct"]
        loo.append({"left_out_season": int(season), "net_vs_market": int(ic-mc)})
    loo_df = pd.DataFrame(loo)

    diagnostics = pd.DataFrame([{
        "games_all_completed": len(data),
        "holdout_games": len(pred),
        "neutral_holdout": int(pred["neutral_game"].sum()),
        "international_holdout": int(pred["international_game"].sum()),
        "neutral_unknown_holdout": int(pred["neutral_unknown"].sum()),
        "early_body_clock_holdout": int((pred["away_body_clock_hour"] < 11).sum()),
        "cross_2plus_holdout": int((pred["away_tz_cross"] >= 2).sum()),
        "long_travel_1500_holdout": int((pred["away_travel_1000"] >= 1.5).sum()),
        "main_penalty": main_penalty,
        "interaction_penalty": int_penalty,
    }])

    pred.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scope_scores.csv", index=False)
    years.to_csv(args.output_dir / "season_scores.csv", index=False)
    disagreement.to_csv(args.output_dir / "disagreements.csv", index=False)
    loo_df.to_csv(args.output_dir / "leave_one_season_out.csv", index=False)
    pd.DataFrame([boot]).to_csv(args.output_dir / "bootstrap.csv", index=False)
    diagnostics.to_csv(args.output_dir / "diagnostics.csv", index=False)
    main_dev.assign(model="travel_main").to_csv(args.output_dir / "development_main.csv", index=False)
    int_dev.assign(model="travel_interactions").to_csv(args.output_dir / "development_interactions.csv", index=False)
    pd.DataFrame({"feature": MAIN_FEATURES, "family": "main"}).to_csv(args.output_dir / "main_features.csv", index=False)
    pd.DataFrame({"feature": INTERACTIONS, "family": "interaction"}).to_csv(args.output_dir / "interaction_features.csv", index=False)

    all_scores = scores.loc[scores["scope"].eq("all")].set_index("model")
    m = all_scores.loc["market"]; g = all_scores.loc["travel_main"]; i = all_scores.loc["travel_interactions"]
    d = disagreement.iloc[0]
    lines = [
        "# Track E: Travel / Body-Clock Study", "",
        "**Research-only. No change to `prospective-v1-frozen-2025`.**", "",
        f"Development OOS: **2016-2018**. Diagnostic holdout: **2019-2025**. Selected ridge penalties: main **{main_penalty:g}**, interactions **{int_penalty:g}**.", "",
        "## Holdout: all games", "",
        f"- Market: **{int(m.correct)}/{int(m.games)} ({100*m.accuracy:.2f}%)**",
        f"- Travel main effects: **{int(g.correct)}/{int(g.games)} ({100*g.accuracy:.2f}%)**, net **{int(g.net_vs_market):+d}** vs market",
        f"- Travel/body-clock interactions: **{int(i.correct)}/{int(i.games)} ({100*i.accuracy:.2f}%)**, net **{int(i.net_vs_market):+d}** vs market",
        f"- Paired lift: **{boot['lift_pp']:+.3f} pp**, 95% CI **[{boot['ci_low_pp']:+.3f}, {boot['ci_high_pp']:+.3f}]**, P(lift>0) **{100*boot['p_lift_gt_0']:.1f}%**", "",
        "## Disagreements", "",
        f"- Interactions vs market: **{int(d.interactions_vs_market_disagreements)}** disagreements; interaction/market wins **{int(d.interaction_wins_when_disagree)}/{int(d.market_wins_when_disagree)}**.",
        f"- Leave-one-holdout-season-out net advantage range: **{int(loo_df.net_vs_market.min()):+d} to {int(loo_df.net_vs_market.max()):+d}** correct picks vs market.", "",
        "## Guardrail", "",
        "Travel and kickoff timing are knowable pregame, but any historical lift still must clear the same stability/proof standard before production use. This study does not alter the frozen 2026 model.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nScope scores")
    print(scores.to_string(index=False))
    print("\nSeason scores")
    print(years.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
