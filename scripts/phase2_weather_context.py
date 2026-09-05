"""Phase 2 near-kickoff weather context experiment.

Historical schedule temperature/wind are treated as a value-of-information test
for conditions knowable near kickoff. They are NOT assumed to be available from
nflverse before a future game; a production implementation must use a timestamped
forecast/observation source available before the pick deadline.

The experiment tests whether weather improves a football model and whether a
market-anchored residual can use that information to make better close-game
choices without broadly abandoning the market.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase1_backtest import BASE_TEAM_METRICS, ROLL_WINDOWS, build_model_table
from phase1_market_residual import EPS, logit, sigmoid

PENALTIES = (10.0, 30.0, 100.0, 300.0, 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2009)
    p.add_argument("--first-test-season", type=int, default=2016)
    p.add_argument("--residual-first-test-season", type=int, default=2019)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_weather_context"))
    return p.parse_args()


def add_weather_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    x["temp_num"] = pd.to_numeric(x.get("temp"), errors="coerce")
    x["wind_num"] = pd.to_numeric(x.get("wind"), errors="coerce")
    roof = x["roof"].astype("string").str.lower()
    x["weather_exposed"] = roof.isin(["outdoors", "open", "retractable"]).astype(int)
    x["weather_reported"] = (x["temp_num"].notna() | x["wind_num"].notna()).astype(int)
    x["cold_degrees"] = (40.0 - x["temp_num"]).clip(lower=0)
    x["very_cold_degrees"] = (32.0 - x["temp_num"]).clip(lower=0)
    x["hot_degrees"] = (80.0 - x["temp_num"]) * -1.0
    x["hot_degrees"] = x["hot_degrees"].clip(lower=0)
    x["wind_over_10"] = (x["wind_num"] - 10.0).clip(lower=0)
    x["wind_over_15"] = (x["wind_num"] - 15.0).clip(lower=0)
    x["market_closeness"] = 1.0 - 2.0 * np.abs(x["market_home_prob"].astype(float) - 0.5)
    x["wind_x_closeness"] = x["wind_num"] * x["market_closeness"]
    x["wind_x_total"] = x["wind_num"] * pd.to_numeric(x["total_line"], errors="coerce") / 50.0
    weather = [
        "temp_num", "wind_num", "weather_exposed", "weather_reported", "cold_degrees",
        "very_cold_degrees", "hot_degrees", "wind_over_10", "wind_over_15",
        "market_closeness", "wind_x_closeness", "wind_x_total",
    ]
    return x, weather


def make_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.15, C=0.5,
            max_iter=5000, random_state=42,
        )),
    ])


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    yy = np.asarray(y, int)
    prob = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    pick = prob >= 0.5
    return {
        "games": int(len(yy)),
        "correct": int(np.sum(pick == yy)),
        "accuracy": float(np.mean(pick == yy)),
        "log_loss": float(log_loss(yy, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(yy, prob)),
    }


def base_features() -> tuple[list[str], list[str]]:
    diffs = [f"diff_{m}_r{w}" for m in BASE_TEAM_METRICS for w in ROLL_WINDOWS]
    numeric = [
        "market_home_prob", "spread_line", "total_line", "rest_diff", "elo_diff",
        "elo_home_prob", *diffs,
    ]
    categorical = ["roof", "surface", "location", "div_game"]
    return numeric, categorical


def oof_weather_models(data: pd.DataFrame, weather: list[str], first_test: int, end_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_numeric, categorical = base_features()
    weather_numeric = [*base_numeric, *weather]
    completed = data.loc[data["home_win"].notna()].copy()
    pred_rows, fold_rows = [], []

    for season in range(first_test, end_season + 1):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if train.empty or test.empty:
            continue
        y_train = train["home_win"].astype(int)
        y_test = test["home_win"].astype(int).to_numpy()
        probs = {"market": test["market_home_prob"].to_numpy(float)}

        for name, numeric in (("base_logistic", base_numeric), ("weather_logistic", weather_numeric)):
            cols = [*numeric, *categorical]
            model = make_logistic(numeric, categorical)
            model.fit(train[cols], y_train)
            probs[name] = model.predict_proba(test[cols])[:, 1]
            fold_rows.append({"season": season, "model": name, **score(y_test, probs[name])})
        fold_rows.append({"season": season, "model": "market", **score(y_test, probs["market"])})

        keep = ["game_id", "season", "week", "gameday", "away_team", "home_team", "home_win", "temp_num", "wind_num", "weather_exposed", "weather_reported"]
        out = test[keep].copy()
        for name, p in probs.items():
            out[f"p_home_{name}"] = p
        pred_rows.append(out)
        print(
            f"{season}: market={score(y_test, probs['market'])['accuracy']:.3f} "
            f"base={score(y_test, probs['base_logistic'])['accuracy']:.3f} "
            f"weather={score(y_test, probs['weather_logistic'])['accuracy']:.3f}"
        )

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows)


def residual_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    market_l = logit(df["p_home_market"].to_numpy(float))
    base_delta = logit(df["p_home_base_logistic"].to_numpy(float)) - market_l
    weather_delta = logit(df["p_home_weather_logistic"].to_numpy(float)) - market_l
    incremental = logit(df["p_home_weather_logistic"].to_numpy(float)) - logit(df["p_home_base_logistic"].to_numpy(float))
    x = np.column_stack([np.ones(len(df)), base_delta, weather_delta, incremental])
    return market_l, x


def offset_objective(theta: np.ndarray, market_l: np.ndarray, x: np.ndarray, y: np.ndarray, penalty: float):
    eta = market_l + x @ theta
    p = sigmoid(eta)
    nll = -np.sum(y * np.log(np.clip(p, EPS, 1)) + (1-y) * np.log(np.clip(1-p, EPS, 1)))
    weights = np.ones_like(theta)
    weights[0] = 0.10
    reg = 0.5 * penalty * np.sum(weights * theta * theta)
    grad = x.T @ (p-y) + penalty * weights * theta
    return float(nll + reg), grad


def fit_offset(df: pd.DataFrame, penalty: float) -> np.ndarray:
    market_l, x = residual_design(df)
    y = df["home_win"].astype(int).to_numpy()
    res = minimize(
        fun=lambda t: offset_objective(t, market_l, x, y, penalty),
        x0=np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def predict_offset(df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
    market_l, x = residual_design(df)
    return sigmoid(market_l + x @ theta)


def choose_penalty(train: pd.DataFrame) -> float:
    seasons = sorted(int(s) for s in train["season"].unique())
    if len(seasons) < 2:
        return 100.0
    rows = []
    for penalty in PENALTIES:
        ps, ys = [], []
        for valid in seasons[1:]:
            tr = train.loc[train["season"] < valid]
            va = train.loc[train["season"] == valid]
            if tr.empty or va.empty:
                continue
            theta = fit_offset(tr, penalty)
            ps.append(predict_offset(va, theta))
            ys.append(va["home_win"].astype(int).to_numpy())
        if ps:
            p = np.concatenate(ps); y = np.concatenate(ys)
            rows.append((float(log_loss(y, p, labels=[0,1])), float(brier_score_loss(y,p)), penalty))
    return min(rows)[2] if rows else 100.0


def nested_residual(oof: pd.DataFrame, first_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, folds = [], []
    for season in sorted(int(s) for s in oof["season"].unique() if int(s) >= first_test):
        train = oof.loc[oof["season"] < season]
        test = oof.loc[oof["season"] == season]
        if train.empty or test.empty:
            continue
        penalty = choose_penalty(train)
        theta = fit_offset(train, penalty)
        p = predict_offset(test, theta)
        y = test["home_win"].astype(int).to_numpy()
        market = test["p_home_market"].to_numpy(float)
        folds.append({
            "season": season, "penalty": penalty,
            **{f"residual_{k}": v for k,v in score(y,p).items()},
            **{f"market_{k}": v for k,v in score(y,market).items()},
            "net_correct_vs_market": score(y,p)["correct"] - score(y,market)["correct"],
            "theta_intercept": theta[0], "theta_base": theta[1],
            "theta_weather": theta[2], "theta_incremental": theta[3],
        })
        out = test.copy()
        out["p_home_weather_residual"] = p
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds)


def summaries(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pred["home_win"].astype(int).to_numpy()
    rows = []
    for name, col in (
        ("market", "p_home_market"),
        ("base_logistic", "p_home_base_logistic"),
        ("weather_logistic", "p_home_weather_logistic"),
        ("weather_residual", "p_home_weather_residual"),
    ):
        rows.append({"model": name, **score(y, pred[col].to_numpy(float))})
    overall = pd.DataFrame(rows).sort_values(["accuracy", "brier"], ascending=[False, True])

    fav = np.maximum(pred["p_home_market"].to_numpy(float), 1 - pred["p_home_market"].to_numpy(float))
    close = pred.loc[fav < 0.525].copy()
    cy = close["home_win"].astype(int).to_numpy()
    crows = []
    for name, col in (
        ("market", "p_home_market"),
        ("base_logistic", "p_home_base_logistic"),
        ("weather_logistic", "p_home_weather_logistic"),
        ("weather_residual", "p_home_weather_residual"),
    ):
        crows.append({"model": name, **score(cy, close[col].to_numpy(float))})
    return overall, pd.DataFrame(crows).sort_values("accuracy", ascending=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model_table(args.start_season, args.end_season)
    data, weather = add_weather_features(raw)
    oof, folds = oof_weather_models(data, weather, args.first_test_season, args.end_season)
    residual_pred, residual_folds = nested_residual(oof, args.residual_first_test_season)
    overall, close = summaries(residual_pred)

    diagnostics = pd.DataFrame([{
        "games": len(data.loc[data["home_win"].notna()]),
        "temp_report_rate": float(data.loc[data["home_win"].notna(), "temp_num"].notna().mean()),
        "wind_report_rate": float(data.loc[data["home_win"].notna(), "wind_num"].notna().mean()),
        "weather_exposed_rate": float(data.loc[data["home_win"].notna(), "weather_exposed"].mean()),
    }])

    oof.to_csv(args.output_dir / "weather_oof_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "weather_oof_folds.csv", index=False)
    residual_pred.to_csv(args.output_dir / "weather_residual_predictions.csv", index=False)
    residual_folds.to_csv(args.output_dir / "weather_residual_folds.csv", index=False)
    overall.to_csv(args.output_dir / "weather_summary.csv", index=False)
    close.to_csv(args.output_dir / "weather_close_games.csv", index=False)
    diagnostics.to_csv(args.output_dir / "weather_data_diagnostics.csv", index=False)

    print("\nWeather context summary")
    print(overall.to_string(index=False))
    print("\nClose games")
    print(close.to_string(index=False))
    print("\nWeather data diagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
