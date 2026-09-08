"""Helpers for exporting and loading the frozen 2026 model artifacts.

The production model architecture and learned weights remain prospective-v1-frozen-2025.
These helpers only make the already-frozen pieces cheap to load during live runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_manifest_files(root: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names}


def export_base_logistic(model, numeric: list[str], categorical: list[str], path: Path) -> None:
    pre = model.named_steps["preprocess"]
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    clf = model.named_steps["model"]

    num_imp = num.named_steps["impute"]
    scaler = num.named_steps["scale"]
    cat_imp = cat.named_steps["impute"]
    onehot = cat.named_steps["onehot"]

    payload = {
        "kind": "base_logistic_pipeline_v1",
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "numeric_imputer_statistics": [float(v) for v in num_imp.statistics_],
        "numeric_scaler_mean": [float(v) for v in scaler.mean_],
        "numeric_scaler_scale": [float(v) for v in scaler.scale_],
        "categorical_imputer_statistics": [str(v) for v in cat_imp.statistics_],
        "categorical_categories": [[str(v) for v in cats] for cats in onehot.categories_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
    }
    write_json(path, payload)


def predict_base_logistic(spec: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    numeric = list(spec["numeric_features"])
    categorical = list(spec["categorical_features"])

    num = frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    stats = np.asarray(spec["numeric_imputer_statistics"], dtype=float)
    missing = ~np.isfinite(num)
    if missing.any():
        num[missing] = np.take(stats, np.where(missing)[1])
    mean = np.asarray(spec["numeric_scaler_mean"], dtype=float)
    scale = np.asarray(spec["numeric_scaler_scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    blocks: list[np.ndarray] = [(num - mean) / scale]

    fills = list(spec["categorical_imputer_statistics"])
    categories = list(spec["categorical_categories"])
    for idx, col in enumerate(categorical):
        values: list[str] = []
        for value in frame[col].tolist():
            if pd.isna(value):
                values.append(str(fills[idx]))
            else:
                values.append(str(value))
        arr = np.asarray(values, dtype=object)
        for cat in categories[idx]:
            blocks.append((arr == str(cat)).astype(float).reshape(-1, 1))

    x = np.column_stack(blocks)
    coef = np.asarray(spec["coef"], dtype=float)
    if x.shape[1] != coef.shape[0]:
        raise RuntimeError(f"Base logistic transformed width {x.shape[1]} != coef width {coef.shape[0]}")
    return sigmoid(x @ coef + float(spec["intercept"]))


def export_simple_logistic(model, features: list[str], path: Path) -> None:
    imp = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    clf = model.named_steps["model"]
    payload = {
        "kind": "simple_logistic_pipeline_v1",
        "features": list(features),
        "imputer_statistics": [float(v) for v in imp.statistics_],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
    }
    write_json(path, payload)


def predict_simple_logistic(spec: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = list(spec["features"])
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    stats = np.asarray(spec["imputer_statistics"], dtype=float)
    missing = ~np.isfinite(x)
    if missing.any():
        x[missing] = np.take(stats, np.where(missing)[1])
    mean = np.asarray(spec["scaler_mean"], dtype=float)
    scale = np.asarray(spec["scaler_scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - mean) / scale
    coef = np.asarray(spec["coef"], dtype=float)
    return sigmoid(z @ coef + float(spec["intercept"]))


def load_catboost(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path), format="json")
    return model


def artifact_manifest_hash(path: Path) -> str:
    return sha256_file(path)
