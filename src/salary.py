"""
salary.py — Salary estimation module.

Trains a simple XGBRegressor on placed students from the Ben Roshan dataset
(the only dataset with real salary data).  At inference time, returns:
  - salary_estimate: predicted annual CTC in INR (integer)
  - salary_band:     "low" | "mid" | "high"
  - salary_range:    [lower_bound, upper_bound] confidence interval

The model is saved to models/salary_bundle.joblib and loaded once at startup.
When no salary training data is available the module stubs out gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import add_feature_engineering, build_preprocessor

# ── Band thresholds (annual INR) ─────────────────────────────────────────────
BAND_LOW = config.SALARY_BAND_LOW   # < 2.5L → "low"
BAND_HIGH = config.SALARY_BAND_HIGH  # > 4.0L → "high"


def _band(salary: float) -> str:
    if salary < BAND_LOW:
        return "low"
    if salary > BAND_HIGH:
        return "high"
    return "mid"


def _salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only columns useful for salary regression (placed students)."""
    return add_feature_engineering(df)


# ── Training ──────────────────────────────────────────────────────────────────

def train_salary_model(df: pd.DataFrame) -> dict[str, Any] | None:
    """
    Train a salary regressor on placed students.

    Returns a metadata dict on success, None if there are too few labelled rows.
    Saves salary_bundle.joblib and salary_meta.json to the models directory.
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError as exc:
        print(f"  Salary model skipped (import error): {exc}")
        return None

    # Filter to placed students with known salary
    placed = df[
        (df[config.TARGET] == "Placed")
        & df[config.SALARY_TARGET].notna()
        & (df[config.SALARY_TARGET] > 0)
    ].copy()

    if len(placed) < 30:
        print(f"  Salary model skipped — only {len(placed)} labelled rows (need ≥ 30).")
        return None

    y_sal = placed[config.SALARY_TARGET].astype(float)
    X_sal = placed.drop(columns=[config.TARGET, config.SALARY_TARGET, "_source"],
                         errors="ignore")
    X_sal = _salary_features(X_sal)

    available_cols = list(X_sal.columns)
    preprocessor = build_preprocessor(available_cols)
    X_t = preprocessor.fit_transform(X_sal)

    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_sal, test_size=0.2, random_state=config.RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Confidence margin: 1 std of residuals (used for salary_range)
    residuals = y_test - y_pred
    margin = float(np.std(residuals))

    print(f"  Salary model: MAE = ₹{mae:,.0f}, R² = {r2:.3f}, margin ±₹{margin:,.0f}")

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "available_cols": available_cols,
        "margin": margin,
    }
    joblib.dump(bundle, config.SALARY_BUNDLE_PATH)

    meta = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": mae,
        "r2": r2,
        "margin": margin,
        "band_low": BAND_LOW,
        "band_high": BAND_HIGH,
    }
    config.SALARY_META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


# ── Inference ─────────────────────────────────────────────────────────────────

class SalaryPredictor:
    """Thin wrapper around the saved salary bundle."""

    def __init__(self, bundle_path: Path | None = None):
        bundle_path = bundle_path or config.SALARY_BUNDLE_PATH
        self._bundle = joblib.load(bundle_path)

    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Predict salary from a raw input dict.

        Returns a dict with salary_estimate, salary_band, salary_range.
        """
        bundle = self._bundle
        # Build a single-row DataFrame from raw input
        df = pd.DataFrame([raw])
        X = add_feature_engineering(df)
        # Keep only columns the preprocessor was trained on
        available = bundle["available_cols"]
        for col in available:
            if col not in X.columns:
                X[col] = np.nan
        X = X[available]

        X_t = bundle["preprocessor"].transform(X)
        estimate = float(bundle["model"].predict(X_t)[0])
        margin = bundle["margin"]

        return {
            "salary_estimate": int(round(estimate, -3)),    # round to nearest ₹1,000
            "salary_band": _band(estimate),
            "salary_range": [
                int(max(0, estimate - margin)),
                int(estimate + margin),
            ],
        }


def load_salary_predictor() -> SalaryPredictor | None:
    """Return a SalaryPredictor if the model artifact exists, else None."""
    if config.SALARY_BUNDLE_PATH.exists():
        try:
            return SalaryPredictor()
        except Exception as exc:
            print(f"Warning: salary model could not be loaded: {exc}")
    return None
