"""
features.py — feature engineering + sklearn preprocessor.

Handles both the original 5-column Ben Roshan feature set and the expanded
Kumbhar columns.  When Kumbhar columns are all-NaN (Ben Roshan-only run) the
engineered activity/skills features fall back to 0.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Percentage columns that all datasets share
PCT_COLS_CORE = ["ssc_p", "hsc_p", "degree_p", "mba_p"]


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered features in-place (non-destructive copy)."""
    df = df.copy()

    # ── academic_avg: mean of all available percentage / CGPA columns ──────
    pct_cols = [c for c in PCT_COLS_CORE + ["cgpa", "etest_p"] if c in df.columns]
    available = df[pct_cols].copy()
    # Scale CGPA (0–10) to 0–100 for averaging with percentage columns
    if "cgpa" in available.columns:
        available["cgpa"] = available["cgpa"] * 9.5
    df["academic_avg"] = available.mean(axis=1)

    # ── performance_gap: spread across academic scores ──────────────────────
    df["performance_gap"] = available.max(axis=1) - available.min(axis=1)

    # ── stem_flag: Science / Sci&Tech background ────────────────────────────
    stem = pd.Series(False, index=df.index)
    if "hsc_s" in df.columns:
        stem |= df["hsc_s"].eq("Science")
    if "degree_t" in df.columns:
        stem |= df["degree_t"].eq("Sci&Tech")
    df["stem_flag"] = stem.astype(int)

    # ── activity_score: sum of practical-experience proxies ─────────────────
    activity_cols = [c for c in ["internships", "projects", "workshops"] if c in df.columns]
    if activity_cols:
        df["activity_score"] = df[activity_cols].fillna(0).sum(axis=1)
    else:
        df["activity_score"] = 0.0

    # ── skills_composite: combined aptitude + soft-skills (0-100 scale) ─────
    if "aptitude_score" in df.columns and "soft_skills" in df.columns:
        apt_col = df["aptitude_score"]
        ss_col = df["soft_skills"]
        apt_med = float(apt_col.median()) if apt_col.notna().any() else 0.0
        ss_med = float(ss_col.median()) if ss_col.notna().any() else 0.0
        apt = apt_col.fillna(apt_med)
        ss = ss_col.fillna(ss_med) * 10   # 0-10 → 0-100
        df["skills_composite"] = (apt + ss) / 2
    elif "aptitude_score" in df.columns:
        df["skills_composite"] = df["aptitude_score"].fillna(0)
    else:
        df["skills_composite"] = 0.0

    return df


def _columns_present(df_or_cols, column_list: list[str]) -> list[str]:
    """Return only those columns from column_list that exist in df_or_cols."""
    cols = df_or_cols if isinstance(df_or_cols, (list, set)) else list(df_or_cols.columns)
    return [c for c in column_list if c in cols]


def build_preprocessor(available_columns: list[str] | None = None) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles missing Kumbhar columns gracefully.

    If available_columns is given, only those numeric features that are actually
    present will be included.  Unknown columns are silently skipped so that a
    Ben Roshan-only run still works without modification.
    """
    numeric_cols = (
        _columns_present(available_columns, config.ALL_NUMERIC)
        if available_columns is not None
        else config.ALL_NUMERIC
    )
    categorical_cols = config.CATEGORICAL_FEATURES

    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return flattened output feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())

