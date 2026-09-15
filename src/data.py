"""
data.py — dual-dataset loader with graceful fallback.

Priority:
  1. Raw Ben Roshan CSV  (data/raw/placement_benroshan.csv)
  2. Raw Kumbhar CSV     (data/raw/placement_kumbhar.csv)
  3. Legacy merged CSV   (data/placement_merged.csv) — written here on first run
  4. Legacy single file  (data/placement_data.csv)   — last-resort fallback

When both raw files are present, they are merged into a unified schema and
cached as `data/placement_merged.csv` for faster subsequent loads.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from . import config


# ── Target normalisation ──────────────────────────────────────────────────────

def _normalise_target(series: pd.Series) -> pd.Series:
    """Map any raw target string to 'Placed' or 'Not Placed'."""
    mapping = {
        "placed": "Placed",
        "1": "Placed",
        "yes": "Placed",
        "notplaced": "Not Placed",
        "not placed": "Not Placed",
        "not_placed": "Not Placed",
        "0": "Not Placed",
        "no": "Not Placed",
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(series)


# ── Ben Roshan loader ─────────────────────────────────────────────────────────

def load_benroshan(path=None) -> pd.DataFrame:
    """Load the 215-row Ben Roshan Campus Placement dataset."""
    path = path or config.BENROSHAN_PATH
    if not path.exists():
        # Try the legacy file location
        if config.LEGACY_DATA_PATH.exists():
            path = config.LEGACY_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Ben Roshan dataset not found at {path}.\n"
                "Download with: kaggle datasets download -d benroshan/factors-affecting-campus-placement"
            )
    df = pd.read_csv(path)

    # Drop ID column if present
    for id_col in config.ID_FEATURES:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    # Normalise target column name
    for candidate in ("status", "Status", "placement_status", "PlacementStatus"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: config.TARGET})
            break

    df[config.TARGET] = _normalise_target(df[config.TARGET])

    # Retain salary column if present
    for sal_col in ("salary", "Salary"):
        if sal_col in df.columns and sal_col != config.SALARY_TARGET:
            df = df.rename(columns={sal_col: config.SALARY_TARGET})

    # Kumbhar columns will be NaN for this dataset — that is intentional
    for col in config.KUMBHAR_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan

    df["_source"] = "benroshan"
    return df


# ── Kumbhar loader ────────────────────────────────────────────────────────────

def load_kumbhar(path=None) -> pd.DataFrame:
    """Load the Ruchika Kumbhar placement dataset (~10k rows)."""
    path = path or config.KUMBHAR_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Kumbhar dataset not found at {path}.\n"
            "Download with: kaggle datasets download -d ruchikakumbhar/placement-prediction-dataset"
        )
    df = pd.read_csv(path)

    # Rename columns to the unified schema
    df = df.rename(columns=config.KUMBHAR_COL_MAP)

    # Normalise target
    df[config.TARGET] = _normalise_target(df[config.TARGET])

    # Map PlacementTraining / ExtracurricularActivities → workex proxy
    # (Kumbhar has no strict "work experience" field; use ExtracurricularActivities)
    if "workex" not in df.columns:
        if "placement_training" in df.columns:
            df["workex"] = df["placement_training"].map(
                {"Yes": "Yes", "No": "No", 1: "Yes", 0: "No"}
            ).fillna("No")
        else:
            df["workex"] = "No"

    # Fill Ben Roshan-style percentage columns from CGPA (rough conversion)
    if "cgpa" in df.columns:
        for pct_col in ("ssc_p", "hsc_p", "degree_p"):
            if pct_col not in df.columns:
                df[pct_col] = (df["cgpa"] * 9.5).clip(0, 100)

    # Fill etest_p from aptitude_score (both ~0-100)
    if "etest_p" not in df.columns and "aptitude_score" in df.columns:
        df["etest_p"] = df["aptitude_score"]

    # Fill missing Ben Roshan categoricals with sensible defaults
    for col, default in [
        ("gender", "M"),
        ("ssc_b", "Others"),
        ("hsc_b", "Others"),
        ("hsc_s", "Commerce"),
        ("degree_t", "Comm&Mgmt"),
        ("specialisation", "Mkt&HR"),
        ("mba_p", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    # Salary is unavailable in this dataset
    if config.SALARY_TARGET not in df.columns:
        df[config.SALARY_TARGET] = np.nan

    df["_source"] = "kumbhar"
    return df


# ── Merge helper ──────────────────────────────────────────────────────────────

def merge_datasets(ben: pd.DataFrame, kumbhar: pd.DataFrame) -> pd.DataFrame:
    """Align both datasets to the unified feature schema and concatenate."""
    all_cols = (
        config.BENROSHAN_NUMERIC
        + config.KUMBHAR_NUMERIC
        + config.BENROSHAN_CATEGORICAL
        + [config.TARGET, config.SALARY_TARGET, "_source"]
    )
    ben_aligned = ben.reindex(columns=all_cols)
    kum_aligned = kumbhar.reindex(columns=all_cols)
    merged = pd.concat([ben_aligned, kum_aligned], ignore_index=True)

    # Fill remaining NaN in percentage columns with column medians
    for col in config.BENROSHAN_NUMERIC:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    return merged


# ── Public entry point ────────────────────────────────────────────────────────

def load_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the best available dataset combination.

    Returns a DataFrame with at minimum all BENROSHAN_NUMERIC +
    BENROSHAN_CATEGORICAL + TARGET columns.
    Kumbhar columns will be NaN when not available.
    """
    # Use cached merged file if available (and not forced)
    if not force_reload and config.DATA_PATH.exists():
        return pd.read_csv(config.DATA_PATH)

    ben_available = (
        config.BENROSHAN_PATH.exists() or config.LEGACY_DATA_PATH.exists()
    )
    kum_available = config.KUMBHAR_PATH.exists()

    if ben_available and kum_available:
        print("Loading dual-dataset (Ben Roshan + Kumbhar)…")
        ben = load_benroshan()
        kum = load_kumbhar()
        df = merge_datasets(ben, kum)
        df.to_csv(config.DATA_PATH, index=False)
        print(f"  Merged → {len(df)} rows, cached to {config.DATA_PATH}")
    elif ben_available:
        print("Loading Ben Roshan dataset only (Kumbhar not found)…")
        df = load_benroshan()
        df.to_csv(config.DATA_PATH, index=False)
        print(f"  {len(df)} rows loaded.")
    else:
        raise FileNotFoundError(
            "No placement dataset found. Place CSVs in data/raw/ or data/. "
            "See README for download instructions."
        )

    return df


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return binary-encoded target column."""
    return df[config.TARGET].map(config.LABEL_MAP).astype(int)