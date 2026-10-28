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

import numpy as np
import pandas as pd

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


def _rename_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Migrate SSC/HSC-style column names to the current 10th/12th schema."""
    renames = {old: new for old, new in config.LEGACY_COLUMN_RENAMES.items()
               if old in df.columns and new not in df.columns}
    return df.rename(columns=renames) if renames else df


def add_masters_fields(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """
    Ensure `has_masters` and `masters_type` exist.

    The public Kaggle datasets do not record non-MBA master's programmes, so
    when the fields are absent this function synthesises them deterministically
    from a fixed seed.  Rows that *do* carry an MBA score (the Ben Roshan MBA
    cohort) are always marked as master's holders; the programme type is
    sampled as MBA/MS/Other.  Rows without any master's information are
    assigned Yes/No with equal probability.

    Important: this is synthetic augmentation for the academic demo and is
    intentionally independent of the target, so it cannot leak label signal.
    Replace it with real programme data when available.
    """
    df = df.copy()
    rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
    n = len(df)
    if n == 0:
        df["has_masters"] = pd.Series(dtype=object)
        df["masters_type"] = pd.Series(dtype=object)
        return df

    def _existing(column: str) -> pd.Series:
        if column in df.columns:
            return df[column].astype(object).where(df[column].notna(), None)
        return pd.Series([None] * n, index=df.index, dtype=object)

    has = _existing("has_masters")
    mtype = _existing("masters_type")

    # Normalise any provided values (e.g. "1"/"0", "yes"/"no").
    norm_has = has.astype(str).str.strip().str.lower().map(
        {"yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes",
         "no": "No", "n": "No", "0": "No", "false": "No"}
    )
    has = has.where(has.isna(), norm_has)
    norm_type = mtype.astype(str).str.strip().str.title().map(
        {"Mba": "MBA", "Ms": "MS", "Other": "Other", "None": "None"}
    )
    mtype = mtype.where(mtype.isna(), norm_type)

    known_mba = (
        df["mba_p"].notna() if "mba_p" in df.columns
        else pd.Series(False, index=df.index)
    )

    # 1. Fill missing has_masters: MBA cohort -> Yes, others 50/50.
    missing_has = has.isna()
    yes_prob = np.where(known_mba, 1.0, 0.5)
    synth_has = np.where(rng.random(n) < yes_prob, "Yes", "No")
    has = has.where(~missing_has, synth_has)

    # 2. Fill missing masters_type for master's holders.
    draws = rng.random(n)
    mba_cut, ms_cut = np.where(known_mba, 0.60, 0.50), np.where(known_mba, 0.85, 0.80)
    synth_type = np.where(
        draws < mba_cut, "MBA", np.where(draws < ms_cut, "MS", "Other")
    )
    missing_type = mtype.isna()
    mtype = mtype.where(~missing_type, synth_type)

    # 3. Enforce consistency: no masters -> type "None"; masters -> never None.
    mtype = mtype.where(has.eq("Yes"), "None")
    mtype = mtype.where(~(has.eq("Yes") & mtype.eq("None")), "MBA")

    df["has_masters"] = has
    df["masters_type"] = mtype
    return df



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

    # Migrate legacy SSC/HSC column names to the 10th/12th schema.
    df = _rename_legacy_columns(df)

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
    return add_masters_fields(df)


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
    df = _rename_legacy_columns(df)

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
        for pct_col in ("tenth_p", "twelfth_p", "degree_p"):
            if pct_col not in df.columns:
                df[pct_col] = (df["cgpa"] * 9.5).clip(0, 100)

    # Fill etest_p from aptitude_score (both ~0-100)
    if "etest_p" not in df.columns and "aptitude_score" in df.columns:
        df["etest_p"] = df["aptitude_score"]

    # Fill missing Ben Roshan categoricals with sensible defaults
    for col, default in [
        ("gender", "M"),
        ("tenth_b", "Others"),
        ("twelfth_b", "Others"),
        ("twelfth_s", "Commerce"),
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
    return add_masters_fields(df)


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

    # Missing values are intentionally left untouched here.  Imputation is
    # fitted inside the sklearn pipeline on X_train only (see features.py),
    # which prevents validation/test information from leaking into training.
    return add_masters_fields(merged)


# ── Public entry point ────────────────────────────────────────────────────────

_CACHE_REQUIRED_COLUMNS = set(
    config.BENROSHAN_NUMERIC
    + config.BENROSHAN_CATEGORICAL
    + [config.TARGET]
)


def _cache_is_current(df: pd.DataFrame) -> bool:
    """Return True when a cached CSV matches the current schema."""
    return _CACHE_REQUIRED_COLUMNS.issubset(df.columns)


def load_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the best available dataset combination.

    Returns a DataFrame with at minimum all BENROSHAN_NUMERIC +
    BENROSHAN_CATEGORICAL + TARGET columns.
    Kumbhar columns will be NaN when not available.
    """
    # Use cached merged file if available and schema-compatible.
    if not force_reload and config.DATA_PATH.exists():
        # Keep "None" as a real masters_type category instead of pandas' NA.
        cached = pd.read_csv(config.DATA_PATH, keep_default_na=False, na_values=[""])
        if _cache_is_current(cached):
            return cached
        print(f"Cached dataset schema is outdated (v{config.SCHEMA_VERSION}); reloading…")

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
        print(f"  Merged -> {len(df)} rows, cached to {config.DATA_PATH}")
    elif ben_available:
        print("Loading Ben Roshan dataset only (Kumbhar not found)…")
        df = add_masters_fields(load_benroshan())
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
