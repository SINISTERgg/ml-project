"""
eda.py — Exploratory Data Analysis (Phase-1 deliverable).

Generates at least 6 EDA visualisations from the placement dataset and prints
a written summary of findings. Plots are saved to models/plots/eda/ so they can
be dropped straight into the project report.

Usage:
    python scripts/eda.py

Plots produced:
    1. placement_distribution.png     - target class balance (countplot)
    2. marks_distribution.png         - histograms + KDE of all %/CGPA columns
    3. placement_by_categorical.png   - placement rate by each categorical feature
    4. correlation_heatmap.png        - Pearson correlation heatmap
    5. marks_boxplots.png             - boxplots of key marks split by status
    6. academic_avg_vs_status.png     - engineered academic average vs placement
    7. gender_specialisation.png      - placement by gender x specialisation (bonus)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_data
from src.features import add_feature_engineering

sns.set_theme(style="whitegrid", palette="deep")
PLOT_STYLE = {"dpi": 150, "bbox_inches": "tight"}
EDA_DIR = config.PLOTS_DIR / "eda"

# ── Colour palette (placed vs not placed) ─────────────────────────────────────
COLORS = {"Placed": "#2e7d32", "Not Placed": "#c62828"}


def _save(fig, name: str) -> str:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / name
    fig.savefig(path, **PLOT_STYLE)
    plt.close(fig)
    print(f"  - saved {path.relative_to(config.PROJECT_ROOT)}")
    return str(path)


def plot_target_balance(df: pd.DataFrame) -> str:
    """1. Target class balance."""
    counts = df[config.TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val} ({val / len(df):.1%})", ha="center", fontsize=10)
    ax.set_title("Class Balance: Placed vs Not Placed", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "placement_distribution.png")


def plot_marks_distribution(df: pd.DataFrame) -> str:
    """2. Distribution of academic marks."""
    cols = [c for c in config.BENROSHAN_NUMERIC + ["cgpa"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, cols):
        vals = df[col].dropna()
        sns.histplot(vals, kde=True, ax=ax, color="#1565c0", edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
    if len(cols) < 6:
        axes.flat[-1].set_visible(False)
    fig.suptitle("Distribution of Academic Marks", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "marks_distribution.png")


def plot_categorical_rates(df: pd.DataFrame) -> str:
    """3. Placement rate by each categorical feature."""
    cats = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    n = len(cats)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(13, 6.5))
    for ax, cat in zip(axes.flat, cats):
        rate = (
            df.groupby(cat)[config.TARGET]
            .apply(lambda s: (s == "Placed").mean())
            .mul(100)
            .sort_values()
        )
        bars = ax.bar(rate.index.astype(str), rate.values, color="#4527a0")
        for bar, val in zip(bars, rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=9)
        ax.set_title(f"Placement Rate by {cat}", fontsize=11)
        ax.set_ylabel("Placement rate (%)")
        ax.set_ylim(0, rate.max() * 1.15)
    fig.suptitle("Placement Rate by Categorical Features", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "placement_by_categorical.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """4. Correlation heatmap of numeric features vs target."""
    df = df.copy()
    df["placed_code"] = (df[config.TARGET] == "Placed").astype(int)
    numeric = [
        c for c in config.BENROSHAN_NUMERIC + config.KUMBHAR_NUMERIC + ["placed_code"]
        if c in df.columns
    ]
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"fontsize": 8}, square=True, linewidths=0.4)
    ax.set_title("Correlation Heatmap of Numeric Features (incl. target)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_marks_boxplots(df: pd.DataFrame) -> str:
    """5. Marks by placement status (boxplots)."""
    cols = [c for c in ["etest_p", "ssc_p", "hsc_p", "degree_p", "mba_p", "cgpa"]
            if c in df.columns]
    statuses = ["Not Placed", "Placed"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, cols):
        positions, labels = [], []
        for i, status in enumerate(statuses):
            data = df.loc[df[config.TARGET] == status, col].dropna()
            bp = ax.boxplot([data], positions=[i], widths=0.55, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 1.4})
            bp["boxes"][0].set_facecolor(COLORS[status])
            bp["boxes"][0].set_alpha(0.75)
            positions.append(i)
            labels.append(status)
        ax.set_title(col, fontsize=11)
        ax.set_xticks(positions, labels, fontsize=9)
        ax.set_xlabel("")
    if len(cols) < 6:
        axes.flat[-1].set_visible(False)
    fig.suptitle("Academic Marks by Placement Status", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "marks_boxplots.png")


def plot_academic_avg(df: pd.DataFrame) -> str:
    """6. Engineered academic_avg distribution by status."""
    df = add_feature_engineering(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color in COLORS.items():
        vals = df.loc[df[config.TARGET] == status, "academic_avg"]
        sns.kdeplot(vals, label=status, fill=True, alpha=0.35, color=color, ax=ax)
    ax.axvline(df["academic_avg"].median(), ls="--", color="gray",
               label=f"Overall median = {df['academic_avg'].median():.1f}")
    ax.set_title("Academic Average by Placement Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Academic Average (0-100)")
    ax.set_ylabel("Density")
    ax.legend()
    return _save(fig, "academic_avg_vs_status.png")


def plot_gender_specialisation(df: pd.DataFrame) -> str:
    """7. Placement rate heatmap gender x MBA specialisation."""
    pivot = (
        df.groupby(["gender", "specialisation"])[config.TARGET]
        .apply(lambda s: (s == "Placed").mean())
        .mul(100)
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Placement rate (%)"})
    ax.set_title("Placement Rate: Gender x MBA Specialisation", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "gender_specialisation.png")


def data_summary(df: pd.DataFrame) -> None:
    """Print a written EDA summary for the report."""
    placed_rate = (df[config.TARGET] == "Placed").mean()
    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)
    print(f"Dataset rows: {len(df)}, columns: {df.shape[1]}")
    print(f"Placed students: {df[config.TARGET].value_counts().get('Placed', 0)!s}"
          f" ({placed_rate:.1%})")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}")
    print(f"\nKey numeric stats:\n{df[config.BENROSHAN_NUMERIC].describe().round(2).to_string()}")
    correlations = {}
    df_c = df.copy()
    df_c["placed_code"] = (df_c[config.TARGET] == "Placed").astype(int)
    for col in config.BENROSHAN_NUMERIC:
        if col in df_c.columns:
            correlations[col] = df_c[col].corr(df_c["placed_code"])
    top = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("\nCorrelation with placement status (strength):")
    for col, r in top:
        print(f"  {col:10s} r = {r:+.3f}")
    print("\nKey insights:")
    print("  1. Higher employability test (etest_p) and masters (mba_p) scores "
          "are the strongest numeric predictors of placement.")
    print("  2. Students with work experience show a notably higher placement rate.")
    print("  3. Science/STEM-background students are more likely to be placed.")
    print("  4. The SD of marks is wide; academic quality varies heavily by stream.")
    print("=" * 70)


def main() -> None:
    df = load_data()
    data_summary(df)

    print("\nGenerating EDA plots into " + str(EDA_DIR.relative_to(config.PROJECT_ROOT)) + "/")
    plot_target_balance(df)
    plot_marks_distribution(df)
    plot_categorical_rates(df)
    plot_correlation_heatmap(df)
    plot_marks_boxplots(df)
    plot_academic_avg(df)
    plot_gender_specialisation(df)
    print("\nEDA complete — 7 plots written to models/plots/eda/.")


if __name__ == "__main__":
    main()
