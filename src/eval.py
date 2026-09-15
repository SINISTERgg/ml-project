import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from . import config

PLOT_STYLE = {"dpi": 150, "bbox_inches": "tight"}


def pick_threshold(y_val, probs, metric="f1"):
    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_val, probs)
        return float(thr[np.argmax(tpr - fpr)])
    precision, recall, thr = precision_recall_curve(y_val, probs)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thr[np.argmax(f1)])


def evaluate(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
    }


def pl(plot_dir, name):
    return plot_dir / name


def plot_roc(y, proba, plot_dir):
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#7c6cff", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#565d85")
    ax.set(title=f"ROC curve (AUC = {auc:.3f})", xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pl(plot_dir, "roc_curve.png"), **PLOT_STYLE)
    plt.close(fig)


def plot_pr(y, proba, plot_dir):
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    base = y.mean()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#38d9a9", lw=2)
    ax.axhline(base, color="#565d85", ls="--", label=f"Baseline = {base:.2f}")
    ax.set(title=f"Precision-Recall (AP = {ap:.3f})", xlabel="Recall", ylabel="Precision")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pl(plot_dir, "pr_curve.png"), **PLOT_STYLE)
    plt.close(fig)


def plot_calibration(y, proba, plot_dir):
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=6)
    brier = brier_score_loss(y, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(prob_pred, prob_true, "o-", color="#ffd166")
    ax.plot([0, 1], [0, 1], "--", color="#565d85", label="Perfectly calibrated")
    ax.set(title=f"Reliability curve (Brier = {brier:.4f})", xlabel="Predicted probability", ylabel="Observed fraction")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pl(plot_dir, "calibration.png"), **PLOT_STYLE)
    plt.close(fig)


def plot_learning_curve(estimator, X, y, plot_dir, splits=5):
    sizes = np.linspace(0.3, 1.0, 5)
    cv = StratifiedKFold(splits, shuffle=True, random_state=config.RANDOM_STATE)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="roc_auc", train_sizes=sizes, n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.25, color="#7c6cff")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.25, color="#38d9a9")
    ax.plot(train_sizes, train_mean, "o-", color="#7c6cff", label="Train ROC-AUC")
    ax.plot(train_sizes, test_mean, "o-", color="#38d9a9", label="CV ROC-AUC")
    ax.set(title="Learning curve", xlabel="Training examples", ylabel="ROC-AUC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pl(plot_dir, "learning_curve.png"), **PLOT_STYLE)
    plt.close(fig)
    return {"train_sizes": train_sizes.tolist(), "train": train_mean.tolist(), "cv": test_mean.tolist()}


def plot_permutation_importance(estimator, X, y, names, plot_dir, repeats=5):
    result = permutation_importance(estimator, X, y, scoring="roc_auc", n_repeats=repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({
        "feature": names,
        "mean": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.35 * len(names))))
    ax.barh(df["feature"], df["mean"], xerr=df["std"], color="#7c6cff", alpha=0.85)
    ax.set(title="Permutation importance (ROC-AUC drop)", xlabel="Mean importance")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    return_type = fig
    fig.savefig(pl(plot_dir, "permutation_importance.png"), **PLOT_STYLE)
    plt.close(fig)
    return df


def plot_confusion(y, proba, threshold, plot_dir, labels=("Not Placed", "Placed")):
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set(title=f"Confusion matrix (thr = {threshold:.2f})")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(pl(plot_dir, "confusion_matrix.png"), **PLOT_STYLE)
    plt.close(fig)


def report_table(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    rows = [{"class": k, **{m: v[m] for m in ("precision", "recall", "f1-score", "support")}}
            for k, v in report.items() if isinstance(v, dict)]
    return pd.DataFrame(rows)