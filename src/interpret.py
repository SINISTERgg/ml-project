import json

import joblib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None
from sklearn.calibration import CalibratedClassifierCV

PLOT_STYLE = {"dpi": 150, "bbox_inches": "tight"}


def unwrap(estimator):
    """Unwrap one level of pipeline to expose the final estimator."""
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list) and steps:
        return steps[-1][1]
    return estimator


def deep_unwrap(estimator):
    """Recursively unwrap pipelines, SMOTE wrappers and calibrated models."""
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list) and steps:
        return deep_unwrap(steps[-1][1])
    if isinstance(estimator, CalibratedClassifierCV):
        return deep_unwrap(getattr(estimator, "estimator", estimator))
    return estimator


def is_tree_estimator(estimator):
    name = type(deep_unwrap(estimator)).__name__
    return any(tok in name for tok in ("Tree", "Forest", "Boost", "XGB", "Xgb"))


def shap_compatible(estimator):
    """True when SHAP has a built-in exact explainer for this estimator.

    Tree models -> TreeExplainer; linear models (coef_) -> LinearExplainer.
    Anything else (RBF SVC, calibration wrappers, voting/stacking) needs a
    surrogate, so train.py swaps in a logistic surrogate for explanation.
    """
    model = deep_unwrap(estimator)
    return is_tree_estimator(model) or hasattr(model, "coef_")


def build_explainer(model, Xt):
    model = deep_unwrap(model)
    name = type(model).__name__
    if any(tok in name for tok in ("Tree", "Forest", "Boost", "XGB", "Xgb")):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, Xt)


def positive_class_shap(shap_values):
    """Return a 2-D (n_samples, n_features) array for the positive class.

    SHAP returns different container types across explainers and versions:
    a list of per-class arrays (multi-class / binary TreeExplainer) or a 3-D
    array.  Reshaping those directly silently selects class 0 ("Not Placed").
    """
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) — take the positive class.
        if arr.shape[-1] > 1:
            return arr[:, :, 1]
        return arr[:, :, 0]
    return arr


def compute_shap_values(explainer, Xt):
    if explainer is None:
        ncols = Xt.shape[1] if hasattr(Xt, "shape") else 10
        return np.zeros((len(Xt), ncols))
    try:
        if "Linear" in type(explainer).__name__:
            return positive_class_shap(explainer.shap_values(Xt))
        return positive_class_shap(explainer.shap_values(Xt, check_additivity=False))
    except Exception:
        ncols = Xt.shape[1] if hasattr(Xt, "shape") else 10
        return np.zeros((len(Xt), ncols))


def expected_value(explainer):
    ev = getattr(explainer, "expected_value", None)
    if ev is None:
        return None
    arr = np.asarray(ev).ravel()
    if arr.size == 0:
        return None
    return float(arr[-1])


def to_probability(value):
    """Convert a log-odds margin to probability; leave probabilities alone."""
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    return float(1.0 / (1.0 + np.exp(-value)))


def explain_and_save(estimator, preprocessor, names, X_engineered, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    plots = run_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    Xt = pd.DataFrame(preprocessor.transform(X_engineered), columns=names)
    explainer = build_explainer(estimator, Xt)
    shap_values = compute_shap_values(explainer, Xt)

    ev = expected_value(explainer)
    info = {
        "expected_value": ev,
        "base_probability": None if ev is None else to_probability(ev),
        "explainer_type": type(explainer).__name__,
        "shap_class": "positive",
    }
    run_root.joinpath("explainer_info.json").write_text(json.dumps(info, indent=2))
    joblib.dump(explainer, run_root / "shap_explainer.joblib")
    np.save(run_root / "shap_values.npy", shap_values)

    plt.figure()
    shap.summary_plot(shap_values, Xt, show=False)
    plt.tight_layout()
    plt.savefig(plots / "shap_summary.png", **PLOT_STYLE)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, Xt, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(plots / "shap_importance.png", **PLOT_STYLE)
    plt.close()

    return info
