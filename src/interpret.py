import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config

PLOT_STYLE = {"dpi": 150, "bbox_inches": "tight"}


def unwrap(estimator):
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list) and steps:
        return steps[-1][1]
    return estimator


def build_explainer(model, Xt):
    model = unwrap(model)
    name = type(model).__name__
    if any(tok in name for tok in ("Tree", "Forest", "Boost", "XGB", "Xgb")):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, Xt)


def compute_shap_values(explainer, Xt):
    if "Linear" in type(explainer).__name__:
        return explainer.shap_values(Xt)
    return explainer.shap_values(Xt, check_additivity=False)


def expected_value(explainer):
    ev = getattr(explainer, "expected_value", None)
    if ev is None:
        return None
    arr = np.asarray(ev).ravel()
    if arr.size == 0:
        return None
    return float(arr[-1] if arr.size == 1 else arr[-1])


def explain_and_save(estimator, preprocessor, names, X_engineered, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    plots = run_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    Xt = pd.DataFrame(preprocessor.transform(X_engineered), columns=names)
    explainer = build_explainer(estimator, Xt)
    shap_values = compute_shap_values(explainer, Xt)

    info = {
        "expected_value": expected_value(explainer),
        "explainer_type": type(explainer).__name__,
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