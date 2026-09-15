import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, experiments
from .data import get_target, load_data
from .ensemble import build_candidates, calibrate, compare_candidates
from .eval import (
    evaluate,
    pick_threshold,
    plot_calibration,
    plot_confusion,
    plot_learning_curve,
    plot_permutation_importance,
    plot_pr,
    plot_roc,
    report_table,
)
from .features import add_feature_engineering, build_preprocessor, feature_names
from .interpret import explain_and_save, unwrap
from .monitor import save_reference
from .salary import train_salary_model
from .settings import TrainSettings
from .tuning import tune


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train placement prediction models")
    parser.add_argument("--config", type=str, default=str(config.TRAIN_CONFIG_PATH))
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--threshold-metric", choices=["f1", "youden"], default=None)
    parser.add_argument("--force-reload", action="store_true",
                        help="Re-merge datasets even if cached merged CSV exists")
    return parser.parse_args(args)


def main():
    cli = parse_args()
    settings = TrainSettings.load(cli.config)
    if cli.trials:
        settings.n_trials = cli.trials
    if cli.no_smote:
        settings.use_smote = False
    if cli.no_calibration:
        settings.use_calibration = False
    if cli.threshold_metric:
        settings.threshold_metric = cli.threshold_metric

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data(force_reload=getattr(cli, "force_reload", False))
    print(f"Dataset: {len(df)} rows, source(s): {df.get('_source', pd.Series()).value_counts().to_dict() if '_source' in df.columns else 'unknown'}")

    y = get_target(df).values
    drop_cols = [config.TARGET, config.SALARY_TARGET, "_source"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Feature engineering ───────────────────────────────────────────────────
    engineer = add_feature_engineering
    X_engineered = engineer(X)

    # Determine which numeric columns are actually non-trivial (not all-NaN)
    available_numeric = [
        c for c in config.ALL_NUMERIC
        if c in X_engineered.columns and X_engineered[c].notna().any()
    ]
    print(f"Active numeric features ({len(available_numeric)}): {available_numeric}")

    # ── Train / val / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=settings.test_size, stratify=y,
        random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=settings.test_size, stratify=y_train,
        random_state=config.RANDOM_STATE
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(list(X_engineered.columns))
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    names = feature_names(preprocessor)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"Class balance — Placed: {pos}, Not Placed: {neg}, scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Optuna hyperparameter tuning ──────────────────────────────────────────
    start = time.time()
    print("\nOptuna hyperparameter tuning:")
    algo_params = {}
    study_scores = {}
    for algo in settings.tune_models:
        result = tune(algo, X_train_t, y_train, settings, scale_pos_weight)
        algo_params[algo] = result["best_params"]
        study_scores[algo] = result["best_score"]
        print(f"  {algo:<18} CV ROC-AUC = {result['best_score']:.4f}  {result['best_params']}")

    # ── Candidate comparison ──────────────────────────────────────────────────
    print("\nBuilding candidates and comparing with repeated CV:")
    candidates = build_candidates(algo_params, settings, scale_pos_weight)
    cv_table = compare_candidates(candidates, X_train_t, y_train, settings)
    print(pd.DataFrame(cv_table).to_string(index=False))

    best_row = max(cv_table, key=lambda r: r["cv_roc_auc_mean"])
    champion_name = best_row["model"]
    print(f"\nChampion candidate: {champion_name} (CV ROC-AUC {best_row['cv_roc_auc_mean']:.4f})")

    champion_estimator = candidates[champion_name]["estimator"]
    champion_estimator.fit(X_train_t, y_train)

    # ── Probability calibration ───────────────────────────────────────────────
    calibration_name = "none"
    calibrator = None
    calibration_results = {}
    if settings.use_calibration:
        best_cal, calibration_results = calibrate(
            unwrap(champion_estimator), settings.calibration_methods,
            X_train_t, y_train, X_val_t, y_val
        )
        calibration_name, calibrator = best_cal
        print(f"Calibration: {dict(calibration_results)} → chosen {calibration_name}")

    def final_proba(X):
        if calibrator is not None:
            return calibrator.predict_proba(X)[:, 1]
        return champion_estimator.predict_proba(X)[:, 1]

    # ── Threshold selection + evaluation ─────────────────────────────────────
    val_proba = final_proba(X_val_t)
    threshold = pick_threshold(y_val, val_proba, settings.threshold_metric)
    test_proba = final_proba(X_test_t)
    metrics = evaluate(y_test, test_proba, threshold)
    print(f"\nThreshold ({settings.threshold_metric}): {threshold:.3f}")
    print("Hold-out test metrics:")
    print(json.dumps(metrics, indent=2))

    # ── SHAP surrogate (for ensemble models) ─────────────────────────────────
    surrogate = None
    surrogate_name = None
    explanation_surrogate = False
    if champion_name in (config.MODEL_NAMES["voting"], config.MODEL_NAMES["stacking"]):
        singles = {c: candidates[c]["estimator"] for c in candidates if c not in
                   (config.MODEL_NAMES["voting"], config.MODEL_NAMES["stacking"])}
        best_single = max(singles, key=lambda c: next(r["cv_roc_auc_mean"] for r in cv_table if r["model"] == c))
        surrogate = singles[best_single]
        surrogate.fit(X_train_t, y_train)
        surrogate_name = best_single
        explanation_surrogate = True
        print(f"SHAP surrogate: {best_single}")

    # ── Run artefacts ─────────────────────────────────────────────────────────
    run_id = experiments.new_run_id(champion_name.lower().replace(" ", "_"))
    run_root = config.RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    plots = run_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plot_roc(y_test, test_proba, plots)
    plot_pr(y_test, test_proba, plots)
    plot_calibration(y_test, test_proba, plots)
    curve_data = plot_learning_curve(champion_estimator, X_train_t, y_train, plots,
                                     splits=settings.cv_splits)
    perm_df = plot_permutation_importance(champion_estimator, X_test_t, y_test, names, plots)
    perm_df.to_csv(run_root / "permutation_importance.csv", index=False)
    plot_confusion(y_test, test_proba, threshold, plots)

    y_pred = (test_proba >= threshold).astype(int)
    report = report_table(y_test, y_pred)
    report.to_csv(run_root / "classification_report.csv", index=False)

    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "run_id": run_id,
        "created_at": trained_at,
        "model_name": champion_name,
        "estimator_key": next(k for k, v in config.MODEL_NAMES.items() if v == champion_name),
        "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
        "cv_roc_auc_std": best_row["cv_roc_auc_std"],
        "study_scores": study_scores,
        "tuned_params": algo_params,
        "threshold": threshold,
        "threshold_metric": settings.threshold_metric,
        "test_metrics": metrics,
        "calibration_method": calibration_name,
        "calibration_results": calibration_results,
        "explanation_mode": "surrogate" if explanation_surrogate else "exact",
        "surrogate_model": surrogate_name,
        "feature_names": names,
        "active_numeric_features": available_numeric,
        "label_map": config.LABEL_MAP,
        "n_train": int(len(X_train_t)),
        "n_val": int(len(X_val_t)),
        "n_test": int(len(X_test_t)),
        "n_total": int(len(df)),
        "settings": settings.to_dict(),
        "elapsed_seconds": round(time.time() - start, 1),
        "learning_curve": curve_data,
    }
    run_root.joinpath("meta.json").write_text(json.dumps(meta, indent=2))

    bundle = {
        "model": champion_estimator,
        "calibrator": calibrator,
        "preprocessor": preprocessor,
        "threshold": threshold,
        "feature_names": names,
        "available_numeric": available_numeric,
        "model_name": champion_name,
        "run_id": run_id,
        "trained_at": trained_at,
        "calibration_method": calibration_name,
        "label_map": config.LABEL_MAP,
        "surrogate": surrogate,
        "surrogate_name": surrogate_name,
        "explanation_surrogate": explanation_surrogate,
    }
    joblib.dump(bundle, run_root / "bundle.joblib")

    explain_estimator = surrogate if surrogate is not None else champion_estimator
    X_all_engineered = engineer(
        df.drop(columns=[c for c in drop_cols if c in df.columns])
    )
    explain_and_save(explain_estimator, preprocessor, names, X_all_engineered, run_root)

    save_reference(df.assign(**{c: df[c] for c in df.columns}), run_root / "reference_features.json")

    cv_df = pd.DataFrame(cv_table).set_index("model").sort_values("cv_roc_auc_mean", ascending=False)
    cv_df.to_csv(run_root / "cv_comparison.csv")

    experiments.register(run_id, meta)
    result = experiments.select_champion(run_id, meta, run_root)
    print(f"\nRun {run_id} registered. Champion promoted: {result['promoted']}")
    print(f"Artifacts → {run_root}")

    # ── Salary regression (optional, runs after main pipeline) ───────────────
    print("\nTraining salary regression model…")
    salary_meta = train_salary_model(df)
    if salary_meta:
        meta["salary_model"] = salary_meta
        print(f"  Salary model saved → {config.SALARY_BUNDLE_PATH}")
    else:
        print("  Salary model skipped (insufficient data).")

    return meta


if __name__ == "__main__":
    main()