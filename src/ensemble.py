from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .tuning import build_raw, with_smote


def single_estimator(algo, params, settings, scale_pos_weight=1.0):
    raw = build_raw(algo, params, scale_pos_weight)
    return with_smote(raw, settings.use_smote)


def _probable_single(algo, params, settings, scale_pos_weight=1.0):
    raw = build_raw(algo, params, scale_pos_weight)
    if algo == "svc":
        estimator = CalibratedClassifierCV(
            SVC(C=params["C"], gamma=params["gamma"], class_weight="balanced",
                random_state=config.RANDOM_STATE),
            ensemble=False,
        )
    else:
        estimator = raw
    return with_smote(estimator, settings.use_smote)


def build_candidates(algo_params, settings, scale_pos_weight=1.0):
    candidates = {}
    for algo, params in algo_params.items():
        display = config.MODEL_NAMES[algo]
        candidates[display] = {
            "estimator": _probable_single(algo, params, settings, scale_pos_weight),
            "tuned_params": params,
        }

    stackers = [a for a in settings.stackers if a in algo_params]
    if settings.use_voting and len(stackers) >= 2:
        estimators = [(a, _probable_single(a, algo_params[a], settings, scale_pos_weight)) for a in stackers]
        candidates[config.MODEL_NAMES["voting"]] = {
            "estimator": VotingClassifier(estimators=estimators, voting="soft"),
            "tuned_params": None,
        }

    if settings.use_stacking and len(stackers) >= 2:
        estimators = [(a, _probable_single(a, algo_params[a], settings, scale_pos_weight)) for a in stackers]
        meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=config.RANDOM_STATE)
        cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
        candidates[config.MODEL_NAMES["stacking"]] = {
            "estimator": StackingClassifier(estimators=estimators, final_estimator=meta,
                                            cv=cv, stack_method="predict_proba", n_jobs=-1),
            "tuned_params": None,
        }
    return candidates


def compare_candidates(candidates, X, y, settings):
    cv = RepeatedStratifiedKFold(
        n_splits=settings.cv_splits, n_repeats=settings.n_repeats,
        random_state=config.RANDOM_STATE,
    )
    rows = []
    for name, entry in candidates.items():
        scores = cross_val_score(entry["estimator"], X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc_mean": float(scores.mean()),
                     "cv_roc_auc_std": float(scores.std())})
    return rows


class ProbaCalibrator:
    def __init__(self, base_estimator, mapping, kind):
        self.base_estimator = base_estimator
        self.mapping = mapping
        self.kind = kind

    def predict_proba(self, X):
        p1 = np.asarray(self.base_estimator.predict_proba(X))[:, 1]
        if self.kind == "sigmoid":
            raw = self.mapping.predict_proba(p1.reshape(-1, 1))[:, 1]
        else:
            raw = self.mapping.predict(p1)
        raw = np.clip(raw, 1e-9, 1 - 1e-9)
        return np.column_stack([1 - raw, raw])


def calibrate(base_estimator, methods, X_cal, y_cal, X_val, y_val):
    cal_proba = np.asarray(base_estimator.predict_proba(X_cal))[:, 1]
    val_proba = np.asarray(base_estimator.predict_proba(X_val))[:, 1]

    results = {"none": float(brier_score_loss(y_val, val_proba))}
    best = ("none", None)
    errors = {}
    for method in methods:
        try:
            if method == "sigmoid":
                mapping = LogisticRegression(max_iter=2000).fit(cal_proba.reshape(-1, 1), y_cal)
                fitted = mapping.predict_proba(val_proba.reshape(-1, 1))[:, 1]
                calibrator = ProbaCalibrator(base_estimator, mapping, "sigmoid")
            else:
                mapping = IsotonicRegression(out_of_bounds="clip").fit(cal_proba, y_cal)
                fitted = mapping.predict(val_proba)
                calibrator = ProbaCalibrator(base_estimator, mapping, "isotonic")
            fitted = np.clip(fitted, 1e-9, 1 - 1e-9)
            score = float(brier_score_loss(y_val, fitted))
            results[method] = score
            if score <= results[best[0]]:
                best = (method, calibrator)
        except Exception as exc:
            results[method] = float("nan")
            errors[method] = str(exc)
    if errors:
        print(f"  calibration errors: {errors}")
    return best, results