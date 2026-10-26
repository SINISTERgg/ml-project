import numpy as np
import optuna
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_raw(algo, params, scale_pos_weight=1.0):
    rs = config.RANDOM_STATE
    if algo == "logistic":
        return LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced",
            random_state=rs, C=params["C"],
        )
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params["max_features"],
            class_weight="balanced_subsample", random_state=rs,
        )
    if algo == "gradient_boosting":
        return GradientBoostingClassifier(
            learning_rate=params["learning_rate"], n_estimators=params["n_estimators"],
            max_depth=params["max_depth"], subsample=params["subsample"],
            min_samples_leaf=params.get("min_samples_leaf", 1), random_state=rs,
        )
    if algo == "xgboost":
        return XGBClassifier(
            learning_rate=params["learning_rate"], n_estimators=params["n_estimators"],
            max_depth=params["max_depth"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params.get("min_child_weight", 1.0),
            gamma=params.get("gamma", 0.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=rs,
        )
    if algo == "svc":
        return SVC(C=params["C"], gamma=params["gamma"], class_weight="balanced", random_state=rs)
    raise ValueError(f"Unknown algorithm: {algo}")


def suggest_params(algo, trial):
    if algo == "logistic":
        return {"C": trial.suggest_float("C", 1e-4, 1e3, log=True)}
    if algo == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if algo == "gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 40, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if algo == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 40, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if algo == "svc":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        }
    raise ValueError(f"Unknown algorithm: {algo}")


def with_smote(estimator, use_smote, categorical_indices=None):
    """Wrap an estimator with a categorical-safe resampler.

    SMOTENC is used whenever categorical feature indices are supplied, which
    prevents standard SMOTE from creating fractional values in one-hot /
    categorical columns.  If no categorical indices are available we fall back
    to regular SMOTE (all-numeric feature spaces).
    """
    if not use_smote:
        return estimator
    if categorical_indices:
        sampler = SMOTENC(
            categorical_features=list(categorical_indices),
            random_state=config.RANDOM_STATE,
        )
    else:
        sampler = SMOTE(random_state=config.RANDOM_STATE)
    return ImbPipeline([("smote", sampler), ("clf", estimator)])


def tune(algo, X, y, settings, scale_pos_weight=1.0, categorical_indices=None):
    def objective(trial):
        params = suggest_params(algo, trial)
        estimator = with_smote(
            build_raw(algo, params, scale_pos_weight), settings.use_smote,
            categorical_indices,
        )
        cv = StratifiedKFold(settings.cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
        scores = []
        for step, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            fold = clone(estimator)
            fold.fit(X[train_idx], y[train_idx])
            proba = fold.predict_proba(X[valid_idx])[:, 1]
            scores.append(float(roc_auc_score(y[valid_idx], proba)))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f"placement-{algo}",
    )
    study.optimize(objective, n_trials=settings.n_trials, show_progress_bar=False)

    try:
        importances = {
            k: float(v) for k, v in optuna.importance.get_param_importances(study).items()
        }
    except (RuntimeError, ValueError, TypeError):
        importances = {}

    return {
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "param_importances": importances,
        "trials": [t.value for t in study.trials if t.value is not None],
    }
