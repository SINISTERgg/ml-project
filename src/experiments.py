import json
import shutil
from datetime import datetime, timezone

from . import config


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def new_run_id(tag):
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + tag


def load_registry():
    if not config.REGISTRY_PATH.exists():
        return {"runs": [], "champion": None}
    return json.loads(config.REGISTRY_PATH.read_text())


def save_registry(registry):
    config.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def register(run_id, meta):
    registry = load_registry()
    entry = {
        "run_id": run_id,
        "model_name": meta["model_name"],
        "created_at": meta["created_at"],
        "cv_roc_auc_mean": meta["cv_roc_auc_mean"],
        "cv_brier_mean": meta.get("cv_brier_mean"),
        "test_metrics": meta["test_metrics"],
        "tuned_params": meta.get("tuned_params"),
        "calibration": meta.get("calibration_method"),
    }
    registry["runs"].append(entry)
    save_registry(registry)
    return entry, registry


def champion_score(meta):
    """Rank candidates by CV ROC-AUC, then Brier score, then F1."""
    cv = meta["cv_roc_auc_mean"]
    brier = meta["test_metrics"].get("brier", 1.0)
    f1 = meta["test_metrics"].get("f1", 0.0)
    return (cv, -brier, f1)


def _beats(current, previous):
    cur_cv, cur_brier_neg, cur_f1 = current
    prev_cv, prev_brier_neg, prev_f1 = previous
    if cur_cv > prev_cv + 1e-12:
        return True
    if abs(cur_cv - prev_cv) <= 1e-12:
        if cur_brier_neg > prev_brier_neg + 1e-12:
            return True
        if abs(cur_brier_neg - prev_brier_neg) <= 1e-12 and cur_f1 >= prev_f1:
            return True
    return False


def select_champion(run_id, meta, run_root):
    registry = load_registry()
    current = champion_score(meta)
    current_champion = registry.get("champion")
    is_better = False
    schema_changed = False
    if current_champion is not None:
        prev_meta_path = config.RUNS_DIR / current_champion["run_id"] / "meta.json"
        if prev_meta_path.exists():
            prev_meta = json.loads(prev_meta_path.read_text())
            prev_features = prev_meta.get("feature_names")
            new_features = meta.get("feature_names")
            schema_changed = (
                prev_features is not None
                and new_features is not None
                and prev_features != new_features
            )
    if current_champion is None or schema_changed:
        is_better = True
    else:
        prev = next((r for r in registry["runs"] if r["run_id"] == current_champion["run_id"]), None)
        if prev is not None:
            prev_cv = prev["cv_roc_auc_mean"]
            prev_brier = prev["test_metrics"].get("brier", 1.0)
            prev_f1 = prev["test_metrics"].get("f1", 0.0)
            is_better = _beats(current, (prev_cv, -prev_brier, prev_f1))
        else:
            is_better = True

    if is_better:
        for rel, dst in [
            ("bundle.joblib", config.BUNDLE_PATH),
            ("shap_explainer.joblib", config.EXPLAINER_PATH),
            ("explainer_info.json", config.EXPLAINER_INFO_PATH),
            ("reference_features.json", config.REFERENCE_PATH),
            ("meta.json", config.META_PATH),
            ("shap_values.npy", config.SHAP_VALUES_PATH),
        ]:
            src = run_root / rel
            if src.exists():
                shutil.copyfile(src, dst)
        for plot in (run_root / "plots").glob("*"):
            shutil.copyfile(plot, config.PLOTS_DIR / plot.name)
        registry["champion"] = {
            "run_id": run_id,
            "model_name": meta["model_name"],
            "updated_at": now_iso(),
        }
        champion_path = {
            "run_id": run_id,
            "model_name": meta["model_name"],
            "bundle_path": str(config.BUNDLE_PATH),
            "explainer_path": str(config.EXPLAINER_PATH),
            "explainer_info": str(config.EXPLAINER_INFO_PATH),
            "updated_at": now_iso(),
        }
        config.CHAMPION_PATH.write_text(json.dumps(champion_path, indent=2))
        save_registry(registry)
        return {"promoted": True, "run_id": run_id}
    return {"promoted": False, "run_id": run_id}
