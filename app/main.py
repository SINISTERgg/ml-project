import json
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, experiments, monitor
from src.counterfactual import suggest
from src.predict import Predictor
from src.salary import load_salary_predictor
from src.settings import DEFAULT_CATEGORY_OPTIONS

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Placement Predictor",
    version="3.0",
    description=(
        "Advanced ML placement predictor with SHAP explanations, salary estimation, "
        "counterfactuals, batch scoring, analytics and drift monitoring."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
app.mount("/plots", StaticFiles(directory=str(config.PLOTS_DIR)), name="plots")


# ── Model loading (once at startup) ──────────────────────────────────────────

def _load_predictor():
    try:
        return Predictor()
    except Exception as exc:
        print(f"Classification model not ready: {exc}")
        return None


def _load_salary():
    try:
        return load_salary_predictor()
    except Exception as exc:
        print(f"Salary model not ready: {exc}")
        return None


predictor = _load_predictor()
salary_predictor = _load_salary()


def require_model():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train` first.",
        )
    return predictor


# ── Shared helpers ────────────────────────────────────────────────────────────

def _log_count() -> int:
    if not config.PREDICTION_LOG_PATH.exists():
        return 0
    return sum(
        1 for line in config.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PlacementInput(BaseModel):
    # Ben Roshan features (always required)
    gender: Literal["M", "F"]
    ssc_p: float = Field(ge=0, le=100)
    ssc_b: Literal["Central", "Others"]
    hsc_p: float = Field(ge=0, le=100)
    hsc_b: Literal["Central", "Others"]
    hsc_s: Literal["Commerce", "Science", "Arts"]
    degree_p: float = Field(ge=0, le=100)
    degree_t: Literal["Sci&Tech", "Comm&Mgmt", "Others"]
    workex: Literal["Yes", "No"]
    etest_p: float = Field(ge=0, le=100)
    specialisation: Literal["Mkt&Fin", "Mkt&HR"]
    mba_p: float = Field(ge=0, le=100)
    # Kumbhar features (optional — improves predictions when present)
    cgpa: Optional[float] = Field(default=None, ge=0, le=10)
    internships: Optional[int] = Field(default=None, ge=0)
    projects: Optional[int] = Field(default=None, ge=0)
    workshops: Optional[int] = Field(default=None, ge=0)
    aptitude_score: Optional[float] = Field(default=None, ge=0, le=100)
    soft_skills: Optional[float] = Field(default=None, ge=0, le=10)


class BatchRequest(BaseModel):
    items: List[PlacementInput] = Field(min_length=1, max_length=1000)


class CounterfactualRequest(PlacementInput):
    target: Optional[Literal["Placed", "Not Placed"]] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    model = require_model()
    return {
        "status": "ok",
        "model": model.model_name,
        "run_id": model.run_id,
        "salary_model": salary_predictor is not None,
        "predictions_logged": _log_count(),
    }


@app.get("/api/model/meta")
def model_meta():
    model = require_model()
    registry = experiments.load_registry()
    champion = registry.get("champion")
    meta = None
    if config.META_PATH.exists():
        try:
            meta = json.loads(config.META_PATH.read_text())
        except Exception:
            pass
    return {
        "model": model.model_name,
        "run_id": model.run_id,
        "champion": champion,
        "test_metrics": (meta or {}).get("test_metrics"),
        "cv_roc_auc_mean": (meta or {}).get("cv_roc_auc_mean"),
        "threshold": (meta or {}).get("threshold"),
        "calibration": (meta or {}).get("calibration_method"),
        "explanation_mode": (meta or {}).get("explanation_mode"),
        "tuned_params": (meta or {}).get("tuned_params"),
        "n_runs": len(registry.get("runs", [])),
        "runs": registry.get("runs", [])[-5:][::-1],
        "n_total": (meta or {}).get("n_total"),
        "active_numeric_features": (meta or {}).get("active_numeric_features"),
        "salary_model": (meta or {}).get("salary_model"),
        "learning_curve": (meta or {}).get("learning_curve"),
    }


@app.get("/api/model/comparison")
def model_comparison():
    """Return the CV comparison table from the current champion run as JSON."""
    model = require_model()
    run_root = config.RUNS_DIR / model.run_id
    csv_path = run_root / "cv_comparison.csv"
    # Fallback to top-level copy
    if not csv_path.exists():
        csv_path = config.CV_COMPARISON_PATH
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No comparison data available yet.")
    df = pd.read_csv(csv_path).reset_index()
    return {
        "run_id": model.run_id,
        "rows": df.to_dict(orient="records"),
    }


@app.post("/api/predict")
def predict(payload: PlacementInput):
    model = require_model()
    raw = payload.model_dump()
    try:
        response = model.predict(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Augment with salary estimate when student is predicted placed
    if response.get("prediction") == "Placed" and salary_predictor is not None:
        try:
            salary_info = salary_predictor.predict(raw)
            response.update(salary_info)
        except Exception as exc:
            response["salary_note"] = f"Salary estimate unavailable: {exc}"

    monitor.log_prediction(raw, response)
    return response


@app.post("/api/predict/batch")
def predict_batch(batch: BatchRequest):
    model = require_model()
    results = []
    for item in batch.items:
        payload = item.model_dump()
        proba, pred = model.probability(payload)
        item_result = {
            "prediction": pred,
            "probability": round(float(proba), 4),
            "confidence": round(max(proba, 1 - proba), 4),
        }
        # Add salary band for placed predictions
        if pred == "Placed" and salary_predictor is not None:
            try:
                salary_info = salary_predictor.predict(payload)
                item_result["salary_band"] = salary_info.get("salary_band")
                item_result["salary_estimate"] = salary_info.get("salary_estimate")
            except Exception:
                pass
        monitor.log_prediction(payload, {
            "model": model.model_name, "run_id": model.run_id,
            "prediction": pred, "probability": round(float(proba), 4),
        })
        results.append(item_result)

    placed = sum(1 for r in results if r["prediction"] == "Placed")
    return {
        "n": len(results),
        "placed": placed,
        "not_placed": len(results) - placed,
        "mean_probability": round(sum(r["probability"] for r in results) / len(results), 4),
        "results": results,
    }


@app.post("/api/explain/counterfactual")
def counterfactual(payload: CounterfactualRequest):
    model = require_model()
    raw = payload.model_dump(exclude_unset=True)
    raw.pop("target", None)
    _, pred = model.probability(raw)
    target = payload.target or ("Not Placed" if pred == "Placed" else "Placed")
    try:
        return suggest(model, raw, target=target, max_edits=6)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Counterfactual search failed: {exc}")


@app.get("/api/admin/analytics")
def analytics(window: int = Query(default=500, ge=1, le=100000)):
    """
    Aggregate analytics from the prediction log.

    Returns:
      - placement_rate: fraction of recent predictions that were "Placed"
      - daily_counts: list of {date, placed, not_placed} buckets for a bar chart
      - top_shap: list of {feature, mean_abs_shap} from global SHAP importance
      - model_versions: distribution of model versions in the log
      - n_predictions: total predictions logged
    """
    if not config.PREDICTION_LOG_PATH.exists():
        return {
            "available": False,
            "message": "No predictions logged yet.",
            "n_predictions": 0,
        }

    lines = [
        json.loads(line)
        for line in config.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return {"available": False, "message": "Log is empty.", "n_predictions": 0}

    recent = lines[-window:]
    df = pd.DataFrame(recent)

    # Placement rate
    placement_rate = (df["prediction"] == "Placed").mean()

    # Daily placement counts
    df["date"] = pd.to_datetime(df["ts"]).dt.date.astype(str)
    daily = (
        df.groupby(["date", "prediction"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    daily_counts = []
    for _, row in daily.iterrows():
        daily_counts.append({
            "date": str(row["date"]),
            "placed": int(row.get("Placed", 0)),
            "not_placed": int(row.get("Not Placed", 0)),
        })

    # Model version distribution
    model_versions = df["model"].value_counts().to_dict() if "model" in df.columns else {}

    # Global SHAP feature importance from saved file
    top_shap: list[dict] = []
    if config.SHAP_VALUES_PATH.exists() and config.BUNDLE_PATH.exists():
        try:
            shap_vals = __import__("numpy").load(config.SHAP_VALUES_PATH)
            bundle = __import__("joblib").load(config.BUNDLE_PATH)
            names = bundle.get("feature_names", [])
            mean_abs = __import__("numpy").abs(shap_vals).mean(axis=0).ravel()
            if len(names) == len(mean_abs):
                pairs = sorted(
                    zip(names, mean_abs.tolist()), key=lambda x: x[1], reverse=True
                )
                top_shap = [
                    {"feature": n.split("__")[-1].replace("_", " ").title(), "importance": round(v, 5)}
                    for n, v in pairs[:12]
                ]
        except Exception:
            pass

    return {
        "available": True,
        "n_predictions": len(lines),
        "window": len(recent),
        "placement_rate": round(float(placement_rate), 4),
        "daily_counts": daily_counts,
        "top_shap": top_shap,
        "model_versions": {str(k): int(v) for k, v in model_versions.items()},
    }


@app.get("/api/monitor/psi")
def psi_report(window: int = Query(default=config.DRIFT_WINDOW, ge=1, le=100000)):
    recent = monitor.load_recent(window)
    report = monitor.drift_report(recent)
    report["window"] = min(window, len(recent)) if len(recent) else 0
    return report