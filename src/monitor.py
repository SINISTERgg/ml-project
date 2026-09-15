import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

EPS = 1e-6


def _numeric_reference(series, bins=8):
    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(series.dropna(), q))
    if len(edges) < 2:
        edges = np.array([float(series.min()) - 1, float(series.max()) + 1])
    counts, _ = np.histogram(series.dropna(), bins=edges)
    return {"type": "numeric", "edges": edges.tolist(), "counts": counts.tolist()}


def _categorical_reference(series):
    counts = series.value_counts().to_dict()
    return {"type": "categorical", "counts": counts}


def build_reference(df):
    ref = {}
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        reference = _numeric_reference(df[col]) if col in config.NUMERIC_FEATURES else _categorical_reference(df[col])
        ref[col] = reference
    return ref


def save_reference(df, path=None):
    path = path or config.REFERENCE_PATH
    path.write_text(json.dumps(build_reference(df), indent=2))


def load_reference(path=None):
    path = path or config.REFERENCE_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _bucket_frequencies(values, reference):
    if reference["type"] == "numeric":
        counts, _ = np.histogram(values.dropna(), bins=np.asarray(reference["edges"], dtype=float))
        total = counts.sum()
        freq = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)
        return np.asarray(reference["counts"], dtype=float) / max(sum(reference["counts"]), 1), freq
    values = values.fillna("__missing__")
    cats = set(reference["counts"]) | set(values.unique())
    ref_total = max(sum(reference["counts"].values()), 1)
    cur_total = max(len(values), 1)
    ref_freq = np.array([reference["counts"].get(c, 0) / ref_total for c in cats])
    cur_freq = np.array([(values == c).sum() / cur_total for c in cats])
    return ref_freq, cur_freq


def psi(feature_ref, values):
    if feature_ref["type"] == "numeric":
        edges = np.asarray(feature_ref["edges"], dtype=float)
        counts, _ = np.histogram(values.dropna(), bins=edges)
        total = max(counts.sum(), 1)
        cur = counts / total
        ref = np.asarray(feature_ref["counts"], dtype=float) / max(sum(feature_ref["counts"]), 1)
    else:
        ref, cur = _bucket_frequencies(values, feature_ref)
    ref = np.maximum(ref, EPS)
    cur = np.maximum(cur, EPS)
    cur = cur / cur.sum()
    return float(((cur - ref) * np.log(cur / ref)).sum())


def drift_report(recent_df, reference=None):
    reference = reference or load_reference()
    if reference is None:
        return {"available": False, "message": "No reference distribution trained yet."}
    features = []
    for col, ref in reference.items():
        if col not in recent_df.columns:
            continue
        value = psi(ref, recent_df[col])
        if value >= config.PSI_HIGH:
            severity = "high"
        elif value >= config.PSI_MODERATE:
            severity = "moderate"
        else:
            severity = "low"
        features.append({"feature": col, "psi": round(value, 5), "severity": severity})
    features.sort(key=lambda f: f["psi"], reverse=True)
    return {
        "available": True,
        "n_samples": int(len(recent_df)),
        "features": features,
        "alerts": [f["feature"] for f in features if f["severity"] == "high"],
    }


def log_prediction(payload, response):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "model": response.get("model"),
        "run_id": response.get("run_id"),
        "prediction": response.get("prediction"),
        "probability": response.get("probability"),
    }
    with open(config.PREDICTION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def load_recent(window=config.DRIFT_WINDOW):
    if not config.PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()
    lines = [json.loads(l) for l in config.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    df = pd.DataFrame(lines)
    payloads = pd.json_normalize(list(df["payload"]))
    return payloads.tail(window)