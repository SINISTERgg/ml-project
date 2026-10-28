import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import add_feature_engineering
from .interpret import compute_shap_values


def pretty_name(col):
    base = col.split("__")[-1]
    if base in config.FEATURE_LABELS:
        return config.FEATURE_LABELS[base]
    for cat, mapping in config.CATEGORY_LABELS.items():
        prefix = f"{cat}_"
        if base.startswith(prefix):
            value = base[len(prefix):]
            return mapping.get(value, base)
    return base.replace("_", " ").title()


class Predictor:
    def __init__(self, bundle_path=None):
        self.bundle = joblib.load(bundle_path or config.BUNDLE_PATH)
        self.explainer = joblib.load(config.EXPLAINER_PATH)
        info_path = config.EXPLAINER_INFO_PATH
        self.explainer_info = json.loads(info_path.read_text()) if info_path.exists() else {}
        # Only core Ben Roshan fields are strictly required.  Masters fields
        # are normalised with sensible defaults for backwards compatibility.
        self.required = [
            c for c in config.BENROSHAN_NUMERIC if c != "mba_p"
        ] + [
            c for c in config.CATEGORICAL_FEATURES
            if c not in ("has_masters", "masters_type")
        ]
        self.optional = config.KUMBHAR_NUMERIC

    @property
    def run_id(self):
        return self.bundle.get("run_id")

    @property
    def model_name(self):
        return self.bundle.get("model_name", "Unknown")

    def _validate(self, raw):
        missing = [c for c in self.required if c not in raw or raw.get(c) in (None, "")]
        if missing:
            raise ValueError(f"Missing input fields: {', '.join(missing)}")

    @staticmethod
    def _normalise_masters(raw):
        """Return (has_masters, masters_type, mba_p) consistent values."""
        value = str(raw.get("has_masters", "No")).strip().lower()
        has_masters = "Yes" if value in {"yes", "y", "1", "true"} else "No"
        mba_p = raw.get("mba_p")
        if mba_p in (None, ""):
            mba_p = 0.0
        if has_masters == "No":
            return has_masters, "None", float(mba_p)
        mtype = str(raw.get("masters_type", "")).strip().title()
        mtype = {"Mba": "MBA", "Ms": "MS", "Other": "Other"}.get(mtype, "MBA")
        return has_masters, mtype, float(mba_p)

    def _table(self, raw):
        self._validate(raw)
        fields = {k: raw.get(k) for k in self.required}
        has_masters, masters_type, mba_p = self._normalise_masters(raw)
        fields["mba_p"] = mba_p
        fields["has_masters"] = has_masters
        fields["masters_type"] = masters_type
        for k in self.optional:
            if k in raw:
                fields[k] = raw[k]
        return add_feature_engineering(pd.DataFrame([fields]))

    def _proba_array(self, raw):
        Xt = self.bundle["preprocessor"].transform(self._table(raw))
        calibrator = self.bundle.get("calibrator")
        if calibrator is not None:
            return calibrator.predict_proba(Xt)[0]
        return self.bundle["model"].predict_proba(Xt)[0]

    def probability(self, raw):
        proba = self._proba_array(raw)
        p1 = float(proba[1])
        threshold = float(self.bundle["threshold"])
        pred = config.INV_LABEL_MAP[int(p1 >= threshold)]
        return p1, pred

    def predict(self, raw, include_explanation=True):
        proba = self._proba_array(raw)
        p1 = float(proba[1])
        threshold = float(self.bundle["threshold"])
        pred_label = config.INV_LABEL_MAP[int(p1 >= threshold)]

        response = {
            "model": self.model_name,
            "run_id": self.run_id,
            "trained_at": self.bundle.get("trained_at"),
            "calibration": self.bundle.get("calibration_method", "none"),
            "prediction": pred_label,
            "probability": round(p1, 4),
            "confidence": round(max(p1, 1 - p1), 4),
            "threshold": threshold,
        }

        if not include_explanation:
            return response

        Xt = self.bundle["preprocessor"].transform(self._table(raw))
        surrogate = self.bundle.get("surrogate")
        response["explanation_mode"] = "surrogate" if surrogate is not None else "exact"
        if surrogate is not None:
            response["explanation_model"] = self.bundle.get("surrogate_name")

        shap_values = compute_shap_values(self.explainer, Xt)
        shap_arr = np.asarray(shap_values)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_arr = np.asarray(shap_values[1])
        elif shap_arr.ndim == 3 and shap_arr.shape[-1] >= 2:
            shap_arr = shap_arr[:, :, 1]
        flat = np.asarray(shap_arr).reshape(-1)
        contributors = [
            {"feature": pretty_name(n), "value": float(v), "shap": float(s)}
            for n, v, s in zip(self.bundle["feature_names"], Xt[0], flat)
        ]
        contributors.sort(key=lambda c: abs(c["shap"]), reverse=True)
        response["contributors"] = contributors[:12]

        base_prob = self.explainer_info.get("base_probability")
        if base_prob is None:
            base_prob = self.explainer_info.get("expected_value")
        if base_prob is not None:
            base_prob = float(base_prob)
            if not 0.0 <= base_prob <= 1.0:
                base_prob = float(1.0 / (1.0 + np.exp(-base_prob)))
            response["base_probability"] = round(base_prob, 4)
        return response
