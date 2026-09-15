"""
test_api.py — FastAPI endpoint tests.

The model is mocked so tests run without a trained bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers to build a minimal mock predictor ────────────────────────────────

def _make_mock_predictor(prediction="Placed", probability=0.82):
    mock = MagicMock()
    mock.model_name = "XGBoost"
    mock.run_id = "test-run-001"
    mock.predict.return_value = {
        "model": "XGBoost",
        "run_id": "test-run-001",
        "trained_at": "2026-01-01T00:00:00Z",
        "calibration": "isotonic",
        "prediction": prediction,
        "probability": probability,
        "confidence": max(probability, 1 - probability),
        "threshold": 0.5,
        "explanation_mode": "exact",
        "contributors": [
            {"feature": "Degree Percentage", "value": 58.0, "shap": 0.312},
            {"feature": "MBA Percentage",    "value": 58.8, "shap": 0.211},
            {"feature": "Employability Test %", "value": 55.0, "shap": -0.095},
        ],
        "base_probability": 0.67,
    }
    mock.probability.return_value = (probability, prediction)
    return mock


def _make_mock_salary_predictor():
    mock = MagicMock()
    mock.predict.return_value = {
        "salary_estimate": 310000,
        "salary_band": "mid",
        "salary_range": [260000, 360000],
    }
    return mock


# ── App fixture (patched globals) ─────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    """TestClient with mocked predictor and salary_predictor."""
    import src.config as cfg

    # Point prediction log to a temp path so tests don't write real files
    original_log = cfg.PREDICTION_LOG_PATH
    cfg.PREDICTION_LOG_PATH = tmp_path / "predictions.jsonl"

    mock_pred = _make_mock_predictor()
    mock_salary = _make_mock_salary_predictor()

    with (
        patch("app.main.predictor", mock_pred),
        patch("app.main.salary_predictor", mock_salary),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c

    cfg.PREDICTION_LOG_PATH = original_log


@pytest.fixture
def client_no_salary(tmp_path):
    """TestClient without salary predictor (tests graceful absence)."""
    import src.config as cfg
    original_log = cfg.PREDICTION_LOG_PATH
    cfg.PREDICTION_LOG_PATH = tmp_path / "predictions.jsonl"

    mock_pred = _make_mock_predictor()

    with (
        patch("app.main.predictor", mock_pred),
        patch("app.main.salary_predictor", None),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c

    cfg.PREDICTION_LOG_PATH = original_log


# ── /api/health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "XGBoost"

    def test_health_includes_salary_flag(self, client):
        resp = client.get("/api/health")
        assert "salary_model" in resp.json()
        assert resp.json()["salary_model"] is True

    def test_health_salary_false_when_absent(self, client_no_salary):
        resp = client_no_salary.get("/api/health")
        assert resp.json()["salary_model"] is False


# ── /api/predict ─────────────────────────────────────────────────────────────

VALID_PAYLOAD: dict[str, Any] = {
    "gender": "M",
    "ssc_p": 67.0, "ssc_b": "Others",
    "hsc_p": 91.0, "hsc_b": "Others", "hsc_s": "Commerce",
    "degree_p": 58.0, "degree_t": "Sci&Tech",
    "workex": "No",
    "etest_p": 55.0,
    "specialisation": "Mkt&HR",
    "mba_p": 58.8,
}


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/api/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_has_required_keys(self, client):
        body = client.post("/api/predict", json=VALID_PAYLOAD).json()
        for key in ("prediction", "probability", "confidence", "threshold", "contributors"):
            assert key in body, f"Missing key: {key}"

    def test_predict_placed_includes_salary(self, client):
        """When prediction is Placed and salary model exists, salary fields appear."""
        body = client.post("/api/predict", json=VALID_PAYLOAD).json()
        assert body["prediction"] == "Placed"
        assert "salary_estimate" in body
        assert "salary_band" in body
        assert "salary_range" in body

    def test_predict_no_salary_when_predictor_absent(self, client_no_salary):
        body = client_no_salary.post("/api/predict", json=VALID_PAYLOAD).json()
        assert "salary_estimate" not in body

    def test_predict_missing_required_field_422(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "mba_p"}
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_out_of_range_422(self, client):
        bad = {**VALID_PAYLOAD, "ssc_p": 150}  # > 100
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_literal_422(self, client):
        bad = {**VALID_PAYLOAD, "gender": "X"}
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_with_optional_kumbhar_fields(self, client):
        payload = {**VALID_PAYLOAD, "cgpa": 7.8, "internships": 2, "projects": 3}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_logs_to_jsonl(self, client, tmp_path):
        """Prediction should append one entry to the JSONL log."""
        import src.config as cfg
        client.post("/api/predict", json=VALID_PAYLOAD)
        assert cfg.PREDICTION_LOG_PATH.exists()
        lines = cfg.PREDICTION_LOG_PATH.read_text().strip().splitlines()
        assert len(lines) >= 1
        row = json.loads(lines[-1])
        assert "ts" in row and "prediction" in row


# ── /api/predict/batch ────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_returns_200(self, client):
        resp = client.post("/api/predict/batch", json={"items": [VALID_PAYLOAD]})
        assert resp.status_code == 200

    def test_batch_summary_keys(self, client):
        body = client.post("/api/predict/batch", json={"items": [VALID_PAYLOAD, VALID_PAYLOAD]}).json()
        assert body["n"] == 2
        assert "placed" in body
        assert "not_placed" in body
        assert "mean_probability" in body
        assert "results" in body

    def test_batch_empty_items_422(self, client):
        resp = client.post("/api/predict/batch", json={"items": []})
        assert resp.status_code == 422

    def test_batch_result_has_salary_band_for_placed(self, client):
        body = client.post("/api/predict/batch", json={"items": [VALID_PAYLOAD]}).json()
        # Mock returns "Placed" — check that salary_band is present
        for r in body["results"]:
            if r["prediction"] == "Placed":
                assert "salary_band" in r


# ── /api/model/meta ───────────────────────────────────────────────────────────

class TestModelMeta:
    def test_meta_returns_model_name(self, client):
        # meta.json not present in test — endpoint should still return basic fields
        resp = client.get("/api/model/meta")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "XGBoost"

    def test_meta_includes_run_id(self, client):
        body = client.get("/api/model/meta").json()
        assert body["run_id"] == "test-run-001"


# ── /api/admin/analytics ──────────────────────────────────────────────────────

class TestAnalytics:
    def test_analytics_empty_log(self, client):
        """Analytics with no predictions logged returns not-available gracefully."""
        resp = client.get("/api/admin/analytics")
        assert resp.status_code == 200
        body = resp.json()
        # Either no data or has the expected structure
        assert "n_predictions" in body

    def test_analytics_after_prediction(self, client):
        """After one prediction, analytics should have n_predictions >= 1."""
        client.post("/api/predict", json=VALID_PAYLOAD)
        body = client.get("/api/admin/analytics").json()
        assert body["n_predictions"] >= 1

    def test_analytics_placement_rate_range(self, client):
        client.post("/api/predict", json=VALID_PAYLOAD)
        body = client.get("/api/admin/analytics").json()
        if body.get("available"):
            assert 0.0 <= body["placement_rate"] <= 1.0


# ── /api/monitor/psi ──────────────────────────────────────────────────────────

class TestPSI:
    def test_psi_returns_200(self, client):
        resp = client.get("/api/monitor/psi")
        assert resp.status_code == 200

    def test_psi_window_param(self, client):
        resp = client.get("/api/monitor/psi?window=50")
        assert resp.status_code == 200

    def test_psi_invalid_window_422(self, client):
        resp = client.get("/api/monitor/psi?window=0")
        assert resp.status_code == 422
