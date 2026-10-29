import json

import numpy as np
import pandas as pd

from src import config, monitor


class TestMonitor:
    def test_log_prediction_is_safe_with_lock(self, tmp_path, monkeypatch):
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(config, "PREDICTION_LOG_PATH", log_path)
        payload = {"gender": "M", "tenth_p": 70}
        response = {"model": "XGBoost", "run_id": "r1", "prediction": "Placed", "probability": 0.8}
        monitor.log_prediction(payload, response)
        monitor.log_prediction(payload, response)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "r1"

    def test_drift_report_without_reference(self, tmp_path, monkeypatch):
        ref_path = tmp_path / "reference.json"
        monkeypatch.setattr(config, "REFERENCE_PATH", ref_path)
        recent = pd.DataFrame({"tenth_p": [70, 80]})
        report = monitor.drift_report(recent)
        assert report["available"] is False

    def test_psi_in_range(self):
        ref = {"type": "numeric", "edges": [0, 50, 100], "counts": [10, 10]}
        values = pd.Series([10, 60, 90])
        psi = monitor.psi(ref, values)
        assert np.isfinite(psi)
        assert psi >= 0

    def test_drift_report_waits_for_minimum_samples(self):
        ref = {"tenth_p": {"type": "numeric", "edges": [0, 50, 100], "counts": [10, 10]}}
        recent = pd.DataFrame({"tenth_p": [10, 60, 90]})
        report = monitor.drift_report(recent, reference=ref)
        assert report["status"] == "collecting"
        assert report["alerts"] == []
        assert report["features"] == []
        assert report["n_samples"] == 3

    def test_drift_report_with_no_predictions(self):
        ref = {"tenth_p": {"type": "numeric", "edges": [0, 50, 100], "counts": [10, 10]}}
        report = monitor.drift_report(pd.DataFrame(), reference=ref)
        assert report["status"] == "no_data"
        assert report["alerts"] == []
