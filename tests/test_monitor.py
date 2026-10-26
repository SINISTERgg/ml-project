import json

import numpy as np
import pandas as pd

from src import config, monitor


class TestMonitor:
    def test_log_prediction_is_safe_with_lock(self, tmp_path, monkeypatch):
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(config, "PREDICTION_LOG_PATH", log_path)
        payload = {"gender": "M", "ssc_p": 70}
        response = {"model": "XGBoost", "run_id": "r1", "prediction": "Placed", "probability": 0.8}
        monitor.log_prediction(payload, response)
        monitor.log_prediction(payload, response)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "r1"

    def test_drift_report_without_reference(self, tmp_path, monkeypatch):
        ref_path = tmp_path / "reference.json"
        monkeypatch.setattr(config, "REFERENCE_PATH", ref_path)
        recent = pd.DataFrame({"ssc_p": [70, 80]})
        report = monitor.drift_report(recent)
        assert report["available"] is False

    def test_psi_in_range(self):
        ref = {"type": "numeric", "edges": [0, 50, 100], "counts": [10, 10]}
        values = pd.Series([10, 60, 90])
        psi = monitor.psi(ref, values)
        assert np.isfinite(psi)
        assert psi >= 0
