import numpy as np
import pandas as pd

from src import config
from src.salary import _band, train_salary_model


class TestSalaryBand:
    def test_low_band(self):
        assert _band(config.SALARY_BAND_LOW - 1000) == "low"

    def test_high_band(self):
        assert _band(config.SALARY_BAND_HIGH + 1000) == "high"

    def test_mid_band(self):
        assert _band((config.SALARY_BAND_LOW + config.SALARY_BAND_HIGH) / 2) == "mid"


class TestSalaryTraining:
    def test_train_salary_model_returns_none_when_insufficient_data(self, tmp_path, monkeypatch):
        # Force paths to temp dir so we don't overwrite real artifacts
        monkeypatch.setattr(config, "SALARY_BUNDLE_PATH", tmp_path / "salary_bundle.joblib")
        monkeypatch.setattr(config, "SALARY_META_PATH", tmp_path / "salary_meta.json")

        df = pd.DataFrame(
            {
                config.TARGET: ["Placed"] * 10 + ["Not Placed"] * 20,
                config.SALARY_TARGET: [300000.0] * 10 + [np.nan] * 20,
                "gender": ["M"] * 30,
                "tenth_p": [70.0] * 30,
                "tenth_b": ["Central"] * 30,
                "twelfth_p": [70.0] * 30,
                "twelfth_b": ["Central"] * 30,
                "twelfth_s": ["Science"] * 30,
                "degree_p": [70.0] * 30,
                "degree_t": ["Sci&Tech"] * 30,
                "workex": ["Yes"] * 30,
                "etest_p": [70.0] * 30,
                "specialisation": ["Mkt&Fin"] * 30,
                "mba_p": [70.0] * 30,
            }
        )
        result = train_salary_model(df)
        assert result is None
        assert not (tmp_path / "salary_bundle.joblib").exists()
