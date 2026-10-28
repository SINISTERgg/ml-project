import json
import sys

from src import config, train
from src.settings import TrainSettings


class TestPipelineE2E:
    def test_quick_train_predict_explain_cycle(self, tmp_path, monkeypatch):
        # Use the bundled sample data but write outputs to temp dirs
        runs_dir = tmp_path / "runs"
        models_dir = tmp_path / "models"
        plots_dir = models_dir / "plots"
        for d in (runs_dir, models_dir, plots_dir, tmp_path / "logs"):
            d.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(config, "RUNS_DIR", runs_dir)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "PLOTS_DIR", plots_dir)
        monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr(config, "BUNDLE_PATH", models_dir / "bundle.joblib")
        monkeypatch.setattr(config, "EXPLAINER_PATH", models_dir / "shap_explainer.joblib")
        monkeypatch.setattr(config, "META_PATH", models_dir / "meta.json")
        monkeypatch.setattr(config, "CV_COMPARISON_PATH", models_dir / "cv_comparison.csv")
        monkeypatch.setattr(config, "EXPLAINER_INFO_PATH", models_dir / "champion_explainer_info.json")
        monkeypatch.setattr(config, "SHAP_VALUES_PATH", models_dir / "shap_values.npy")
        monkeypatch.setattr(config, "REGISTRY_PATH", models_dir / "registry.json")
        monkeypatch.setattr(config, "CHAMPION_PATH", models_dir / "champion.json")
        monkeypatch.setattr(config, "REFERENCE_PATH", models_dir / "reference_features.json")
        monkeypatch.setattr(config, "SALARY_BUNDLE_PATH", models_dir / "salary_bundle.joblib")
        monkeypatch.setattr(config, "SALARY_META_PATH", models_dir / "salary_meta.json")
        monkeypatch.setattr(config, "PREDICTION_LOG_PATH", models_dir / "logs" / "predictions.jsonl")

        # Fast config
        cfg_path = tmp_path / "train.json"
        settings = TrainSettings(
            test_size=0.25,
            cv_splits=3,
            n_repeats=1,
            n_trials=2,
            use_smote=True,
            use_calibration=True,
            calibration_methods=["isotonic"],
            threshold_metric="f1",
            tune_models=["logistic"],
            stackers=["logistic"],
            use_voting=False,
            use_stacking=False,
        )
        cfg_path.write_text(json.dumps(settings.to_dict()))
        monkeypatch.setattr(config, "TRAIN_CONFIG_PATH", cfg_path)
        monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path)])

        meta = train.main()
        assert meta is not None
        assert (models_dir / "bundle.joblib").exists()
        assert (models_dir / "meta.json").exists()
        assert (models_dir / "plots" / "roc_curve.png").exists()
        assert (models_dir / "plots" / "confusion_matrix.png").exists()
        assert (models_dir / "plots" / "confusion_matrix_normalized.png").exists()
        assert (models_dir / "plots" / "importance_comparison.png").exists()

        from src.predict import Predictor

        pred = Predictor(bundle_path=models_dir / "bundle.joblib")
        raw = {
            "gender": "M",
            "tenth_p": 67.0,
            "tenth_b": "Others",
            "twelfth_p": 91.0,
            "twelfth_b": "Others",
            "twelfth_s": "Commerce",
            "degree_p": 58.0,
            "degree_t": "Sci&Tech",
            "workex": "No",
            "etest_p": 55.0,
            "specialisation": "Mkt&HR",
            "mba_p": 58.8,
            "has_masters": "Yes",
            "masters_type": "MBA",
        }
        out = pred.predict(raw)
        assert out["prediction"] in {"Placed", "Not Placed"}
        assert "contributors" in out
        assert len(out["contributors"]) >= 1
        # Positive-class SHAP should not be all zeros and should be finite
        vals = [abs(c["shap"]) for c in out["contributors"]]
        assert any(v > 0 for v in vals)
        assert all((c["shap"] == c["shap"]) for c in out["contributors"])
