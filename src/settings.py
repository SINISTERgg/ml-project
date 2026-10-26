import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config


@dataclass
class TrainSettings:
    test_size: float = 0.2
    cv_splits: int = 5
    n_repeats: int = 3
    n_trials: int = 20
    use_smote: bool = True
    use_calibration: bool = True
    calibration_methods: list = field(default_factory=lambda: ["isotonic", "sigmoid"])
    threshold_metric: str = "f1"
    tune_models: list = field(
        default_factory=lambda: [
            "logistic", "random_forest", "gradient_boosting", "xgboost", "svc"
        ]
    )
    stackers: list = field(
        default_factory=lambda: ["random_forest", "gradient_boosting", "xgboost"]
    )
    use_voting: bool = True
    use_stacking: bool = True

    def validate(self):
        numeric = ("test_size", "cv_splits", "n_repeats", "n_trials")
        bad = [n for n in numeric if getattr(self, n) <= 0]
        if bad:
            raise ValueError(f"Settings must be positive: {bad}")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if self.threshold_metric not in ("f1", "youden"):
            raise ValueError("threshold_metric must be 'f1' or 'youden'")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None):
        path = Path(path or config.TRAIN_CONFIG_PATH)
        if path.exists():
            data = json.loads(path.read_text())
            known = {f for f in cls.__dataclass_fields__}
            settings = cls(**{k: v for k, v in data.items() if k in known})
        else:
            settings = cls()
        settings.validate()
        return settings


# ── Default category options (used by the counterfactual engine) ──────────────
DEFAULT_CATEGORY_OPTIONS = {
    "gender":         ["M", "F"],
    "ssc_b":          ["Central", "Others"],
    "hsc_b":          ["Central", "Others"],
    "hsc_s":          ["Commerce", "Science", "Arts"],
    "degree_t":       ["Sci&Tech", "Comm&Mgmt", "Others"],
    "workex":         ["Yes", "No"],
    "specialisation": ["Mkt&Fin", "Mkt&HR"],
}
