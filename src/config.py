from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
RUNS_DIR = MODELS_DIR / "runs"
LOGS_DIR = MODELS_DIR / "logs"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Raw dataset file paths ────────────────────────────────────────────────────
# Ben Roshan dataset (215 rows) — ssc_p … mba_p, status, salary
BENROSHAN_PATH = DATA_RAW_DIR / "placement_benroshan.csv"
# Ruchika Kumbhar dataset (~10k rows) — CGPA, Internships, Projects, … PlacementStatus
KUMBHAR_PATH = DATA_RAW_DIR / "placement_kumbhar.csv"
# Legacy / fallback path (the file already in data/)
LEGACY_DATA_PATH = DATA_DIR / "placement_data.csv"

# ── Derived / model artifact paths ───────────────────────────────────────────
DATA_PATH = DATA_DIR / "placement_merged.csv"   # cached merged set written by data.py
BUNDLE_PATH = MODELS_DIR / "bundle.joblib"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
META_PATH = MODELS_DIR / "meta.json"
CV_COMPARISON_PATH = MODELS_DIR / "cv_comparison.csv"
EXPLAINER_INFO_PATH = MODELS_DIR / "champion_explainer_info.json"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.npy"
REGISTRY_PATH = MODELS_DIR / "registry.json"
CHAMPION_PATH = MODELS_DIR / "champion.json"
REFERENCE_PATH = MODELS_DIR / "reference_features.json"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.jsonl"
TRAIN_CONFIG_PATH = CONFIGS_DIR / "train.json"

# ── Salary model artifacts ────────────────────────────────────────────────────
SALARY_BUNDLE_PATH = MODELS_DIR / "salary_bundle.joblib"
SALARY_META_PATH = MODELS_DIR / "salary_meta.json"

# ── Ensure directories exist at import time ───────────────────────────────────
for _d in (MODELS_DIR, PLOTS_DIR, RUNS_DIR, LOGS_DIR, DATA_RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Training hyper-parameters ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
N_TRIALS = 20
BATCH_LIMIT = 1000
DRIFT_WINDOW = 200
PSI_MODERATE = 0.1
PSI_HIGH = 0.25

# ── Target column + label encoding ────────────────────────────────────────────
TARGET = "status"
SALARY_TARGET = "salary"
ID_FEATURES: list[str] = ["sl_no"]
DROP_FEATURES: list[str] = ["salary"]          # dropped before classification training
LABEL_MAP = {"Not Placed": 0, "Placed": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── Ben Roshan feature schema (original columns) ──────────────────────────────
BENROSHAN_NUMERIC = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
BENROSHAN_CATEGORICAL = [
    "gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"
]

# ── Kumbhar-only columns (optional; present only when that dataset is loaded) ──
KUMBHAR_NUMERIC = ["cgpa", "internships", "projects", "workshops", "aptitude_score", "soft_skills"]

# ── Unified feature schema (superset; preprocessor handles missingness) ────────
NUMERIC_FEATURES = BENROSHAN_NUMERIC + KUMBHAR_NUMERIC
CATEGORICAL_FEATURES = BENROSHAN_CATEGORICAL
ENGINEERED_FEATURES = [
    "academic_avg",      # mean of all available percentage/CGPA columns
    "performance_gap",   # max% − academic_avg
    "stem_flag",         # Science/Sci&Tech background
    "activity_score",    # internships + projects + workshops (0 when absent)
    "skills_composite",  # (aptitude_score + soft_skills * 10) / 2 normalised to 0-100
]
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES

# ── Kumbhar → unified column name mapping ────────────────────────────────────
# Maps raw Kumbhar CSV column names to the shared schema names.
KUMBHAR_COL_MAP = {
    # Academic
    "CGPA": "cgpa",
    "Cgpa": "cgpa",
    "cgpa": "cgpa",
    "SSC_Marks": "ssc_p",
    "ssc_marks": "ssc_p",
    "HSC_Marks": "hsc_p",
    "hsc_marks": "hsc_p",
    # Skills / activity
    "Internships": "internships",
    "internships": "internships",
    "Projects": "projects",
    "projects": "projects",
    "Workshops": "workshops",
    "workshops": "workshops",
    "AptitudeTestScore": "aptitude_score",
    "aptitudeTestScore": "aptitude_score",
    "SoftSkillsRating": "soft_skills",
    "softSkillsRating": "soft_skills",
    # Categorical equivalents
    "ExtracurricularActivities": "workex",
    "extracurricularActivities": "workex",
    "PlacementTraining": "placement_training",
    # Target
    "PlacementStatus": "status",
    "placementStatus": "status",
    "Placed": "status",          # possible raw value — handled in normaliser
}

# ── Model registry names ──────────────────────────────────────────────────────
MODEL_NAMES = {
    "logistic": "LogisticRegression",
    "random_forest": "RandomForest",
    "gradient_boosting": "GradientBoosting",
    "xgboost": "XGBoost",
    "svc": "SVC",
    "voting": "VotingEnsemble",
    "stacking": "StackingEnsemble",
}

# ── Human-readable feature labels (for SHAP display) ─────────────────────────
FEATURE_LABELS = {
    "ssc_p": "SSC Percentage",
    "hsc_p": "HSC Percentage",
    "degree_p": "Degree Percentage",
    "etest_p": "Employability Test %",
    "mba_p": "MBA Percentage",
    "cgpa": "CGPA",
    "internships": "Internships",
    "projects": "Projects",
    "workshops": "Workshops / Certs",
    "aptitude_score": "Aptitude Score",
    "soft_skills": "Soft Skills Rating",
    "academic_avg": "Academic Average",
    "performance_gap": "Performance Gap",
    "stem_flag": "STEM Background",
    "activity_score": "Activity Score",
    "skills_composite": "Skills Composite",
}

CATEGORY_LABELS = {
    "gender": {"F": "Female", "M": "Male"},
    "ssc_b": {"Central": "SSC: Central Board", "Others": "SSC: Other Board"},
    "hsc_b": {"Central": "HSC: Central Board", "Others": "HSC: Other Board"},
    "hsc_s": {"Commerce": "HSC: Commerce", "Science": "HSC: Science", "Arts": "HSC: Arts"},
    "degree_t": {
        "Sci&Tech": "Degree: Sci & Tech",
        "Comm&Mgmt": "Degree: Comm & Mgmt",
        "Others": "Degree: Other",
    },
    "workex": {"Yes": "Work Experience", "No": "No Work Experience"},
    "specialisation": {"Mkt&Fin": "MBA: Mkt & Fin", "Mkt&HR": "MBA: Mkt & HR"},
}

# ── Salary band thresholds (annual CTC in INR lakhs) ─────────────────────────
SALARY_BAND_LOW = 250_000     # < 2.5L
SALARY_BAND_HIGH = 400_000    # > 4L
