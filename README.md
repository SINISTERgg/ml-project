# PlaceIQ — Student Placement Predictor

[![CI](https://github.com/<YOUR_USERNAME>/placement-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_USERNAME>/placement-predictor/actions)

Advanced, portfolio-grade ML project that predicts campus placement probability,
estimates salary, and explains every prediction with SHAP — served through a
**Streamlit app**, a premium multi-tab web UI, and a documented REST API.

CIE Machine Learning project · 100 marks (scaled to 10) · 10 weeks.
Full write-up and marking-scheme mapping: [`reports/project_report.md`](reports/project_report.md).

---

## What makes this "advanced"

| Basic version | This project |
|---|---|
| Single model | 5+ models compared + auto-selected champion |
| No explanation | Per-prediction SHAP waterfall + top-factor bars |
| No salary | XGBRegressor salary estimate + band (low/mid/high) |
| No login | (JWT auth stub, ready to wire) |
| No storage | Predictions logged to JSONL + PSI drift monitoring |
| Manual redeploy | GitHub Actions CI + Docker Compose |
| Static model | Experiment registry, champion auto-promotion, run history |
| One dataset | Two Kaggle datasets merged with graceful fallback |

---

## Datasets

| Dataset | Rows | Key features |
|---|---|---|
| [Ben Roshan — Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) | 215 | tenth_p, twelfth_p, degree_p, mba_p, workex, salary |
| [Ruchika Kumbhar — Placement Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset) | ~10k | CGPA, Internships, Projects, Workshops, AptitudeScore, SoftSkills |

Download both:

```powershell
pip install kaggle
# place kaggle.json in ~/.kaggle/
kaggle datasets download -d benroshan/factors-affecting-campus-placement -p data/raw/ --unzip
kaggle datasets download -d ruchikakumbhar/placement-prediction-dataset    -p data/raw/ --unzip

# Rename if needed:
# data/raw/Placement_Data_Full_Class.csv → data/raw/placement_benroshan.csv
# data/raw/placement.csv                 → data/raw/placement_kumbhar.csv
```

When only the Ben Roshan file is present the pipeline falls back gracefully —
Kumbhar-only features (CGPA, Internships, etc.) are filled with column medians.

---

## Quick start

```powershell
pip install -r requirements.txt

# 0. Phase-1 EDA (7 visualisations → models/plots/eda/)
python scripts/eda.py

# 1. Train all models (Optuna tuning + SHAP + salary regression)
python scripts/train_all.py

# 2a. Streamlit app (recommended, brief-compliant deployment)
streamlit run app/streamlit_app.py

# 2b. REST API + premium UI (bonus artefact)
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8501** (Streamlit) or **http://127.0.0.1:8000** (API/UI).

### Docker Compose

```powershell
docker-compose up --build
```

The `trainer` service runs the training pipeline automatically if no model is
found in `models/`. The `backend` service starts the API on port 8000.

---

## CIE deliverable mapping (100 marks → 10)

| Deliverable | Where |
|---|---|
| ≥ 5 EDA visualisations | `scripts/eda.py` → 8 plots in `models/plots/eda/` |
| ≥ 3 ML algorithms | 5 models + voting/stacking in `src/ensemble.py` |
| Hyperparameter tuning | Optuna in `src/tuning.py`, config `configs/train.json` |
| Evaluation + interpretation | `reports/project_report.md` §8 |
| Deployment (Streamlit/Gradio/Flask) | `app/streamlit_app.py` (Streamlit) |
| Public GitHub repo | this repository + `.github/workflows/ci.yml` |

**10-week plan (17 Aug – 19 Oct 2026)** and the full report:
[`reports/project_report.md`](reports/project_report.md)

---

## Feature engineering

Five engineered features are derived from both datasets:

| Feature | Formula |
|---|---|
| `academic_avg` | Mean of all available academic percentages / CGPA×9.5 |
| `performance_gap` | Max score − academic_avg |
| `stem_flag` | 1 if Science stream or Sci&Tech degree, else 0 |
| `activity_score` | internships + projects + workshops (0 when absent) |
| `skills_composite` | (aptitude_score + soft_skills×10) / 2 |

The schema also carries a master's pathway: `has_masters` (`Yes`/`No`) and
`masters_type` (`None`/`MBA`/`MS`/`Other`). The public Kaggle files do not
record non-MBA programmes, so the training set fills these deterministically
from a fixed seed (documented as synthetic augmentation). Rows with an MBA
score are always master's holders; programme type is sampled as MBA/MS/Other.
Replace with real programme data when available.

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check; shows model name, salary model flag |
| GET | `/api/model/meta` | Champion run info, test metrics, learning curve data |
| GET | `/api/model/comparison` | CV comparison table across all candidate models |
| POST | `/api/predict` | Single prediction + SHAP + salary (if Placed) |
| POST | `/api/predict/batch` | Batch scoring up to 1,000 profiles |
| POST | `/api/explain/counterfactual` | Minimal edits to flip the outcome |
| GET | `/api/admin/analytics` | Aggregate trends from prediction log |
| GET | `/api/monitor/psi` | PSI feature drift report |

Full interactive docs at **http://localhost:8000/docs**

### Predict example

```json
POST /api/predict
{
  "gender": "M",
  "tenth_p": 67, "tenth_b": "Others",
  "twelfth_p": 91, "twelfth_b": "Others", "twelfth_s": "Commerce",
  "degree_p": 58, "degree_t": "Sci&Tech",
  "workex": "No",
  "etest_p": 55, "specialisation": "Mkt&HR", "mba_p": 58.8,
  "has_masters": "Yes", "masters_type": "MBA",
  "cgpa": 7.8, "internships": 2, "projects": 3
}
```

Master's fields: `has_masters` is `Yes`/`No`; when `Yes`, `masters_type`
is one of `MBA`, `MS` or `Other` (and `mba_p` is the master's percentage).
When `No`, `masters_type` is forced to `None` and `mba_p` may be omitted.

Optional Kumbhar fields (`cgpa`, `internships`, `projects`, `workshops`,
`aptitude_score`, `soft_skills`) improve accuracy when provided but are not
required.

### Response (Placed student)

```json
{
  "prediction": "Placed",
  "probability": 0.8412,
  "confidence": 0.8412,
  "salary_estimate": 310000,
  "salary_band": "mid",
  "salary_range": [260000, 360000],
  "contributors": [...],
  ...
}
```

---

## Frontend tabs

| Tab | Content |
|---|---|
| **Predict** | Form → animated SVG gauge → SHAP waterfall → salary badge → counterfactual |
| **Batch** | Textarea → batch table with salary band column |
| **Analytics** | KPI cards, daily placement bar chart, global SHAP importance chart |
| **Model Info** | CV comparison table, learning curve, experiment run history |

---

## Training configuration

Edit `configs/train.json`:

```json
{
  "n_trials": 25,
  "use_smote": true,
  "use_calibration": true,
  "calibration_methods": ["isotonic", "sigmoid"],
  "threshold_metric": "f1",
  "tune_models": ["logistic", "random_forest", "gradient_boosting", "xgboost", "svc"],
  "stackers": ["random_forest", "gradient_boosting", "xgboost"],
  "use_voting": true,
  "use_stacking": true
}
```

CLI overrides: `--trials 10 --no-smote --no-calibration --threshold-metric youden`

---

## Running tests

```powershell
pytest tests/ -v
```

Tests mock the trained model, so they run without a training step.

---

## Project layout

```
configs/train.json            training configuration
data/raw/                     raw Kaggle CSVs (gitignored)
data/placement_merged.csv     auto-generated merged dataset cache
src/
  config.py                   paths, constants, column mappings
  data.py                     dual-dataset loader (Ben Roshan + Kumbhar)
  features.py                 feature engineering + sklearn preprocessor
  salary.py                   XGBRegressor salary model
  train.py                    orchestrator (tune → explain → register → salary)
  ensemble.py                 candidates, stacking/voting, calibration
  eval.py                     thresholding + evaluation plots
  experiments.py              registry + champion auto-promotion
  interpret.py                SHAP (TreeExplainer / LinearExplainer)
  monitor.py                  prediction log + PSI drift
  counterfactual.py           hill-climbing minimal-edit engine
  predict.py                  inference wrapper (prob + SHAP + salary)
  settings.py                 TrainSettings dataclass
app/
  main.py                     FastAPI (predict, batch, analytics, comparison)
  static/
    index.html                multi-tab premium dark UI
    style.css                 Inter font, glassmorphism, SVG gauge
    app.js                    tabs, charts, salary badge, waterfall
models/                       runs/, bundle.joblib, salary_bundle.joblib, ...
tests/
  conftest.py                 shared fixtures (mock DataFrames + payloads)
  test_features.py            feature engineering unit tests
  test_api.py                 API endpoint tests (mocked predictor)
scripts/train_all.py          pipeline entrypoint
scripts/migrate_schema.py     rewrite CSVs to the current 10th/12th + masters schema
scripts/promote_run.py        force-promote a run after a schema change
docker-compose.yml            backend + auto-trainer services
.github/workflows/ci.yml      lint → test → docker-build CI pipeline
```

---

## Docker

```powershell
docker build -t placeiq .
docker run --rm -p 8000:8000 -v ${PWD}/models:/app/models placeiq
```

---

## Success criteria

- Classification: **ROC-AUC ≥ 0.85** on held-out test set
- Salary: **MAE < ₹40,000** on placed-students held-out set
- API: p95 latency < 200ms for `/api/predict` (model loaded once at startup)
- UI: all four tabs load and render without errors in Chromium/Firefox

---

## Future enhancements

- JWT authentication (student vs. admin roles) — PostgreSQL stub already in `docker-compose.yml`
- Resume upload + auto-fill (pyresparser)
- Model monitoring with Evidently AI
- A/B testing between challenger model versions
- React/Vite frontend rewrite for richer interactivity
- Fairness audit for gender/board bias