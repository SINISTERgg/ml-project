# PlaceIQ — Demo Runbook (Phase 2 Viva)

## Pre-checks
- [ ] `python scripts/train_all.py` completed (or artifacts exist in `models/`)
- [ ] `streamlit run app/streamlit_app.py` running on http://localhost:8501
- [ ] (optional) `python -m uvicorn app.main:app --reload` on http://127.0.0.1:8000/docs
- [ ] `git log --oneline -3` shows recent commits

## 3-Minute Script

### 1) Single Prediction (40s)
1. Open **Predict** tab
2. Click **Pre-fill example** (or fill manually)
3. Click **Predict Placement**
4. Show animated gauge + **prediction note** (model/run/calibration/surrogate)
5. If Placed → point out **salary estimate + band + range**
6. Show **SHAP waterfall** and **Top Influencing Factors**

### 2) Explainability & Counterfactual (40s)
1. Scroll to **Suggest changes to flip the outcome**
2. Click it → show actionable steps (only internships/projects/workshops/etest_p/degree_p/mba_p/cgpa/soft_skills)
3. Note immutable attributes are never suggested
4. Click **Apply** on a step to auto-fill and re-predict

### 3) Batch Scoring (25s)
1. Switch to **Batch Scoring**
2. Paste CSV/JSON lines or click example
3. Show results table with prediction, probability, confidence, salary band

### 4) Model Info, EDA & Monitoring (40s)
1. **Model Info & EDA** → **Model comparison** table (ranked by AUC/Brier/F1)
2. **Evaluation metrics** + learning curve
3. (API tab/analytics) show **/api/monitor/psi** or drift line in footer
4. Show **/docs** endpoints: `/api/health`, `/api/ready`, `/api/admin/reload`

### 5) CI/GitHub (35s)
1. Show `.github/workflows/ci.yml` (lint → test → docker smoke)
2. Run `pytest tests/ -v` locally (or show GitHub Actions green)
3. Show `docker build` concept / `docker-compose up`

## Talking Points
- **Leakage fixed**: imputation inside pipeline, OOF calibration
- **Fair comparison**: repeated stratified CV with ROC-AUC/F1/Brier
- **Explainable**: positive-class SHAP fixed for multi-class
- **Ethical**: counterfactuals don't suggest changing gender/boards/scores
- **Production-minded**: health/ready probes, locked logging, PSI drift, CI
- **Metrics**: ROC-AUC 0.913, Precision 0.960, Recall 0.800, F1 0.873
