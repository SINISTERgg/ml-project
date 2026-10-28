# ruff: noqa: E702
from pptx import Presentation

prs = Presentation()
title_layout = prs.slide_layouts[0]
content_layout = prs.slide_layouts[1]

def add_title(slide, txt):
    if slide.shapes.title:
        slide.shapes.title.text = txt

# 1 Cover
s = prs.slides.add_slide(title_layout)
s.shapes.title.text = 'PlaceIQ — Phase 2 Review'
s.placeholders[1].text = 'CIE ML Project | Weeks 6–10 | 2026\nModel Comparison • Tuning • Evaluation • Deployment • CI/CD'

# 2 Agenda
s = prs.slides.add_slide(content_layout)
add_title(s, 'Agenda')
tf = s.placeholders[1].text_frame
tf.clear()
for t in ['Week 6: Model Comparison & Experimentation', 'Week 7: Tuning, Calibration & Selection',
          'Week 8: Evaluation, SHAP & Counterfactuals', 'Week 9: Deployment, CI/CD & Monitoring',
          'Week 10: Testing, Report & Viva', 'Results, Demo & Q&A']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 3 Week 6
s = prs.slides.add_slide(content_layout)
add_title(s, 'Week 6 — Leakage Fix, SMOTENC & Benchmarking')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['Removed pre-split imputation leakage (fit in pipeline)',
          'Categorical-safe resampling (SMOTENC) with fallback',
          'Benchmarked LR, RF, GB, XGBoost, SVC + Voting/Stacking',
          'Evaluation: ROC-AUC, F1, Brier | RepeatedStratifiedKFold',
          'Best CV: SVC 0.948 (stable)']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 4 Week 7
s = prs.slides.add_slide(content_layout)
add_title(s, 'Week 7 — Optuna Tuning, OOF Calibration & Selection')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['Expanded search spaces + MedianPruner, track param importances',
          'OOF calibration via cross_val_predict (fixes leakage)',
          'Champion selection: CV AUC → Brier → F1',
          'Salary regressor hardened (log-target + P80 residual margin)',
          'Threshold tuned to maximise F1 (0.553)']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 5 Week 8
s = prs.slides.add_slide(content_layout)
add_title(s, 'Week 8 — SHAP, Counterfactuals & Error Analysis')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['Fixed SHAP to positive-class slicing for multi-class/tree outputs',
          'Ethical counterfactual engine (immutable vs actionable)',
          'Diagnostics: ROC/PR, calibration, confusion (raw+normalised), learning curve',
          'Permutation vs SHAP comparison plot',
          'Actionable insights surfaced in UI']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 6 Week 9
s = prs.slides.add_slide(content_layout)
add_title(s, 'Week 9 — Deployment, CI/CD & Drift')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['/api/health, /api/ready, /api/admin/reload (decoupled probes)',
          'Streamlit refresh button + sidebar metadata',
          'Static UI: CSV drag-drop, toasts, theme persistence',
          'Locked prediction logging + PSI drift monitoring',
          'GitHub Actions + Docker Compose hardening']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 7 Week 10
s = prs.slides.add_slide(content_layout)
add_title(s, 'Week 10 — Testing, Report & Demo Prep')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['New tests: salary, counterfactual, monitor, end-to-end pipeline',
          '57 tests passing | lint clean',
          'Report updated with Phase 1+2 mapping',
          'Demo runbook ready',
          'Final integration verified']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 8 Results
s = prs.slides.add_slide(content_layout)
add_title(s, 'Final Results (Held-out Test)')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['Champion: SVC (surrogate XGBoost for SHAP)',
          'ROC-AUC: 0.926 | PR-AUC: 0.968 | F1: 0.830',
          'Precision: 0.957 | Recall: 0.733 | Brier: 0.116',
          'Threshold: 0.745 (F1-optimised)',
          'SHAP + Counterfactuals + Salary estimation']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 9 Demo
s = prs.slides.add_slide(content_layout)
add_title(s, 'Live Demo Plan')
tf = s.placeholders[1].text_frame; tf.clear()
for t in ['1) Single prediction with gauge + SHAP waterfall',
          '2) Counterfactual: flip outcome with actionable steps',
          '3) Batch scoring (CSV)',
          '4) Model comparison, learning curve & drift monitor',
          '5) GitHub CI green + API docs']:
    p = tf.add_paragraph(); p.text = t; p.level = 0

# 10 Qs
s = prs.slides.add_slide(content_layout)
add_title(s, 'Q&A')
tf = s.placeholders[1].text_frame; tf.clear()
p = tf.add_paragraph(); p.text = 'Thank You'; p.level = 0

prs.save('PlaceIQ_Phase2_Review.pptx')
print('done')
