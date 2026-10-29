# -*- coding: utf-8 -*-
"""
Front matter and preliminaries for 25-30 page Academic Report.
"""

def get_front_matter():
    return r"""# A MINOR PROJECT REPORT ON

# PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning

---

### Submitted in partial fulfillment of the requirements for the award of the degree of
### BACHELOR OF TECHNOLOGY / BACHELOR OF SCIENCE
### in
### COMPUTER SCIENCE AND ENGINEERING / ARTIFICIAL INTELLIGENCE & MACHINE LEARNING

---

<div class="cover-meta-grid">
  <div class="cover-meta-block">
    <strong>Submitted by:</strong><br>
    <strong>Student Name:</strong> M S<br>
    <strong>Register / Roll No:</strong> 2023-CS-ML-042<br>
    <strong>Semester & Batch:</strong> VII Semester, 2026–2027
  </div>
  <div class="cover-meta-block" style="text-align: right;">
    <strong>Under the Guidance of:</strong><br>
    <strong>Project Guide:</strong> Dr. Academic Guide, Ph.D.<br>
    <strong>Designation:</strong> Associate Professor<br>
    <strong>Department:</strong> Computer Science and Engineering
  </div>
</div>

<div class="cover-dept-footer">
  <strong>DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING</strong><br>
  COLLEGE OF ENGINEERING & TECHNOLOGY<br>
  ACADEMIC YEAR: 2026 – 2027
</div>

<div class="page-break"></div>

## CERTIFICATE OF APPROVAL

This is to certify that the minor project report entitled **"PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning"**, submitted by **M S** (Register No: **2023-CS-ML-042**) in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**, is a bona fide record of the work carried out under my supervision and guidance during the academic semester (August 2026 – October 2026).

The results embodied in this report have not been submitted to any other University or Institute for the award of any degree, diploma, or certificate.

<br><br><br>
<div style="display: flex; justify-content: space-between; text-align: center; font-family: var(--font-sans); font-size: 10pt;">
  <div style="flex: 1;">
    _______________________________<br>
    <strong>Dr. Academic Guide, Ph.D.</strong><br>
    Project Supervisor / Guide<br>
    Dept. of Computer Science & Engg.
  </div>
  <div style="flex: 1;">
    _______________________________<br>
    <strong>Head of Department</strong><br>
    Dept. of Computer Science & Engg.<br>
    College of Engineering & Technology
  </div>
</div>

<br><br>
<div style="text-align: center; font-family: var(--font-sans); font-size: 10pt;">
  _______________________________<br>
  <strong>External Examiner</strong><br>
  Date of Examination: 28-October-2026
</div>

<div class="page-break"></div>

## DECLARATION & ACKNOWLEDGEMENTS

### Student Declaration
I hereby declare that this minor project report entitled **"PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning"** is an authentic record of the original research and development carried out by me under the supervision of my faculty guide. All cited ideas, algorithms, and benchmark datasets have been attributed in accordance with IEEE referencing standards. This work has not formed the basis for the award of any other academic degree or qualification.

<br>
<div style="text-align: right; font-family: var(--font-sans); font-size: 10pt;">
  <strong>M S</strong> (Reg. No: 2023-CS-ML-042)<br>
  Department of Computer Science and Engineering<br>
  Date: 28-October-2026
</div>

---

### Acknowledgements
I express my profound gratitude to my project supervisor for their intellectual mentorship, constructive criticism, and encouragement throughout this project. I am grateful to the **Head of the Department of Computer Science and Engineering** and the faculty evaluation committee for their academic resources, computing infrastructure, and critical feedback during the Continuous Internal Evaluation (CIE) Phase 1 and Phase 2 reviews. Finally, I extend my heartfelt appreciation to my family and peers for their continuous moral support.

---

## ABSTRACT

Campus recruitment drives represent the defining transition for technical graduates entering the professional knowledge economy. However, students frequently recognize their academic vulnerabilities and employability gaps too late to enact remedial interventions. This project presents **PlaceIQ**, an end-to-end, interpretable machine learning system engineered to predict campus placement probabilities, estimate post-recruitment salary packages, and generate transparent, actionable feature attributions for academic counseling.

Formulated as a supervised binary classification problem with tabular academic, demographic, and experiential attributes, PlaceIQ benchmarks five distinct classification algorithms: L2-Regularized Logistic Regression, Support Vector Classifier with Radial Basis Function kernel (RBF-SVC), Random Forest, Gradient Tree Boosting, and Extreme Gradient Boosting (XGBoost), alongside soft-voting and stacking ensembles. To ensure clinical rigor and prevent data leakage, the pipeline enforces an encapsulated transformer architecture integrating Synthetic Minority Over-sampling Technique (SMOTE) in training folds, domain-engineered composite features, out-of-fold Platt sigmoid calibration, and decision threshold optimization via $F_1$-score maximization.

Under 5-fold repeated stratified cross-validation ($5 \times 3 = 15$ folds), the tuned, sigmoid-calibrated RBF-SVC achieved champion status with a cross-validated ROC-AUC of **0.9417 \pm 0.0385** and a Brier calibration loss of **0.0987**. On an independent held-out test partition ($N = 43$), the model delivers an accuracy of **79.07%**, an exceptional precision of **95.65%**, a recall of **73.33%**, an $F_1$-score of **0.8302**, an ROC-AUC of **0.9256**, and a PR-AUC of **0.9680** at an optimal calibrated threshold of $\tau^* = 0.745$. Post-hoc explainability is achieved via an XGBoost surrogate explainer delivering TreeSHAP attributions and actionable counterfactual recommendations, establishing that secondary schooling marks (`tenth_p`), undergraduate degree marks (`degree_p`), and work experience (`workex`) govern placement likelihood. The system is containerized with Docker and operationalized via a Streamlit interactive dashboard and a FastAPI REST microservice with Population Stability Index (PSI) drift monitoring, fulfilling all Continuous Internal Evaluation (CIE) academic requirements.

**Keywords:** Educational Data Mining, Campus Placement Prediction, Interpretable Machine Learning, Support Vector Machines, SHAP, Probability Calibration, Optuna, MLOps.

<div class="page-break"></div>

## TABLE OF CONTENTS

- **Preliminary Pages:** Certificate of Approval, Declaration & Acknowledgements, Abstract, Abbreviations
- **Chapter 1: Introduction** (§1.1 Background, §1.2 Problem Definition, §1.3 Objectives, §1.4 CIE Roadmap)
- **Chapter 2: Literature Review & Related Work** (§2.1 EDM Survey, §2.2 Tabular Classifiers, §2.3 XAI, §2.4 Table 1: Literature Matrix, §2.5 Gaps)
- **Chapter 3: Dataset Description & Exploratory Data Analysis** (§3.1 Provenance, §3.2 Table 2: Data Dictionary, §3.3 Univariate/Bivariate EDA, §3.4 Figures 1–8, §3.5 Table 3: Hypotheses)
- **Chapter 4: Data Preprocessing & Feature Engineering** (§4.1 Leakage Prevention, §4.2 Imputation & Encoding, §4.3 SMOTE, §4.4 Table 4: Feature Formulas, §4.5 Table 5: Data Splitting)
- **Chapter 5: Methodology & Machine Learning Modeling** (§5.1 Formulation, §5.2 LR, §5.3 RBF-SVC, §5.4 RF, §5.5 GBM, §5.6 XGBoost, §5.7 Ensembling, §5.8 Salary Regression)
- **Chapter 6: Experimental Setup, Hyperparameter Tuning & Calibration** (§6.1 CV Protocol, §6.2 Table 6 & 7: Optuna Tuning, §6.3 Calibration Table 8 & Fig 9, §6.4 Table 9: Threshold)
- **Chapter 7: Results, Performance Evaluation & Error Analysis** (§7.1 Table 10: CV Benchmark, §7.2 Table 11: Test Generalization, §7.3 Confusion Matrix & Figures 10–14, §7.4 Failure Modes)
- **Chapter 8: Explainable AI (XAI) & Feature Attribution** (§8.1 Surrogate Explainer, §8.2 Figs 15 & 16: Importance, §8.3 Figs 17 & 18: SHAP, §8.4 Counterfactuals, §8.5 Fairness)
- **Chapter 9: Software Engineering, System Architecture & Deployment** (§9.1 Architecture, §9.2 Streamlit, §9.3 Table 13: FastAPI, §9.4 Table 14: CI/CD Suite, §9.5 Docker & PSI)
- **Chapter 10: Conclusion, Limitations & Future Work** (§10.1 Summary, §10.2 CIE Mapping, §10.3 Limitations, §10.4 Roadmap)
- **References:** IEEE Formatted Citations [1]–[12]
- **Appendices:** Appendix A (Source Layout), Appendix B (Execution Runbook), Appendix C (Metadata Snapshot)

---

## LIST OF FIGURES & TABLES

<div style="display: flex; gap: 20px; font-family: var(--font-sans); font-size: 8.5pt;">
  <div style="flex: 1;">
    <strong>Figures (18 Visualizations):</strong><br>
    • Fig 1: Placement Distribution (Base Corpus)<br>
    • Fig 2: Marks Distributions (Histograms & KDE)<br>
    • Fig 3: Academic Boxplots by Placement Status<br>
    • Fig 4: Pearson Correlation Heatmap Matrix<br>
    • Fig 5: Academic Average Density Separation<br>
    • Fig 6: Placement Rates by Categorical Predictors<br>
    • Fig 7: Gender × MBA Specialization Interaction<br>
    • Fig 8: Postgraduate Programme Placement Ratios<br>
    • Fig 9: Reliability Calibration Diagram<br>
    • Fig 10: Absolute Confusion Matrix ($N=43$)<br>
    • Fig 11: Normalized Confusion Matrix (TPR/TNR)<br>
    • Fig 12: Receiver Operating Characteristic (ROC)<br>
    • Fig 13: Precision-Recall (PR) Curve<br>
    • Fig 14: Empirical Learning Curves ($N=32$ to $109$)<br>
    • Fig 15: Permutation Feature Importance<br>
    • Fig 16: Permutation vs Impurity Importance<br>
    • Fig 17: SHAP Global Mean |SHAP| Attributions<br>
    • Fig 18: SHAP Beeswarm Distribution Summary
  </div>
  <div style="flex: 1;">
    <strong>Tables (14 Comprehensive Tables):</strong><br>
    • Table 1: Literature Review Comparison Matrix (§2.4)<br>
    • Table 2: 19-Variable Dataset Data Dictionary (§3.2)<br>
    • Table 3: Summary of Empirical EDA Hypotheses (§3.5)<br>
    • Table 4: Mathematical Definitions of Features (§4.4)<br>
    • Table 5: Stratified Dataset Splitting Protocol (§4.5)<br>
    • Table 6: Optuna Search Spaces & Distributions (§6.2)<br>
    • Table 7: Optimal Hyperparameters & Importances (§6.2)<br>
    • Table 8: Calibration Brier Score Comparison (§6.3)<br>
    • Table 9: Decision Threshold Tuning Grid (§6.4)<br>
    • Table 10: 5-Fold Repeated CV Benchmark (§7.1)<br>
    • Table 11: Held-Out Test Set Generalization (§7.2)<br>
    • Table 12: Test Confusion Matrix Contingency (§7.3)<br>
    • Table 13: FastAPI REST Endpoint Specifications (§9.3)<br>
    • Table 14: CI/CD Quality Assurance Test Suite (§9.4)
  </div>
</div>

---

## LIST OF ABBREVIATIONS

| Acronym | Definition | Acronym | Definition |
|---|---|---|---|
| **API** | Application Programming Interface | **OOF** | Out-of-Fold Cross-Validation |
| **AUC** | Area Under the Curve | **PSI** | Population Stability Index |
| **CIE** | Continuous Internal Evaluation | **RBF** | Radial Basis Function (Gaussian) |
| **CTC** | Cost-to-Company (Annual Remuneration) | **ROC** | Receiver Operating Characteristic |
| **EDM** | Educational Data Mining | **SHAP** | SHapley Additive exPlanations |
| **F1** | Harmonic Mean of Precision and Recall | **SMOTE** | Synthetic Minority Over-sampling Technique |
| **HSC** | Higher Secondary Certificate (12th Grade) | **SSC** | Secondary School Certificate (10th Grade) |
| **MLOps**| Machine Learning Operations | **SVC/SVM**| Support Vector Classifier / Machine |
| **PR**  | Precision-Recall | **XGBoost**| Extreme Gradient Boosting |

<div class="page-break"></div>
"""

