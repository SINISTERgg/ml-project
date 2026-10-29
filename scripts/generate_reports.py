# -*- coding: utf-8 -*-
"""
PlaceIQ Academic Minor Project Report Builder.
Generates:
  1. reports/project_report.md
  2. reports/project_report.html
"""

import os
import sys

WORKSPACE = r"c:\Users\M S\Desktop\ml project"
REPORTS_DIR = os.path.join(WORKSPACE, "reports")
MD_PATH = os.path.join(REPORTS_DIR, "project_report.md")
HTML_PATH = os.path.join(REPORTS_DIR, "project_report.html")

# Define all section contents systematically

def get_front_matter_md():
    return """# A MINOR PROJECT REPORT ON

# **PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning**

---

### **Submitted in partial fulfillment of the requirements for the award of the degree of**
### **BACHELOR OF TECHNOLOGY / BACHELOR OF SCIENCE**
### **in**
### **COMPUTER SCIENCE AND ENGINEERING / ARTIFICIAL INTELLIGENCE & MACHINE LEARNING**

---

**Submitted by:**  
**Student Name:** M S  
**University Roll No / Register No:** 2023-CS-ML-042  
**Semester & Batch:** VII Semester, 2026–2027  

**Under the Guidance of:**  
**Project Supervisor / Faculty Guide:** Dr. Academic Guide, Ph.D.  
*Department of Computer Science and Engineering*  

**Academic Year:** 2026 – 2027  
**Department of Computer Science and Engineering**  
*College of Engineering & Technology*  

---

<div class="page-break"></div>

## CERTIFICATE OF APPROVAL

This is to certify that the minor project report titled **"PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning"**, submitted by **M S** (Register No: **2023-CS-ML-042**) in partial fulfillment of the requirements for the award of the Degree of **Bachelor of Technology in Computer Science and Engineering**, is a bona fide record of the work carried out by them under my supervision and guidance during the academic semester (August 2026 – October 2026).

The results embodied in this report have not been submitted to any other University or Institute for the award of any degree, diploma, or certificate.

<br><br>

----------------------------------------- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -----------------------------------------  
**Project Guide / Supervisor** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Head of Department (CSE)**  
Department of Computer Science & Engg. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Department of Computer Science & Engg.  

<br><br>

-----------------------------------------  
**External Examiner**  
Date: 28-October-2026  

---

<div class="page-break"></div>

## DECLARATION

I hereby declare that the minor project entitled **"PlaceIQ: Predicting Campus Placement Probability and Salary Expectations with Interpretable Machine Learning"**, submitted to the **Department of Computer Science and Engineering**, is an authentic record of the original work carried out by me under the esteemed supervision of my project guide.

I also declare that this report represents my own ideas in my own words and where others' ideas or words have been included, I have adequately cited and referenced the original sources in accordance with IEEE referencing standards. To the best of my knowledge, this work contains no material previously published or written by another person, nor has it been submitted for the award of any other academic degree or qualification.

<br><br>
**M S**  
Register No: 2023-CS-ML-042  
Department of Computer Science and Engineering  
Place: Campus Center  
Date: 28-October-2026  

---

<div class="page-break"></div>

## ACKNOWLEDGEMENTS

I express my deepest gratitude and sincere appreciation to my project guide for their invaluable guidance, insightful suggestions, constant encouragement, and intellectual rigor throughout the conceptualization, experimentation, and drafting phases of this project.

I am profoundly grateful to the **Head of the Department of Computer Science and Engineering** for providing the infrastructural resources, computational laboratories, and academic support required to successfully execute this research.

I extend my heartfelt thanks to the faculty members, technical staff, and peer reviewers of the department who offered constructive criticisms and invaluable feedback during the Phase 1 and Phase 2 continuous internal evaluations (CIE).

Finally, I wish to convey my warmest appreciation to my parents, family, and peers whose encouragement, support, and patience have been a pillar of strength throughout this academic journey.

<br>
**M S**  

---

<div class="page-break"></div>

## ABSTRACT

Campus recruitment drives constitute one of the most critical milestone events in higher educational institutions. Despite their paramount importance, students frequently discover their employability gaps and placement vulnerabilities late in their academic cycles, leaving negligible scope for timely remedial intervention. While academic placement cells systematically archive historical academic transcripts, secondary schooling scores, undergraduate specializations, and qualitative skill assessments, these records predominantly remain static and underutilized.

This project presents **PlaceIQ**, an end-to-end, interpretable machine learning system engineered to predict student campus placement probabilities, estimate post-recruitment salary tiers, and provide transparent, actionable feature attributions for academic counseling. Formulated as a supervised binary classification problem with tabular demographic, academic, and skills attributes, the system evaluates five diverse classification architectures: L2-Regularized Logistic Regression, Support Vector Classifier with Radial Basis Function kernel (RBF-SVC), Random Forest, Gradient Boosting, and Extreme Gradient Boosting (XGBoost), supplemented by soft-voting and stacking ensembles. 

To overcome severe limitations of conventional predictive modeling—specifically data leakage, class imbalance, and model overconfidence—the preprocessing pipeline enforces a strictly partitioned transformer architecture incorporating Synthetic Minority Over-sampling Technique (SMOTE), domain-engineered aggregate features (`academic_avg`, `performance_gap`, `stem_flag`, `activity_score`, `skills_composite`), out-of-fold probability calibration via Platt scaling (sigmoid), and decision threshold optimization via $F_1$-score maximization.

Rigorous evaluation via 5-fold repeated stratified cross-validation ($5 \\times 3$ repeats) demonstrates that the tuned, sigmoid-calibrated RBF-SVC achieves champion performance with a cross-validated ROC-AUC of **0.942 \\pm 0.039** and a Brier calibration score of **0.099**. On an independent, strictly isolated held-out test partition ($N=43$), the final system delivers an accuracy of **79.07%**, an exceptional precision of **95.65%**, a recall of **73.33%**, an $F_1$-score of **0.8302**, an ROC-AUC of **0.9256**, and a Precision-Recall AUC of **0.9680** at an optimal calibrated decision threshold of $\\tau = 0.745$. Because RBF-SVC functions as a non-linear kernel machine lacking direct tree-based attributions, model explainability is achieved via a dedicated XGBoost surrogate explainer combined with global and local SHapley Additive exPlanations (SHAP) and permutation importance analyses. Top predictive signals include secondary education percentage (`tenth_p`), undergraduate degree marks (`degree_p`), prior work experience (`workex`), and candidate academic aggregate (`academic_avg`).

The predictive engine is containerized with Docker and operationalized via a dual-delivery deployment architecture featuring an interactive web dashboard (Streamlit) and a high-performance RESTful API microservice (FastAPI), complemented by Population Stability Index (PSI) data-drift monitoring. This project fulfills all requirements of the Phase 1 and Phase 2 Continuous Internal Evaluation (CIE) academic curriculum.

**Keywords:** Educational Data Mining, Campus Placement Prediction, Interpretable Machine Learning, Support Vector Classification, SHAP (SHapley Additive exPlanations), Probability Calibration, Optuna Bayesian Optimization, MLOps.

---

<div class="page-break"></div>

## TABLE OF CONTENTS

- **Certificate of Approval**
- **Declaration**
- **Acknowledgements**
- **Abstract**
- **List of Figures**
- **List of Tables**
- **List of Symbols and Abbreviations**
- **Chapter 1: Introduction**
  - 1.1 Background and Context
  - 1.2 Problem Definition and Challenges
  - 1.3 Project Motivation
  - 1.4 Objectives of the Study
  - 1.5 Scope and Delimitations
  - 1.6 Continuous Internal Evaluation (CIE) Roadmap
  - 1.7 Organization of the Report
- **Chapter 2: Literature Review and Related Work**
  - 2.1 Overview of Educational Data Mining (EDM)
  - 2.2 Survey of Machine Learning in Placement Forecasting
  - 2.3 Tree Ensembles vs. Kernel Methods on Small Tabular Corpora
  - 2.4 The Interpretability Imperative: eXplainable AI (XAI)
  - 2.5 Comparative Literature Survey Matrix
  - 2.6 Critical Analysis and Identified Research Gaps
- **Chapter 3: Dataset Description & Exploratory Data Analysis**
  - 3.1 Data Acquisition and Provenance
  - 3.2 Feature Schema and Comprehensive Data Dictionary
  - 3.3 Exploratory Data Analysis Methodology
  - 3.4 Target Variable Distribution & Imbalance
  - 3.5 Academic Metric Distributions & Boxplot Disparity
  - 3.6 Multivariate Correlation & Collinearity Analysis
  - 3.7 Demographic, Stream, and Specialization Interactions
  - 3.8 Postgraduate Stream & Synthetic Pathway Analysis
  - 3.9 Key Empirical Hypotheses Derived from EDA
- **Chapter 4: Data Preprocessing & Feature Engineering**
  - 4.1 Data Leakage Prevention Framework
  - 4.2 Missing Value Imputation Strategy
  - 4.3 Categorical Feature Encoding
  - 4.4 Feature Scaling and Normalization
  - 4.5 Class Imbalance Remediation via SMOTE
  - 4.6 Domain-Driven Feature Engineering Formulations
  - 4.7 Dataset Partitioning Protocol
- **Chapter 5: Methodology & Machine Learning Modeling**
  - 5.1 Formal Mathematical Formulation of Binary Placement Classification
  - 5.2 Logistic Regression (L2-Regularized Linear Baseline)
  - 5.3 Support Vector Classification (RBF Kernel & Platt Scaling)
  - 5.4 Random Forest Classifier (Bagging Ensemble)
  - 5.5 Gradient Tree Boosting (Sequential Residual Learning)
  - 5.6 Extreme Gradient Boosting (XGBoost)
  - 5.7 Heterogeneous Ensembling (Soft Voting & Stacking)
  - 5.8 Post-Placement Salary Regression Subsystem
- **Chapter 6: Experimental Setup, Hyperparameter Optimization & Calibration**
  - 6.1 Validation Protocol: 5-Fold Repeated Stratified Cross-Validation
  - 6.2 Bayesian Hyperparameter Optimization with Optuna
  - 6.3 Probability Calibration Framework (Platt Scaling vs. Isotonic)
  - 6.4 Decision Threshold Optimization via F1 Maximization
  - 6.5 Champion Model Promotion Architecture
- **Chapter 7: Results, Performance Evaluation & Error Analysis**
  - 7.1 Cross-Validation Benchmark Comparison
  - 7.2 Independent Held-Out Test Set Generalization
  - 7.3 Confusion Matrix & Misclassification Dynamics
  - 7.4 Discriminative Power: ROC and Precision-Recall Analyses
  - 7.5 Generalization Dynamics: Empirical Learning Curves
  - 7.6 Error Analysis & Failure Mode Identification
- **Chapter 8: Explainable AI (XAI) & Feature Attribution**
  - 8.1 The Black-Box Dilemma in Academic Predictive Systems
  - 8.2 Surrogate Model Architecture for RBF-SVC
  - 8.3 Permutation Importance vs. Impurity Importance
  - 8.4 Global Feature Importance via SHAP Values
  - 8.5 Directional Feature Impact: SHAP Beeswarm Distribution
  - 8.6 Actionable Counterfactual Explanations
  - 8.7 Algorithmic Fairness, Demographic Disparity & Ethical Considerations
- **Chapter 9: Software Engineering, System Architecture & Deployment**
  - 9.1 High-Level Architecture Overview
  - 9.2 Interactive Dashboard (Streamlit Frontend)
  - 9.3 Production RESTful API (FastAPI Microservice)
  - 9.4 Continuous Integration, Testing & Quality Assurance
  - 9.5 Containerization with Docker & Multi-Service Compose
  - 9.6 Drift Monitoring: Population Stability Index (PSI)
- **Chapter 10: Conclusion, Limitations & Future Work**
  - 10.1 Summary of Contributions
  - 10.2 Assessment Against CIE Milestones & Rubrics
  - 10.3 Identified Project Limitations
  - 10.4 Roadmap for Future Extensions
- **References**
- **Appendices**
  - Appendix A: Full System Source Code Layout
  - Appendix B: Reproduction and Environment Execution Runbook
  - Appendix C: Complete Training Run Logs & Metadata Snapshot

---

<div class="page-break"></div>

## LIST OF FIGURES

| Figure No. | Figure Caption | Location / Relative Path |
|---|---|---|
| Figure 1 | Distribution of Student Placement Status in the Base Corpus | `../models/plots/eda/placement_distribution.png` |
| Figure 2 | Academic Percentage Distributions Across Academic Milestones | `../models/plots/eda/marks_distribution.png` |
| Figure 3 | Comparative Boxplots of Academic Marks Partitioned by Placement Outcome | `../models/plots/eda/marks_boxplots.png` |
| Figure 4 | Pearson Correlation Heatmap Matrix of Academic and Demographic Predictors | `../models/plots/eda/correlation_heatmap.png` |
| Figure 5 | Density and Separation Analysis of Academic Composite Average vs. Status | `../models/plots/eda/academic_avg_vs_status.png` |
| Figure 6 | Placement Ratios Partitioned by Categorical Student Attributes | `../models/plots/eda/placement_by_categorical.png` |
| Figure 7 | Interaction Between Gender, MBA Specialization, and Placement Success | `../models/plots/eda/gender_specialisation.png` |
| Figure 8 | Analysis of Postgraduate Programme Enrolment and Placement Outcome | `../models/plots/eda/masters_placement.png` |
| Figure 9 | Model Calibration Reliability Diagram Comparing Raw, Isotonic, and Sigmoid Scaling | `../models/plots/calibration.png` |
| Figure 10 | Absolute Confusion Matrix on Held-Out Test Set | `../models/plots/confusion_matrix.png` |
| Figure 11 | Normalized Confusion Matrix Illustrating True Positive and Negative Rates | `../models/plots/confusion_matrix_normalized.png` |
| Figure 12 | Receiver Operating Characteristic (ROC) Curve for Tuned Champion Model | `../models/plots/roc_curve.png` |
| Figure 13 | Precision-Recall (PR) Curve Highlighting Superior Positive Class Retention | `../models/plots/pr_curve.png` |
| Figure 14 | Empirical Learning Curves Demonstrating Training and Cross-Validation Generalization | `../models/plots/learning_curve.png` |
| Figure 15 | Permutation Feature Importance Evaluated Across Multiple Resampling Passes | `../models/plots/permutation_importance.png` |
| Figure 16 | Comparative Evaluation of Permutation Importance versus Tree Impurity Importances | `../models/plots/importance_comparison.png` |
| Figure 17 | SHAP Global Feature Attributions (Mean Absolute SHAP Value) | `../models/plots/shap_importance.png` |
| Figure 18 | SHAP Beeswarm Distribution Plot Highlighting Directional Feature Contributions | `../models/plots/shap_summary.png` |

---

<div class="page-break"></div>

## LIST OF TABLES

| Table No. | Table Title | Section |
|---|---|---|
| Table 1 | Literature Survey and Comparative Summary of Prior Educational Data Mining Works | § 2.5 |
| Table 2 | PlaceIQ Dataset Variable Specifications and Comprehensive Data Dictionary | § 3.2 |
| Table 3 | Summary of Empirical Hypotheses Extracted from Exploratory Data Analysis | § 3.9 |
| Table 4 | Mathematical Definitions and Theoretical Rationale for Engineered Features | § 4.6 |
| Table 5 | Stratified Dataset Partitioning Protocol and Class Representation | § 4.7 |
| Table 6 | Hyperparameter Search Spaces Configured for Bayesian Optimization in Optuna | § 6.2 |
| Table 7 | Optimal Hyperparameters and Parameter Importance Fractions Discovered via Optuna | § 6.2 |
| Table 8 | Probability Calibration Performance Comparison (Brier Score Analysis) | § 6.3 |
| Table 9 | Decision Threshold Evaluation Grid for Precision-Recall-F1 Optimization | § 6.4 |
| Table 10 | 5-Fold Repeated Stratified Cross-Validation Benchmark Comparison | § 7.1 |
| Table 11 | Final Generalization Metrics on Independent Held-Out Test Set ($N=43$) | § 7.2 |
| Table 12 | Confusion Matrix Contingency Table on Held-Out Test Set | § 7.3 |
| Table 13 | PlaceIQ Production Microservice REST API Endpoint Specifications | § 9.3 |
| Table 14 | Automated Continuous Integration Verification and Quality Assurance Test Suite | § 9.4 |

---

<div class="page-break"></div>

## LIST OF SYMBOLS AND ABBREVIATIONS

| Acronym / Symbol | Expansion / Description |
|---|---|
| **AI** | Artificial Intelligence |
| **API** | Application Programming Interface |
| **AUC** | Area Under the Curve |
| **Brier** | Brier Calibration Loss Metric |
| **CIE** | Continuous Internal Evaluation |
| **CTC** | Cost to Company (Annual Remuneration in INR) |
| **CV** | Cross-Validation |
| **EDM** | Educational Data Mining |
| **F1** | Harmonic Mean of Precision and Recall |
| **GBM** | Gradient Boosting Machine |
| **HSC** | Higher Secondary Certificate (12th Grade) |
| **K-Fold** | K-Fold Partitioning Cross-Validation |
| **ML** | Machine Learning |
| **MLOps** | Machine Learning Operations |
| **OOF** | Out-of-Fold (Cross-Validated Probabilities) |
| **PR** | Precision-Recall |
| **PSI** | Population Stability Index |
| **RBF** | Radial Basis Function (Gaussian Kernel) |
| **REST** | Representational State Transfer |
| **ROC** | Receiver Operating Characteristic |
| **SHAP** | SHapley Additive exPlanations |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **SSC** | Secondary School Certificate (10th Grade) |
| **SVC** | Support Vector Classifier |
| **SVM** | Support Vector Machine |
| **TPE** | Tree-structured Parzen Estimator |
| **XAI** | eXplainable Artificial Intelligence |
| **XGBoost** | Extreme Gradient Boosting |
| **$\\tau$** | Decision Classification Threshold |
| **$\\sigma(z)$** | Logistic Sigmoid Function $\\frac{1}{1 + e^{-z}}$ |
| **$\\phi(x)$** | Feature Mapping Function into Reproducing Kernel Hilbert Space |

---

<div class="page-break"></div>
"""

print("Front matter generated.")
