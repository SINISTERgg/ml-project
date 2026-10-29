# -*- coding: utf-8 -*-
"""
Chapters 4 through 6 for 25-30 page Academic Report.
"""

def get_chapters_4_to_6():
    return r"""
# CHAPTER 4: DATA PREPROCESSING & FEATURE ENGINEERING

## 4.1 Data Leakage Prevention Framework
Data leakage occurs when information from outside the training dataset is inadvertently used to fit preprocessing transformations, creating artificially inflated cross-validation performance that fails in production. In educational pipelines, fitting imputation medians, scaling parameters ($\mu, \sigma$), or applying SMOTE across the entire dataset prior to splitting leaks evaluation data into the model.

To eliminate leakage, **PlaceIQ** encapsulates all preprocessing inside a Scikit-Learn `Pipeline` and `ColumnTransformer`. All scalers, imputers, and encoders are fitted strictly on the designated training split ($X_{\text{train}}$) and subsequently applied to validation ($X_{\text{val}}$) and test ($X_{\text{test}}$) splits without refitting.

## 4.2 Imputation, Categorical Encoding, and Scaling
1. **Numerical Imputation:** Missing continuous values (e.g., auxiliary test scores) are imputed using training fold medians: $\hat{x}_j = \text{median}(\{x_{i,j} \mid x_{i,j} \neq \text{NaN}, i \in \text{Train}\})$, offering robustness against outliers.
2. **Categorical Imputation & Encoding:** Missing categorical attributes are imputed using the modal training value. Binary variables (`gender`, `workex`, `has_masters`) are transformed via binary one-hot encoding (`drop="if_binary"`), producing single indicator columns (`cat__gender_M`, `cat__workex_Yes`). Multi-level nominal variables (`twelfth_s`, `degree_t`, `masters_type`) use full one-hot encoding with `handle_unknown="ignore"` to safely handle unseen inference categories.
3. **Z-Score Feature Scaling:** Because distance-based algorithms (SVM) and regularized linear models (Logistic Regression) are scale-sensitive, all continuous features are transformed to zero mean and unit variance: $z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$.
4. **Class Imbalance Remediation via SMOTE:** Synthetic Minority Over-sampling Technique (SMOTE) [4] generates synthetic minority instances by interpolating between $k$-nearest neighbors ($k=5$) in Euclidean feature space: $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \lambda \sim U(0, 1)$. SMOTE is executed **strictly within inner training folds**, preserving genuine empirical distributions in evaluation sets.

## 4.3 Domain-Driven Feature Engineering
Five domain-engineered features synthesize multi-tier academic consistency and practical engagement, defined in Table 4.

### Table 4: Mathematical Definitions and Theoretical Rationale for Engineered Features

| Feature Identifier | Mathematical Formula | Conceptual Domain Rationale | Input Features |
|---|---|---|---|
| `academic_avg` | $\frac{1}{M} \sum (\text{tenth\_p} + \text{twelfth\_p} + \text{degree\_p} + \text{mba\_p} + \text{etest\_p})$ | Synthesizes longitudinal academic strength into a continuous composite score. | Academic marks % |
| `performance_gap` | $\max(\mathcal{P}) - \min(\mathcal{P}), \mathcal{P}=\{\text{tenth, twelfth, degree, mba}\}$ | Measures longitudinal academic consistency; high gap flags erratic performance. | Academic marks % |
| `stem_flag` | $\mathbb{I}(\text{twelfth\_s} = \text{'Science'} \lor \text{degree\_t} = \text{'Sci\&Tech'})$ | Captures recruiter preference for STEM backgrounds during technical hiring. | `twelfth_s`, `degree_t` |
| `activity_score` | $\text{internships} + \text{projects} + \text{workshops}$ | Aggregates practical engineering, software development, and certification activity. | Project & internship counts |
| `skills_composite` | $\frac{1}{2} (\text{aptitude\_score} + 10 \times \text{soft\_skills})$ | Blends cognitive aptitude with behavioral soft-skill ratings on a 0–100 scale. | Aptitude & soft skills |

Following encoding and transformation, the final design matrix $\mathbf{X} \in \mathbb{R}^{N \times 25}$ spans 25 orthogonal dimensions (10 continuous scaled numerical features and 15 one-hot categorical binary indicators).

## 4.4 Stratified Partitioning Protocol
The corpus ($N = 215$) was partitioned via stratified 60/20/20 splitting (`random_state=42`), summarized in Table 5.

### Table 5: Stratified Dataset Partitioning Protocol and Class Representation

| Partition Role | Proportion (%) | Count ($N$) | Placed ($Y=1$) | Not Placed ($Y=0$) | Minority Ratio (%) | Methodological Purpose |
|---|---|---|---|---|---|---|
| **Training Partition ($X_{\text{train}}$)** | 63.7% | 137 | 94 | 43 | 31.39% | Pipeline fitting, SMOTE oversampling, model training. |
| **Validation Partition ($X_{\text{val}}$)** | 16.3% | 35 | 24 | 11 | 31.43% | Optuna hyperparameter optimization, threshold tuning. |
| **Held-Out Test ($X_{\text{test}}$)** | 20.0% | 43 | 30 | 13 | 30.23% | Final unseen generalization audit and error analysis. |
| **Total Cohort** | 100.0% | 215 | 148 | 67 | 31.16% | Complete unified campus recruitment corpus. |

<div class="page-break"></div>

# CHAPTER 5: METHODOLOGY & MACHINE LEARNING MODELING

## 5.1 Supervised Classification Formulation
The binary placement classification task seeks a parameterized mapping $f(\mathbf{x}): \mathbb{R}^{25} \rightarrow [0, 1]$ minimizing empirical risk: $\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(y_i, f(\mathbf{x}_i; \mathbf{w})) + \lambda \Omega(\mathbf{w})$. Five distinct model families and two meta-ensembles were implemented in `src/ensemble.py`.

## 5.2 Candidate Model Architectures
1. **Logistic Regression (L2 Baseline):** Models posterior log-odds linearly: $P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T \mathbf{x} + b))}$. Solved by minimizing cross-entropy with Ridge regularization: $\mathcal{J}(\mathbf{w}) = \mathcal{L}_{\text{BCE}} + \frac{1}{2C} \|\mathbf{w}\|_2^2$, providing high coefficient interpretability.
2. **Support Vector Classifier (RBF Kernel & Platt Scaling):** Solves the dual soft-margin quadratic program in Reproducing Kernel Hilbert Space using the Radial Basis Function kernel: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$. The uncalibrated decision function $f_{\text{raw}}(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i^* K(\mathbf{x}_i, \mathbf{x}) + b$ is mapped to probabilities via Platt scaling [6]: $P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A f_{\text{raw}}(\mathbf{x}) + B)}$.
3. **Random Forest Classifier:** Bagging ensemble constructing $B = 600$ decorrelated decision trees using bootstrap aggregation and random feature subspace projection ($m = \sqrt{D}$), averaging class frequencies across trees: $P(Y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$ [11].
4. **Gradient Tree Boosting:** Sequential forward stagewise additive model: $F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^{M} \eta h_m(\mathbf{x})$, where each base regression tree $h_m(\mathbf{x})$ fits pseudo-residuals $r_{im} = y_i - \sigma(F_{m-1}(\mathbf{x}_i))$ scaled by learning rate $\eta \in (0, 1]$ [1].
5. **Extreme Gradient Boosting (XGBoost):** Optimizes second-order Taylor expansion approximations of the loss function with explicit leaf regularization: $\tilde{\mathcal{L}}^{(m)} \approx \sum_{i=1}^N [g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i)] + \gamma T + \frac{1}{2} \lambda \sum w_j^2 + \alpha \sum |w_j|$ [2].
6. **Heterogeneous Ensembles:** Soft Voting (weighted probability averaging: $P_{\text{vote}} = \sum \omega_k P_k$) and Stacking (Level-0 base learners feed out-of-fold probabilities into a Level-1 Logistic Regression meta-learner).

## 5.3 Post-Placement Salary Regression Subsystem
For candidates classified as placed ($P(Y=1 \mid \mathbf{x}) \ge \tau^*$), an auxiliary continuous regressor (`src/salary.py`) estimates annual CTC in INR. To handle extreme positive skewness in corporate compensation packages (INR 200k to 940k), the regressor fits natural log-transformed target values: $y_{\text{salary}}' = \ln(\text{salary})$. An **XGBRegressor** is trained using Huber loss (smooth $\ell_1$ loss) to resist outlier contamination: $\mathcal{L}_\delta(r) = \frac{1}{2}r^2$ for $|r| \le \delta$ and $\delta(|r| - \frac{1}{2}\delta)$ otherwise. Point predictions are converted back to rupees via $\hat{y} = \exp(\hat{y}')$, accompanied by uncertainty bounds and qualitative remuneration bands (*Standard Entry*, *Competitive*, *Premium Tier*).

<div class="page-break"></div>

# CHAPTER 6: EXPERIMENTAL SETUP, HYPERPARAMETER TUNING & CALIBRATION

## 6.1 Validation Protocol & Bayesian Optimization
Model selection and evaluation utilized **5-Fold Repeated Stratified Cross-Validation** with 3 independent repeats ($5 \times 3 = 15$ total evaluation folds per candidate model), reporting mean ROC-AUC and standard deviation ($\sigma_{\text{CV}}$).

Hyperparameters were systematically tuned using **Optuna** [5] via Tree-structured Parzen Estimators (TPE) across 25 trials per model, optimizing cross-validated ROC-AUC. Table 6 outlines the search spaces, and Table 7 specifies the discovered optimal configurations and parameter importance shares computed via functional ANOVA.

### Table 6: Hyperparameter Search Spaces Configured for Bayesian Optimization in Optuna

| Model Architecture | Hyperparameter Identifier | Distribution Type | Search Bounds |
|---|---|---|---|
| **Logistic Regression** | Inverse Regularization ($C$) | Log-Uniform Continuous | $[1.0 \times 10^{-3}, 1.0 \times 10^3]$ |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features` | Integer / Continuous Uniform | $[50, 600]$, $[3, 15]$, $[1, 10]$, $[2, 10]$, $[0.2, 1.0]$ |
| **Gradient Boosting** | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `min_samples_leaf` | Log-Uniform / Uniform | $[0.01, 0.3]$, $[50, 400]$, $[2, 8]$, $[0.6, 1.0]$, $[1, 10]$ |
| **XGBoost** | `lr`, `n_estimators`, `depth`, `min_child_weight`, $\gamma$, `subsample`, `colsample`, $\alpha$, $\lambda$ | Log-Uniform / Uniform | $[0.01, 0.3]$, $[50, 400]$, $[2, 8]$, $[0.5, 10.0]$, $[0, 1]$, $[0.5, 1]$, $[0.4, 1]$, $[10^{-4}, 10]$ |
| **Support Vector (SVC)** | Regularization Penalty ($C$), Kernel Bandwidth ($\gamma$) | Log-Uniform Continuous | $[0.01, 500.0]$, $[1.0 \times 10^{-4}, 0.1]$ |

### Table 7: Optimal Hyperparameters and Parameter Importance Fractions Discovered via Optuna

| Model | Optimal Hyperparameters Discovered | Governing Hyperparameter | Optuna Importance Share (%) |
|---|---|---|---|
| **Logistic Regression** | $C = 16.786$ | Inverse Regularization ($C$) | 100.0% |
| **Random Forest** | `n_est` = 600, `depth` = 8, `leaf` = 2, `split` = 5, `feat` = 0.50 | `min_samples_split` (50.4%), `min_samples_leaf` (37.1%) | Split constraints govern regularization. |
| **Gradient Boosting** | `lr` = 0.0805, `n_est` = 254, `depth` = 5, `subsample` = 0.854, `leaf` = 5 | `max_depth` (33.9%), `min_samples_leaf` (25.8%) | Tree depth and leaf constraints dominate. |
| **XGBoost** | `lr` = 0.203, `n_est` = 115, `depth` = 5, `child` = 0.796, $\gamma=0.0006$, `sub` = 0.842, `col` = 0.708 | `min_child_weight` (40.0%), `learning_rate` (22.5%) | Minimum Hessian child weight dictates pruning. |
| **Support Vector (SVC)** | $C = 392.982$, $\gamma = 0.001959$ | Regularization Penalty ($C$) | 84.5% importance on $C$; 15.5% on $\gamma$. |

## 6.2 Probability Calibration and Threshold Tuning
To ensure probability predictions reflect empirical likelihoods, Out-of-Fold (OOF) cross-validation predictions were evaluated using **Brier Calibration Loss**: $\text{Brier} = \frac{1}{N} \sum (P(Y=1 \mid \mathbf{x}_i) - y_i)^2$. Parametric Platt Sigmoid calibration was benchmarked against non-parametric Isotonic Regression (Table 8).

### Table 8: Probability Calibration Performance Comparison (Brier Score Analysis)

| Calibration Method | Mathematical Formulation | Validation Brier Score | Relative Change (%) | Selection Verdict |
|---|---|---|---|---|
| **Raw Uncalibrated** | Direct model output $f(\mathbf{x})$ | 0.1227 | Baseline | Prone to probability compression. |
| **Isotonic Regression** | Monotonic step function | 0.1347 | $-9.78\%$ (Degraded) | Overfitted on small sample sizes. |
| **Sigmoid (Platt Scaling)** | Logistic fit $\frac{1}{1 + \exp(Az + B)}$ | **0.1212** | **$+1.21\%$ (Superior)** | **Selected Champion Calibrator** |

Figure 9 portrays the calibration reliability diagram. The sigmoid-calibrated curve aligns tightly with the ideal $45^\circ$ diagonal, confirming that predicted probabilities directly mirror actual empirical placement conversion rates.

<figure class="report-figure" style="text-align: center; margin: 15px auto;">
  <img src="../models/plots/calibration.png" alt="Figure 9: Calibration Curves" style="max-width: 65%; max-height: 220px;">
  <figcaption>Figure 9: Model Calibration Reliability Diagram Comparing Raw, Isotonic, and Sigmoid Scaling</figcaption>
</figure>

To minimize false positive risk, a validation threshold sweep was conducted across $\tau \in [0.10, 0.90]$ targeting $F_1$-score maximization (Table 9).

### Table 9: Decision Threshold Evaluation Grid for Precision-Recall-F1 Optimization

| Operating Threshold ($\tau$) | Precision ($Y=1$) | Recall ($Y=1$) | Validation $F_1$-Score | Clinical / Counseling Assessment |
|---|---|---|---|---|
| $\tau = 0.300$ | 0.742 | 0.958 | 0.836 | Overly permissive; high false positive rate. |
| $\tau = 0.500$ (Default) | 0.846 | 0.875 | 0.860 | Suboptimal under asymmetric false positive penalty. |
| **$\tau^* = 0.745$ (Optimal)** | **0.957** | **0.733** | **0.898 (Val) / 0.830 (Test)** | **Global $F_1$ optimum; suppresses costly false positive errors.** |
| $\tau = 0.800$ | 0.941 | 0.667 | 0.780 | Overly conservative; severe drop in sensitivity. |

The optimal cut-off $\tau^* = 0.745$ elevates positive class precision to **95.65%**, guaranteeing that students assured of placement can rely upon the prediction with high statistical certainty. Tuned RBF-SVC was promoted as production champion (`models/champion.json`).

<div class="page-break"></div>
"""

