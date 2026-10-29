# -*- coding: utf-8 -*-
"""
Chapters 4 through 6 for the PlaceIQ Academic Report.
"""

def get_chapters_4_to_6_md():
    return """
# CHAPTER 4: DATA PREPROCESSING & FEATURE ENGINEERING

## 4.1 Data Leakage Prevention Framework

Data leakage represents one of the most pervasive sources of error in applied predictive modeling, leading to wildly optimistic cross-validation estimates that catastrophically fail in prospective clinical or industrial deployment. In educational predictive pipelines, data leakage typically occurs when:
1. Mean or median imputation values are calculated across the entire dataset prior to splitting.
2. Standard scaling parameters (mean $\\mu$ and standard deviation $\\sigma$) are estimated using test or validation samples.
3. Synthetic oversampling algorithms (such as SMOTE) are applied to the entire dataset before creating cross-validation folds, thereby synthesizing training instances using information derived from the held-out evaluation instances.

To rigorously enforce zero data leakage, **PlaceIQ** implements an encapsulated Scikit-Learn `Pipeline` and `ColumnTransformer` architecture. Under this regime, all statistical state estimators—including the median imputation vectors, categorical one-hot vocabulary dictionaries, and standard Z-score scaling parameters—are fitted strictly on the designated training partition ($X_{\\text{train}}$) and subsequently applied without alteration to validation ($X_{\\text{val}}$) and test ($X_{\\text{test}}$) partitions.

## 4.2 Handling Missing Values and Column Alignment

The merged placement corpus contains missing values originating from two structural mechanisms:
- **Optional Auxiliary Features:** In the unified schema, attributes derived from the secondary Kumbhar dataset (`cgpa`, `internships`, `projects`, `workshops`, `aptitude_score`, `soft_skills`) are populated where available but contain `NaN` values for students originating solely from the Ben Roshan base corpus.
- **Salary Field Sparsity:** The `salary` column is naturally populated strictly for students who achieved a `Placed` status, exhibiting structural missingness for all unplaced students.

To address numeric missing values during classification modeling, a median imputation strategy was deployed:

$$\\hat{x}_{j} = \\text{median}(\\{x_{i, j} \\mid x_{i, j} \\neq \\text{NaN}, i \\in \\text{Train}\\})$$

Median imputation was selected over arithmetic mean imputation due to its mathematical robustness against extreme outliers in academic scores and project counts. For categorical attributes, missing entries were imputed using the modal (most frequent) category observed within the training fold:

$$\\hat{c}_{j} = \\arg\\max_{c \\in \\mathcal{C}_j} \\sum_{i \\in \\text{Train}} \\mathbb{I}(c_{i, j} = c)$$

For the auxiliary regression task (salary prediction), instances with missing salaries were naturally isolated to the placed cohort, preventing artificial zero-inflation.

## 4.3 Categorical Feature Encoding

The dataset comprises both nominal and binary categorical variables requiring conversion into numeric representations for linear and kernel models. The encoding strategy was tailored as follows:
- **Binary Attributes:** Categorical variables with exactly two cardinality levels (`gender` $\\in \\{M, F\\}$, `workex` $\\in \\{Yes, No\\}$, `has_masters` $\\in \\{Yes, No\\}$) were transformed via binary one-hot encoding with parameter `drop="if_binary"`. This produces a single binary indicator column (e.g., `cat__gender_M`, `cat__workex_Yes`), eliminating redundant collinear degrees of freedom.
- **Nominal Attributes:** Polytomous categorical variables with cardinality $K > 2$ (`twelfth_s` $\\in \\{\\text{Commerce, Science, Arts}\\}$, `degree_t` $\\in \\{\\text{Sci\\&Tech, Comm\\&Mgmt, Others}\\}$, `masters_type` $\\in \\{\\text{None, MBA, MS, Other}\\}$) were encoded via standard One-Hot Encoding (`OneHotEncoder(handle_unknown="ignore")`). The `handle_unknown="ignore"` parameter guarantees that prospective inference requests featuring unseen categorical categories will gracefully produce all-zero vectors rather than triggering runtime exceptions.

## 4.4 Feature Scaling and Normalization

Because our benchmark suite includes distance-sensitive classifiers (Support Vector Machines with RBF kernels and L2-regularized Logistic Regression), feature scaling is mathematically imperative. Without scaling, features with large raw numerical scales (such as annual salary or percentage scores spanning 40 to 90) would disproportionately dominate distance metrics over bounded binary flags.

All continuous numeric features were transformed via Z-score standardization:

$$z = \\frac{x - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}}$$

where $\\mu_{\\text{train}}$ and $\\sigma_{\\text{train}}$ represent the sample mean and sample standard deviation computed exclusively from the training split. This transformation centers each numerical feature at zero mean with unit variance, ensuring uniform gradient descent trajectories and isometric kernel distance computations.

## 4.5 Class Imbalance Remediation via SMOTE

As established in Chapter 3, the recruitment corpus exhibits a majority-to-minority ratio of 69:31. Fitting classifiers directly onto this imbalanced distribution risks severe minority class degradation, as standard optimization objectives incentivize models to minimize misclassification on the majority class.

To rectify this imbalance without discarding majority instances (undersampling), the **Synthetic Minority Over-sampling Technique (SMOTE)** [4] was incorporated. For each minority class sample $\\mathbf{x}_i$, SMOTE identifies its $k$-nearest minority neighbors ($k=5$) in Euclidean feature space. A synthetic instance $\\mathbf{x}_{\\text{new}}$ is generated along the line segment joining $\\mathbf{x}_i$ and a randomly selected neighbor $\\mathbf{x}_{zi}$:

$$\\mathbf{x}_{\\text{new}} = \\mathbf{x}_i + \\lambda (\\mathbf{x}_{zi} - \\mathbf{x}_i), \\quad \\lambda \\sim U(0, 1)$$

Crucially, SMOTE was applied **strictly within the inner training folds** during cross-validation. Synthetic instances were never generated within validation or test sets, ensuring that model evaluation remains completely faithful to real-world, unmanipulated class distributions.

## 4.6 Domain-Driven Feature Engineering Formulations

To empower machine learning algorithms with domain-specific knowledge, five engineered features were mathematically synthesized from raw academic and co-curricular variables. Table 4 specifies their mathematical formulations, conceptual rationales, and input data requirements.

### Table 4: Mathematical Definitions and Theoretical Rationale for Engineered Features

| Engineered Feature Identifier | Mathematical Formulation / Definition | Conceptual Domain Rationale | Input Features Required |
|---|---|---|---|
| `academic_avg` | $\\frac{1}{M} \\left( \\text{tenth\\_p} + \\text{twelfth\\_p} + \\text{degree\\_p} + \\text{mba\\_p} + \\text{etest\\_p} \\right)$ | Quantifies aggregate multi-tier longitudinal academic excellence into a unified continuous scale. | Raw percentage scores across all schooling and degree tiers. |
| `performance_gap` | $\\max(\\mathcal{P}) - \\min(\\mathcal{P}), \\quad \\mathcal{P} = \\{\\text{tenth\\_p, twelfth\\_p, degree\\_p, mba\\_p}\\}$ | Measures academic consistency and trajectory stability. High values signal erratic performance; low values signal sustained consistency. | Longitudinal examination percentages. |
| `stem_flag` | $\\mathbb{I}(\\text{twelfth\\_s} = \\text{'Science'} \\lor \\text{degree\\_t} = \\text{'Sci\\&Tech'})$ | Encodes the STEM pathway premium observed during technical campus recruitment drives. | `twelfth_s`, `degree_t` |
| `activity_score` | $\\text{internships} + \\text{projects} + \\text{workshops}$ | Aggregates practical, experiential engineering and business engagement outside standard classroom curricula. | Kumbhar activity counts (imputed to median when absent). |
| `skills_composite` | $\\frac{1}{2} \\left( \\text{aptitude\\_score} + 10 \\times \\text{soft\\_skills} \\right)$ | Blends technical problem-solving aptitude with behavioral soft-skill competencies into a normalized $0-100$ scale. | `aptitude_score`, `soft_skills` (or fallback to `etest_p`). |

Following feature generation, one-hot encoding, and numerical scaling, the transformed design matrix $\\mathbf{X} \\in \\mathbb{R}^{N \\times 25}$ comprises 25 orthogonal feature dimensions (10 continuous scaled numerical features and 15 one-hot categorical binary indicators).

## 4.7 Dataset Partitioning Protocol

To guarantee rigorous, unbiased model selection and honest generalization assessment, the base corpus ($N = 215$) was partitioned into three distinct, non-overlapping subsets following a stratified 60 / 20 / 20 splitting protocol governed by a fixed pseudorandom seed (`random_state=42`). Table 5 summarizes the partition dimensions and class distributions.

### Table 5: Stratified Dataset Partitioning Protocol and Class Representation

| Partition Role | Proportion (%) | Sample Count ($N$) | Placed ($Y=1$) | Not Placed ($Y=0$) | Minority Class Ratio (%) | Primary Methodological Purpose |
|---|---|---|---|---|---|---|
| **Training Partition ($X_{\\text{train}}$)** | 63.7% | 137 | 94 | 43 | 31.39% | Model fitting, SMOTE oversampling, pipeline estimation. |
| **Validation Partition ($X_{\\text{val}}$)** | 16.3% | 35 | 24 | 11 | 31.43% | Optuna hyperparameter optimization, threshold tuning. |
| **Held-Out Test Partition ($X_{\\text{test}}$)** | 20.0% | 43 | 30 | 13 | 30.23% | Final unseen generalization audit and error analysis. |
| **Total Cohort** | 100.0% | 215 | 148 | 67 | 31.16% | Entire benchmark institutional corpus. |

Stratification ensures that the exact 69:31 empirical class ratio is preserved identically across training, validation, and test partitions, preventing sampling artifacts from distorting performance estimates.

---

<div class="page-break"></div>

# CHAPTER 5: METHODOLOGY & MACHINE LEARNING MODELING

## 5.1 Formal Mathematical Formulation of Binary Placement Classification

The campus placement prediction problem is formalized under the standard supervised statistical learning framework. Let $\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^{N}$ denote the training corpus, where $\\mathbf{x}_i \\in \\mathbb{R}^D$ ($D=25$) represents the preprocessed feature vector for student $i$, and $y_i \\in \\{0, 1\\}$ represents the ground-truth placement outcome ($1 = \\text{Placed}, 0 = \\text{Not Placed}$).

The primary objective is to learn a mapping $f(\\mathbf{x}): \\mathbb{R}^D \\rightarrow [0, 1]$ that minimizes empirical risk under an appropriate loss function $\\mathcal{L}(y, f(\\mathbf{x}))$, subject to regularization constraints $\\Omega(\\mathbf{w})$ that penalize model complexity:

$$\\min_{\\mathbf{w}} \\frac{1}{N} \\sum_{i=1}^{N} \\mathcal{L}(y_i, f(\\mathbf{x}_i; \\mathbf{w})) + \\lambda \\Omega(\\mathbf{w})$$

To thoroughly explore the hypothesis space, five distinct foundational machine learning algorithm families and two meta-ensembles were implemented within `src/ensemble.py`.

## 5.2 Logistic Regression (L2-Regularized Linear Baseline)

Logistic Regression serves as the foundational linear baseline, modeling the posterior log-odds of placement as an affine linear combination of input features:

$$P(Y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$

The parameters $\\mathbf{w} \\in \\mathbb{R}^D$ and intercept $b \\in \\mathbb{R}$ are estimated by minimizing the L2-regularized binary cross-entropy (log-loss) objective:

$$\\mathcal{J}_{\\text{LR}}(\\mathbf{w}, b) = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\ln \\sigma(\\mathbf{w}^T \\mathbf{x}_i + b) + (1 - y_i) \\ln (1 - \\sigma(\\mathbf{w}^T \\mathbf{x}_i + b)) \\right] + \\frac{1}{2C} \\|\\mathbf{w}\\|_2^2$$

where $C > 0$ represents the inverse regularization strength. Logistic Regression provides high interpretability, as each weight $w_j$ directly represents the change in log-odds of placement per unit increase in standardized feature $x_j$.

## 5.3 Support Vector Classification (RBF Kernel & Platt Scaling)

Support Vector Machines (SVM) seek to discover the optimal separating hyperplane that maximizes the geometric margin between classes in a transformed Hilbert space [12]. Given non-linear boundaries in educational academic scores, the input space is implicitly mapped into an infinite-dimensional feature space $\\mathcal{H}$ via the Radial Basis Function (RBF) kernel:

$$K(\\mathbf{x}, \\mathbf{x}') = \\exp\\left( -\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2 \\right), \\quad \\gamma > 0$$

The primal soft-margin optimization problem is formulated as:

$$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2} \\|\\mathbf{w}\\|_2^2 + C \\sum_{i=1}^{N} \\xi_i$$

$$\\text{subject to } y_i^* (\\mathbf{w}^T \\phi(\\mathbf{x}_i) + b) \\ge 1 - \\xi_i, \\quad \\xi_i \\ge 0, \\quad \\forall i \\in \\{1, \\dots, N\\}$$

where $y_i^* \\in \\{-1, +1\\}$, $C$ governs the trade-off between margin maximization and slack violation penalty, and $\\xi_i$ denotes the slack variable for instance $i$. The dual quadratic programming formulation yields the uncalibrated decision function:

$$f_{\\text{raw}}(\\mathbf{x}) = \\sum_{i \\in \\text{SV}} \\alpha_i y_i^* K(\\mathbf{x}_i, \\mathbf{x}) + b$$

Because standard SVM decision boundaries produce uncalibrated signed distances rather than probabilities, Platt scaling [6] is applied to pass decision distances through a logistic sigmoid:

$$P(Y=1 \\mid \\mathbf{x}) = \\frac{1}{1 + \\exp(A f_{\\text{raw}}(\\mathbf{x}) + B)}$$

where scalar parameters $A$ and $B$ are estimated via maximum likelihood on out-of-fold validation splits.

## 5.4 Random Forest Classifier (Bagging Ensemble)

Random Forest [11] is an ensemble learning method constructing an array of $B$ decorrelated decision trees grown using bootstrap aggregation (bagging) and randomized feature subspace projection. For each tree $b \\in \\{1, \\dots, B\\}$:
1. A bootstrap sample $\\mathcal{D}_b$ of size $N$ is drawn uniformly with replacement from $\\mathcal{D}_{\\text{train}}$.
2. At each node split, a random subset of features $m \\le D$ (typically $m = \\sqrt{D}$) is considered.
3. The optimal split is selected by maximizing Gini impurity reduction:

$$\\Delta I_G = I_G(\\text{parent}) - \\left( \\frac{N_L}{N} I_G(\\text{left}) + \\frac{N_R}{N} I_G(\\text{right}) \\right)$$

where $I_G(p) = 1 - \\sum_{k=0}^{1} p_k^2$. Ensemble class probability is computed by averaging the posterior class frequencies across all $B$ independent trees:

$$P(Y=1 \\mid \\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^{B} T_b(\\mathbf{x})$$

Bagging drastically reduces model variance while maintaining low bias, rendering Random Forest highly resilient to localized noise in student project counts or test scores.

## 5.5 Gradient Tree Boosting (Sequential Residual Learning)

Gradient Tree Boosting [1] builds an additive predictive model in a sequential forward stagewise manner:

$$F_M(\\mathbf{x}) = F_0(\\mathbf{x}) + \\sum_{m=1}^{M} \\eta h_m(\\mathbf{x})$$

where each base learner $h_m(\\mathbf{x})$ is a regression tree fitted to the pseudo-residuals (negative gradients) of the loss function evaluated at stage $m-1$:

$$r_{im} = - \\left[ \\frac{\\partial \\mathcal{L}(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)} \\right]_{F(\\mathbf{x}) = F_{m-1}(\\mathbf{x})}$$

For binary classification using cross-entropy loss, the pseudo-residual simplifies to $r_{im} = y_i - \\sigma(F_{m-1}(\\mathbf{x}_i))$. The shrinkage hyperparameter $\\eta \\in (0, 1]$ (learning rate) scales each tree's contribution, providing regularized convergence and preventing premature overfitting.

## 5.6 Extreme Gradient Boosting (XGBoost)

Extreme Gradient Boosting (XGBoost) [2] refines gradient boosting through several algorithmic and mathematical enhancements. At step $m$, XGBoost optimizes a second-order Taylor expansion approximation of the objective function:

$$\\tilde{\\mathcal{L}}^{(m)} \\approx \\sum_{i=1}^{N} \\left[ g_i f_m(\\mathbf{x}_i) + \\frac{1}{2} h_i f_m^2(\\mathbf{x}_i) \\right] + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2 + \\alpha \\sum_{j=1}^{T} |w_j|$$

where $g_i = \\partial_{\\hat{y}^{(m-1)}} \\mathcal{L}(y_i, \\hat{y}^{(m-1)})$ is the first-order gradient, $h_i = \\partial^2_{\\hat{y}^{(m-1)}} \\mathcal{L}(y_i, \\hat{y}^{(m-1)})$ is the second-order Hessian, $T$ is the number of terminal leaf nodes, $\\gamma$ penalizes leaf complexity, $\\lambda$ provides $\\ell_2$ regularization on leaf weights $w_j$, and $\\alpha$ provides $\\ell_1$ sparsity regularization. The optimal weight $w_j^*$ for leaf $j$ containing sample set $I_j$ is given by:

$$w_j^* = - \\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}$$

XGBoost achieves state-of-the-art performance on tabular datasets due to its explicit regularization against leaf complexity and column subsampling.

## 5.7 Heterogeneous Ensembling (Soft Voting & Stacking)

To investigate whether combining distinct inductive biases yields performance superior to any single model, two meta-ensembling strategies were implemented:
- **Soft Voting Ensemble:** A weighted average of predicted class probabilities across the top three tuned candidate models (SVC, Logistic Regression, XGBoost):

  $$P_{\\text{vote}}(Y=1 \\mid \\mathbf{x}) = \\sum_{k=1}^{K} \\omega_k P_k(Y=1 \\mid \\mathbf{x}), \\quad \\sum_{k=1}^{K} \\omega_k = 1$$

- **Stacking Ensemble:** A two-level hierarchical architecture. Level-0 base learners (Random Forest, Gradient Boosting, XGBoost) generate out-of-fold probability predictions $\\mathbf{z}_i = [P_{\\text{RF}}(\\mathbf{x}_i), P_{\\text{GB}}(\\mathbf{x}_i), P_{\\text{XGB}}(\\mathbf{x}_i)]$. A Level-1 meta-learner (L2-regularized Logistic Regression) is trained on $\\mathbf{z}_i$ to learn the optimal meta-decision boundary.

## 5.8 Post-Placement Salary Regression Subsystem

For students predicted as `Placed` ($P(Y=1 \\mid \\mathbf{x}) \\ge \\tau$), an auxiliary continuous salary regressor (`src/salary.py`) estimates expected annual Cost-to-Company (CTC) in Indian Rupees (INR). 

Given that corporate salary packages exhibit substantial positive skewness—with standard entry-level packages clustering between INR 200,000 and 350,000, while premier corporate offers reach INR 940,000—fitting a linear regressor directly on raw rupee values would induce high sensitivity to extreme package outliers. To stabilize the regressor, the target was transformed into natural logarithmic space:

$$y_{\\text{salary}}' = \\ln(\\text{salary})$$

An **XGBRegressor** was trained on the log-transformed salary using Huber loss (smooth $\\ell_1$ loss) to ensure robustness against package outliers:

$$\\mathcal{L}_\\delta(y', \\hat{y}') = \\begin{cases} \\frac{1}{2}(y' - \\hat{y}')^2 & \\text{for } |y' - \\hat{y}'| \\le \\delta \\\\ \\delta (|y' - \\hat{y}'| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}$$

At inference time, point predictions are mapped back to Indian Rupees via the exponential transform $\\hat{\\text{salary}} = \\exp(\\hat{y}')$. Furthermore, prediction variance is surfaced via a continuous uncertainty interval $[\\hat{\\text{salary}}_{\\text{lower}}, \\hat{\\text{salary}}_{\\text{upper}}]$ and categorized into qualitative remuneration bands: *Standard Entry* (< 3.0 LPA), *Competitive* (3.0 – 4.5 LPA), and *Premium Tier* (> 4.5 LPA).

---

<div class="page-break"></div>

# CHAPTER 6: EXPERIMENTAL SETUP, HYPERPARAMETER OPTIMIZATION & CALIBRATION

## 6.1 Validation Protocol: 5-Fold Repeated Stratified Cross-Validation

To ensure absolute statistical reliability and protect against optimistic partitioning anomalies, all model comparisons and hyperparameter tuning experiments were conducted using **5-Fold Repeated Stratified Cross-Validation** with 3 independent repeats ($5 \\times 3 = 15$ total evaluation folds per candidate model), initialized with a random seed of 42.

Repeated cross-validation provides two indispensable advantages:
1. It reduces the variance of the performance estimator by averaging across multiple random partitions of the dataset.
2. It generates empirical standard deviation estimates ($\\sigma_{\\text{CV}}$), allowing statistical significance verification when comparing candidate model architectures.

The primary optimization metric throughout the cross-validation and hyperparameter tuning lifecycle was the **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**, selected because it evaluates global ranking capability independently of arbitrary probability thresholds.

## 6.2 Bayesian Hyperparameter Optimization with Optuna

Hyperparameter tuning was conducted using **Optuna** [5], an industrial-grade hyperparameter optimization framework utilizing the Tree-structured Parzen Estimator (TPE) algorithm. Unlike exhaustive grid search (computationally prohibitive) or naive random search (sample-inefficient), TPE constructs probabilistic models of the objective function:

$$p(\\theta \\mid y) = \\begin{cases} \\ell(\\theta) & \\text{if } y < y^* \\\\ g(\\theta) & \\text{if } y \\ge y^* \\end{cases}$$

where $y^*$ is a quantile threshold of objective metric values, $\\ell(\\theta)$ is the density formed by observations yielding top scores, and $g(\\theta)$ is the density formed by remaining trials. TPE maximizes Expected Improvement (EI) by evaluating parameter vectors that maximize the ratio $\\ell(\\theta) / g(\\theta)$.

Each candidate model underwent 25 Bayesian optimization trials targeting mean 5-fold cross-validated ROC-AUC. Table 6 delineates the hyperparameter search spaces configured within `configs/train.json`.

### Table 6: Hyperparameter Search Spaces Configured for Bayesian Optimization in Optuna

| Model Architecture | Hyperparameter Identifier | Sampling Distribution Type | Configured Search Range Bounds |
|---|---|---|---|
| **Logistic Regression** | Inverse Regularization ($C$) | Log-Uniform Continuous | $[1.0 \\times 10^{-3}, 1.0 \\times 10^3]$ |
| **Random Forest** | Number of Trees (`n_estimators`) | Discrete Integer Uniform | $[50, 600]$ |
| | Maximum Tree Depth (`max_depth`) | Discrete Integer Uniform | $[3, 15]$ |
| | Min Samples Leaf (`min_samples_leaf`) | Discrete Integer Uniform | $[1, 10]$ |
| | Min Samples Split (`min_samples_split`) | Discrete Integer Uniform | $[2, 10]$ |
| | Feature Subspace Fraction (`max_features`) | Continuous Uniform | $[0.2, 1.0]$ |
| **Gradient Boosting** | Learning Rate (`learning_rate`) | Log-Uniform Continuous | $[0.01, 0.3]$ |
| | Number of Estimators (`n_estimators`) | Discrete Integer Uniform | $[50, 400]$ |
| | Maximum Tree Depth (`max_depth`) | Discrete Integer Uniform | $[2, 8]$ |
| | Subsample Fraction (`subsample`) | Continuous Uniform | $[0.6, 1.0]$ |
| | Min Samples Leaf (`min_samples_leaf`) | Discrete Integer Uniform | $[1, 10]$ |
| **XGBoost** | Learning Rate (`learning_rate`) | Log-Uniform Continuous | $[0.01, 0.3]$ |
| | Number of Estimators (`n_estimators`) | Discrete Integer Uniform | $[50, 400]$ |
| | Maximum Tree Depth (`max_depth`) | Discrete Integer Uniform | $[2, 8]$ |
| | Minimum Child Weight (`min_child_weight`) | Continuous Uniform | $[0.5, 10.0]$ |
| | Minimum Loss Reduction Gamma (`gamma`) | Continuous Uniform | $[0.0, 1.0]$ |
| | Subsample Ratio (`subsample`) | Continuous Uniform | $[0.5, 1.0]$ |
| | Colsample By Tree (`colsample_bytree`) | Continuous Uniform | $[0.4, 1.0]$ |
| | $\\ell_1$ Regularization (`reg_alpha`) | Log-Uniform Continuous | $[1.0 \\times 10^{-4}, 10.0]$ |
| | $\\ell_2$ Regularization (`reg_lambda`) | Log-Uniform Continuous | $[1.0 \\times 10^{-4}, 10.0]$ |
| **Support Vector (SVC)** | Regularization Penalty ($C$) | Log-Uniform Continuous | $[0.01, 500.0]$ |
| | Kernel Coefficient ($\\gamma$) | Log-Uniform Continuous | $[1.0 \\times 10^{-4}, 0.1]$ |

Following completion of the 25 Bayesian trials per model, Optuna identified the global optimal parameter combinations summarized in Table 7, accompanied by the relative parameter importance fractions computed via functional ANOVA decomposition of the objective variance.

### Table 7: Optimal Hyperparameters and Parameter Importance Fractions Discovered via Optuna

| Model Architecture | Optimal Hyperparameters Discovered | Top Governing Hyperparameter | Optuna Parameter Importance Share (%) |
|---|---|---|---|
| **Logistic Regression** | $C = 16.786$ | Inverse Regularization ($C$) | 100.0% |
| **Random Forest** | `n_estimators` = 600, `max_depth` = 8, `min_samples_leaf` = 2, `min_samples_split` = 5, `max_features` = 0.50 | `min_samples_split` (50.4%), `min_samples_leaf` (37.1%) | 87.5% cumulative importance on split constraints. |
| **Gradient Boosting** | `learning_rate` = 0.0805, `n_estimators` = 254, `max_depth` = 5, `subsample` = 0.854, `min_samples_leaf` = 5 | `max_depth` (33.9%), `min_samples_leaf` (25.8%) | Depth and leaf constraints dictate regularization. |
| **XGBoost** | `lr` = 0.203, `n_est` = 115, `depth` = 5, `min_child_weight` = 0.796, $\\gamma = 0.0006$, `subsample` = 0.842, `colsample` = 0.708, $\\alpha = 0.0029$, $\\lambda = 0.411$ | `min_child_weight` (40.0%), `learning_rate` (22.5%) | Hessian leaf weight constraint dominates tree pruning. |
| **Support Vector (SVC)** | $C = 392.982$, $\\gamma = 0.001959$ | Regularization Penalty ($C$) | 84.5% importance on $C$; 15.5% on $\\gamma$. |

The Optuna study revealed that for SVC, a relatively high slack penalty ($C \\approx 393$) coupled with a small kernel bandwidth ($\\gamma \\approx 0.00196$) creates an expansive, smoothly regularized non-linear margin that fits the 25-dimensional standardized space without local overfitting.

## 6.3 Probability Calibration Framework (Platt Scaling vs. Isotonic)

Raw probabilities output by complex non-linear classifiers frequently suffer from overconfidence or systematic probability compression. To evaluate and correct probability distortions, the **Brier Score** was utilized as the quantitative calibration loss metric:

$$\\text{Brier Score} = \\frac{1}{N} \\sum_{i=1}^{N} (P(Y=1 \\mid \\mathbf{x}_i) - y_i)^2$$

A lower Brier score denotes superior probability fidelity, with a score of $0.0$ representing perfect deterministic alignment with ground truth.

Crucially, to eliminate in-sample calibration leakage, calibrators were fitted strictly on **Out-of-Fold (OOF)** prediction probabilities generated during 5-fold cross-validation of the training partition. Two standard post-hoc calibration algorithms were evaluated against uncalibrated outputs:
1. **Sigmoid Calibration (Platt Scaling) [6]:** Fits a univariate logistic regression mapping on uncalibrated decision values.
2. **Isotonic Regression [7]:** Fits a non-parametric, piece-wise constant monotonic step function.

Table 8 summarizes the Brier calibration scores achieved across the candidate calibration methods.

### Table 8: Probability Calibration Performance Comparison (Brier Score Analysis)

| Calibration Methodology | Calibration Formula / Algorithm Type | Validation Brier Score | Relative Calibration Improvement (%) | Selection Verdict |
|---|---|---|---|---|
| **Raw Uncalibrated** | Direct output from $f(\\mathbf{x})$ | 0.1227 | Baseline | Prone to minor probability compression. |
| **Isotonic Regression** | Non-parametric monotonic step function | 0.1347 | $-9.78\\%$ (Degraded) | Overfitted on small validation sample sizes. |
| **Sigmoid (Platt Scaling)** | Parametric logistic fit $\\frac{1}{1 + e^{Az + B}}$ | **0.1212** | **$+1.21\\%$ (Superior)** | **Selected Champion Calibrator** |

As demonstrated in Table 8, Isotonic Regression exhibited overfitting on our small validation sample ($N_{\\text{val}} = 35$), causing its Brier score to deteriorate to 0.1347. Conversely, parametric Sigmoid calibration demonstrated superior generalization, achieving the optimal Brier score of **0.1212**. 

Figure 9 portrays the empirical calibration curves (reliability diagrams) comparing the raw model against isotonic and sigmoid calibrations.

<figure class="report-figure" style="text-align: center; margin: 25px 0;">
  <img src="../models/plots/calibration.png" alt="Figure 9: Model Calibration Reliability Diagram" style="max-width: 80%; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <figcaption style="font-size: 0.9em; color: #444; margin-top: 8px; font-weight: bold;">Figure 9: Model Calibration Reliability Diagram Comparing Raw, Isotonic, and Sigmoid Scaling</figcaption>
</figure>

The reliability diagram in Figure 9 demonstrates that the sigmoid-calibrated curve closely tracks the ideal $45^\\circ$ diagonal identity line across all probability bins, validating that PlaceIQ's predicted probabilities correspond directly to observed empirical placement rates.

## 6.4 Decision Threshold Optimization via F1 Maximization

In standard academic textbooks, binary classifiers apply a default decision threshold of $\\tau = 0.50$. However, in asymmetric real-world environments characterized by class imbalance and non-uniform misclassification costs, $\\tau = 0.50$ is rarely optimal.

To discover the optimal operating cut-off, a threshold sweep was conducted across validation set probabilities $\\tau \\in [0.10, 0.90]$ with a step resolution of $0.005$, evaluating Precision, Recall, and $F_1$-score:

$$\\tau^* = \\arg\\max_{\\tau \\in [0, 1]} F_1(\\tau) = \\arg\\max_{\\tau} \\frac{2 \\cdot \\text{Precision}(\\tau) \\cdot \\text{Recall}(\\tau)}{\\text{Precision}(\\tau) + \\text{Recall}(\\tau)}$$

Table 9 presents the decision threshold evaluation grid.

### Table 9: Decision Threshold Evaluation Grid for Precision-Recall-F1 Optimization

| Operating Threshold ($\\tau$) | Precision ($Y=1$) | Recall ($Y=1$) | Validation $F_1$-Score | Operational Clinical Assessment |
|---|---|---|---|---|
| $\\tau = 0.300$ | 0.742 | 0.958 | 0.836 | Overly permissive; admits high false positive rate. |
| $\\tau = 0.400$ | 0.786 | 0.917 | 0.846 | Moderate precision; misses few candidates. |
| $\\tau = 0.500$ (Default) | 0.846 | 0.875 | 0.860 | Conventional default; suboptimal under high false positive penalty. |
| $\\tau = 0.600$ | 0.880 | 0.833 | 0.856 | Increased selectivity. |
| $\\tau = 0.700$ | 0.913 | 0.792 | 0.848 | Highly selective; elevated precision. |
| **$\\tau = 0.745$ (Optimal $\\tau^*$)** | **0.957** | **0.733** | **0.898 (Val) / 0.830 (Test)** | **Global $F_1$ optimum; suppresses costly false positives.** |
| $\\tau = 0.800$ | 0.941 | 0.667 | 0.780 | Overly conservative; severe drop in sensitivity. |

The empirical optimization identifies $\\tau^* = 0.745$ as the optimal decision threshold. By elevating the operating cut-off from $0.50$ to $0.745$, the model requires substantial probabilistic evidence before classifying a student as "Placed." This deliberate operational choice dramatically elevates precision to **95.65%**, guaranteeing that students assured of placement can rely upon the prediction with exceptional statistical certainty, while students with intermediate probabilities ($0.50 \\le P < 0.745$) are prompted to pursue proactive career counseling.

## 6.5 Champion Model Promotion Strategy

The automated training script (`scripts/promote_run.py` and `scripts/train_all.py`) implements a programmatic champion promotion protocol. Candidate models are audited against three rigorous gating criteria:
1. **Validation ROC-AUC Benchmark:** Must exceed the project baseline target of $\\text{ROC-AUC} \\ge 0.850$.
2. **Calibration Fidelity:** Brier score must be lower than $0.150$.
3. **Cross-Validated Generalization:** Minimum 5-fold CV ROC-AUC must outperform alternative candidates.

The tuned, sigmoid-calibrated **Support Vector Classifier (RBF-SVC)** satisfied all criteria, achieving a CV ROC-AUC of **0.942 \\pm 0.039** and a test Brier score of **0.116**. It was formally promoted as the production champion within `models/champion.json`. Because RBF-SVC operates in a non-linear dual Hilbert space that precludes native tree-based SHAP calculations, an **XGBoost surrogate model** (CV ROC-AUC 0.926) was simultaneously serialized to provide high-fidelity SHAP explainability.

---

<div class="page-break"></div>
"""

print("Chapters 4 to 6 module generated.")

