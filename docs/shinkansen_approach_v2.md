# Shinkansen Passenger Satisfaction Prediction — V2

## Context

V1 achieved 95.56% test accuracy (3rd place) using a single manually-tuned LightGBM model. The 1st place score is 95.85%. V2 targets this 0.29% gap (~104 more correct predictions out of 35,602) through systematic hyperparameter optimization, model diversity via stacking, and variance reduction.

---

## What Changed from V1

### Step 1–3: Data Merging, Encoding, Missing Values

Identical to V1. Inner join on ID, ordinal encoding for survey ratings, label encoding for nominals, native NaN handling by the tree models.

### Step 4: Enhanced Feature Engineering

All V1 engineered features were retained. New features added:

| Feature | Description | Rationale |
|---|---|---|
| `survey_median` | Median of all survey ratings | Robust to outlier ratings unlike mean |
| `survey_skew` | Skewness of survey ratings | Detects asymmetric response patterns — a passenger who rates most things highly but one thing poorly has negative skew |
| `survey_sum` | Sum of all survey ratings | Total satisfaction score, preserves scale unlike mean |
| `mid_ratings_count` | Count of ratings == 2 or 3 | Captures ambivalent responses — passengers who aren't strongly positive or negative |
| `delay_to_distance` | total_delay / (Travel_Distance + 1) | A 30-minute delay on a 200km trip matters more than on a 2000km trip |
| `age_bin` | Age bucketed into decile quantiles | Discretizes age for interaction effects without requiring many tree splits |
| `is_missing_*` | Binary indicators for columns with >1% missingness | Explicitly encodes whether a value was missing, since ~10% missingness in Customer_Type, Type_Travel, and Catering likely represents a distinct passenger segment |
| `TypeTravel_x_TravelClass` | Interaction of Type_Travel and Travel_Class | Business travelers in Business class may have different satisfaction patterns than business travelers in Eco |
| `CustType_x_SeatClass` | Interaction of Customer_Type and Seat_Class | Loyal customers in Green Car vs. Ordinary may weight survey aspects differently |

Total feature count increased from ~34 (V1) to ~45.

### Step 5: Bayesian Hyperparameter Optimization with Optuna

V1 used manually selected hyperparameters. V2 uses Optuna's Tree-structured Parzen Estimator (TPE) to systematically search the hyperparameter space. Optuna is a Bayesian optimization framework — unlike grid search or random search, it learns from previous trials to focus on promising regions of the search space.

Each trial proposes a new hyperparameter combination, evaluates it using 5-fold stratified cross-validation, and feeds the result back to the optimizer. This was run independently for three models:

**LightGBM** — 100 trials over:
- learning_rate: [0.005, 0.1] (log scale)
- max_depth: [4, 10]
- num_leaves: [15, 127]
- subsample, colsample_bytree: [0.5, 1.0]
- min_child_samples: [5, 100]
- reg_alpha, reg_lambda: [0.001, 10.0] (log scale)
- n_estimators: 5000 with early stopping (patience=100)

**XGBoost** — 100 trials over the same structural parameters plus:
- gamma (minimum loss reduction): [0, 5.0]
- min_child_weight: [1, 50]

**CatBoost** — 75 trials over:
- depth: [4, 10]
- learning_rate: [0.005, 0.1]
- l2_leaf_reg: [0.001, 10.0]
- border_count: [32, 255]
- bagging_temperature: [0, 5.0]
- random_strength: [0.1, 10.0]

Early stopping was used across all models to automatically determine the optimal number of trees per trial, preventing both over- and under-fitting.

### Step 6: Stacking Ensemble

V1's soft voting ensemble of LightGBM + XGBoost failed to improve over LightGBM alone because the two models are architecturally similar (both histogram-based gradient boosting) and produce correlated predictions. Simply averaging correlated predictions doesn't add value.

V2 addresses this with two changes:

**1. Adding CatBoost for genuine model diversity.** CatBoost uses ordered target encoding for categorical features and ordered boosting (a permutation-based approach that reduces prediction shift). This is fundamentally different from the histogram-based splitting in LightGBM/XGBoost, producing less correlated predictions.

**2. Stacking instead of voting.** Rather than averaging predictions, stacking uses a two-level architecture:

- **Level 0 (base models):** Each of the three tuned models generates out-of-fold (OOF) predictions using 5-fold CV. For each passenger in the training set, the prediction comes from the fold where that passenger was in the validation set — ensuring no data leakage.
- **Level 1 (meta-learner):** A logistic regression is trained on the three OOF probability columns to predict the target. The meta-learner learns the optimal weighting of each model and can capture situations where, for example, "when LightGBM predicts 0.6 but CatBoost predicts 0.3, the true label is usually 0."

For test predictions, each base model's test probabilities (averaged across all 5 folds) are fed through the trained meta-learner.

As a comparison, optimized weighted averaging (grid search over weight combinations) was also evaluated.

### Step 7: Classification Threshold Optimization

V1 used the default 0.5 threshold for converting probabilities to binary predictions. With a 54.7% / 45.3% class distribution and accuracy as the evaluation metric, the optimal threshold may differ from 0.5.

V2 sweeps thresholds from 0.40 to 0.60 in 0.005 increments on the OOF predictions and selects the threshold maximizing training accuracy. Because this is evaluated on OOF predictions (not in-sample predictions), the threshold generalizes to unseen data.

### Step 8: Seed Averaging

A single model training run contains randomness from:
- Data shuffling in cross-validation fold creation
- Row and column subsampling within each tree
- Random initialization in optimization

V2 retrains the entire stacking pipeline with 5 different random seeds (42, 123, 456, 789, 2024) and averages the test set probabilities across all seeds before applying the optimized threshold. This reduces prediction variance and stabilizes the final output — a prediction that fluctuates between 0.48 and 0.52 across seeds gets smoothed to a more reliable value.

---

## Expected Contribution of Each Improvement

| Technique | Expected Gain | Mechanism |
|---|---|---|
| Optuna hyperparameter tuning | +0.10–0.30% | Finds better parameter combinations than manual selection |
| CatBoost + stacking | +0.05–0.15% | Model diversity + learned blending > simple averaging |
| Enhanced feature engineering | +0.05–0.10% | Missing indicators and interactions provide explicit signal trees would need many splits to learn |
| Threshold optimization | +0.00–0.10% | Adjusts for class imbalance in accuracy metric |
| Seed averaging | +0.00–0.05% | Reduces variance, stabilizes borderline predictions |

These gains are not perfectly additive — some overlap. The combined target is >95.85% test accuracy.

---

## Tools and Libraries

- Python 3, pandas, NumPy, SciPy, scikit-learn
- LightGBM, XGBoost, CatBoost — three gradient boosting frameworks as base models
- Optuna — Bayesian hyperparameter optimization
- JupyterLab — interactive development in `shinkansen_v2.ipynb`
