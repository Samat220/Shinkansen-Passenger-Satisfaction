# Shinkansen Passenger Satisfaction Prediction

## Problem Statement

Predict whether a Shinkansen Bullet Train passenger was satisfied (1) or not satisfied (0) based on travel attributes and post-service survey responses. Evaluation metric is accuracy.

## Dataset Overview

Two source datasets per split (train/test):

- **Travel Data** — passenger demographics and trip logistics: Gender, Customer_Type, Age, Type_Travel, Travel_Class, Travel_Distance, Departure_Delay_in_Mins, Arrival_Delay_in_Mins
- **Survey Data** — 14 post-service satisfaction ratings (ordinal) plus Seat_Class

Training set: 94,379 rows. Test set: 35,602 rows. Target split: ~54.7% satisfied, ~45.3% not satisfied (mild imbalance, not enough to require resampling).

---

## Approach

### Step 1: Data Merging

Inner join of travel and survey data on the `ID` column, producing a single feature matrix per passenger with 24 raw features.

### Step 2: Encoding Categorical Features

**Ordinal survey ratings** were mapped to integers preserving their natural ranking:

| Value | Encoded |
|---|---|
| Extremely Poor | 0 |
| Poor | 1 |
| Needs Improvement | 2 |
| Acceptable | 3 |
| Good | 4 |
| Excellent | 5 |

Platform_Location used its own scale: Very Inconvenient (0) → Inconvenient (1) → Needs Improvement (2) → Manageable (3) → Convenient (4) → Very Convenient (5).

**Nominal categoricals** (Gender, Customer_Type, Type_Travel, Travel_Class, Seat_Class) were label-encoded. Since the final model is tree-based, arbitrary integer assignment doesn't introduce false ordinal relationships — trees split on thresholds, so they effectively treat each value independently.

### Step 3: Missing Value Strategy

No explicit imputation was performed. Both LightGBM and XGBoost handle NaN natively by learning optimal split directions for missing values during training. This approach was chosen because:

- Some columns had significant missingness (Customer_Type ~9.5%, Type_Travel ~9.8%, Catering ~9.3%), where the fact that a value is missing may itself be informative.
- Mean/median imputation can introduce noise and bias the distribution.
- The models' native handling has been shown to outperform simple imputation in most tabular data benchmarks.

### Step 4: Feature Engineering

Engineered features derived from the 14 ordinal survey columns:

| Feature | Description | Rationale |
|---|---|---|
| `survey_mean` | Mean of all survey ratings | Overall satisfaction summary |
| `survey_std` | Standard deviation of ratings | Rating consistency — high variance suggests mixed experience |
| `survey_min` | Minimum rating given | Captures worst pain point |
| `survey_max` | Maximum rating given | Captures best aspect of experience |
| `survey_range` | Max minus min rating | Spread of experience quality |
| `high_ratings_count` | Count of ratings ≥ 4 (Good/Excellent) | How many aspects were positively rated |
| `low_ratings_count` | Count of ratings ≤ 1 (Poor/Extremely Poor) | How many aspects were negatively rated |
| `total_delay` | Departure delay + arrival delay | Cumulative delay impact |
| `delay_diff` | Arrival delay − departure delay | Whether delays compounded or were recovered |
| `has_delay` | Binary: any delay > 0 | Simple delay indicator |

The aggregate survey features (survey_mean, survey_std) ranked among the top 5 most important features, confirming they added signal beyond individual ratings.

### Step 5: Model Selection and Training

Three models were evaluated using 5-fold stratified cross-validation:

| Model | CV Accuracy | CV Std |
|---|---|---|
| **LightGBM** | **95.62%** | ±0.22% |
| XGBoost | 95.38% | ±0.16% |
| Soft Voting Ensemble (LGB + XGB) | 95.57% | ±0.17% |

LightGBM was selected as the final model. The ensemble did not improve over LightGBM alone because the two models are architecturally similar and produce correlated predictions.

### Step 6: Hyperparameters

Final LightGBM configuration:

```
n_estimators: 2000
learning_rate: 0.03
max_depth: 7
num_leaves: 63
subsample: 0.8
colsample_bytree: 0.8
min_child_samples: 30
reg_alpha: 0.1
reg_lambda: 0.1
```

Design decisions:

- **High estimator count with low learning rate** — more trees at smaller steps reduces overfitting and improves generalization vs. fewer aggressive trees.
- **Depth 7 / 63 leaves** — enough complexity to capture feature interactions (e.g., business travelers weighting different survey aspects than personal travelers) without memorizing noise.
- **80% row and column subsampling** — randomization per tree that acts as regularization.
- **L1 and L2 regularization (0.1 each)** — additional overfitting control on leaf weights.

### Step 7: Feature Importance (Top 15)

1. Travel_Distance
2. Age
3. survey_std
4. survey_mean
5. Arrival_Time_Convenient
6. delay_diff
7. Catering
8. Onboard_Service
9. Seat_Comfort
10. Type_Travel
11. Platform_Location
12. total_delay
13. high_ratings_count
14. Arrival_Delay_in_Mins
15. Legroom

Travel_Distance and Age dominate, suggesting passenger profile strongly influences satisfaction. The engineered survey aggregates (survey_std, survey_mean) outperformed most individual survey columns, validating the feature engineering step.

---

## Result

**Cross-validation accuracy: 95.62%** on the training set with 5-fold stratified validation.

---

## Potential Improvements

### 1. Hyperparameter Optimization with Optuna

The current hyperparameters were manually selected based on common defaults for tabular data. A systematic Bayesian search using Optuna over learning rate, tree depth, leaf count, regularization, and subsampling would likely squeeze out 0.1–0.3% additional accuracy. Early stopping on a validation fold would also prevent unnecessary estimators.

### 2. Stacking Ensemble

Instead of simple voting, a stacking approach would train LightGBM, XGBoost, and CatBoost as base models, then feed their out-of-fold predictions into a logistic regression or shallow neural network meta-learner. This captures complementary model strengths more effectively than averaging. CatBoost in particular handles categorical features differently (ordered target encoding) and would add prediction diversity.

### 3. Target Encoding for High-Cardinality Interactions

Creating interaction features like `Type_Travel × Travel_Class` or `Customer_Type × Seat_Class` with target encoding (mean of target per category, with regularization) could capture segment-specific satisfaction patterns that trees would otherwise need many splits to learn.

### 4. More Sophisticated Missing Value Treatment

While native NaN handling works well, explicitly modeling missingness could help. Options include adding binary "is_missing" indicator columns for high-missingness features, or using iterative imputation (MICE) that models each missing feature as a function of the others. The ~10% missingness in Customer_Type, Type_Travel, and several survey columns suggests these aren't random — they may indicate a distinct passenger segment (e.g., didn't complete the survey).

### 5. Additional Feature Engineering

- **Delay-to-distance ratio** — a 30-minute delay on a 200km trip matters more than on a 2000km trip.
- **Age buckets interacted with survey features** — older vs. younger passengers may have different satisfaction thresholds.
- **PCA or factor analysis on survey columns** — reduce the 14 correlated survey features into 3–4 latent factors (e.g., "service quality," "convenience," "digital experience") that may generalize better.
- **Cluster-based features** — k-means clustering on survey responses to identify passenger archetypes, then using cluster membership as a feature.

### 6. Neural Network Approaches

TabNet or FT-Transformer (transformer architecture adapted for tabular data) have shown competitive results with gradient-boosted trees on some benchmarks. They could be added to a stacking ensemble to increase model diversity. These are particularly strong when there are complex non-linear interactions between features.

### 7. Calibrated Probability Thresholding

The current model uses 0.5 as the classification threshold. Since accuracy is the metric and the classes are mildly imbalanced (54.7% / 45.3%), optimizing the threshold on validation data could improve accuracy by 0.1–0.2% if the predicted probability distribution is asymmetric.

### 8. Pseudo-Labeling

Using confident predictions (probability > 0.95 or < 0.05) on the test set as additional training data, then retraining. This semi-supervised approach can improve accuracy when the unlabeled test set is large relative to the training set, which is somewhat the case here (35k test vs. 94k train).

---

## Tools and Libraries

- Python 3, pandas, NumPy, scikit-learn
- LightGBM for the primary model
- XGBoost for comparison and ensemble candidate
