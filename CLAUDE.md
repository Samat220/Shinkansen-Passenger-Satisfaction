# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary classification project for the MIT Applied Data Science Program. Goal: predict Shinkansen bullet train passenger satisfaction (1 = satisfied, 0 = not satisfied) from travel attributes and post-service survey responses. Evaluation metric is **accuracy**.

## Project Structure

```
├── shinkansen_v6.ipynb          # Active notebook
├── CLAUDE.md
├── data/                        # All raw data files
│   ├── Traveldata_train_(1).csv / Traveldata_test_(1).csv
│   ├── Surveydata_train_(1).csv / Surveydata_test_(1).csv
│   ├── Sample_Submission_(1).csv
│   └── Data_Dictionary_(1).xlsx
├── submissions/                 # All submission CSVs (submission_1.csv ... submission_N.csv)
├── notebooks/archive/           # Prior notebook versions (v2, v4, v5)
└── docs/                        # Approach documentation (shinkansen_approach.md, _v2.md)
```

## Dataset Structure

Four CSV data files in `data/` join on the `ID` column:

- `Traveldata_train_(1).csv` / `Traveldata_test_(1).csv` — demographics and trip logistics (Gender, Customer_Type, Age, Type_Travel, Travel_Class, Travel_Distance, delays)
- `Surveydata_train_(1).csv` / `Surveydata_test_(1).csv` — 14 ordinal satisfaction ratings + Seat_Class; train set includes `Overall_Experience` target
- `Sample_Submission_(1).csv` — expected output format: `ID, Overall_Experience`
- `Data_Dictionary_(1).xlsx` — field definitions

Training: 94,379 rows. Test: 35,602 rows. Class split: ~54.7% satisfied / ~45.3% not satisfied.

## Version History & Leaderboard Results

| Version | Notebook | Approach | Test Accuracy |
|---------|----------|----------|---------------|
| V1 | — | Single LightGBM, manual params | 0.9556 (3rd) |
| V2 | `notebooks/archive/shinkansen_v2.ipynb` | Optuna-tuned LGB+XGB+CatBoost stacking, seed averaging | 0.9575 (2nd) |
| V4 | `notebooks/archive/shinkansen_v4.ipynb` | V2 + pseudo-labeling | 0.9570 (worse — pseudo-labeling hurt) |
| V5 | `notebooks/archive/shinkansen_v5.ipynb` | V2 + ExtraTrees 4th model, 7-fold CV | Abandoned (ET caused OOM freeze, 0.9498 OOF) |
| V6 | `shinkansen_v6.ipynb` | Bug fixes + CatBoost cat_features + refined features, 7-fold | 0.9578 (2nd, new best) |

1st place target: **0.9585**. V6 OOF accuracy: **0.958434**.

## Established Pipeline

Documented in `docs/shinkansen_approach.md` (V1) and `docs/shinkansen_approach_v2.md` (V2). The pipeline is:

1. **Merge** travel + survey data via inner join on `ID`
2. **Ordinal encoding** for survey ratings (Extremely Poor=0 through Excellent=5); label encoding for nominal categoricals; restore NaN after label encoding
3. **No explicit imputation** — LightGBM/XGBoost/CatBoost handle NaN natively
4. **Engineered features** (V6: 51 total): survey aggregates (mean, std, min, max, range, median, skew), rating counts (high/low/mid), delay features (total_delay, delay_diff, has_delay, delay_to_distance, log_total_delay, log_arrival_delay, log_departure_delay), age_bin (decile quantiles, edges from train only), is_missing_* indicators, n_missing_total, interaction features (TypeTravel×TravelClass, CustType×SeatClass, Gender×TypeTravel, Gender×TravelClass)
5. **Optuna hyperparameter tuning** (100 trials LGB, 100 XGB, 75 CatBoost) — best params hardcoded in V6 notebook to avoid re-running the 2+ hour optimization
6. **Stacking ensemble** with weighted average blending (optimized via grid search on OOF predictions). V6 best weights: LGB=0.36, XGB=0.35, CB=0.29. Weighted average outperformed logistic regression meta-learner.
7. **Threshold optimization** — sweep 0.40–0.60 on OOF predictions; V6 optimal threshold landed at 0.500 (class imbalance not enough to shift it)
8. **Seed averaging** — 5 seeds (42, 123, 456, 789, 2024), full pipeline retrained per seed, test probabilities averaged

## Key Implementation Details

### Survey columns definition
The `all_survey_cols` list includes **14 survey columns + Platform_Location** (15 total). Platform_Location uses a different ordinal scale (Very Inconvenient→Very Convenient) than the other 13 survey columns (Extremely Poor→Excellent). Both are mapped to 0–5. All 15 columns are used in the survey aggregate features (mean, std, etc.).

### Encoding subtlety
After label encoding nominal categoricals, NaN values are **restored** from the original data. This is because `LabelEncoder` converts NaN to a string "nan" and encodes it as an integer — the pipeline explicitly sets those positions back to `np.nan` so the tree models use their native NaN handling.

### CatBoost cat_features (V6)
CatBoost receives a DataFrame (not float32 numpy) with 9 categorical columns as int64 (NaN → -1 sentinel): Gender, Customer_Type, Type_Travel, Travel_Class, Seat_Class, plus 4 interaction features. CatBoost v1.2.8 requires non-float dtype for `cat_features` — float numpy arrays are rejected. GPU + cat_features works. LGB/XGB continue using float32 numpy arrays.

### age_bin fix (V6)
V2-V5 had a bug: `pd.qcut` was called independently on train and test, producing different bin edges. V6 computes edges on train with `pd.qcut(..., retbins=True)`, then applies to both with `pd.cut()`.

### Submission naming
V6 outputs submissions to `submissions/` directory, auto-detecting existing `submission_*.csv` files and incrementing the number.

## What Didn't Work

- **Pseudo-labeling** (V4): Adding high-confidence test predictions (>95% probability) as training data amplified errors; test accuracy dropped from 0.9575 to 0.9570
- **Logistic regression meta-learner** for stacking: Weighted average blending consistently outperformed it (0.9578 vs 0.9574 OOF)
- **Simple voting ensemble** (V1): LGB+XGB soft voting didn't beat LGB alone because predictions are too correlated
- **ExtraTrees** (V5): 0.9498 OOF — 1.5% below GBDTs, caused system OOM/freeze (2000 trees × depth 30 × 16 workers > 30GB RAM), would get near-zero blend weight

## GPU Configuration

XGBoost and CatBoost use GPU (`device='cuda'` / `task_type='GPU'`) when `USE_GPU=True`. LightGBM runs on CPU (pip build lacks CUDA support). Set `USE_GPU = False` if no CUDA-capable GPU is available.

## Tech Stack

Python 3, pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, matplotlib, seaborn. Code is in Jupyter notebooks.

No build system, package manager lockfile, or test suite — this is a data science analysis project.

## Running the Notebooks

- Notebooks are cell-order-dependent and take significant time to run end-to-end (seed averaging with 3 models × 5 seeds × 7 folds = 105 model fits, ~45 min total)
- The latest active notebook is `shinkansen_v6.ipynb` (in project root). V2, V4, V5 are in `notebooks/archive/` for reference.
- GBDT hyperparameters are hardcoded as `BEST_LGB_PARAMS`, `BEST_XGB_PARAMS`, `BEST_CB_PARAMS` dicts at the top of V6 — these came from V2's Optuna runs and should not be re-tuned without reason
- V6 has a CatBoost Optuna re-tuning cell (cell 9) that was partially run (18/75 trials) but not completed. The best trial found (0.957681 OOF) didn't beat the current params. The cell output persists in the notebook — it can be re-run or skipped
- Individual model OOF accuracies (V6, seed=42): LGB=0.957660, XGB=0.957416, CB=0.956950

## Important Notes

- Data file names contain parentheses — quote or escape them in shell commands: `data/Surveydata_train_\(1\).csv`
- Notable missing values: Customer_Type (~9.5%), Type_Travel (~9.8%), Catering (~9.3%), several survey columns
- Top features by importance: Travel_Distance, Age, survey_std, survey_mean, Arrival_Time_Convenient
- Submissions go in `submissions/` directory
- Consult the approach docs (`docs/`) before proposing new modeling directions — many ideas have already been tried or are documented as potential improvements
