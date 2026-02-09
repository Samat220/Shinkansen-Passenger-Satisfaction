# Shinkansen Bullet Train — Passenger Satisfaction Prediction

**2nd Place Solution** | MIT Applied Data Science Program — Machine Learning Hackathon

## Overview

Binary classification model predicting whether Shinkansen (bullet train) passengers are satisfied with their overall travel experience, using travel attributes and post-service survey responses.

**Test Accuracy: 95.78%** (2nd out of all participants, 0.07% behind 1st place)

## Approach

- **51 engineered features** from 24 raw columns: survey aggregates (mean, std, skew), delay ratios, log-transforms, missing indicators, categorical interactions
- **3-model stacking ensemble**: Optuna-tuned LightGBM + XGBoost + CatBoost with optimized weighted blending
- **CatBoost native categorical handling** via ordered target encoding for 9 categorical columns
- **7-fold stratified CV** with seed averaging across 5 seeds (105 total model fits)

## Results

| Version | Approach | Test Accuracy |
|---------|----------|---------------|
| V1 | Single LightGBM, manual params | 95.56% |
| V2 | Optuna-tuned 3-model stacking | 95.75% |
| V6 | + Bug fixes, cat_features, refined features | **95.78%** |

## Project Structure

```
├── shinkansen_v6.ipynb           # Main notebook (final solution)
├── data/
│   ├── Traveldata_train_(1).csv  # Travel demographics & logistics (train)
│   ├── Traveldata_test_(1).csv   # Travel demographics & logistics (test)
│   ├── Surveydata_train_(1).csv  # Survey ratings + target (train)
│   ├── Surveydata_test_(1).csv   # Survey ratings (test)
│   └── Data_Dictionary_(1).xlsx  # Feature definitions
├── submissions/                  # Competition submission CSVs
├── notebooks/archive/            # Prior notebook versions (V2, V4, V5)
└── docs/                         # Approach documentation
```

## Dataset

- **Training:** 94,379 passengers, 24 features + target
- **Test:** 35,602 passengers, 24 features (target hidden)
- **Target:** `Overall_Experience` — binary (1 = satisfied, 0 = not satisfied), 54.7% / 45.3% split
- **Sources:** Travel data (demographics, trip logistics, delays) and Survey data (14 ordinal satisfaction ratings + Seat Class), joined on passenger ID

## Tech Stack

Python 3, pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, matplotlib, seaborn

GPU acceleration: XGBoost and CatBoost via CUDA

## Running

```bash
# Requires Python 3.8+ with the above libraries installed
# Set USE_GPU = False in the notebook if no CUDA GPU is available
jupyter lab shinkansen_v6.ipynb
```

The notebook runs end-to-end in ~45 minutes (3 models x 5 seeds x 7 folds = 105 model fits).

## Author

Samat Ramazan — February 2026
