# Technical Handoff Document: Stage 1 to Stage 2

**From:** Anish (Stage 1 Lead — EDA, Preprocessing Pipeline, KNN & Decision Tree)  
**To:** Shubh (Stage 2 Lead — Support Vector Machines & Random Forest)  
**CC:** Dev (Stage 3 Lead — Holdout Test Evaluation, Model Ensembling & Deployment)  
**Project:** Credit Card Fraud Detection (BTech ML Mini-Project)  
**Date:** October 2026  
**Status:** Stage 1 Complete, Fully Verified, and Runnable  

---

## 1. Welcome & Stage 1 Deliverables Summary

Hey Shubh, I have completed all Stage 1 tasks. The foundation of our project is locked in, strictly leakage-free, and fully reproducible. All source code, preprocessed datasets, serialized scalers, baseline models, figures, and technical reports are ready in the shared repository.

### What is Completed:
1. **Exhaustive EDA**:
   - Analyzed all 1.85 million transactions.
   - Verified **0 missing values** and **0 duplicate records**.
   - Profiled the severe **0.58% class imbalance** (1 fraud per 171 genuine transactions).
   - Documented the **10x fraud surge during nocturnal hours** (22:00 to 03:00) and the **7.3x median amount discrepancy**.
2. **Leakage-Safe `FraudPreprocessor`**:
   - Engineered 23 features: Haversine distance, cardholder age, log amounts, temporal signals, and 14 one-hot merchant categories.
   - Fitted strictly on $X_{train}$ and serialized to `models/preprocessor.joblib`.
3. **Reproducible Data Splits**:
   - Preprocessed and saved to `data/processed/`.
   - The official Kaggle test set (`fraudTest.csv`, 555,719 rows) is preserved untouched for Dev's Stage 3 evaluation.
4. **Baseline Models Implemented & Tuned**:
   - **KNN ($k=3$, SMOTE-balanced)**: F1 = 0.4541, Recall = 54.02%, Precision = 39.17%.
   - **Decision Tree (Gini, Pruned)**: F1 = 0.6244, Recall = 73.56%, Precision = 54.24%.

---

## 2. Directory Structure & File Manifest

Here is how our project workspace is organized. Please preserve this structure so the repository remains runnable:

```
ML_Project/
├── data/
│   ├── raw/
│   │   ├── fraudTrain.csv            # Raw training dataset (1,296,675 rows)
│   │   └── fraudTest.csv             # Raw holdout test dataset (555,719 rows, preserved for Dev)
│   └── processed/
│       ├── train_processed.joblib     # Preprocessed (X_train, y_train): 60,000 rows, 23 features
│       ├── val_processed.joblib       # Preprocessed (X_val, y_val): 15,000 rows, 23 features
│       ├── test_processed.joblib      # Preprocessed (X_test, y_test): 15,000 rows, 23 features
│       └── dataset_metadata.json      # Metadata on split sizes, fraud counts, feature names
├── models/
│   ├── preprocessor.joblib           # Fitted FraudPreprocessor instance (StandardScaler + OneHotEncoder)
│   ├── knn_best_model.joblib         # Optimal KNN classifier (k=3, weights='distance')
│   ├── dt_best_model.joblib          # Optimal Decision Tree classifier (Gini, min_samples_leaf=5)
│   └── stage1_results.json           # All validation metrics, tuning histories, and metadata
├── reports/
│   ├── figures/                      # 10 publication-quality PNG charts
│   ├── eda_summary.txt               # Raw terminal output of head(), tail(), info(), describe()
│   ├── data_dictionary.md            # Detailed schema and domain justifications
│   ├── stage1_report.md              # Complete engineering report and 10 viva Q&As
│   └── handoff_to_shubh.md           # This document
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── eda.py                        # Reusable EDA and profiling functions
│   ├── preprocessing.py              # Reusable FraudPreprocessor and splitting functions
│   ├── models.py                     # Training, SMOTE, tuning, and evaluation logic
│   └── visualization.py              # Publication-grade plotting functions
├── notebooks/
│   └── 01_stage1_eda_and_baseline_models.ipynb  # Interactive Jupyter notebook for viva defense
├── run_stage1.py                     # Master execution script (runs end-to-end in ~35s)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and viva overview
```

---

## 3. How to Load and Use Preprocessed Data

You do **not** need to re-parse raw CSVs, compute Haversine distances, or fit scalers from scratch. The data is already preprocessed, scaled, encoded, and saved as serialized `joblib` tuples:

```python
import joblib

# Load preprocessed splits
X_train, y_train = joblib.load("data/processed/train_processed.joblib")
X_val, y_val = joblib.load("data/processed/val_processed.joblib")
X_test, y_test = joblib.load("data/processed/test_processed.joblib")

print(f"X_train shape: {X_train.shape}")  # (60000, 23)
print(f"X_val shape:   {X_val.shape}")    # (15000, 23)
print(f"Features: {X_train.columns.tolist()}")
```

### Feature List (23 Columns):
```python
[
    'log_amt',                # Continuous, StandardScaled
    'trans_hour',             # Continuous, StandardScaled
    'trans_day_of_week',      # Continuous, StandardScaled
    'age',                    # Continuous, StandardScaled
    'dist_to_merchant_km',    # Continuous, StandardScaled
    'log_city_pop',           # Continuous, StandardScaled
    'gender_binary',          # Binary (1 = M, 0 = F)
    'is_night',               # Binary (1 = 22h-03h window, 0 = daytime)
    'is_weekend',             # Binary (1 = Sat/Sun, 0 = Mon-Fri)
    'category_entertainment', # Binary One-Hot
    'category_food_dining',
    'category_gas_transport',
    'category_grocery_net',
    'category_grocery_pos',
    'category_health_fitness',
    'category_home',
    'category_kids_pets',
    'category_misc_net',
    'category_misc_pos',
    'category_personal_care',
    'category_shopping_net',
    'category_shopping_pos',
    'category_travel'
]
```

### How to Transform New / Raw Data:
If you or Dev receive new raw transaction data, simply load the fitted preprocessor:
```python
from src.preprocessing import FraudPreprocessor

# Load fitted preprocessor
preprocessor = FraudPreprocessor.load("models/preprocessor.joblib")

# Transform raw DataFrame into the 23-feature matrix
X_new_processed = preprocessor.transform(df_raw_new)
```

---

## 4. Stage 1 Baseline Benchmarks (Targets for Stage 2)

Your models in Stage 2 (**Support Vector Machine** and **Random Forest**) should aim to outperform our Stage 1 baselines on the validation set ($N = 15,000$, 87 frauds):

| Metric | KNN ($k=3$, SMOTE) | Decision Tree (Gini) | Target for SVM | Target for Random Forest |
|---|---|---|---|---|
| **Accuracy** | 99.25% | 99.49% | > 99.30% | > 99.60% |
| **Precision** | 39.17% | 54.24% | > 45.00% | **> 70.00%** |
| **Recall** | 54.02% | 73.56% | > 65.00% | **> 75.00%** |
| **F1-Score** | 0.4541 | 0.6244 | > 0.5500 | **> 0.7500** |
| **ROC-AUC** | 0.7903 | 0.8672 | > 0.8800 | **> 0.9500** |
| **PR-AUC** | 0.3099 | 0.6898 | > 0.5000 | **> 0.7500** |
| **True Positives** | 47 / 87 | 64 / 87 | > 55 / 87 | **> 65 / 87** |
| **False Positives**| 73 | 54 | < 70 | **< 30** |

---

## 5. Technical Instructions for Stage 2 Implementation

### 5.1 Model 1: Support Vector Machine (SVM)
The syllabus requires implementing and tuning an SVM classifier. Keep the following technical points in mind:

1. **Computational Complexity ($O(N^2)$ to $O(N^3)$)**:
   - Kernel SVM (RBF) computes an $N \times N$ Gram matrix. On 60,000 samples, an RBF SVM takes approximately 1 to 2 minutes. On 1.3 million samples, it will run out of memory or take days.
   - **Recommendation**: Train on the provided 60,000 preprocessed training samples (`train_processed.joblib`), or for fast hyperparameter grid search, use a 25,000 stratified subset.
2. **Kernel Selection**:
   - Start with `LinearSVC(class_weight='balanced')` for a fast linear baseline.
   - Then implement `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)` for non-linear decision boundaries.
3. **Hyperparameters to Tune**:
   - Regularization parameter $C \in [0.1, 1.0, 10.0]$: Controls the trade-off between margin width and margin violations.
   - Kernel coefficient $\gamma \in ['scale', 'auto', 0.01, 0.1]$: Defines the radius of influence of support vectors.
   - `class_weight`: Use `'balanced'` to prevent the hyperplane from collapsing to the majority class.
4. **Feature Scaling Requirement**:
   - SVM is a margin-maximization algorithm; distance to the separating hyperplane depends on Euclidean norms. Because our `FraudPreprocessor` has already standardized continuous features, the data is already scaled for SVM.

---

### 5.2 Model 2: Random Forest Classifier
Random Forest is an ensemble of bagging decision trees with feature subspace randomization:

1. **Why It Should Beat Decision Tree**:
   - Single decision trees suffer from high variance and sensitivity to training data noise.
   - Random Forest aggregates $B$ decorrelated decision trees, reducing variance by $\frac{1}{B} \text{Var}(T)$.
2. **Hyperparameters to Tune**:
   - `n_estimators` $\in [100, 200, 300]$: Number of trees in the forest.
   - `max_depth` $\in [10, 15, 20, \text{None}]$: Individual tree depth.
   - `class_weight` $\in ['balanced', 'balanced_subsample']$: `'balanced_subsample'` recalculates weights per bootstrap sample.
   - `max_features` $\in ['sqrt', 'log2']$: Number of features evaluated at each split.
3. **Feature Importance Interpretation**:
   - Extract `.feature_importances_` and map them to `X_train.columns` to present the top fraud predictors during your viva defense.

---

## 6. Ready-to-Run Starter Script for Stage 2

Here is a template you can save as `src/stage2_models.py` or run directly in Python to implement your models and log metrics in the exact same format as Stage 1:

```python
"""
Stage 2 Starter Script: SVM and Random Forest
Author: Shubh (Stage 2 Lead)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Import Stage 1 shared modules
from src.models import evaluate_model_performance
from src.visualization import plot_confusion_matrices, plot_roc_and_pr_curves

# 1. Load preprocessed data
print("Loading preprocessed datasets...")
X_train, y_train = joblib.load("data/processed/train_processed.joblib")
X_val, y_val = joblib.load("data/processed/val_processed.joblib")

# 2. Train and Evaluate SVM
print("\n--- Training Support Vector Machine (RBF Kernel) ---")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm_model.fit(X_train, y_train)

svm_preds = svm_model.predict(X_val)
svm_proba = svm_model.predict_proba(X_val)[:, 1]
svm_metrics = evaluate_model_performance(y_val, svm_preds, svm_proba)
print(f"SVM Validation F1: {svm_metrics['f1']:.4f} | Recall: {svm_metrics['recall']:.4f} | Precision: {svm_metrics['precision']:.4f}")

# 3. Train and Evaluate Random Forest
print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_val)
rf_proba = rf_model.predict_proba(X_val)[:, 1]
rf_metrics = evaluate_model_performance(y_val, rf_preds, rf_proba)
print(f"RF Validation F1:  {rf_metrics['f1']:.4f} | Recall: {rf_metrics['recall']:.4f} | Precision: {rf_metrics['precision']:.4f}")

# 4. Save Models
joblib.dump(svm_model, "models/svm_best_model.joblib")
joblib.dump(rf_model, "models/rf_best_model.joblib")
print("\nStage 2 models saved successfully to models/")
```

---

## 7. Viva Voce Preparation for Stage 2

Examiners will likely ask you how your models compare with Anish's Stage 1 models. Here are the expected talking points:
1. **SVM vs. KNN**:
   - Both are geometric classifiers, but KNN memorizes all training data and computes $O(N)$ distances at inference time. SVM finds a global optimal separating hyperplane defined exclusively by **support vectors**, discarding all non-boundary training points at inference time.
2. **Random Forest vs. Decision Tree**:
   - Single decision trees overfit and exhibit high variance. Random Forest uses **bootstrap aggregation (bagging)** and **random feature subspace selection** to reduce variance without increasing bias.
3. **Class Weighting (`balanced` vs `balanced_subsample`)**:
   - Explain that `balanced_subsample` dynamically re-weights samples for each individual bootstrap draw, making it robust against bootstrap sampling variation.

Good luck with Stage 2, Shubh! Let me know if you need any adjustments to the preprocessing pipeline.
