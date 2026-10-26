# Credit Card Fraud Detection — Machine Learning Mini-Project

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Kartik%20Shenoy-green.svg)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
[![Status](https://img.shields.io/badge/Stage%201-Completed%20%26%20Verified-brightgreen.svg)]()

A modular, reproducible machine learning group mini-project designed for BTech Computer Science & Engineering. The project detects fraudulent credit card transactions under extreme class imbalance using the simulated dataset by Kartik Shenoy on Kaggle.

---

## 👥 Project Team & Work Breakdown

| Stage | Team Lead | Responsibilities & Models | Status | Deliverables |
|---|---|---|---|---|
| **Stage 1** | **Anish** | EDA, Data Cleaning, Outliers & Missing Values, Feature Engineering, Leakage-Safe Pipeline, Data Splits, **K-Nearest Neighbors (KNN)**, **Decision Tree (ID3 / CART)** | **Completed** ✅ | `src/eda.py`, `src/preprocessing.py`, `src/models.py`, `src/visualization.py`, `run_stage1.py`, 10 Figures, Reports, Jupyter Notebook |
| **Stage 2** | **Shubh** | Stage 1 Integration, **Support Vector Machine (SVM)** (Linear & RBF Kernels), **Random Forest Ensemble**, Hyperparameter Tuning | **Handoff Ready** 🚀 | `reports/handoff_to_shubh.md`, Preprocessed Splits (`data/processed/`) |
| **Stage 3** | **Dev** | Final Out-of-Time Test Set Evaluation (`fraudTest.csv`, 555k rows), Model Comparison, Ensemble/Stacking, Final Viva Presentation | **Upcoming** ⏳ | Preserved Test Set, Final Submission |

---

## 📊 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset (Kartik Shenoy)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Total Transactions**: 1,852,394 transactions across 23 raw attributes.
  - `data/raw/fraudTrain.csv`: 1,296,675 transactions (Jan 2019 – June 2020).
  - `data/raw/fraudTest.csv`: 555,719 transactions (July 2020 – Dec 2020), preserved as the final held-out test set.
- **Class Imbalance**: Severe **0.58% fraud prevalence** (1 fraud per 171 legitimate charges).
- **Missing Values**: 0 missing entries across all columns (100% complete).
- **Duplicates**: 0 duplicate transaction hashes.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Stage 1 End-to-End
Run the complete pipeline (EDA, leakage-safe splits, model tuning, plotting, and metric calculation):
```bash
python run_stage1.py
```
*Execution takes approximately 35 seconds.*

---

## 📈 Stage 1 Empirical Results (Validation Set: N = 15,000, 87 Frauds)

| Metric | K-Nearest Neighbors ($k=3$, SMOTE) | Decision Tree (Gini, Pruned) | Preferred Model |
|---|---|---|---|
| **Accuracy** | 99.25% | **99.49%** | Decision Tree |
| **Precision** | 39.17% | **54.24%** (+15.07%) | Decision Tree |
| **Recall (Sensitivity)** | 54.02% | **73.56%** (+19.54%) | Decision Tree |
| **F1-Score** | 0.4541 | **0.6244** (+0.1703) | Decision Tree |
| **Specificity** | 99.51% | **99.64%** | Decision Tree |
| **ROC-AUC** | 0.7903 | **0.8672** | Decision Tree |
| **PR-AUC (Average Precision)**| 0.3099 | **0.6898** | Decision Tree |
| **Frauds Caught (TP)** | 47 / 87 | **64 / 87** | Decision Tree |
| **False Alarms (FP)** | 73 | **54** | Decision Tree |

> **⚠️ The Accuracy Paradox:** Both models achieve >99.2% accuracy. A dummy classifier predicting zero fraud would achieve 99.42% accuracy with 0% recall. Therefore, **F1-Score, Recall, and PR-AUC** are our primary evaluation criteria.

---

## 🛠️ Feature Engineering & Pipeline

Our custom `FraudPreprocessor` (`src/preprocessing.py`) creates **23 predictive features** with zero data leakage:

1. **Haversine Distance (`dist_to_merchant_km`)**: Calculates spherical great-circle distance between customer residence coordinates $(\text{lat}, \text{long})$ and merchant store coordinates $(\text{merch\_lat}, \text{merch\_long})$.
2. **Circadian Anomaly (`is_night`)**: Flags transactions occurring between 10:00 PM and 3:00 AM (which exhibit a 10x surge in fraud prevalence).
3. **Log Monetary Amount (`log_amt`)**: Applies $\log(1 + \text{amt})$ to eliminate severe right skew.
4. **Cardholder Age (`age`)**: Computed from transaction date and customer date of birth (`dob`).
5. **StandardScaler**: Fitted **strictly on $X_{train}$** and applied to $X_{val}$ and $X_{test}$ (prevents data leakage).
6. **OneHotEncoder**: Encodes the 14 merchant categories with unseen category handling.

---

## 📂 Repository Layout

```
ML_Project/
├── data/
│   ├── raw/                           # Raw downloaded Kaggle CSVs (fraudTrain.csv, fraudTest.csv)
│   └── processed/                     # Leakage-safe preprocessed splits (train, val, test)
├── models/
│   ├── preprocessor.joblib            # Serialized FraudPreprocessor
│   ├── knn_best_model.joblib          # Tuned KNN classifier
│   ├── dt_best_model.joblib           # Tuned Decision Tree classifier
│   └── stage1_results.json            # Machine-readable evaluation results
├── reports/
│   ├── figures/                       # 10 generated publication figures
│   ├── eda_summary.txt                # Raw statistical audit outputs
│   ├── data_dictionary.md             # Complete feature definitions & rationale
│   ├── stage1_report.md               # Detailed scientific report & 10 viva Q&As
│   └── handoff_to_shubh.md            # Technical handoff instructions for Stage 2
├── src/
│   ├── __init__.py
│   ├── eda.py                         # Automated EDA & statistical checks
│   ├── preprocessing.py               # Leakage-safe pipeline & dataset splits
│   ├── models.py                      # KNN, Decision Tree, SMOTE, evaluation
│   └── visualization.py               # Confusion matrices, ROC/PR, tuning plots
├── notebooks/
│   └── 01_stage1_eda_and_baseline_models.ipynb  # Interactive Jupyter notebook
├── run_stage1.py                      # Master pipeline executable
├── requirements.txt                   # Dependency manifest
└── README.md                          # Project documentation
```

---

## 🎓 BTech Viva Voce Talking Points

Be prepared to answer these core examiner questions:
1. **Why scale features for KNN but not Decision Trees?** KNN depends on Euclidean distance ($\sqrt{\sum (x_i - y_i)^2}$); without scaling, unscaled features like `city_pop` dominate. Decision Trees make orthogonal threshold cuts ($x_i \le \theta$) invariant to monotonic scaling.
2. **Why does unweighted KNN fail on 0.58% fraud?** Majority class points outnumber fraud 171:1, so the $k$ nearest neighbors will almost always be legitimate. We solved this using **SMOTE** on $X_{train}$ and **distance-weighted voting**.
3. **ID3 vs. CART difference:** ID3 uses Information Gain based on Shannon Entropy ($-\sum p \log_2 p$), whereas CART uses Gini Impurity ($1 - \sum p^2$). Gini is computationally faster and produced our best validation F1 (0.6244).
4. **How was Data Leakage avoided?** `FraudPreprocessor.fit()` was called strictly on the training fold. Scalers and encoders learned means and standard deviations exclusively from $X_{train}$ and applied them to $X_{val}$ and $X_{test}$.
