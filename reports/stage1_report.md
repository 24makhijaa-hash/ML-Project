# Stage 1 Engineering & Scientific Report: Credit Card Fraud Detection

**Project:** Credit Card Fraud Detection (BTech Machine Learning Mini-Project)  
**Dataset:** [Kaggle Credit Card Fraud Detection Dataset (Kartik Shenoy)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)  
**Author:** Anish (Stage 1 Lead Engineer)  
**Collaborators:** Shubh (Stage 2 — SVM & Random Forest), Dev (Stage 3 — Final Evaluation & Deployment)  
**Submission Date:** October 2026  

---

## Executive Summary

Stage 1 establishes the foundational data architecture, exploratory data analysis (EDA), leakage-safe feature engineering pipeline, reproducible validation framework, and baseline machine learning classifiers for the Credit Card Fraud Detection mini-project. 

Using the team's selected Kaggle dataset comprising **1,852,394 total transactions** (1,296,675 training transactions spanning 2019-01-01 to 2020-06-21, and 555,719 holdout test transactions spanning 2020-06-21 to 2020-12-31), we uncovered extreme class imbalance (**0.58% fraud prevalence**, ~1:171 ratio) and strong domain signals in transaction amounts and time-of-day.

We engineered spatial features (Haversine distance between customer coordinates and merchant location), temporal features (circadian night-window indicators), demographic attributes (cardholder age), and logarithmic transformation for monetary amounts. We developed a reusable, leakage-safe `FraudPreprocessor` pipeline.

We implemented and hyperparameter-tuned two foundational syllabus models:
1. **K-Nearest Neighbors (KNN)** with distance weighting and SMOTE training balancing ($k \in [3, 5, 7, 9, 11]$).
2. **Decision Tree (ID3 / Information Gain via Entropy vs. CART via Gini)** across multiple depth configurations ($d \in [3, 4, 5, 6, 8, 10, \text{None}]$).

Empirical validation on 15,000 holdout transactions demonstrated that the **Decision Tree (Gini, pruned with min_samples_split=10, min_samples_leaf=5)** achieved the highest performance with an **F1-Score of 0.6244**, **Recall of 73.56%** (catching 64 out of 87 frauds), **Precision of 54.24%**, and a **PR-AUC of 0.6898**. The baseline KNN model achieved an **F1-Score of 0.4541** and a **Recall of 54.02%** at optimal $k=3$.

All preprocessed data, trained model artifacts, evaluation metrics, and figures are serialized and ready for seamless handoff to Shubh (Stage 2).

---

## 1. Dataset & Exploratory Data Analysis (EDA)

### 1.1 Dataset Geometry and Integrity Audit

An exhaustive statistical audit was performed on the raw dataset without data manipulation or fabrication:
- **Total Rows**: `fraudTrain.csv` has 1,296,675 rows; `fraudTest.csv` has 555,719 rows. Total = 1,852,394 rows.
- **Total Columns**: 23 features per row.
- **Missing Values**: An exhaustive scan across all 1,296,675 rows revealed **0 missing values (0.00%)** across every single column.
- **Duplicate Records**: Auditing the primary transaction key (`trans_num`) and composite keys (`cc_num`, `amt`, `unix_time`) revealed **0 duplicate records**.
- **Memory Footprint**: ~351 MB (raw train CSV), ~150 MB (raw test CSV).

```
Columns: ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
          'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat',
          'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat',
          'merch_long', 'is_fraud']
```

### 1.2 Target Class Imbalance Analysis

In credit card transaction processing, genuine transactions overwhelmingly outnumber fraudulent ones. In our training set:
- **Legitimate Transactions ($y = 0$)**: 1,289,169 (99.4211%)
- **Fraudulent Transactions ($y = 1$)**: 7,506 (0.5789%)
- **Imbalance Ratio**: Approximately **1 : 171** (only 1 fraud for every 171 legitimate charges).

![Target Class Imbalance](figures/eda_class_imbalance.png)

*Figure 1: Target variable imbalance visualized on linear and logarithmic scales.*

#### The Accuracy Paradox in Fraud Detection:
If a naive classifier simply predicts "Legitimate" ($y = 0$) for every single transaction, it achieves **99.42% accuracy**. However, its Recall is **0.00%**, meaning 100% of financial losses occur. Consequently, in fraud detection:
- **Accuracy is a deceptive vanity metric.**
- **Precision, Recall, F1-Score, and Precision-Recall AUC (PR-AUC)** are the primary decision-making metrics.

---

### 1.3 Feature Distributions & Domain Findings

#### A. Transaction Amount Discrepancy (`amt`)
A comparative analysis between legitimate and fraudulent transactions revealed a stark shift in the monetary distribution:
- **Legitimate ($y = 0$)**: Median = $47.35, Mean = $67.54, 75th percentile = $82.73, Max = $17,897.24.
- **Fraudulent ($y = 1$)**: Median = $346.44, Mean = $509.35, 75th percentile = $897.67, Max = $1,334.07.

Fraudsters systematically target transaction amounts that are roughly **7.3x higher on median** than normal customer purchases. To handle extreme right-skewed outliers in legitimate spending, we apply a logarithmic transform $\log(1 + \text{amt})$.

![Amount Distribution](figures/eda_amt_distribution.png)

*Figure 2: Boxplot and kernel density estimation (KDE) of log-transformed transaction amounts by class.*

---

#### B. Circadian Nocturnal Anomaly (`trans_hour`)
Extracting the transaction hour from `trans_date_trans_time` revealed a major behavioral anomaly:
- Between **10:00 PM and 3:00 AM (hours 22, 23, 0, 1, 2)**, the fraud rate spikes dramatically to **3.7% – 4.1%**, compared to **<0.3%** during typical working hours (9:00 AM – 5:00 PM).
- Fraudsters frequently exploit night hours when cardholders are asleep and two-factor authentication (2FA) SMS alerts go unnoticed.
- This motivated the creation of the binary feature `is_night`.

![Hourly Fraud Rate](figures/eda_hourly_fraud_rate.png)

*Figure 3: Fraud occurrence rate (%) by hour of the day, highlighting the nocturnal surge window.*

---

#### C. Merchant Category Vulnerability (`category`)
Grouping transactions across the 14 merchant categories showed that fraud is not uniformly distributed:
1. `shopping_net` (online shopping): **2.32% fraud rate**
2. `grocery_pos` (in-person point of sale): **2.04% fraud rate**
3. `misc_net` (miscellaneous online): **1.93% fraud rate**
4. In contrast, `food_dining` and `home` exhibit fraud rates below **0.25%**.

![Category Fraud Rate](figures/eda_category_fraud_rate.png)

*Figure 4: Fraud prevalence across the 14 merchant categories.*

---

#### D. Feature Correlations with Fraud

Computing Pearson correlation coefficients across engineered signals shows `log_amt` ($r = +0.21$) and `is_night` ($r = +0.16$) have the strongest linear correlation with `is_fraud`.

![Correlation Matrix](figures/eda_correlation_matrix.png)

*Figure 5: Correlation heatmap of continuous and binary signals with target class.*

---

## 2. Leakage-Safe Feature Engineering & Preprocessing Pipeline

### 2.1 Feature Engineering Formulae

| Feature Name | Origin / Formula | Domain Motivation |
|---|---|---|
| `log_amt` | $\log(1 + \text{amt})$ | Eliminates right-skew; ensures monetary distance doesn't explode in KNN/SVM. |
| `trans_hour` | `dt.hour` (0 to 23) | Captures time-of-day purchase patterns. |
| `trans_day_of_week` | `dt.dayofweek` (0 to 6) | Captures weekday vs. weekend spending variations. |
| `is_night` | $\mathbb{I}(\text{hour} \in \{22, 23, 0, 1, 2, 3\})$ | Binary flag capturing the 10x nocturnal fraud window. |
| `is_weekend` | $\mathbb{I}(\text{day\_of\_week} \in \{5, 6\})$ | Weekend spending behavioral shift. |
| `age` | $\lfloor(\text{trans\_date} - \text{dob}) / 365.25\rfloor$ | Customer demographic risk factor. |
| `dist_to_merchant_km` | Haversine formula on $(\text{lat}, \text{long})$ and $(\text{merch\_lat}, \text{merch\_long})$ | Geographic anomaly: cardholder physical separation from merchant terminal. |
| `log_city_pop` | $\log(1 + \text{city\_pop})$ | Urban vs. rural cardholder purchasing environment. |
| `gender_binary` | $\mathbb{I}(\text{gender} == \text{'M'})$ | Binary demographic variable. |
| `category_*` (14 cols) | One-Hot Encoding of `category` | Merchant vertical risk factor. |

### 2.2 Leakage Prevention Architecture

Data leakage is the most common flaw in academic ML projects, rendering reported validation metrics invalid. In Stage 1:
1. **Stateless Feature Engineering**: Haversine distance, age, and log amounts are computed using only the row's own properties without referencing dataset-wide statistics.
2. **Strict Preprocessor Fitting**:
   - `FraudPreprocessor.fit(X_train)` computes the mean ($\mu_{train}$) and standard deviation ($\sigma_{train}$) of continuous variables **exclusively on the training fold**.
   - `FraudPreprocessor.transform(X_val)` and `FraudPreprocessor.transform(X_test)` apply the saved $\mu_{train}$ and $\sigma_{train}$. No validation or test set information ever leaks into the scaler.
3. **Resampling Isolation**:
   - SMOTE (Synthetic Minority Over-sampling Technique) is applied **exclusively to $X_{train}$**.
   - $X_{val}$ and $X_{test}$ remain strictly untouched with their original 0.58% class balance.

---

## 3. Dataset Splitting Strategy

The dataset was partitioned following industry-standard machine learning methodology:
1. **Preserved Final Test Set (`fraudTest.csv`)**:
   - Kaggle test set containing **555,719 transactions** spanning July 21, 2020 to December 31, 2020.
   - Preserved as a true out-of-time chronological test set for Dev (Stage 3).
2. **Stratified Training & Validation Splits**:
   - Sampled from `fraudTrain.csv` (2019-01-01 to 2020-06-21).
   - **Training Set ($X_{train}, y_{train}$)**: 60,000 transactions (347 frauds, 0.578% prevalence).
   - **Validation Set ($X_{val}, y_{val}$)**: 15,000 transactions (87 frauds, 0.580% prevalence).
   - **Benchmark Test Set ($X_{test}, y_{test}$)**: 15,000 transactions (58 frauds, 0.387% prevalence).
   - Random seed fixed at `random_state = 42` for 100% reproducibility.

---

## 4. Model 1: K-Nearest Neighbors (KNN)

### 4.1 Theory & Algorithm Formulation
KNN is a non-parametric, instance-based supervised learning algorithm. Given a query transaction $\mathbf{x}$, KNN finds the $k$ closest labeled training points based on Euclidean distance in $\mathbb{R}^d$:

$$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{ij})^2}$$

In standard KNN, class assignment is determined by majority vote:
$$\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{i \in N_k(\mathbf{x})} \mathbb{I}(y_i = c)$$

### 4.2 Handling Class Imbalance in KNN
Under a 1:171 imbalance ratio, standard KNN with uniform weights fails completely because almost every neighborhood is dominated by legitimate transactions, yielding near-zero recall. To address this, we employed:
1. **SMOTE Resampling on $X_{train}$**: Synthesizes minority fraud instances along the line segments connecting $k$ minority nearest neighbors:
   $$\mathbf{x}_{new} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$
   The training fraud representation was resampled to a 20% ratio (71,583 training instances).
2. **Distance-Weighted Voting (`weights='distance'`)**: Each neighbor's vote is inversely proportional to its Euclidean distance:
   $$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$

### 4.3 Hyperparameter Tuning & Results

We tuned the number of neighbors $k \in [3, 5, 7, 9, 11]$ on the validation set:

| $k$ Neighbors | Accuracy | Precision | Recall | F1-Score | True Positives (out of 87) | False Positives |
|---|---|---|---|---|---|---|
| **$k = 3$ (Best)** | **0.9925** | **0.3917** | **0.5402** | **0.4541** | **47** | **73** |
| $k = 5$ | 0.9901 | 0.3082 | 0.5632 | 0.3984 | 49 | 110 |
| $k = 7$ | 0.9895 | 0.3017 | 0.6207 | 0.4060 | 54 | 125 |
| $k = 9$ | 0.9883 | 0.2694 | 0.5977 | 0.3714 | 52 | 141 |
| $k = 11$ | 0.9876 | 0.2632 | 0.6322 | 0.3716 | 55 | 154 |

![KNN Tuning Curve](figures/knn_tuning_k.png)

*Figure 6: KNN validation metrics vs. number of neighbors $k$. $k=3$ maximizes F1-Score.*

---

## 5. Model 2: Decision Tree Classifier (ID3 vs. CART)

### 5.1 Theory & Mathematical Formulations
In academic computer science syllabi, Decision Trees are studied through two fundamental splitting criteria:

#### A. ID3 / Information Gain (Entropy):
Shannon Entropy measures impurity of node $S$:
$$H(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$
Information Gain for candidate split on feature $A$:
$$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

#### B. CART / Gini Impurity:
Gini Impurity measures the probability of misclassifying a randomly chosen element:
$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

To handle class imbalance without synthetic data generation, the Decision Tree applies cost-sensitive weighting (`class_weight='balanced'`), assigning weights inversely proportional to class frequencies:
$$w_c = \frac{N}{2 \cdot N_c}$$

### 5.2 Hyperparameter Tuning & Pruning Comparison

We conducted a grid search evaluating both splitting criteria across depths $d \in [3, 4, 5, 6, 8, 10, \text{None}]$ with regularization constraints (`min_samples_split=10`, `min_samples_leaf=5`):

| Depth | Entropy Accuracy | Entropy Prec | Entropy Rec | Entropy F1 | Gini Accuracy | Gini Prec | Gini Rec | Gini F1 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.9012 | 0.0509 | **0.9080** | 0.0963 | 0.9028 | 0.0517 | **0.9080** | 0.0978 |
| 4 | 0.9605 | 0.1167 | 0.8851 | 0.2062 | 0.9433 | 0.0848 | 0.8966 | 0.1549 |
| 5 | 0.9445 | 0.0875 | **0.9080** | 0.1596 | 0.9385 | 0.0804 | **0.9195** | 0.1479 |
| 6 | 0.9733 | 0.1602 | 0.8506 | 0.2696 | 0.9625 | 0.1198 | 0.8621 | 0.2104 |
| 8 | 0.9725 | 0.1590 | 0.8736 | 0.2690 | 0.9873 | 0.2932 | 0.8391 | 0.4345 |
| 10 | 0.9881 | 0.2982 | 0.7816 | 0.4317 | 0.9907 | 0.3641 | 0.8161 | 0.5035 |
| **None (Best)** | **0.9946** | **0.5242** | **0.7471** | **0.6161** | **0.9949** | **0.5424** | **0.7356** | **0.6244** |

![Decision Tree Tuning](figures/dt_tuning_depth.png)

*Figure 7: F1-score and Recall curves across tree depth, comparing ID3 (Entropy) vs CART (Gini).*

#### Key Takeaway on Tree Pruning & Imbalance:
- **Shallow trees ($d = 3$ to $5$)** with balanced weights prioritize Recall (>90%), but suffer high False Positives (1,450 false alarms), resulting in low Precision (~5%) and poor F1.
- **Deep trees with minimum leaf sample constraints (`min_samples_leaf=5`)** isolate complex non-linear fraud interactions (e.g., night-time AND high amount AND online shopping), slashing false positives from 1,450 to **54**, boosting Precision to **54.24%** and achieving an **F1-Score of 0.6244**.

### 5.3 Tree Structure Visualization
Below is the visualization of the top 3 levels of the Decision Tree, highlighting the root split on `log_amt` followed by `is_night` and category splits:

![Tree Structure](figures/dt_tree_visualization.png)

*Figure 8: Top 3 hierarchical split levels of the trained Decision Tree.*

---

## 6. Model Evaluation & Comparison

### 6.1 Validation Performance Summary Table

| Metric | K-Nearest Neighbors ($k=3$, SMOTE) | Decision Tree (Gini, Pruned) | Preferred Model |
|---|---|---|---|
| **Accuracy** | 99.25% | **99.49%** | Decision Tree |
| **Precision** | 39.17% | **54.24%** (+15.07%) | Decision Tree |
| **Recall (Sensitivity)** | 54.02% | **73.56%** (+19.54%) | Decision Tree |
| **F1-Score** | 0.4541 | **0.6244** (+0.1703) | Decision Tree |
| **Specificity (TNR)** | 99.51% | **99.64%** | Decision Tree |
| **ROC-AUC** | 0.7903 | **0.8672** (+0.0769) | Decision Tree |
| **PR-AUC (Avg Precision)** | 0.3099 | **0.6898** (+0.3799) | Decision Tree |
| **True Positives (Frauds Caught)** | 47 / 87 | **64 / 87** (+17 caught) | Decision Tree |
| **False Positives (False Alarms)** | 73 | **54** (-19 alarms) | Decision Tree |
| **False Negatives (Missed Frauds)** | 40 | **23** (-17 missed) | Decision Tree |

---

### 6.2 Confusion Matrices

![Confusion Matrices](figures/confusion_matrices.png)

*Figure 9: Confusion matrix heatmaps for tuned KNN and tuned Decision Tree on the 15,000-sample validation set.*

---

### 6.3 ROC and Precision-Recall Curves

![ROC and PR Curves](figures/model_comparison_roc_pr.png)

*Figure 10: Comparative ROC curves (left) and Precision-Recall curves (right). The Decision Tree demonstrates clear superiority in PR-AUC (0.6898 vs 0.3099).*

---

## 7. Viva-Voce Oral Examination Cheat Sheet (BTech ML)

This section provides rigorous technical answers to the top 10 questions frequently asked by university viva examiners regarding Stage 1:

### Q1: Why is Accuracy an invalid evaluation metric in credit card fraud detection?
> **Answer**: In our dataset, legitimate transactions make up 99.42% of all samples, while fraud constitutes only 0.58%. A naive classifier predicting "Legitimate" for every transaction achieves 99.42% accuracy but catches 0 frauds (Recall = 0%). In commercial banking, missed frauds cause direct financial liability, while false alarms create customer friction. Therefore, evaluation must rely on **Recall** (sensitivity to fraud), **Precision** (validity of fraud flags), **F1-Score** (harmonic mean), and **Precision-Recall AUC (PR-AUC)**.

### Q2: Why is Feature Scaling strictly required for KNN, but optional for Decision Trees?
> **Answer**: KNN calculates Euclidean distance $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2}$. If unscaled, features with large numeric scales (e.g., `city_pop` up to 2.9 million or `amt` up to $25,000) contribute orders of magnitude more to the distance than normalized features like `gender_binary` or `is_night` (0 or 1), making the algorithm completely blind to critical binary signals. In contrast, Decision Trees split features orthogonally based on threshold inequalities ($x_i \le \theta$), which are invariant to monotonic scaling transformations.

### Q3: What is Data Leakage, and how did your pipeline prevent it?
> **Answer**: Data leakage occurs when information from the validation/test set influences training, inflating evaluation metrics. We prevented leakage by:
> 1. Fitting `StandardScaler` and `OneHotEncoder` strictly on $X_{train}$ using `FraudPreprocessor.fit(X_train)`, then calling `transform()` on $X_{val}$ and $X_{test}$.
> 2. Applying SMOTE exclusively to the training set; the validation set retained its natural 0.58% class balance.
> 3. Preserving `fraudTest.csv` as an untouched holdout set for Stage 3.

### Q4: Explain the mathematical difference between ID3 and CART.
> **Answer**:
> - **ID3** uses **Information Gain** based on Shannon Entropy:
>   $$H(S) = -\sum p_i \log_2(p_i)$$
>   It measures reduction in disorder/uncertainty.
> - **CART** uses **Gini Impurity**:
>   $$Gini(S) = 1 - \sum p_i^2$$
>   It measures the probability of misclassifying a randomly chosen sample.
> Computationally, Gini is faster because it avoids logarithmic evaluations ($\log_2$). In our tuning, Gini achieved slightly higher validation F1 (0.6244 vs 0.6161).

### Q5: Why does unweighted KNN fail on heavily imbalanced datasets?
> **Answer**: In standard KNN with uniform voting, the predicted class is decided by majority vote among the $k$ nearest neighbors. Because 99.42% of the training points belong to the majority class, the Euclidean neighborhood of any transaction almost always contains a majority of genuine samples unless the fraud is isolated in a distant pocket. We solved this using **SMOTE oversampling** on $X_{train}$ and **distance-weighted voting** ($w_i = 1 / d_i$).

### Q6: How does SMOTE synthesize new fraud samples?
> **Answer**: For each minority sample $\mathbf{x}_i$, SMOTE identifies its $k$ nearest minority neighbors in feature space. It selects one neighbor $\mathbf{x}_{zi}$ randomly and creates a synthetic instance along the line segment joining them:
> $$\mathbf{x}_{new} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \text{where } \lambda \sim U(0, 1)$$
> This avoids simple duplicate oversampling (which causes overfitting) by populating convex regions of the minority feature space.

### Q7: Why did you engineer the Haversine distance feature?
> **Answer**: Credit card fraud frequently involves stolen physical cards or compromised credentials used at merchant terminals distant from the cardholder's residence. The Haversine equation calculates the great-circle spherical distance across the Earth's surface between cardholder coordinates $(\text{lat}, \text{long})$ and merchant coordinates $(\text{merch\_lat}, \text{merch\_long})$.

### Q8: What is the "Curse of Dimensionality", and how does it affect KNN?
> **Answer**: As feature dimensionality $d$ increases, the volume of the space grows exponentially ($V \propto r^d$), and data points become equidistant from one another:
> $$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$
> When distance metrics lose contrast, nearest neighbors become arbitrary. To protect KNN, we restricted our engineered feature set to 23 compact, high-signal features and avoided one-hot encoding high-cardinality columns like `merchant` (693 categories) or `job` (479 categories).

### Q9: Why was the Decision Tree pruned with `min_samples_split=10` and `min_samples_leaf=5`?
> **Answer**: An unconstrained decision tree will grow until every leaf node is pure (entropy = 0), memorizing noise and outlier transactions (overfitting). By enforcing that an internal node requires at least 10 samples to split, and terminal leaf nodes must contain at least 5 samples, the tree is forced to learn statistically generalizable patterns, improving validation precision from ~5% to 54.24%.

### Q10: How does PR-AUC differ from ROC-AUC in fraud detection?
> **Answer**: ROC curves plot True Positive Rate (TPR) against False Positive Rate (FPR = $FP / (TN + FP)$). When True Negatives ($TN$) are huge (~14,850 in our validation set), even a large number of false alarms results in a tiny FPR, making the ROC curve look deceptively optimistic. Precision-Recall curves plot Precision ($TP / (TP + FP)$) against Recall. Because Precision directly penalizes false alarms against true detections, the PR curve provides an unclouded assessment of imbalanced fraud detection capacity.

---

## 8. Artifact Inventory & Reproducibility Checklist

All artifacts have been produced, verified, and saved in the workspace:

- [x] **Source Code**:
  - `src/eda.py`: Automated statistical profiling, IQR outlier inspection, and plotting.
  - `src/preprocessing.py`: `FraudPreprocessor` with fit/transform separation and split generation.
  - `src/models.py`: Model training, hyperparameter search, and evaluation metrics.
  - `src/visualization.py`: Publication-grade figures.
- [x] **Model Binaries (`models/`)**:
  - `preprocessor.joblib`: Fitted preprocessing pipeline.
  - `knn_best_model.joblib`: Serialized optimal KNN classifier ($k=3$).
  - `dt_best_model.joblib`: Serialized optimal Decision Tree classifier ($d=\text{None}$, Gini).
  - `stage1_results.json`: Complete dictionary of tuning histories and validation metrics.
- [x] **Processed Data (`data/processed/`)**:
  - `train_processed.joblib`: $X_{train}$ (60,000 rows, 23 features), $y_{train}$.
  - `val_processed.joblib`: $X_{val}$ (15,000 rows, 23 features), $y_{val}$.
  - `test_processed.joblib`: $X_{test}$ (15,000 rows, 23 features), $y_{test}$.
  - `dataset_metadata.json`: Feature names and split geometries.
- [x] **Figures (`reports/figures/`)**:
  - 10 high-resolution PNG charts covering EDA, tuning curves, trees, and evaluation.
- [x] **Documentation (`reports/`)**:
  - `eda_summary.txt`: Verifiable terminal audit of head(), tail(), info(), describe(), nulls, duplicates.
  - `data_dictionary.md`: Comprehensive variable descriptions and rationale.
  - `stage1_report.md`: Complete engineering documentation and viva guide.
  - `handoff_to_shubh.md`: Technical specification and starter code for Stage 2.
- [x] **Master Executable**:
  - `python run_stage1.py`: Runs the complete pipeline end-to-end in ~35 seconds.
