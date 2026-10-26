# Credit Card Fraud Detection: Data Dictionary & Preprocessing Rationale

**Project:** BTech ML Mini-Project — Credit Card Fraud Detection  
**Dataset Source:** [Kaggle - Credit Card Fraud Detection by Kartik Shenoy](https://www.kaggle.com/datasets/kartik2112/fraud-detection)  
**Stage:** Stage 1 — Exploratory Data Analysis & Leakage-Safe Preprocessing  
**Author:** Anish (Stage 1 Lead)  
**Shared With:** Shubh (Stage 2 — SVM & Random Forest), Dev (Stage 3 — Evaluation & Deployment)

---

## 1. Raw Dataset Schema (Original Features)

The raw dataset contains simulated credit card transactions generated using the Python `Faker` library, mimicking legitimate and fraudulent purchasing patterns across the United States between January 1, 2019 and December 31, 2020.

| # | Column Name | Raw Data Type | Example Value | Description | Preprocessing Decision & Rationale |
|---|-------------|---------------|---------------|-------------|-----------------------------------|
| 0 | `Unnamed: 0` | `int64` | `0` | Original export row index | **Dropped**: Pure artifact of CSV export with no predictive value. |
| 1 | `trans_date_trans_time` | `object` (string) | `2019-01-01 00:00:18` | Timestamp of the transaction | **Transformed**: Parsed into datetime to extract temporal features (`trans_hour`, `trans_day_of_week`, `is_night`, `is_weekend`). Raw string dropped. |
| 2 | `cc_num` | `int64` | `2703186189652095` | Primary Account Number (PAN) / Card Number | **Dropped**: Treating PAN as a continuous numeric feature is statistically invalid; treating it as a categorical ID would cause severe memorization/overfitting on specific training cards. |
| 3 | `merchant` | `object` | `fraud_Rippin, Kub and Mann` | Merchant identifier (693 unique values) | **Dropped**: High-cardinality nominal variable with synthetic names; merchant category (`category`) provides clean, generalizable domain signal without dimensionality explosion. |
| 4 | `category` | `object` | `misc_net` | Merchant category (14 unique values) | **Encoded**: One-Hot Encoded into 14 binary dummy features using `OneHotEncoder(handle_unknown='ignore')`. Essential for capturing category-specific fraud risk (e.g. online shopping vs in-person grocery). |
| 5 | `amt` | `float64` | `4.97` | Transaction amount in USD | **Transformed & Scaled**: Extremely right-skewed (median $48.15, max $28,948). Transformed using $\log(1 + \text{amt})$, then normalized using `StandardScaler`. |
| 6 | `first` | `object` | `Jennifer` | Cardholder first name | **Dropped**: High-cardinality personal identifier; prone to memorization and bias. |
| 7 | `last` | `object` | `Banks` | Cardholder surname | **Dropped**: High-cardinality personal identifier; prone to memorization. |
| 8 | `gender` | `object` | `F` | Cardholder gender (`M` / `F`) | **Encoded**: Binary encoded (`M` = 1, `F` = 0). |
| 9 | `street` | `object` | `561 Perry Cove` | Cardholder billing street address | **Dropped**: High-cardinality nominal text; spatial relationship captured via coordinates. |
| 10 | `city` | `object` | `Moravian Falls` | Cardholder billing city | **Dropped**: High-cardinality nominal text (848+ unique cities). |
| 11 | `state` | `object` | `NC` | Cardholder billing US State | **Dropped**: Geographic location captured continuously by latitude/longitude and population. |
| 12 | `zip` | `int64` | `28654` | Cardholder ZIP code | **Dropped**: Arbitrary postal code numbers lack ordinal distance properties. |
| 13 | `lat` | `float64` | `36.0788` | Cardholder latitude coordinate | **Engineered**: Used to compute great-circle distance to merchant via the Haversine formula. |
| 14 | `long` | `float64` | `-81.1781` | Cardholder longitude coordinate | **Engineered**: Used in Haversine distance computation. |
| 15 | `city_pop` | `int64` | `3495` | Population of cardholder's city | **Transformed & Scaled**: Highly skewed (range 23 to 2,906,700). Transformed using $\log(1 + \text{city\_pop})$ and normalized via `StandardScaler`. |
| 16 | `job` | `object` | `Psychologist, counselling` | Cardholder occupation (479 unique values) | **Dropped**: High cardinality with sparse representation across classes. |
| 17 | `dob` | `object` (string) | `1988-03-09` | Cardholder date of birth | **Engineered**: Combined with `trans_date_trans_time` to calculate precise `age` in years at the moment of the transaction. Raw string dropped. |
| 18 | `trans_num` | `object` | `0b242abb...` | Unique 32-character hex transaction ID | **Dropped**: Non-predictive unique hash. |
| 19 | `unix_time` | `int64` | `1325376018` | Unix epoch timestamp in seconds | **Dropped**: Redundant with `trans_date_trans_time`. |
| 20 | `merch_lat` | `float64` | `36.011293` | Merchant latitude coordinate | **Engineered**: Used in Haversine distance calculation. |
| 21 | `merch_long` | `float64` | `-82.048315` | Merchant longitude coordinate | **Engineered**: Used in Haversine distance calculation. |
| 22 | `is_fraud` | `int64` | `0` | Target binary label (0 = Legitimate, 1 = Fraud) | **Target Variable**: Evaluated on natural distribution in validation/test sets (~0.58% prevalence). |

---

## 2. Engineered Feature Set (Preprocessed Matrix)

The final preprocessed feature matrix $X$ consists of exactly **23 features** ($d = 23$), structured for optimal distance computation in KNN/SVM and split-finding in Decision Tree/Random Forest.

### 2.1 Continuous Numeric Features (Standardized: Mean = 0, Std = 1)

1. **`log_amt`**: $\log(1 + \text{amt})$. Compresses monetary extremes while preserving relative purchasing power differences.
2. **`trans_hour`**: Hour of transaction (0–23), standardized.
3. **`trans_day_of_week`**: Day of the week (0 = Monday, 6 = Sunday), standardized.
4. **`age`**: Cardholder age in years at transaction time:
   $$\text{age} = \left\lfloor \frac{\text{trans\_date} - \text{dob}}{365.25} \right\rfloor$$
5. **`dist_to_merchant_km`**: Great-circle spatial distance between customer residence $(\text{lat}, \text{long})$ and merchant store $(\text{merch\_lat}, \text{merch\_long})$ using the Haversine equation:
   $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
   $$d = 2 R \arcsin(\sqrt{a}) \quad \text{where } R = 6,367 \text{ km}$$
6. **`log_city_pop`**: $\log(1 + \text{city\_pop})$. Normalizes city population variance.

### 2.2 Binary Domain Indicators (Values: 0 or 1)

7. **`gender_binary`**: 1 for Male (`M`), 0 for Female (`F`).
8. **`is_night`**: 1 if the transaction occurred during the peak fraud window ($22:00 \le \text{hour} \le 03:00$), else 0. Captures a 10x surge in fraud occurrence.
9. **`is_weekend`**: 1 if transaction was on Saturday or Sunday, else 0.

### 2.3 One-Hot Encoded Merchant Categories (14 Binary Features)

10. `category_entertainment`
11. `category_food_dining`
12. `category_gas_transport`
13. `category_grocery_net`
14. `category_grocery_pos`
15. `category_health_fitness`
16. `category_home`
17. `category_kids_pets`
18. `category_misc_net`
19. `category_misc_pos`
20. `category_personal_care`
21. `category_shopping_net`
22. `category_shopping_pos`
23. `category_travel`

---

## 3. Leakage Prevention Protocol

Data leakage occurs when information from outside the training dataset is inadvertently used to create the model. In Stage 1, leakage is prevented through the following protocols:

1. **Strict Fit/Transform Separation**: `FraudPreprocessor.fit()` is executed **only on the training split** ($X_{train}$). Means, standard deviations, and category encodings are computed on $X_{train}$ and simply applied via `transform()` to $X_{val}$ and $X_{test}$.
2. **Temporal & Stratified Split Integrity**: The official test set (`fraudTest.csv`, 555,719 transactions) covers a later chronological window (July 2020 – Dec 2020) and is preserved as an untouched holdout set for Stage 3 evaluation.
3. **Resampling Isolation**: SMOTE oversampling is applied **strictly to $X_{train}$**. The validation set ($X_{val}$) and test set ($X_{test}$) remain untouched with their natural 0.58% fraud distribution to guarantee realistic, non-inflated evaluation metrics.
