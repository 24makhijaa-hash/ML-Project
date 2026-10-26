"""
Leakage-Safe Preprocessing Pipeline Module
Credit Card Fraud Detection Project - Stage 1
Author: Anish (Stage 1 Lead)
Shared Component for: Shubh (Stage 2) and Dev (Stage 3)

Provides:
- Domain feature engineering (Haversine distance, temporal signals, age)
- Leakage-safe ColumnTransformer (StandardScaler on numerics, OneHotEncoder on categories)
- Reproducible, stratified train/validation/test splitting
- Preprocessor persistence (save/load via joblib)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between cardholder and merchant in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6367.0 * c


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade, leakage-safe preprocessor for credit card fraud detection.
    Fits all encoders and scalers ONLY on training data.
    """

    NUMERIC_FEATURES = [
        'log_amt',
        'trans_hour',
        'trans_day_of_week',
        'age',
        'dist_to_merchant_km',
        'log_city_pop'
    ]

    BINARY_FEATURES = [
        'gender_binary',
        'is_night',
        'is_weekend'
    ]

    CATEGORICAL_FEATURES = [
        'category'
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names_out_ = None
        self.is_fitted = False

    def engineer_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts mathematical, temporal, demographic, and spatial signals from raw columns.
        Does NOT apply scaling or encoding (prevents data leakage).
        """
        data = df.copy()

        # 1. Parse timestamps
        trans_dt = pd.to_datetime(data['trans_date_trans_time'])
        dob_dt = pd.to_datetime(data['dob'])

        # 2. Temporal signals
        data['trans_hour'] = trans_dt.dt.hour
        data['trans_day_of_week'] = trans_dt.dt.dayofweek
        data['is_night'] = data['trans_hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
        data['is_weekend'] = data['trans_day_of_week'].isin([5, 6]).astype(int)

        # 3. Cardholder Age
        data['age'] = ((trans_dt - dob_dt).dt.days // 365.25).astype(float)

        # 4. Haversine Spatial Distance (Cardholder to Merchant)
        data['dist_to_merchant_km'] = haversine_distance_km(
            data['lat'].values,
            data['long'].values,
            data['merch_lat'].values,
            data['merch_long'].values
        )

        # 5. Skew mitigation (log1p transform)
        data['log_amt'] = np.log1p(data['amt'].values)
        data['log_city_pop'] = np.log1p(data['city_pop'].values)

        # 6. Binary gender encoding (M=1, F=0)
        data['gender_binary'] = (data['gender'] == 'M').astype(int)

        return data

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits StandardScaler on numeric features and OneHotEncoder on categories.
        MUST BE CALLED ONLY ON THE TRAINING SET.
        """
        # First ensure engineered columns exist
        engineered = self.engineer_raw_dataframe(X)

        # Fit numeric scaler
        num_data = engineered[self.NUMERIC_FEATURES].values
        self.scaler.fit(num_data)

        # Fit category encoder
        cat_data = engineered[self.CATEGORICAL_FEATURES]
        self.one_hot_encoder.fit(cat_data)

        # Record feature names
        cat_names = list(self.one_hot_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES))
        self.feature_names_out_ = self.NUMERIC_FEATURES + self.BINARY_FEATURES + cat_names
        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw data into scaled and encoded feature matrix.
        Uses statistics learned during fit() to avoid leakage.
        """
        if not self.is_fitted:
            raise RuntimeError("FraudPreprocessor must be fitted on training data before calling transform()!")

        engineered = self.engineer_raw_dataframe(X)

        # Scale numerics
        num_scaled = self.scaler.transform(engineered[self.NUMERIC_FEATURES].values)

        # Extract binaries (as-is, float64 for compatibility)
        bin_data = engineered[self.BINARY_FEATURES].values

        # Encode categories
        cat_encoded = self.one_hot_encoder.transform(engineered[self.CATEGORICAL_FEATURES])

        # Combine into single 2D array
        transformed_array = np.hstack([num_scaled, bin_data, cat_encoded])

        return pd.DataFrame(transformed_array, columns=self.feature_names_out_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> list:
        if not self.is_fitted:
            raise RuntimeError("FraudPreprocessor is not fitted yet.")
        return self.feature_names_out_

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"FraudPreprocessor saved successfully to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'FraudPreprocessor':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved preprocessor found at {filepath}")
        preprocessor = joblib.load(filepath)
        print(f"FraudPreprocessor loaded successfully from {filepath}")
        return preprocessor


def create_reproducible_splits(
    raw_train_csv: str = "data/raw/fraudTrain.csv",
    raw_test_csv: str = "data/raw/fraudTest.csv",
    sample_train_size: int = 60000,
    sample_val_size: int = 15000,
    sample_test_size: int = 15000,
    random_state: int = 42,
    processed_dir: str = "data/processed",
    models_dir: str = "models"
):
    """
    Creates leakage-safe, stratified train/validation/test splits.
    - Fits FraudPreprocessor strictly on X_train.
    - Transforms X_train, X_val, and X_test.
    - Preserves test set for Dev (Stage 3).
    - Saves all processed sets and the preprocessor artifact.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("CREATING LEAKAGE-SAFE REPRODUCIBLE DATA SPLITS")
    print("=" * 70)

    # 1. Load raw train data
    print(f"Loading raw training dataset from {raw_train_csv}...")
    df_raw_train = pd.read_csv(raw_train_csv)
    print(f"Raw train shape: {df_raw_train.shape}")

    # 2. Extract stratified subset for Stage 1/2 benchmark if requested
    if sample_train_size and sample_train_size + sample_val_size < len(df_raw_train):
        total_sample = sample_train_size + sample_val_size
        print(f"Sampling {total_sample:,} stratified rows for high-efficiency modeling...")
        df_train_pool, _ = train_test_split(
            df_raw_train,
            train_size=total_sample,
            stratify=df_raw_train['is_fraud'],
            random_state=random_state
        )
    else:
        df_train_pool = df_raw_train

    # 3. Split into Train (80%) and Validation (20%)
    val_ratio = sample_val_size / (sample_train_size + sample_val_size) if sample_train_size else 0.20
    df_train_raw, df_val_raw = train_test_split(
        df_train_pool,
        test_size=val_ratio,
        stratify=df_train_pool['is_fraud'],
        random_state=random_state
    )

    # 4. Load raw test data (preserves final evaluation set)
    print(f"Loading raw test dataset from {raw_test_csv}...")
    df_raw_test = pd.read_csv(raw_test_csv)
    print(f"Raw test shape: {df_raw_test.shape}")

    if sample_test_size and sample_test_size < len(df_raw_test):
        print(f"Sampling {sample_test_size:,} stratified test rows for benchmark evaluation...")
        df_test_raw, _ = train_test_split(
            df_raw_test,
            train_size=sample_test_size,
            stratify=df_raw_test['is_fraud'],
            random_state=random_state
        )
    else:
        df_test_raw = df_raw_test

    y_train = df_train_raw['is_fraud'].reset_index(drop=True)
    y_val = df_val_raw['is_fraud'].reset_index(drop=True)
    y_test = df_test_raw['is_fraud'].reset_index(drop=True)

    print(f"Train set: {len(df_train_raw):,} rows (Fraud: {y_train.sum():,} - {y_train.mean()*100:.3f}%)")
    print(f"Val set:   {len(df_val_raw):,} rows (Fraud: {y_val.sum():,} - {y_val.mean()*100:.3f}%)")
    print(f"Test set:  {len(df_test_raw):,} rows (Fraud: {y_test.sum():,} - {y_test.mean()*100:.3f}%)")

    # 5. Fit Preprocessor STRICTLY on Training Set
    print("\nFitting FraudPreprocessor strictly on X_train (Zero Data Leakage)...")
    preprocessor = FraudPreprocessor()
    preprocessor.fit(df_train_raw)

    # 6. Transform Train, Validation, and Test
    print("Transforming X_train, X_val, and X_test using fitted preprocessor...")
    X_train_proc = preprocessor.transform(df_train_raw).reset_index(drop=True)
    X_val_proc = preprocessor.transform(df_val_raw).reset_index(drop=True)
    X_test_proc = preprocessor.transform(df_test_raw).reset_index(drop=True)

    # 7. Save preprocessor and datasets
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    print(f"Saving processed datasets to {processed_dir}...")
    joblib.dump((X_train_proc, y_train), os.path.join(processed_dir, "train_processed.joblib"))
    joblib.dump((X_val_proc, y_val), os.path.join(processed_dir, "val_processed.joblib"))
    joblib.dump((X_test_proc, y_test), os.path.join(processed_dir, "test_processed.joblib"))

    # Also save CSV headers and metadata
    metadata = {
        "train_rows": len(X_train_proc),
        "train_fraud_count": int(y_train.sum()),
        "train_fraud_rate": float(y_train.mean()),
        "val_rows": len(X_val_proc),
        "val_fraud_count": int(y_val.sum()),
        "val_fraud_rate": float(y_val.mean()),
        "test_rows": len(X_test_proc),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "num_features": X_train_proc.shape[1],
        "feature_names": preprocessor.get_feature_names(),
        "random_state": random_state
    }
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Data preparation complete! Total features: {X_train_proc.shape[1]}")
    return preprocessor, (X_train_proc, y_train), (X_val_proc, y_val), (X_test_proc, y_test)
