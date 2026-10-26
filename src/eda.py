"""
Exploratory Data Analysis (EDA) Module
Credit Card Fraud Detection Project - Stage 1
Author: Anish

Provides comprehensive data auditing:
- head(), tail(), describe(), shape, and info() inspection
- Missing values and duplicate detection
- Outlier detection via IQR method
- Class imbalance analysis and domain-specific visualizations
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between two GPS coordinates in kilometers
    using the Haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6367.0 * c
    return km


def inspect_outliers_iqr(series: pd.Series, feature_name: str) -> dict:
    """
    Calculates IQR-based outlier statistics for a numeric feature.
    """
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        "feature": feature_name,
        "q25": float(q25),
        "median": float(series.median()),
        "q75": float(q75),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": float(len(outliers) / len(series) * 100.0),
        "min": float(series.min()),
        "max": float(series.max())
    }


def perform_full_eda(df: pd.DataFrame, figures_dir: str = "reports/figures", report_path: str = "reports/eda_summary.txt") -> dict:
    """
    Performs full exploratory data analysis and generates summary reports and plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CREDIT CARD FRAUD DETECTION - STAGE 1 EXPLORATORY DATA ANALYSIS (EDA)")
    summary_lines.append("Author: Anish (Stage 1 Lead)")
    summary_lines.append("=" * 80)

    # 1. Shape
    shape_str = f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns"
    print(shape_str)
    summary_lines.append("\n[1] DATASET SHAPE:")
    summary_lines.append(shape_str)

    # 2. Head and Tail
    summary_lines.append("\n[2] FIRST 5 ROWS (head()):")
    summary_lines.append(df.head(5).to_string())

    summary_lines.append("\n[3] LAST 5 ROWS (tail()):")
    summary_lines.append(df.tail(5).to_string())

    # 3. Info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    summary_lines.append("\n[4] DATASET INFO (info()):")
    summary_lines.append(info_str)

    # 4. Describe
    summary_lines.append("\n[5] NUMERICAL SUMMARY STATISTICS (describe()):")
    summary_lines.append(df.describe().to_string())

    summary_lines.append("\n[6] CATEGORICAL SUMMARY STATISTICS (describe(include=['O'])):")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        summary_lines.append(df.describe(include=['O']).to_string())

    # 5. Missing values
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100.0
    null_df = pd.DataFrame({"Missing_Count": null_counts, "Percentage": null_pct})
    summary_lines.append("\n[7] MISSING VALUES AUDIT:")
    summary_lines.append(null_df.to_string())

    # 6. Duplicates
    dup_full = df.duplicated().sum()
    dup_trans = df['trans_num'].duplicated().sum() if 'trans_num' in df.columns else 0
    summary_lines.append("\n[8] DUPLICATE ROWS AUDIT:")
    summary_lines.append(f"Full duplicate rows: {dup_full}")
    summary_lines.append(f"Duplicate transaction IDs ('trans_num'): {dup_trans}")

    # 7. Class Imbalance
    if 'is_fraud' in df.columns:
        counts = df['is_fraud'].value_counts()
        pcts = df['is_fraud'].value_counts(normalize=True) * 100.0
        neg_count = counts.get(0, 0)
        pos_count = counts.get(1, 0)
        imbalance_ratio = (neg_count / pos_count) if pos_count > 0 else 0

        summary_lines.append("\n[9] CLASS IMBALANCE AUDIT (Target: 'is_fraud'):")
        summary_lines.append(f"Legitimate Transactions (0): {neg_count:,} ({pcts.get(0, 0):.4f}%)")
        summary_lines.append(f"Fraudulent Transactions (1): {pos_count:,} ({pcts.get(1, 0):.4f}%)")
        summary_lines.append(f"Imbalance Ratio: 1 fraud per {imbalance_ratio:.1f} legitimate transactions (~{imbalance_ratio:.0f}:1)")

    # 8. Outlier Analysis on key numeric features
    summary_lines.append("\n[10] OUTLIER AUDIT (IQR METHOD):")
    outlier_metrics = {}
    for col in ['amt', 'city_pop']:
        if col in df.columns:
            stats = inspect_outliers_iqr(df[col], col)
            outlier_metrics[col] = stats
            summary_lines.append(
                f"- Feature '{col}': Median={stats['median']:.2f}, IQR={stats['iqr']:.2f}, "
                f"Upper Bound={stats['upper_bound']:.2f}, Outliers={stats['outlier_count']:,} "
                f"({stats['outlier_percentage']:.2f}%), Max={stats['max']:.2f}"
            )

    # Save summary report text
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"EDA textual summary written to {report_path}")

    # 9. Generate and save visualizations
    generate_eda_visualizations(df, figures_dir)

    return {
        "shape": df.shape,
        "null_counts": null_counts.to_dict(),
        "duplicates": int(dup_full),
        "class_counts": counts.to_dict() if 'is_fraud' in df.columns else {},
        "outliers": outlier_metrics
    }


def generate_eda_visualizations(df: pd.DataFrame, figures_dir: str = "reports/figures"):
    """
    Creates informative visual charts for the credit card fraud dataset.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Class Imbalance Plot
    if 'is_fraud' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        counts = df['is_fraud'].value_counts()
        labels = ['Legitimate (0)', 'Fraudulent (1)']
        values = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['#2b5c8f', '#d9534f']

        # Linear count
        bars = axes[0].bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
        axes[0].set_title("Class Distribution (Count Scale)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Number of Transactions", fontsize=11)
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f"{height:,}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Log scale with percentage
        bars_log = axes[1].bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
        axes[1].set_yscale('log')
        axes[1].set_title("Class Distribution (Log Scale - Severe Imbalance)", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Number of Transactions (Log10)", fontsize=11)
        total = sum(values)
        for bar in bars_log:
            height = bar.get_height()
            pct = (height / total) * 100
            axes[1].annotate(f"{pct:.3f}% ({height:,})",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.suptitle("Target Variable Imbalance Analysis ('is_fraud')", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "eda_class_imbalance.png"), dpi=300)
        plt.close()

    # 2. Transaction Amount Distribution
    if 'amt' in df.columns and 'is_fraud' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Boxplot log amount
        df_copy = df[['amt', 'is_fraud']].copy()
        df_copy['log_amt'] = np.log1p(df_copy['amt'])
        df_copy['Label'] = df_copy['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})

        sns.boxplot(data=df_copy, x='Label', y='log_amt', hue='Label', palette=['#2b5c8f', '#d9534f'], legend=False, ax=axes[0])
        axes[0].set_title("Log(Amount + 1) by Class", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Log(Amount + 1) in USD", fontsize=11)
        axes[0].set_xlabel("Transaction Class", fontsize=11)

        # KDE plot of log amount
        sns.kdeplot(data=df_copy[df_copy['is_fraud'] == 0]['log_amt'], ax=axes[1],
                    label='Legitimate', color='#2b5c8f', fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(data=df_copy[df_copy['is_fraud'] == 1]['log_amt'], ax=axes[1],
                    label='Fraudulent', color='#d9534f', fill=True, alpha=0.4, linewidth=2)
        axes[1].set_title("Density Distribution: Log(Amount) Shift", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Log(Amount + 1)", fontsize=11)
        axes[1].set_ylabel("Density", fontsize=11)
        axes[1].legend()

        plt.suptitle("Transaction Amount Comparison: Legitimate vs Fraudulent", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "eda_amt_distribution.png"), dpi=300)
        plt.close()

    # 3. Hourly Fraud Rate
    if 'trans_date_trans_time' in df.columns and 'is_fraud' in df.columns:
        trans_dt = pd.to_datetime(df['trans_date_trans_time'])
        hours = trans_dt.dt.hour
        hourly_stats = df.groupby(hours)['is_fraud'].agg(['count', 'mean']).reset_index()
        hourly_stats.columns = ['Hour', 'Total_Transactions', 'Fraud_Rate']
        hourly_stats['Fraud_Rate_Pct'] = hourly_stats['Fraud_Rate'] * 100.0

        plt.figure(figsize=(12, 5))
        bars = plt.bar(hourly_stats['Hour'], hourly_stats['Fraud_Rate_Pct'],
                       color=['#d9534f' if h in [22, 23, 0, 1, 2, 3] else '#2b5c8f' for h in hourly_stats['Hour']],
                       edgecolor='black', linewidth=1)
        plt.title("Fraud Occurrence Rate by Hour of Day (Night-Time Anomaly)", fontsize=14, fontweight='bold')
        plt.xlabel("Hour of Day (00:00 - 23:00)", fontsize=12)
        plt.ylabel("Fraud Rate (%)", fontsize=12)
        plt.xticks(range(0, 24))
        plt.axvspan(21.5, 23.5, color='orange', alpha=0.15, label='High-Risk Night Window (22h - 03h)')
        plt.axvspan(-0.5, 3.5, color='orange', alpha=0.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "eda_hourly_fraud_rate.png"), dpi=300)
        plt.close()

    # 4. Category-Wise Fraud Rate
    if 'category' in df.columns and 'is_fraud' in df.columns:
        cat_stats = df.groupby('category')['is_fraud'].agg(['count', 'mean']).reset_index()
        cat_stats['Fraud_Rate_Pct'] = cat_stats['mean'] * 100.0
        cat_stats = cat_stats.sort_values('Fraud_Rate_Pct', ascending=True)

        plt.figure(figsize=(12, 6))
        plt.barh(cat_stats['category'], cat_stats['Fraud_Rate_Pct'], color='#4a90e2', edgecolor='black')
        plt.title("Fraud Rate (%) Across Merchant Categories", fontsize=14, fontweight='bold')
        plt.xlabel("Fraud Rate (%)", fontsize=12)
        plt.ylabel("Merchant Category", fontsize=12)
        for i, val in enumerate(cat_stats['Fraud_Rate_Pct']):
            plt.text(val + 0.05, i, f"{val:.2f}%", va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "eda_category_fraud_rate.png"), dpi=300)
        plt.close()

    # 5. Correlation Heatmap of Engineered Signals
    # Compute quick engineered features for correlation visualization
    corr_df = pd.DataFrame()
    if 'amt' in df.columns:
        corr_df['log_amt'] = np.log1p(df['amt'])
    if 'trans_date_trans_time' in df.columns:
        dt = pd.to_datetime(df['trans_date_trans_time'])
        corr_df['hour'] = dt.dt.hour
        corr_df['day_of_week'] = dt.dt.dayofweek
        corr_df['is_night'] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3]).astype(int)
    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        dob = pd.to_datetime(df['dob'])
        dt = pd.to_datetime(df['trans_date_trans_time'])
        corr_df['age'] = ((dt - dob).dt.days // 365.25).astype(int)
    if 'lat' in df.columns and 'merch_lat' in df.columns:
        corr_df['dist_km'] = compute_haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    if 'city_pop' in df.columns:
        corr_df['log_city_pop'] = np.log1p(df['city_pop'])
    if 'gender' in df.columns:
        corr_df['gender_binary'] = (df['gender'] == 'M').astype(int)
    if 'is_fraud' in df.columns:
        corr_df['is_fraud'] = df['is_fraud']

    if not corr_df.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = corr_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    vmin=-0.2, vmax=0.4, linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
        plt.title("Feature Correlation Matrix with Target ('is_fraud')", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "eda_correlation_matrix.png"), dpi=300)
        plt.close()

    print(f"All EDA visualizations successfully saved in {figures_dir}")
