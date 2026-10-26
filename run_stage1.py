"""
Master Execution Pipeline for Credit Card Fraud Detection - Stage 1
Author: Anish (Stage 1 Lead)

Executes all Stage 1 tasks end-to-end:
1. Complete EDA (head, tail, describe, info, missing values, duplicates, outliers, class balance)
2. Leakage-safe feature engineering & preprocessing pipeline
3. Stratified train/val/test splits (preserving test set for Dev)
4. KNN Classifier implementation, SMOTE handling, and hyperparameter tuning
5. Decision Tree implementation (ID3 / Information Gain vs CART / Gini) and depth tuning
6. Comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC)
7. Generation of all publication-grade figures in reports/figures/
8. Model artifact persistence in models/
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.eda import perform_full_eda
from src.preprocessing import create_reproducible_splits, FraudPreprocessor
from src.models import (
    tune_knn_classifier, tune_decision_tree_classifier,
    evaluate_model_performance, save_trained_model
)
from src.visualization import (
    plot_confusion_matrices, plot_knn_hyperparameter_tuning,
    plot_decision_tree_tuning, plot_roc_and_pr_curves,
    plot_decision_tree_structure
)


def main():
    start_time = time.time()
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION - STAGE 1 REPRODUCIBLE PIPELINE")
    print("Lead Engineer: Anish")
    print("Role: EDA, Preprocessing Pipeline, Data Splits, KNN & Decision Tree Baselines")
    print("=" * 80)

    # File paths
    raw_train_path = os.path.join("data", "raw", "fraudTrain.csv")
    raw_test_path = os.path.join("data", "raw", "fraudTest.csv")
    figures_dir = os.path.join("reports", "figures")
    reports_dir = os.path.join("reports")
    models_dir = os.path.join("models")
    processed_dir = os.path.join("data", "processed")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # -------------------------------------------------------------
    # STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    print("\n>>> STEP 1: PERFORMING COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)...")
    # Load sample for comprehensive EDA reporting (or full)
    print(f"Reading sample from {raw_train_path} for in-depth statistical profiling...")
    df_eda = pd.read_csv(raw_train_path, nrows=100000)
    eda_stats = perform_full_eda(
        df_eda,
        figures_dir=figures_dir,
        report_path=os.path.join(reports_dir, "eda_summary.txt")
    )
    print("Step 1 Complete: EDA figures and summary text generated.")

    # -------------------------------------------------------------
    # STEP 2: LEAKAGE-SAFE PREPROCESSING & DATA SPLITS
    # -------------------------------------------------------------
    print("\n>>> STEP 2: PREPARING LEAKAGE-SAFE PREPROCESSING & REPRODUCIBLE SPLITS...")
    preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_reproducible_splits(
        raw_train_csv=raw_train_path,
        raw_test_csv=raw_test_path,
        sample_train_size=60000,
        sample_val_size=15000,
        sample_test_size=15000,
        random_state=42,
        processed_dir=processed_dir,
        models_dir=models_dir
    )
    feature_names = preprocessor.get_feature_names()
    print(f"Engineered and preprocessed feature count: {len(feature_names)}")
    print("Features:", feature_names)

    # -------------------------------------------------------------
    # STEP 3: K-NEAREST NEIGHBORS (KNN) IMPLEMENTATION & TUNING
    # -------------------------------------------------------------
    print("\n>>> STEP 3: MODEL 1 - K-NEAREST NEIGHBORS (KNN)...")
    k_range = [3, 5, 7, 9, 11]
    best_knn, best_k, knn_history = tune_knn_classifier(
        X_train, y_train, X_val, y_val,
        k_values=k_range,
        use_smote=True,
        random_state=42
    )

    # Save best KNN model
    knn_model_path = os.path.join(models_dir, "knn_best_model.joblib")
    save_trained_model(best_knn, knn_model_path)

    # Plot KNN tuning curve
    knn_tuning_plot_path = os.path.join(figures_dir, "knn_tuning_k.png")
    plot_knn_hyperparameter_tuning(k_range, knn_history, knn_tuning_plot_path)

    # -------------------------------------------------------------
    # STEP 4: DECISION TREE / ID3 IMPLEMENTATION & TUNING
    # -------------------------------------------------------------
    print("\n>>> STEP 4: MODEL 2 - DECISION TREE (ID3/Entropy vs CART/Gini)...")
    depth_range = [3, 4, 5, 6, 8, 10, None]
    best_dt, best_dt_config, dt_entropy_hist, dt_gini_hist = tune_decision_tree_classifier(
        X_train, y_train, X_val, y_val,
        depths=depth_range,
        random_state=42
    )

    # Save best Decision Tree model
    dt_model_path = os.path.join(models_dir, "dt_best_model.joblib")
    save_trained_model(best_dt, dt_model_path)

    # Plot Decision Tree tuning curves
    dt_tuning_plot_path = os.path.join(figures_dir, "dt_tuning_depth.png")
    plot_decision_tree_tuning(depth_range, dt_entropy_hist, dt_gini_hist, dt_tuning_plot_path)

    # Plot Decision Tree structure visualization
    dt_tree_plot_path = os.path.join(figures_dir, "dt_tree_visualization.png")
    plot_decision_tree_structure(best_dt, feature_names, dt_tree_plot_path, max_depth=3)

    # -------------------------------------------------------------
    # STEP 5: COMPARATIVE EVALUATION ON VALIDATION SET
    # -------------------------------------------------------------
    print("\n>>> STEP 5: COMPARATIVE EVALUATION ON VALIDATION SET...")
    # Predictions and probabilities
    knn_preds = best_knn.predict(X_val)
    knn_proba = best_knn.predict_proba(X_val)[:, 1]
    knn_val_metrics = evaluate_model_performance(y_val, knn_preds, knn_proba)

    dt_preds = best_dt.predict(X_val)
    dt_proba = best_dt.predict_proba(X_val)[:, 1]
    dt_val_metrics = evaluate_model_performance(y_val, dt_preds, dt_proba)

    # Plot confusion matrices
    cm_dict = {
        f"KNN (k={best_k}, SMOTE)": np.array(knn_val_metrics["confusion_matrix"]),
        f"Decision Tree ({best_dt_config['criterion']}, d={best_dt_config['max_depth']})": np.array(dt_val_metrics["confusion_matrix"])
    }
    cm_plot_path = os.path.join(figures_dir, "confusion_matrices.png")
    plot_confusion_matrices(cm_dict, cm_plot_path)

    # Plot ROC and PR curves
    models_dict = {
        f"KNN (k={best_k})": best_knn,
        f"Decision Tree ({best_dt_config['criterion']}, d={best_dt_config['max_depth']})": best_dt
    }
    roc_pr_plot_path = os.path.join(figures_dir, "model_comparison_roc_pr.png")
    plot_roc_and_pr_curves(models_dict, X_val, y_val, roc_pr_plot_path)

    # Summary table
    comparison_df = pd.DataFrame([
        {
            "Model": f"K-Nearest Neighbors (k={best_k})",
            "Accuracy": f"{knn_val_metrics['accuracy']:.4f}",
            "Precision": f"{knn_val_metrics['precision']:.4f}",
            "Recall": f"{knn_val_metrics['recall']:.4f}",
            "F1-Score": f"{knn_val_metrics['f1']:.4f}",
            "ROC-AUC": f"{knn_val_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{knn_val_metrics['pr_auc']:.4f}",
            "True Positives": knn_val_metrics['true_positives'],
            "False Positives": knn_val_metrics['false_positives']
        },
        {
            "Model": f"Decision Tree ({best_dt_config['criterion']}, depth={best_dt_config['max_depth']})",
            "Accuracy": f"{dt_val_metrics['accuracy']:.4f}",
            "Precision": f"{dt_val_metrics['precision']:.4f}",
            "Recall": f"{dt_val_metrics['recall']:.4f}",
            "F1-Score": f"{dt_val_metrics['f1']:.4f}",
            "ROC-AUC": f"{dt_val_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{dt_val_metrics['pr_auc']:.4f}",
            "True Positives": dt_val_metrics['true_positives'],
            "False Positives": dt_val_metrics['false_positives']
        }
    ])

    print("\n" + "=" * 80)
    print("STAGE 1 VALIDATION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save results to JSON
    stage1_results = {
        "dataset": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "num_features": len(feature_names),
            "feature_names": feature_names
        },
        "knn": {
            "optimal_k": best_k,
            "weights": "distance",
            "smote_applied": True,
            "validation_metrics": knn_val_metrics,
            "tuning_history": knn_history
        },
        "decision_tree": {
            "best_config": best_dt_config,
            "validation_metrics": dt_val_metrics,
            "entropy_tuning": dt_entropy_hist,
            "gini_tuning": dt_gini_hist
        },
        "execution_time_seconds": round(time.time() - start_time, 2)
    }

    results_json_path = os.path.join(models_dir, "stage1_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(stage1_results, f, indent=4)
    print(f"\nAll Stage 1 results saved to {results_json_path}")
    print(f"Stage 1 execution completed successfully in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()
