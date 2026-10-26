"""
Model Implementation, Hyperparameter Tuning, and Evaluation Module
Credit Card Fraud Detection Project - Stage 1
Author: Anish (Stage 1 Lead)

Algorithms implemented:
1. K-Nearest Neighbors (KNN) Classifier (Distance-weighted, SMOTE-balanced)
2. Decision Tree Classifier (ID3/Information Gain via Entropy vs CART via Gini)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, classification_report
)
from imblearn.over_sampling import SMOTE


def evaluate_model_performance(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes all standard BTech evaluation metrics for imbalanced classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist()
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def tune_knn_classifier(X_train, y_train, X_val, y_val, k_values=None, use_smote=True, random_state=42):
    """
    Evaluates KNN over a range of neighbors (k).
    Applies SMOTE strictly on X_train to overcome minority class starvation.
    """
    if k_values is None:
        k_values = [3, 5, 7, 9, 11]

    print("=" * 70)
    print("TUNING K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
    print("=" * 70)

    # Apply SMOTE only to training data to handle severe class imbalance
    if use_smote:
        print("Applying SMOTE oversampling to training set (sampling_strategy=0.20)...")
        smote = SMOTE(sampling_strategy=0.20, random_state=random_state)
        X_tr_fit, y_tr_fit = smote.fit_resample(X_train, y_train)
        print(f"Resampled training set: {len(X_tr_fit):,} samples (Fraud: {y_tr_fit.sum():,})")
    else:
        X_tr_fit, y_tr_fit = X_train, y_train

    tuning_history = {"k": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    best_k = None
    best_f1 = -1.0
    best_model = None

    for k in k_values:
        print(f"Evaluating KNN with k={k} (weights='distance')...")
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        knn.fit(X_tr_fit, y_tr_fit)

        preds = knn.predict(X_val)
        proba = knn.predict_proba(X_val)[:, 1]
        m = evaluate_model_performance(y_val, preds, proba)

        tuning_history["k"].append(k)
        tuning_history["accuracy"].append(m["accuracy"])
        tuning_history["precision"].append(m["precision"])
        tuning_history["recall"].append(m["recall"])
        tuning_history["f1"].append(m["f1"])

        print(f"  k={k:2d} -> Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | "
              f"Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | TP: {m['true_positives']} | FP: {m['false_positives']}")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_k = k
            best_model = knn

    print(f"\nOptimal KNN hyperparameter: k={best_k} with Validation F1 = {best_f1:.4f}")
    return best_model, best_k, tuning_history


def tune_decision_tree_classifier(X_train, y_train, X_val, y_val, depths=None, random_state=42):
    """
    Compares ID3 / Information Gain (criterion='entropy') vs CART (criterion='gini')
    across tree depths to study overfitting, pruning, and recall vs precision.
    """
    if depths is None:
        depths = [3, 4, 5, 6, 8, 10, None]

    print("=" * 70)
    print("TUNING DECISION TREE CLASSIFIER (ID3/Entropy vs CART/Gini)")
    print("=" * 70)

    results_entropy = {"depth": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    results_gini = {"depth": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    best_model = None
    best_f1 = -1.0
    best_config = None

    for criterion in ['entropy', 'gini']:
        print(f"\n--- Testing Criterion: {criterion.upper()} ({'ID3 / Info Gain' if criterion == 'entropy' else 'CART / Gini Index'}) ---")
        history = results_entropy if criterion == 'entropy' else results_gini

        for d in depths:
            d_str = str(d) if d is not None else "None"
            dt = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=d,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=random_state
            )
            dt.fit(X_train, y_train)

            preds = dt.predict(X_val)
            proba = dt.predict_proba(X_val)[:, 1]
            m = evaluate_model_performance(y_val, preds, proba)

            history["depth"].append(d)
            history["accuracy"].append(m["accuracy"])
            history["precision"].append(m["precision"])
            history["recall"].append(m["recall"])
            history["f1"].append(m["f1"])

            print(f"  Depth: {d_str:>4} -> Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | "
                  f"Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | TP: {m['true_positives']} | FP: {m['false_positives']}")

            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_config = {"criterion": criterion, "max_depth": d}
                best_model = dt

    print(f"\nOptimal Decision Tree: {best_config} with Validation F1 = {best_f1:.4f}")
    return best_model, best_config, results_entropy, results_gini


def save_trained_model(model, filepath: str):
    """Saves serialized model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to {filepath}")


def load_trained_model(filepath: str):
    """Loads serialized model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded successfully from {filepath}")
    return model
