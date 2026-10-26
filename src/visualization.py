"""
Visualization Module for Model Evaluation and Hyperparameter Tuning
Credit Card Fraud Detection Project - Stage 1
Author: Anish (Stage 1 Lead)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.tree import plot_tree


def plot_confusion_matrices(cm_dict: dict, save_path: str = "reports/figures/confusion_matrices.png"):
    """
    Plots side-by-side annotated confusion matrix heatmaps for multiple models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        total = np.sum(cm)
        labels = [
            f"TN\n{cm[0, 0]:,}\n({cm[0, 0]/total*100:.1f}%)",
            f"FP (False Alarm)\n{cm[0, 1]:,}\n({cm[0, 1]/total*100:.2f}%)",
            f"FN (Missed Fraud)\n{cm[1, 0]:,}\n({cm[1, 0]/total*100:.2f}%)",
            f"TP (Caught Fraud)\n{cm[1, 1]:,}\n({cm[1, 1]/total*100:.2f}%)"
        ]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    linewidths=1.5, linecolor='white', annot_kws={"fontsize": 11, "fontweight": "bold"})
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"])
        ax.set_yticklabels(["Legitimate (0)", "Fraudulent (1)"])

    plt.suptitle("Model Evaluation: Confusion Matrices on Validation Set", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_knn_hyperparameter_tuning(k_values: list, metrics: dict, save_path: str = "reports/figures/knn_tuning_k.png"):
    """
    Plots KNN performance metrics (Precision, Recall, F1) across k neighbors.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))

    plt.plot(k_values, metrics['f1'], marker='o', linewidth=2.5, color='#2b5c8f', label='F1-Score')
    plt.plot(k_values, metrics['recall'], marker='s', linewidth=2, linestyle='--', color='#28a745', label='Recall (Fraud Caught)')
    plt.plot(k_values, metrics['precision'], marker='^', linewidth=2, linestyle=':', color='#d9534f', label='Precision (Low False Alarms)')

    # Highlight best k by F1
    best_idx = np.argmax(metrics['f1'])
    best_k = k_values[best_idx]
    best_f1 = metrics['f1'][best_idx]
    plt.scatter([best_k], [best_f1], s=160, color='gold', edgecolor='black', zorder=5,
                label=f'Optimal k={best_k} (F1={best_f1:.4f})')

    plt.title("KNN Hyperparameter Tuning: Metrics vs. Number of Neighbors (k)", fontsize=13, fontweight='bold')
    plt.xlabel("k (Number of Nearest Neighbors)", fontsize=11)
    plt.ylabel("Validation Metric Score", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"KNN tuning plot saved to {save_path}")


def plot_decision_tree_tuning(depths: list, entropy_metrics: dict, gini_metrics: dict,
                              save_path: str = "reports/figures/dt_tuning_depth.png"):
    """
    Plots Decision Tree F1 and Recall across depths, comparing Information Gain (entropy) vs Gini Index.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth_labels = [str(d) if d is not None else "None" for d in depths]
    x = np.arange(len(depths))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # F1 Score comparison
    axes[0].plot(x, entropy_metrics['f1'], marker='o', linewidth=2.5, color='#2b5c8f', label='Entropy / ID3 (Info Gain)')
    axes[0].plot(x, gini_metrics['f1'], marker='s', linewidth=2.5, color='#d9534f', linestyle='--', label='Gini Index (CART)')
    axes[0].set_title("Decision Tree F1-Score vs. Max Depth", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Max Tree Depth", fontsize=11)
    axes[0].set_ylabel("F1-Score", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(depth_labels)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Recall comparison
    axes[1].plot(x, entropy_metrics['recall'], marker='o', linewidth=2.5, color='#2b5c8f', label='Entropy / ID3 (Info Gain)')
    axes[1].plot(x, gini_metrics['recall'], marker='s', linewidth=2.5, color='#d9534f', linestyle='--', label='Gini Index (CART)')
    axes[1].set_title("Decision Tree Recall vs. Max Depth", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Max Tree Depth", fontsize=11)
    axes[1].set_ylabel("Recall (Fraud Caught)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(depth_labels)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.suptitle("Hyperparameter Tuning: Decision Tree Depth & Splitting Criterion", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision Tree tuning plot saved to {save_path}")


def plot_roc_and_pr_curves(models_dict: dict, X_val, y_val, save_path: str = "reports/figures/model_comparison_roc_pr.png"):
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
    Essential for highly imbalanced credit card fraud datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#2b5c8f', '#28a745', '#d9534f', '#6f42c1', '#fd7e14']

    for (name, model), color in zip(models_dict.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = model.predict(X_val)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_score = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.4f})", linewidth=2.2, color=color)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_val, y_proba)
        ap_score = average_precision_score(y_val, y_proba)
        axes[1].plot(recall, precision, label=f"{name} (PR-AUC = {ap_score:.4f})", linewidth=2.2, color=color)

    # ROC formatting
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    axes[0].set_title("ROC Curves (False Positive Rate vs True Positive Rate)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # PR formatting
    baseline_pr = y_val.mean()
    axes[1].axhline(baseline_pr, color='k', linestyle='--', alpha=0.6, label=f'Baseline Rate ({baseline_pr*100:.2f}%)')
    axes[1].set_title("Precision-Recall Curves (Critical for Imbalanced Data)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Recall (Fraction of Frauds Caught)", fontsize=11)
    axes[1].set_ylabel("Precision (Fraction of Valid Flags)", fontsize=11)
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Model Evaluation: Discrimination Capacity on Validation Set", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC and PR curves saved to {save_path}")


def plot_decision_tree_structure(model, feature_names: list, save_path: str = "reports/figures/dt_tree_visualization.png", max_depth: int = 3):
    """
    Visualizes the first few levels of the Decision Tree to demonstrate
    ID3 / Information Gain split rules for BTech viva defense.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["Legitimate", "Fraud"],
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2,
        proportion=False
    )
    plt.title(f"Decision Tree Splitting Logic (Top {max_depth} Levels - Information Gain / Entropy)",
              fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision tree structure saved to {save_path}")
