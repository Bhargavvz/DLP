"""
DeepFinDLP - Evaluation Module
Comprehensive metrics computation for all models.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import label_binarize
import time


def compute_metrics(y_true, y_pred, y_proba=None, class_names=None,
                    model_name="Model"):
    """Compute comprehensive evaluation metrics."""
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
        "y_true": y_true,
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_class_f1"] = dict(zip(class_names[:len(per_class_f1)], per_class_f1))

    # AUC-ROC (one-vs-rest)
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])

            # Ensure y_proba has correct shape
            if y_proba.shape[1] >= num_classes:
                y_proba_use = y_proba[:, :num_classes]
            else:
                y_proba_use = y_proba

            metrics["auc_roc_macro"] = roc_auc_score(
                y_bin, y_proba_use, average="macro", multi_class="ovr"
            )
            metrics["auc_roc_weighted"] = roc_auc_score(
                y_bin, y_proba_use, average="weighted", multi_class="ovr"
            )

            # Per-class AUC
            try:
                per_class_auc = roc_auc_score(y_bin, y_proba_use, average=None)
                metrics["per_class_auc"] = dict(zip(
                    class_names[:len(per_class_auc)], per_class_auc
                ))
            except Exception:
                pass

            metrics["y_proba"] = y_proba_use
        except Exception as e:
            print(f"    Warning: Could not compute AUC-ROC: {e}")
            metrics["auc_roc_macro"] = 0.0
            metrics["auc_roc_weighted"] = 0.0
    else:
        metrics["auc_roc_macro"] = 0.0
        metrics["auc_roc_weighted"] = 0.0

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=class_names[:num_classes],
        zero_division=0
    )

    return metrics


def print_metrics_table(all_results: dict, class_names=None):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - DeepFinDLP Results")
    print("=" * 100)

    header = f"{'Model':<25} {'Acc':>8} {'Prec(W)':>8} {'Rec(W)':>8} {'F1(W)':>8} {'F1(M)':>8} {'AUC':>8} {'Time':>8}"
    print(header)
    print("-" * 100)

    for model_key, res in all_results.items():
        if res is None:
            continue

        name = res.get("model_name", model_key)
        acc = res.get("accuracy", 0)
        prec = res.get("precision_weighted", 0)
        rec = res.get("recall_weighted", 0)
        f1w = res.get("f1_weighted", 0)
        f1m = res.get("f1_macro", 0)
        auc = res.get("auc_roc_weighted", 0)
        train_time = res.get("train_time", 0)

        print(f"  {name:<23} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1w:>8.4f} {f1m:>8.4f} {auc:>8.4f} {train_time:>7.1f}s")

    print("=" * 100)

    # Print per-class F1 for each model
    if class_names:
        print(f"\nPer-Class F1 Scores:")
        print("-" * 100)
        header = f"{'Class':<25}"
        for model_key, res in all_results.items():
            if res is not None:
                short_name = res.get("model_name", model_key)[:12]
                header += f" {short_name:>12}"
        print(header)
        print("-" * 100)

        for cls in class_names:
            row = f"  {cls:<23}"
            for model_key, res in all_results.items():
                if res is not None:
                    f1 = res.get("per_class_f1", {}).get(cls, 0)
                    row += f" {f1:>12.4f}"
            print(row)

    print("=" * 100)
