"""
DeepFinDLP - Traditional ML Baselines
Random Forest, SVM, XGBoost for comparison.
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def _evaluate(model, X_test, y_test, model_name):
    """Evaluate a sklearn model and return metrics."""
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
        "inference_time": inference_time,
    }

    # Try to get probability predictions for ROC
    if hasattr(model, "predict_proba"):
        metrics["y_proba"] = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        metrics["y_proba"] = model.decision_function(X_test)

    return metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest."""
    print("\n  Training Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(**config.RF_CONFIG)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"    Training time: {train_time:.1f}s")

    metrics = _evaluate(rf, X_test, y_test, "Random Forest")
    metrics["train_time"] = train_time
    metrics["model"] = rf
    metrics["feature_importances"] = rf.feature_importances_

    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    return metrics


def train_svm(X_train, y_train, X_test, y_test, max_samples=50000):
    """Train and evaluate SVM (subsampled for speed)."""
    print("\n  Training SVM (RBF kernel)...")

    # SVM doesn't scale well — subsample for feasibility
    if len(X_train) > max_samples:
        print(f"    Subsampling {max_samples} samples for SVM (from {len(X_train):,})")
        indices = np.random.RandomState(config.RANDOM_STATE).choice(
            len(X_train), max_samples, replace=False
        )
        X_sub, y_sub = X_train[indices], y_train[indices]
    else:
        X_sub, y_sub = X_train, y_train

    start = time.time()
    svm = SVC(**config.SVM_CONFIG, probability=True)
    svm.fit(X_sub, y_sub)
    train_time = time.time() - start
    print(f"    Training time: {train_time:.1f}s")

    metrics = _evaluate(svm, X_test, y_test, "SVM (RBF)")
    metrics["train_time"] = train_time
    metrics["model"] = svm

    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost with GPU acceleration."""
    if not HAS_XGB:
        print("\n  [SKIP] XGBoost not installed. Install with: pip install xgboost")
        return None

    print("\n  Training XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(**config.XGB_CONFIG)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - start
    print(f"    Training time: {train_time:.1f}s")

    metrics = _evaluate(xgb_model, X_test, y_test, "XGBoost")
    metrics["train_time"] = train_time
    metrics["model"] = xgb_model
    metrics["feature_importances"] = xgb_model.feature_importances_

    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    return metrics


def train_baselines(X_train, y_train, X_test, y_test):
    """Train all baseline models and return results."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Training Traditional ML Baselines")
    print("=" * 70)

    results = {}
    results["rf"] = train_random_forest(X_train, y_train, X_test, y_test)
    results["svm"] = train_svm(X_train, y_train, X_test, y_test)
    results["xgb"] = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n" + "-" * 50)
    print("Baseline Summary:")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 (W)':>10}")
    print("-" * 50)
    for name, res in results.items():
        if res is not None:
            print(f"  {res['model_name']:<20} {res['accuracy']:>10.4f} {res['f1_weighted']:>10.4f}")

    return results


def evaluate_baseline(model, X_test, y_test, model_name):
    """Evaluate a pre-trained baseline model."""
    return _evaluate(model, X_test, y_test, model_name)
