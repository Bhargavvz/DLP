"""
DeepFinDLP - Main Pipeline Orchestrator
End-to-end: download → preprocess → feature engineering → train → evaluate → visualize.
"""
import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data_loader import get_full_pipeline, compute_class_weights
from src.feature_engineering import full_feature_engineering
from src.trainer import Trainer
from src.evaluator import compute_metrics, print_metrics_table
from src.models.dnn_model import BasicDNN
from src.models.cnn_model import CNN1DModel
from src.models.lstm_model import BiLSTMModel
from src.models.cnn_lstm_model import CNNBiLSTMModel
from src.models.deep_fin_dlp import DeepFinDLPModel
from src.models.baselines import train_baselines
from visualization.generate_figures import generate_all_figures


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ██████╗ ███████╗███████╗██████╗ ███████╗██╗███╗   ██╗                ║
║   ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██║████╗  ██║                ║
║   ██║  ██║█████╗  █████╗  ██████╔╝█████╗  ██║██╔██╗ ██║                ║
║   ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██║██║╚██╗██║                ║
║   ██████╔╝███████╗███████╗██║     ██║     ██║██║ ╚████║                ║
║   ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═╝╚═╝  ╚═══╝                ║
║                                                                        ║
║   Deep Learning-Driven Data Leakage Prevention                         ║
║   For Mitigating Financial Instability                                  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_info():
    print("\n" + "=" * 70)
    print("System Information")
    print("=" * 70)
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Num GPUs: {torch.cuda.device_count()}")
        print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
    print(f"  Device: {config.DEVICE}")
    print(f"  AMP Enabled: {config.USE_AMP}")
    print(f"  AMP Dtype: {config.AMP_DTYPE}")
    print("=" * 70)


def run_data_pipeline(args):
    """Step 1: Data loading and preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 1: Data Pipeline")
    print("=" * 70)

    # Check if data exists
    if not os.path.exists(config.PARQUET_FILE):
        print("  Dataset not found. Downloading...")
        from download_data import download_dataset, prepare_dataset
        download_dataset()
        prepare_dataset()

    # Run the full pipeline
    batch_size = 256 if args.quick_test else config.BATCH_SIZE
    result = get_full_pipeline(batch_size=batch_size)

    return result


def run_feature_engineering(data_result, args):
    """Step 2: Feature engineering (subsampled for speed)."""
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)

    X_train, y_train = data_result["train_data"]

    # Subsample for speed — MI and correlation don't need all 11M rows
    max_fe_samples = 100000
    if len(X_train) > max_fe_samples:
        print(f"  Subsampling {max_fe_samples:,} rows for feature engineering (from {len(X_train):,})")
        indices = np.random.RandomState(42).choice(len(X_train), max_fe_samples, replace=False)
        X_sub, y_sub = X_train[indices], y_train[indices]
    else:
        X_sub, y_sub = X_train, y_train

    fe_result = full_feature_engineering(
        X_sub, y_sub,
        feature_names=data_result["artifacts"]["feature_names"],
        top_k=config.TOP_K_FEATURES,
    )

    data_result["mi_scores"] = fe_result.get("mi_scores")
    return data_result


def run_baselines(data_result, args):
    """Step 3: Train traditional ML baselines."""
    print("\n" + "=" * 70)
    print("STEP 3: Traditional ML Baselines")
    print("=" * 70)

    X_train, y_train = data_result["train_data"]
    X_test, y_test = data_result["test_data"]

    # Subsample for quick test
    if args.quick_test:
        max_samples = 10000
        if len(X_train) > max_samples:
            indices = np.random.RandomState(42).choice(len(X_train), max_samples, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]

    baseline_results = train_baselines(X_train, y_train, X_test, y_test)

    # Compute full metrics for baselines
    class_names = data_result["artifacts"]["class_names"]
    for key, res in baseline_results.items():
        if res is not None:
            full_metrics = compute_metrics(
                y_test, res["y_pred"],
                y_proba=res.get("y_proba"),
                class_names=class_names,
                model_name=res["model_name"]
            )
            baseline_results[key].update(full_metrics)

    return baseline_results


def run_dl_training(data_result, args):
    """Step 4: Train deep learning models."""
    print("\n" + "=" * 70)
    print("STEP 4: Deep Learning Models")
    print("=" * 70)

    train_loader = data_result["train_loader"]
    val_loader = data_result["val_loader"]
    test_loader = data_result["test_loader"]
    num_features = data_result["artifacts"]["num_features"]
    num_classes = data_result["artifacts"]["num_classes"]
    class_names = data_result["artifacts"]["class_names"]
    class_weights = data_result["class_weights"]

    epochs = 10 if args.quick_test else config.EPOCHS
    dl_results = {}
    all_histories = {}

    # ── Model configurations ──
    models_config = {
        "dnn": {
            "class": BasicDNN,
            "kwargs": {
                "num_features": num_features,
                "num_classes": num_classes,
                **config.DNN_CONFIG,
            },
        },
        "cnn": {
            "class": CNN1DModel,
            "kwargs": {
                "num_features": num_features,
                "num_classes": num_classes,
                **config.CNN_CONFIG,
            },
        },
        "lstm": {
            "class": BiLSTMModel,
            "kwargs": {
                "num_features": num_features,
                "num_classes": num_classes,
                **config.LSTM_CONFIG,
            },
        },
        "cnn_lstm": {
            "class": CNNBiLSTMModel,
            "kwargs": {
                "num_features": num_features,
                "num_classes": num_classes,
                **config.CNN_LSTM_CONFIG,
            },
        },
        "deep_fin_dlp": {
            "class": DeepFinDLPModel,
            "kwargs": {
                "num_features": num_features,
                "num_classes": num_classes,
                **config.DEEP_FIN_DLP_CONFIG,
            },
        },
    }

    for model_key, model_cfg in models_config.items():
        print(f"\n{'─'*60}")
        display_name = config.MODEL_NAMES.get(model_key, model_key)

        # Create model
        model = model_cfg["class"](**model_cfg["kwargs"])

        if hasattr(model, "summary"):
            model.summary()

        # Train
        trainer = Trainer(
            model=model,
            model_name=model_key,
            class_weights=class_weights,
            epochs=epochs,
        )

        history, train_time = trainer.train(train_loader, val_loader)
        all_histories[model_key] = history

        # Predict on test set
        predictions = trainer.predict(test_loader)

        # Compute metrics
        metrics = compute_metrics(
            predictions["y_true"],
            predictions["y_pred"],
            y_proba=predictions["y_proba"],
            class_names=class_names,
            model_name=display_name,
        )
        metrics["train_time"] = train_time

        # Store features for t-SNE
        if "features" in predictions:
            metrics["features"] = predictions["features"]

        # Get attention weights for the proposed model
        if model_key == "deep_fin_dlp":
            try:
                attn_weights = model.get_attention_weights()
                if attn_weights is not None:
                    dl_results["attention_weights"] = attn_weights
            except Exception:
                pass

        dl_results[model_key] = metrics

        print(f"\n  {display_name} Results:")
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"    F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"    AUC-ROC:      {metrics['auc_roc_weighted']:.4f}")

    return dl_results, all_histories


def run_evaluation(baseline_results, dl_results, data_result):
    """Step 5: Comprehensive evaluation and comparison."""
    print("\n" + "=" * 70)
    print("STEP 5: Comprehensive Evaluation")
    print("=" * 70)

    class_names = data_result["artifacts"]["class_names"]

    # Combine all results
    all_results = {}
    for key, res in baseline_results.items():
        if res is not None:
            all_results[key] = res
    for key, res in dl_results.items():
        if key != "attention_weights":
            all_results[key] = res

    # Print comparison table
    print_metrics_table(all_results, class_names)

    # Print detailed classification report for the proposed model
    if "deep_fin_dlp" in all_results:
        print("\n" + "=" * 70)
        print("Classification Report — DeepFinDLP (Proposed)")
        print("=" * 70)
        print(all_results["deep_fin_dlp"].get("classification_report", "N/A"))

    # Save results to JSON
    results_summary = {}
    for key, res in all_results.items():
        results_summary[key] = {
            "model_name": res.get("model_name", key),
            "accuracy": float(res.get("accuracy", 0)),
            "precision_weighted": float(res.get("precision_weighted", 0)),
            "recall_weighted": float(res.get("recall_weighted", 0)),
            "f1_weighted": float(res.get("f1_weighted", 0)),
            "f1_macro": float(res.get("f1_macro", 0)),
            "auc_roc_weighted": float(res.get("auc_roc_weighted", 0)),
            "train_time": float(res.get("train_time", 0)),
        }

    results_path = os.path.join(config.RESULTS_DIR, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_results


def run_visualization(all_results, all_histories, data_result):
    """Step 6: Generate all figures."""
    class_names = data_result["artifacts"]["class_names"]
    generate_all_figures(data_result, all_results, all_histories, class_names)


def main():
    parser = argparse.ArgumentParser(description="DeepFinDLP - Full Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "data", "baselines", "train", "evaluate", "visualize"],
                       help="Pipeline stage to run")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with reduced data and epochs")
    parser.add_argument("--skip-baselines", action="store_true",
                       help="Skip traditional ML baselines")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    args = parser.parse_args()

    print_banner()
    print_system_info()

    total_start = time.time()

    # ── Step 1: Data Pipeline ──
    data_result = run_data_pipeline(args)

    # ── Step 2: Feature Engineering ──
    data_result = run_feature_engineering(data_result, args)

    # ── Step 3: Baselines ──
    baseline_results = {}
    if not args.skip_baselines:
        baseline_results = run_baselines(data_result, args)

    # ── Step 4: DL Training ──
    dl_results, all_histories = run_dl_training(data_result, args)

    # ── Step 5: Evaluation ──
    all_results = run_evaluation(baseline_results, dl_results, data_result)

    # ── Step 6: Visualization ──
    run_visualization(all_results, all_histories, data_result)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {config.RESULTS_DIR}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
