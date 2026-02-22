"""
DeepFinDLP - Comprehensive Visualization Module
Generates 16+ publication-quality figures for IEEE research paper.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless for server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Seaborn theme
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": config.FIGURE_DPI,
    "savefig.dpi": config.FIGURE_DPI,
    "savefig.bbox_inches": "tight",
    "font.size": config.FONT_SIZE,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 8),
})


def save_figure(fig, name):
    """Save figure to the figures directory."""
    path = os.path.join(config.FIGURES_DIR, f"{name}.{config.FIGURE_FORMAT}")
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Dataset Class Distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_class_distribution(df, label_col="DLP_Label"):
    """Bar chart of attack type frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    counts = df[label_col].value_counts()
    colors = sns.color_palette("husl", len(counts))

    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.9)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel("DLP Threat Category", fontsize=12)
    ax.set_title("CIC-IDS2017 Dataset — DLP Threat Category Distribution", fontsize=14, fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Number of Samples (log scale)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    return save_figure(fig, "01_class_distribution")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Feature Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df, top_n=30):
    """Correlation heatmap of top features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:top_n]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True)

    ax.set_title(f"Feature Correlation Heatmap (Top {top_n})", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    return save_figure(fig, "02_correlation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Feature Importance (Mutual Information)
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_importance(mi_scores=None, feature_importances=None,
                            feature_names=None, top_n=20):
    """Top-N feature importance from mutual information or RF."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if mi_scores is not None:
        scores = mi_scores.head(top_n)
        title = "Top Features by Mutual Information Score"
    elif feature_importances is not None and feature_names is not None:
        idx = np.argsort(feature_importances)[-top_n:]
        scores = dict(zip(
            [feature_names[i] for i in idx],
            feature_importances[idx]
        ))
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        title = "Top Features by Random Forest Importance"
    else:
        return None

    names = list(scores.keys()) if isinstance(scores, dict) else scores.index.tolist()
    values = list(scores.values()) if isinstance(scores, dict) else scores.values

    colors = sns.color_palette("viridis", len(names))
    ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    return save_figure(fig, "03_feature_importance")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES 4-6: Training Curves (Loss, Accuracy, F1)
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves(all_histories: dict):
    """Plot loss, accuracy, and F1 curves for all DL models."""
    metrics = [
        ("loss", "Training & Validation Loss", "Loss"),
        ("acc", "Training & Validation Accuracy", "Accuracy"),
        ("f1", "Training & Validation F1-Score", "F1-Score"),
    ]

    colors = sns.color_palette("husl", len(all_histories))
    paths = []

    for metric_key, title, ylabel in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (model_name, history) in enumerate(all_histories.items()):
            color = colors[idx]
            display_name = config.MODEL_NAMES.get(model_name, model_name)

            train_key = f"train_{metric_key}"
            val_key = f"val_{metric_key}"

            if train_key in history and val_key in history:
                epochs = range(1, len(history[train_key]) + 1)

                # Training curves
                axes[0].plot(epochs, history[train_key], label=display_name,
                           color=color, linewidth=2)
                axes[0].set_title(f"Training {ylabel}", fontsize=13, fontweight="bold")

                # Validation curves
                axes[1].plot(epochs, history[val_key], label=display_name,
                           color=color, linewidth=2, linestyle="--")
                axes[1].set_title(f"Validation {ylabel}", fontsize=13, fontweight="bold")

        for ax in axes:
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_num = {"loss": "04", "acc": "05", "f1": "06"}[metric_key]
        paths.append(save_figure(fig, f"{fig_num}_training_{metric_key}"))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: ROC Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_roc_curves(all_results: dict, class_names: list):
    """Multi-class ROC curves for all models."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("husl", len(all_results))

    for idx, (model_key, res) in enumerate(all_results.items()):
        if res is None or "y_proba" not in res:
            continue

        y_true = res["y_true"] if "y_true" in res else None
        y_proba = res["y_proba"]
        if y_true is None:
            continue

        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        # Compute micro-average ROC
        y_proba_use = y_proba[:, :num_classes] if y_proba.shape[1] >= num_classes else y_proba
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba_use.ravel())
        roc_auc = auc(fpr, tpr)

        display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))
        ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                label=f'{display_name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models (Micro-Average)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    return save_figure(fig, "07_roc_curves")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Precision-Recall Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_precision_recall_curves(all_results: dict, class_names: list):
    """Precision-Recall curves for all models."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("husl", len(all_results))

    for idx, (model_key, res) in enumerate(all_results.items()):
        if res is None or "y_proba" not in res:
            continue

        y_true = res.get("y_true")
        y_proba = res["y_proba"]
        if y_true is None:
            continue

        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        y_proba_use = y_proba[:, :num_classes] if y_proba.shape[1] >= num_classes else y_proba
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba_use.ravel())
        ap = average_precision_score(y_bin, y_proba_use, average="micro")

        display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))
        ax.plot(recall, precision, color=colors[idx], linewidth=2,
                label=f'{display_name} (AP = {ap:.4f})')

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models (Micro-Average)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    return save_figure(fig, "08_precision_recall_curves")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES 9-10: Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, fig_name):
    """Plot a single confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={"label": "Proportion"})

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    return save_figure(fig, fig_name)


def plot_confusion_matrices(all_results: dict, class_names: list):
    """Plot confusion matrices for the proposed model and baselines."""
    paths = []

    # Proposed model (DeepFinDLP)
    if "deep_fin_dlp" in all_results and all_results["deep_fin_dlp"] is not None:
        res = all_results["deep_fin_dlp"]
        paths.append(plot_confusion_matrix(
            res["y_true"], res["y_pred"], class_names,
            "DeepFinDLP (Proposed)", "09_confusion_matrix_proposed"
        ))

    # All models comparison (2x3 or 2x4 grid)
    models_to_plot = {k: v for k, v in all_results.items()
                      if v is not None and k != "deep_fin_dlp" and "y_pred" in v}

    if models_to_plot:
        n = len(models_to_plot)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, (model_key, res) in enumerate(models_to_plot.items()):
            if idx >= len(axes):
                break
            cm = confusion_matrix(res["y_true"], res["y_pred"])
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))

            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[idx], linewidths=0.5, vmin=0, vmax=1)
            axes[idx].set_title(display_name, fontsize=11, fontweight="bold")
            axes[idx].set_xlabel("Predicted", fontsize=9)
            axes[idx].set_ylabel("True", fontsize=9)
            axes[idx].tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(len(models_to_plot), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Confusion Matrices — Baseline Models", fontsize=14, fontweight="bold")
        plt.tight_layout()
        paths.append(save_figure(fig, "10_confusion_matrices_baselines"))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11: Model Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_comparison(all_results: dict):
    """Bar chart comparing Accuracy, F1, and AUC across all models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    model_names = []
    accuracies = []
    f1_scores = []
    auc_scores = []

    for model_key, res in all_results.items():
        if res is None:
            continue
        display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))
        model_names.append(display_name)
        accuracies.append(res.get("accuracy", 0))
        f1_scores.append(res.get("f1_weighted", 0))
        auc_scores.append(res.get("auc_roc_weighted", 0))

    x = np.arange(len(model_names))
    width = 0.25

    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy",
                   color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, f1_scores, width, label="F1-Score (Weighted)",
                   color="#4CAF50", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, auc_scores, width, label="AUC-ROC",
                   color="#FF9800", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim([0.8, 1.01])
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight="bold")

    return save_figure(fig, "11_model_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12: t-SNE Embedding
# ═══════════════════════════════════════════════════════════════════════════
def plot_tsne_embedding(features, labels, class_names, model_name="DeepFinDLP",
                        max_samples=5000):
    """2D t-SNE visualization of learned features."""
    if features is None or len(features) == 0:
        print("  [SKIP] No features available for t-SNE")
        return None

    # Subsample for speed
    if len(features) > max_samples:
        indices = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"  Computing t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors[i]], label=name, alpha=0.6, s=10, edgecolors="none")

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(f"t-SNE Visualization of Learned Features — {model_name}",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, markerscale=3)

    return save_figure(fig, "12_tsne_embedding")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13: Attention Weight Visualization
# ═══════════════════════════════════════════════════════════════════════════
def plot_attention_weights(attention_weights, head_idx=0):
    """Heatmap of attention patterns from the multi-head attention."""
    if attention_weights is None:
        print("  [SKIP] No attention weights available")
        return None

    # attention_weights: (batch, heads, seq, seq)
    # Take mean over batch, select a specific head
    if len(attention_weights.shape) == 4:
        attn = attention_weights[0].cpu().numpy()  # First sample
    else:
        attn = attention_weights.cpu().numpy()

    num_heads = attn.shape[0]
    fig, axes = plt.subplots(2, min(4, num_heads), figsize=(16, 8))
    axes = axes.flatten()

    for h in range(min(8, num_heads)):
        if h >= len(axes):
            break
        sns.heatmap(attn[h], cmap="YlOrRd", ax=axes[h], cbar=False,
                    square=True, linewidths=0, xticklabels=False, yticklabels=False)
        axes[h].set_title(f"Head {h+1}", fontsize=10)

    for idx in range(min(8, num_heads), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Multi-Head Self-Attention Weights — DeepFinDLP",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, "13_attention_weights")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14: Per-Class F1 Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_class_f1(all_results: dict, class_names: list):
    """Grouped bar chart of per-class F1 for all models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    models = {}
    for model_key, res in all_results.items():
        if res is None or "per_class_f1" not in res:
            continue
        display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))
        models[display_name] = res["per_class_f1"]

    if not models:
        return None

    n_models = len(models)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / n_models

    colors = sns.color_palette("husl", n_models)

    for i, (model_name, f1_dict) in enumerate(models.items()):
        f1_values = [f1_dict.get(cls, 0) for cls in class_names]
        ax.bar(x + i * width - 0.4 + width/2, f1_values, width,
               label=model_name, color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1-Score Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    return save_figure(fig, "14_per_class_f1")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 15: Training Time Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_time(all_results: dict):
    """Bar chart of wall-clock training times."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = []
    times = []

    for model_key, res in all_results.items():
        if res is None or "train_time" not in res:
            continue
        display_name = config.MODEL_NAMES.get(model_key, res.get("model_name", model_key))
        model_names.append(display_name)
        times.append(res["train_time"])

    colors = sns.color_palette("coolwarm", len(model_names))
    bars = ax.bar(model_names, times, color=colors, edgecolor="black", linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=9, fontweight="bold")

    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    return save_figure(fig, "15_training_time")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 16: Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════
def plot_architecture_diagram():
    """Generate a visual architecture diagram of DeepFinDLP."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Define block positions and sizes
    blocks = [
        {"name": "Input\n(80 features)", "x": 0.3, "y": 3.0, "w": 1.8, "h": 2.0, "color": "#E3F2FD"},
        {"name": "BatchNorm", "x": 2.5, "y": 3.3, "w": 1.3, "h": 1.4, "color": "#F3E5F5"},
        {"name": "1D-CNN\nBlock\n(3 layers)", "x": 4.2, "y": 2.5, "w": 1.5, "h": 3.0, "color": "#E8F5E9"},
        {"name": "SE\nBlock", "x": 6.1, "y": 3.0, "w": 1.2, "h": 2.0, "color": "#FFF3E0"},
        {"name": "BiLSTM\n(2 layers)", "x": 7.7, "y": 2.5, "w": 1.5, "h": 3.0, "color": "#E1F5FE"},
        {"name": "Multi-Head\nAttention\n(8 heads)", "x": 9.6, "y": 2.5, "w": 1.6, "h": 3.0, "color": "#FCE4EC"},
        {"name": "Global\nPool", "x": 11.6, "y": 3.0, "w": 1.2, "h": 2.0, "color": "#F1F8E9"},
        {"name": "Residual\nFC Head", "x": 13.2, "y": 3.0, "w": 1.3, "h": 2.0, "color": "#E8EAF6"},
        {"name": "Output\n(8 classes)", "x": 14.9, "y": 3.3, "w": 1.0, "h": 1.4, "color": "#FFEBEE"},
    ]

    for b in blocks:
        rect = plt.Rectangle((b["x"], b["y"]), b["w"], b["h"],
                            linewidth=2, edgecolor="black", facecolor=b["color"],
                            alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]/2, b["name"],
               ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

    # Draw arrows
    for i in range(len(blocks) - 1):
        x_start = blocks[i]["x"] + blocks[i]["w"]
        x_end = blocks[i+1]["x"]
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2

        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                   arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    # Add residual connection arc (Attention input to output)
    ax.annotate("", xy=(11.6, 5.8), xytext=(9.6, 5.8),
               arrowprops=dict(arrowstyle="->", lw=1.5, color="red",
                             connectionstyle="arc3,rad=0.3", linestyle="--"))
    ax.text(10.6, 6.5, "Residual", ha="center", fontsize=8, color="red", fontstyle="italic")

    ax.set_title("DeepFinDLP — Temporal Convolutional Transformer Architecture",
                fontsize=16, fontweight="bold", pad=20)

    return save_figure(fig, "16_architecture_diagram")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION: Generate All Figures
# ═══════════════════════════════════════════════════════════════════════════
def generate_all_figures(data_pipeline_result, all_results, all_histories,
                          class_names):
    """Generate all 16 figures."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Generating Publication Figures")
    print("=" * 70)

    generated = []

    # 1. Class distribution
    try:
        df = data_pipeline_result.get("dataframe")
        if df is not None:
            p = plot_class_distribution(df)
            generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Class distribution: {e}")

    # 2. Correlation heatmap
    try:
        if df is not None:
            p = plot_correlation_heatmap(df)
            generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Correlation heatmap: {e}")

    # 3. Feature importance
    try:
        mi_scores = data_pipeline_result.get("mi_scores")
        rf_importance = None
        rf_features = None
        if "rf" in all_results and all_results["rf"] is not None:
            rf_importance = all_results["rf"].get("feature_importances")
            rf_features = data_pipeline_result.get("artifacts", {}).get("feature_names")
        p = plot_feature_importance(mi_scores=mi_scores,
                                     feature_importances=rf_importance,
                                     feature_names=rf_features)
        if p:
            generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Feature importance: {e}")

    # 4-6. Training curves
    try:
        if all_histories:
            paths = plot_training_curves(all_histories)
            generated.extend(paths)
    except Exception as e:
        print(f"  [ERROR] Training curves: {e}")

    # 7. ROC curves
    try:
        p = plot_roc_curves(all_results, class_names)
        generated.append(p)
    except Exception as e:
        print(f"  [ERROR] ROC curves: {e}")

    # 8. Precision-Recall curves
    try:
        p = plot_precision_recall_curves(all_results, class_names)
        generated.append(p)
    except Exception as e:
        print(f"  [ERROR] PR curves: {e}")

    # 9-10. Confusion matrices
    try:
        paths = plot_confusion_matrices(all_results, class_names)
        generated.extend(paths)
    except Exception as e:
        print(f"  [ERROR] Confusion matrices: {e}")

    # 11. Model comparison
    try:
        p = plot_model_comparison(all_results)
        generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Model comparison: {e}")

    # 12. t-SNE
    try:
        proposed = all_results.get("deep_fin_dlp")
        if proposed is not None and "features" in proposed:
            p = plot_tsne_embedding(
                proposed["features"],
                proposed["y_true"][:len(proposed["features"])],
                class_names
            )
            if p:
                generated.append(p)
    except Exception as e:
        print(f"  [ERROR] t-SNE: {e}")

    # 13. Attention weights
    try:
        attn = all_results.get("attention_weights")
        if attn is not None:
            p = plot_attention_weights(attn)
            if p:
                generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Attention weights: {e}")

    # 14. Per-class F1
    try:
        p = plot_per_class_f1(all_results, class_names)
        if p:
            generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Per-class F1: {e}")

    # 15. Training time
    try:
        p = plot_training_time(all_results)
        if p:
            generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Training time: {e}")

    # 16. Architecture diagram
    try:
        p = plot_architecture_diagram()
        generated.append(p)
    except Exception as e:
        print(f"  [ERROR] Architecture diagram: {e}")

    print(f"\n  Total figures generated: {len(generated)}")
    print("=" * 70)
    return generated
