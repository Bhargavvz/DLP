"""
DeepFinDLP - Feature Engineering Pipeline
Feature selection, correlation removal, and statistical feature generation.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm


def compute_mutual_information(X: np.ndarray, y: np.ndarray,
                                feature_names: list,
                                n_jobs: int = -1) -> pd.Series:
    """Compute mutual information between features and target."""
    print("  Computing mutual information scores...")
    mi_scores = mutual_info_classif(
        X, y, discrete_features=False,
        random_state=42, n_neighbors=5
    )
    mi_series = pd.Series(mi_scores, index=feature_names)
    mi_series = mi_series.sort_values(ascending=False)
    return mi_series


def remove_correlated_features(X: np.ndarray, feature_names: list,
                                threshold: float = 0.95) -> tuple:
    """Remove highly correlated features."""
    print(f"  Removing features with correlation > {threshold}...")
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"    Removed {len(to_drop)} highly correlated features")

    keep_cols = [c for c in feature_names if c not in to_drop]
    keep_indices = [feature_names.index(c) for c in keep_cols]

    return X[:, keep_indices], keep_cols


def select_top_k_features(X: np.ndarray, y: np.ndarray,
                           feature_names: list, k: int = 50) -> tuple:
    """Select top-K features based on mutual information."""
    mi_scores = compute_mutual_information(X, y, feature_names)

    top_k = min(k, len(feature_names))
    selected_features = mi_scores.head(top_k).index.tolist()
    selected_indices = [feature_names.index(f) for f in selected_features]

    print(f"  Selected top {top_k} features by mutual information")
    print(f"    Top 10: {selected_features[:10]}")

    return X[:, selected_indices], selected_features, mi_scores


def add_statistical_features(X: np.ndarray, feature_names: list) -> tuple:
    """Add statistical meta-features (ratios, logs)."""
    print("  Generating statistical meta-features...")
    new_features = []
    new_names = []

    # Row-wise statistics
    row_mean = X.mean(axis=1, keepdims=True)
    row_std = X.std(axis=1, keepdims=True) + 1e-8
    row_max = X.max(axis=1, keepdims=True)
    row_min = X.min(axis=1, keepdims=True)

    new_features.extend([row_mean, row_std, row_max, row_min])
    new_names.extend(["row_mean", "row_std", "row_max", "row_min"])

    # Range and coefficient of variation
    row_range = row_max - row_min
    row_cv = row_std / (np.abs(row_mean) + 1e-8)
    new_features.extend([row_range, row_cv])
    new_names.extend(["row_range", "row_cv"])

    # Skewness approximation
    row_skew = np.mean(((X - row_mean) / row_std) ** 3, axis=1, keepdims=True)
    new_features.append(row_skew)
    new_names.append("row_skew")

    # Kurtosis approximation
    row_kurt = np.mean(((X - row_mean) / row_std) ** 4, axis=1, keepdims=True) - 3
    new_features.append(row_kurt)
    new_names.append("row_kurt")

    X_augmented = np.hstack([X] + new_features)
    augmented_names = feature_names + new_names

    # Replace NaN/Inf
    X_augmented = np.nan_to_num(X_augmented, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Added {len(new_names)} meta-features. Total: {X_augmented.shape[1]}")
    return X_augmented, augmented_names


def full_feature_engineering(X: np.ndarray, y: np.ndarray,
                              feature_names: list,
                              top_k: int = 50,
                              corr_threshold: float = 0.95) -> dict:
    """Run the full feature engineering pipeline."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Feature Engineering Pipeline")
    print("=" * 70)
    print(f"  Input: {X.shape[1]} features")

    # Step 1: Remove correlated features
    X_clean, clean_names = remove_correlated_features(
        X, feature_names, threshold=corr_threshold
    )

    # Step 2: Select top-K features by MI
    X_selected, selected_names, mi_scores = select_top_k_features(
        X_clean, y, clean_names, k=top_k
    )

    # Step 3: Add statistical meta-features
    X_final, final_names = add_statistical_features(X_selected, selected_names)

    print(f"\n  Final feature count: {X_final.shape[1]}")
    print("=" * 70)

    return {
        "X": X_final,
        "feature_names": final_names,
        "mi_scores": mi_scores,
        "selected_features": selected_names,
        "num_features": X_final.shape[1],
    }
