"""
DeepFinDLP - Data Loading & Preprocessing Pipeline
Handles loading, cleaning, splitting, balancing, and DataLoader creation.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FinDLPDataset(Dataset):
    """PyTorch Dataset for financial DLP data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_clean_data(parquet_path: str = None) -> pd.DataFrame:
    """Load the Parquet dataset and perform initial cleaning."""
    if parquet_path is None:
        parquet_path = config.PARQUET_FILE

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Strip column names
    df.columns = df.columns.str.strip()

    # Ensure 'Label' column exists
    assert "Label" in df.columns, "Label column not found!"

    # Separate features and labels
    label_col = df["Label"].copy()

    # Drop non-numeric and label columns
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove any ID-like columns if present
    exclude_cols = ["Flow ID", "Source IP", "Destination IP",
                    "Source Port", "Destination Port", "Timestamp"]
    feature_cols = [c for c in feature_cols if c not in exclude_cols]

    df_features = df[feature_cols].copy()

    # Replace infinite values with NaN
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with column median
    nan_counts = df_features.isna().sum()
    if nan_counts.any():
        print(f"  Filling {nan_counts.sum()} NaN values with column medians")
        df_features.fillna(df_features.median(), inplace=True)

    # Fill any remaining NaN with 0
    df_features.fillna(0, inplace=True)

    # Remove columns with zero variance
    variance = df_features.var()
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        print(f"  Removing {len(zero_var_cols)} zero-variance columns")
        df_features.drop(columns=zero_var_cols, inplace=True)

    # Add label back
    df_features["Label"] = label_col.values

    print(f"  After cleaning: {df_features.shape[0]:,} rows × {df_features.shape[1]-1} features")
    return df_features


def map_labels_to_dlp(df: pd.DataFrame) -> pd.DataFrame:
    """Map original CIC-IDS2017 labels to DLP context categories."""
    df = df.copy()
    original_labels = df["Label"].unique()
    print(f"\n  Original labels ({len(original_labels)}):")
    for lbl in sorted(original_labels):
        count = (df["Label"] == lbl).sum()
        print(f"    {lbl}: {count:,}")

    # Apply mapping
    df["DLP_Label"] = df["Label"].map(config.LABEL_MAP)

    # Handle unmapped labels
    unmapped = df["DLP_Label"].isna()
    if unmapped.any():
        print(f"\n  Unmapped labels found ({unmapped.sum()} rows). Mapping to 'Normal Traffic'")
        # Try fuzzy mapping
        for idx in df[unmapped].index:
            orig = df.loc[idx, "Label"]
            mapped = False
            for key, val in config.LABEL_MAP.items():
                if key.lower() in orig.lower() or orig.lower() in key.lower():
                    df.loc[idx, "DLP_Label"] = val
                    mapped = True
                    break
            if not mapped:
                # Check for common patterns
                orig_lower = orig.lower()
                if "dos" in orig_lower or "ddos" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "DoS Attack"
                elif "web" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Web Attack"
                elif "brute" in orig_lower or "patator" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Credential Theft"
                elif "bot" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Botnet Exfiltration"
                elif "infiltr" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Data Infiltration"
                elif "heartbleed" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Vulnerability Exploit"
                elif "portscan" in orig_lower or "scan" in orig_lower:
                    df.loc[idx, "DLP_Label"] = "Reconnaissance"
                else:
                    df.loc[idx, "DLP_Label"] = "Normal Traffic"

    # Remove the Vulnerability Exploit and Data Infiltration if too few samples
    label_counts = df["DLP_Label"].value_counts()
    min_samples = 50
    rare_classes = label_counts[label_counts < min_samples].index.tolist()
    if rare_classes:
        print(f"\n  Merging rare classes (<{min_samples} samples) into nearest category:")
        for rc in rare_classes:
            print(f"    {rc}: {label_counts[rc]} samples → merged into 'Normal Traffic'")
        df.loc[df["DLP_Label"].isin(rare_classes), "DLP_Label"] = "Normal Traffic"

    print(f"\n  DLP-mapped labels:")
    dlp_counts = df["DLP_Label"].value_counts()
    for lbl, count in dlp_counts.items():
        print(f"    {lbl}: {count:,}")

    return df


def prepare_splits(df: pd.DataFrame, use_smote: bool = None):
    """Split data into train/val/test and apply SMOTE if configured."""
    if use_smote is None:
        use_smote = config.USE_SMOTE

    # Encode labels
    le = LabelEncoder()
    feature_cols = [c for c in df.columns if c not in ["Label", "DLP_Label"]]
    X = df[feature_cols].values.astype(np.float32)
    y = le.fit_transform(df["DLP_Label"].values)

    print(f"\n  Total samples: {len(X):,} | Features: {X.shape[1]} | Classes: {len(le.classes_)}")
    print(f"  Classes: {list(le.classes_)}")

    # Stratified split: train+val | test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE
    )

    # Split train | val
    val_frac = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        stratify=y_trainval, random_state=config.RANDOM_STATE
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Apply SMOTE on training set only
    if use_smote:
        print("\n  Applying SMOTE oversampling on training set...")
        counter_before = Counter(y_train)
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        counter_after = Counter(y_train)
        print(f"  Before SMOTE: {dict(counter_before)}")
        print(f"  After SMOTE:  {dict(counter_after)}")
        print(f"  Train samples after SMOTE: {len(X_train):,}")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler and label encoder
    artifacts = {
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_cols,
        "num_features": X_train.shape[1],
        "num_classes": len(le.classes_),
        "class_names": list(le.classes_),
    }

    artifact_path = os.path.join(config.PROCESSED_DATA_DIR, "artifacts.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Saved preprocessing artifacts to {artifact_path}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), artifacts


def create_dataloaders(train_data, val_data, test_data,
                       batch_size: int = None):
    """Create PyTorch DataLoaders with H200 optimization."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_dataset = FinDLPDataset(X_train, y_train)
    val_dataset = FinDLPDataset(X_val, y_val)
    test_dataset = FinDLPDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    print(f"\n  DataLoaders created:")
    print(f"    Train: {len(train_loader)} batches (batch_size={batch_size})")
    print(f"    Val:   {len(val_loader)} batches (batch_size={batch_size*2})")
    print(f"    Test:  {len(test_loader)} batches (batch_size={batch_size*2})")

    return train_loader, val_loader, test_loader


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for loss function."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # Avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.FloatTensor(weights).to(config.DEVICE)


def get_full_pipeline(parquet_path: str = None, use_smote: bool = None,
                      batch_size: int = None):
    """Run the full data pipeline: load → clean → map → split → dataloaders."""
    print("=" * 70)
    print("DeepFinDLP - Data Pipeline")
    print("=" * 70)

    # Step 1: Load and clean
    df = load_and_clean_data(parquet_path)

    # Step 2: Map labels to DLP categories
    df = map_labels_to_dlp(df)

    # Step 3: Split and balance
    train_data, val_data, test_data, artifacts = prepare_splits(df, use_smote)

    # Step 4: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size
    )

    # Step 5: Compute class weights
    class_weights = compute_class_weights(train_data[1], artifacts["num_classes"])

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "artifacts": artifacts,
        "class_weights": class_weights,
        "dataframe": df,
    }

    print("\n" + "=" * 70)
    print("Data Pipeline Complete!")
    print("=" * 70)

    return result
