"""
DeepFinDLP - Configuration
All hyperparameters and paths centralized here.
"""
import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Create directories
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
          FIGURES_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Device Configuration ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_AMP = True  # Automatic Mixed Precision (BF16 on H200)
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
USE_COMPILE = True  # torch.compile for kernel fusion

# ─── Data Configuration ──────────────────────────────────────────────────────
PARQUET_FILE = os.path.join(PROCESSED_DATA_DIR, "cicids2017_full.parquet")
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
TOP_K_FEATURES = 50  # Number of features to keep after selection
USE_SMOTE = True
SMOTE_STRATEGY = "auto"  # or dict for custom ratios

# Label mapping: CIC-IDS2017 → DLP context
LABEL_MAP = {
    "BENIGN": "Normal Traffic",
    "DoS Hulk": "DoS Attack",
    "PortScan": "Reconnaissance",
    "DDoS": "DDoS Attack",
    "DoS GoldenEye": "DoS Attack",
    "FTP-Patator": "Credential Theft",
    "SSH-Patator": "Credential Theft",
    "DoS slowloris": "DoS Attack",
    "DoS Slowhttptest": "DoS Attack",
    "Bot": "Botnet Exfiltration",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Infiltration": "Data Infiltration",
    "Heartbleed": "Vulnerability Exploit",
}

# Simplified multi-class categories for training
DLP_CATEGORIES = [
    "Normal Traffic",
    "DoS Attack",
    "DDoS Attack",
    "Reconnaissance",
    "Credential Theft",
    "Botnet Exfiltration",
    "Web Attack",
    "Data Infiltration",
]

NUM_CLASSES = len(DLP_CATEGORIES)

# ─── DataLoader Configuration ────────────────────────────────────────────────
BATCH_SIZE = 4096  # Large batch for H200 (141GB VRAM)
NUM_WORKERS = 16
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ─── Model Hyperparameters ───────────────────────────────────────────────────

# 1D-CNN
CNN_CONFIG = {
    "channels": [128, 256, 512],
    "kernel_sizes": [5, 3, 3],
    "dropout": 0.3,
}

# BiLSTM
LSTM_CONFIG = {
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
}

# CNN-BiLSTM
CNN_LSTM_CONFIG = {
    "cnn_channels": [128, 256],
    "cnn_kernel_sizes": [5, 3],
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "dropout": 0.3,
}

# DeepFinDLP (Proposed)
DEEP_FIN_DLP_CONFIG = {
    "cnn_channels": [128, 256, 512],
    "cnn_kernel_sizes": [7, 5, 3],
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "attention_heads": 8,
    "attention_dim": 512,
    "se_reduction": 16,
    "fc_dims": [512, 256],
    "dropout": 0.3,
}

# Basic DNN
DNN_CONFIG = {
    "hidden_dims": [512, 256, 128],
    "dropout": 0.3,
}

# ─── Training Hyperparameters ────────────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15  # Early stopping
MIN_DELTA = 1e-4  # Minimum improvement for early stopping
GRADIENT_CLIP_MAX_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 1

# LR Scheduler
SCHEDULER_TYPE = "cosine_warm_restarts"  # "cosine" | "cosine_warm_restarts" | "step"
COSINE_T0 = 10
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-6

# ─── Baseline Hyperparameters ────────────────────────────────────────────────
RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": 30,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

SVM_CONFIG = {
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "scale",
    "max_iter": 5000,
}

XGB_CONFIG = {
    "n_estimators": 500,
    "max_depth": 10,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ─── Visualization ───────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = "husl"
FONT_SIZE = 12

# Model names for labeling
MODEL_NAMES = {
    "rf": "Random Forest",
    "svm": "SVM (RBF)",
    "xgb": "XGBoost",
    "dnn": "Basic DNN",
    "cnn": "1D-CNN",
    "lstm": "BiLSTM",
    "cnn_lstm": "CNN-BiLSTM",
    "deep_fin_dlp": "DeepFinDLP (Proposed)",
}
