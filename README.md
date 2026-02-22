# DeepFinDLP: Deep Learning-Driven Data Leakage Prevention for Financial Instability Mitigation

A novel **Temporal Convolutional Transformer (TCT)** architecture for detecting and preventing data leakage in financial network traffic. This project implements an end-to-end pipeline from data preprocessing to model training and evaluation, optimized for NVIDIA H200 GPUs.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_data.py
```
This downloads the CIC-IDS2017 dataset (~2.8M records, 80+ features).

**Manual download alternative**: Visit https://www.unb.ca/cic/datasets/ids-2017.html and place the CSV files in `data/raw/`.

### 3. Run Full Pipeline
```bash
# Full pipeline (all models, all figures)
python main.py --mode all

# Quick test (subset data, 10 epochs)
python main.py --quick-test

# Skip baselines (only DL models)
python main.py --skip-baselines
```

## Project Structure

```
├── main.py                    # Main pipeline orchestrator
├── config.py                  # All hyperparameters & paths
├── download_data.py           # Dataset download script
├── requirements.txt           # Dependencies
├── src/
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── feature_engineering.py # Feature selection & engineering
│   ├── trainer.py             # H200-optimized training loop
│   ├── evaluator.py           # Comprehensive metrics
│   └── models/
│       ├── baselines.py       # RF, SVM, XGBoost
│       ├── dnn_model.py       # Basic DNN (MLP)
│       ├── cnn_model.py       # 1D-CNN
│       ├── lstm_model.py      # BiLSTM
│       ├── cnn_lstm_model.py  # CNN-BiLSTM Hybrid
│       └── deep_fin_dlp.py    # Proposed DeepFinDLP (TCT)
├── visualization/
│   └── generate_figures.py    # 16 publication-quality figures
└── results/
    ├── figures/               # Generated plots (PNG, 300 DPI)
    ├── checkpoints/           # Model checkpoints
    └── logs/                  # Training histories
```

## Architecture

**DeepFinDLP** uses a hierarchical feature extraction approach:

```
Input (80 features) → BatchNorm → 1D-CNN (3 layers) → SE Block
→ BiLSTM (2 layers) → Multi-Head Self-Attention (8 heads)
→ Residual FC Head → Softmax (8 classes)
```

## H200 GPU Optimizations

- BFloat16 mixed precision training
- Batch size: 4096 (leveraging 141GB VRAM)
- `torch.compile(mode='max-autotune')`
- 16 DataLoader workers with pin_memory
- Cosine annealing warm restarts LR scheduler

## Generated Figures (16)

1. Class distribution
2. Feature correlation heatmap
3. Feature importance (MI)
4. Training loss curves
5. Training accuracy curves
6. Training F1 curves
7. ROC curves (all models)
8. Precision-Recall curves
9. Confusion matrix (proposed)
10. Confusion matrices (baselines)
11. Model comparison bar chart
12. t-SNE embedding
13. Attention weight heatmap
14. Per-class F1 comparison
15. Training time comparison
16. Architecture diagram

## Citation

If you use this code, please cite:
```bibtex
@article{deepfindlp2025,
  title={Mitigating Financial Instability Through Deep Learning-Driven Data Leakage Prevention},
  year={2025},
}
```
