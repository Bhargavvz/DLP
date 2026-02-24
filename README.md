# DeepFinDLP — Full Stack Application

Deep Learning-Driven Data Leakage Prevention for Financial Instability Mitigation.

## Project Structure

```
├── ML/                    # Machine Learning Pipeline
│   ├── main.py            # Training orchestrator
│   ├── config.py          # Hyperparameters
│   ├── src/               # Models, trainer, evaluator
│   ├── visualization/     # Figure generation
│   ├── data/              # Dataset (CIC-IDS2017)
│   └── results/           # Checkpoints, figures, logs
│
├── backend/               # FastAPI Backend
│   └── main.py            # API server
│
├── frontend/              # React + Vite Dashboard
│   ├── src/pages/         # Dashboard, Models, Training, Figures, Architecture
│   └── src/index.css      # Design system
│
└── README.md
```

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### ML Training
```bash
cd ML
pip install -r requirements.txt
python download_data.py
python main.py --mode all
```
