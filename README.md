# Heart Disease Prediction — MLOps Pipeline

**Course:** Machine Learning Operations (MLOps) AIMLCZG523 | **Assignment 01** | **Total Marks: 50**

End-to-end MLOps solution: data → EDA → training → MLflow tracking → FastAPI → Docker → Kubernetes → CI/CD → Prometheus/Grafana monitoring.

**GitHub Repository:** [https://github.com/2024ac05660-hub/MLOPS](https://github.com/2024ac05660-hub/MLOPS)

---

## Project Structure

```text
heart-disease-mlops/
├── .github/
│   └── workflows/ci.yml              # GitHub Actions — Lint → Test → Train → Docker
├── api/
│   └── app.py                        # FastAPI application with /predict + /metrics
├── data/
│   ├── raw/processed.cleveland.data  # Raw UCI Cleveland dataset
│   └── processed/                    # Cleaned CSV (auto-generated)
├── helm/heart-disease-api/           # Helm chart for Kubernetes deployment
├── k8s/
│   ├── deployment.yaml               # Kubernetes Deployment (2 replicas)
│   ├── service.yaml                  # LoadBalancer Service
│   ├── hpa.yaml                      # Horizontal Pod Autoscaler
│   └── ingress.yaml                  # Nginx Ingress
├── models/                           # Saved model artifacts (auto-generated)
├── monitoring/
│   ├── prometheus.yml                # Prometheus scrape config
│   └── grafana/provisioning/         # Auto-provisioned Grafana dashboard
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA with visualisations
│   ├── 02_training.ipynb             # Training + MLflow tracking
│   └── 03_inference.ipynb            # Inference + batch prediction
├── report/
│   └── MLOps_Assignment_Report.md    # Full 10-section assignment report
├── screenshots/                      # EDA + model plots (19 PNG files)
├── scripts/
│   ├── download_data.py              # Dataset download script
│   └── run_eda.py                    # Headless EDA script (used in CI)
├── src/
│   ├── data_processing.py            # Data cleaning + ColumnTransformer pipeline
│   ├── train.py                      # Training + GridSearchCV + MLflow logging
│   └── predict.py                    # Inference helper
├── tests/
│   ├── test_data_processing.py       # 19 unit tests — data pipeline
│   ├── test_model.py                 # 15 unit tests — model + inference
│   ├── test_api.py                   # 11 unit tests — FastAPI endpoints
│   └── sample_input.json             # Sample patient JSON for smoke tests
├── conftest.py                       # pytest sys.path configuration
├── docker-compose.yml                # API + Prometheus + Grafana stack
├── Dockerfile                        # Multi-stage production Docker image
├── environment.yml                   # Conda environment specification
└── requirements.txt                  # pip dependencies (pinned versions)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker Desktop / OrbStack / Colima (for Tasks 6–8)
- Minikube + kubectl + Helm (for Task 7)
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/2024ac05660-hub/MLOPS.git
cd MLOPS
```

### 2 — Create virtual environment and install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Download / verify dataset

Dataset is already included at `data/raw/processed.cleveland.data`. To re-download:

```bash
python scripts/download_data.py
```

### 4 — Preprocess data

```bash
PYTHONPATH=src python src/data_processing.py
# Output: data/processed/heart_disease_clean.csv
```

### 5 — Train all models (with MLflow tracking)

```bash
PYTHONPATH=src python src/train.py
# Trains: Logistic Regression, Random Forest, Gradient Boosting
# Logs: parameters, metrics, artifacts to ./mlruns
# Saves: models/best_model.pkl + models/model_meta.json
```

### 6 — View MLflow UI

```bash
python -m mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

### 7 — Run the API locally

```bash
PYTHONPATH=src:api uvicorn api.app:app --host 0.0.0.0 --port 8080
# Swagger UI:  http://localhost:8080/docs
# Health:      http://localhost:8080/health
# Metrics:     http://localhost:8080/metrics
```

### 8 — Run unit tests

```bash
pytest tests/test_data_processing.py tests/test_model.py -v
# Expected: 34 passed
```

### Conda alternative

```bash
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## API Usage

### Health check

```bash
curl http://localhost:8080/health
```

### Single prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_input.json
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.931,
  "probabilities": {"no_disease": 0.069, "disease": 0.931},
  "model_version": "1.0.0",
  "timestamp": "2026-07-12T10:00:00+00:00"
}
```

### API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Service info |
| `/health` | GET | Liveness probe |
| `/predict` | POST | Heart disease prediction + confidence |
| `/sample` | GET | Sample input JSON |
| `/docs` | GET | Interactive Swagger UI |
| `/metrics` | GET | Prometheus metrics |

---

## Docker

### Build and run

```bash
# Build image (train first to generate models/best_model.pkl)
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8080:8080 --name hd-api heart-disease-api:latest

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_input.json

# Stop
docker stop hd-api && docker rm hd-api
```

### Full stack — API + Prometheus + Grafana

```bash
docker-compose up -d
```

| Service | URL | Credentials |
| --- | --- | --- |
| API + Swagger | [http://localhost:8080/docs](http://localhost:8080/docs) | — |
| Prometheus | [http://localhost:9090](http://localhost:9090) | — |
| Grafana | [http://localhost:3000](http://localhost:3000) | admin / admin123 |

```bash
docker-compose down   # stop all services
```

---

## Kubernetes Deployment (Minikube)

### Using raw manifests

```bash
# Start Minikube
minikube start --driver=docker --memory=4096

# Load local image into Minikube
minikube image load heart-disease-api:latest

# Apply all manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Check pods are running
kubectl get pods

# Get service URL and test
minikube service heart-disease-api --url
```

### Using Helm

```bash
helm install heart-disease ./helm/heart-disease-api
helm status heart-disease

# Scale up
helm upgrade heart-disease ./helm/heart-disease-api --set replicaCount=3
```

---

## CI/CD Pipeline (GitHub Actions)

File: `.github/workflows/ci.yml` — triggers on every push to `main`/`develop` and pull requests.

| Stage | What it does | Fails on |
| --- | --- | --- |
| **1. Lint** | flake8 (PEP8) + black --check | Any style violation |
| **2. Test** | pytest with coverage report (34 tests) | Any test failure |
| **3. Train** | Preprocess + train all models, verify artifact | Model not created |
| **4. Docker** | Build image → run → /health + /predict smoke test | Container fails to respond |

Each stage uploads artifacts: test results, coverage report, trained model, MLflow runs, training plots.

---

## Models Trained

| Model | Tuning | Best Params | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | GridSearchCV — C | C=10.0 | 0.949 |
| **Random Forest** ✓ | GridSearchCV — n_estimators, max_depth, min_samples_split | n=200, depth=6 | **0.952** |
| Gradient Boosting | GridSearchCV — n_estimators, learning_rate, max_depth | n=100, lr=0.05 | 0.948 |

**Best model: Random Forest** — saved to `models/best_model.pkl`

---

## Evaluation Metrics

All models evaluated on a stratified 80/20 train/test split plus 5-fold cross-validation:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC (primary selection metric)
- Confusion matrix
- Feature importance

---

## Dataset

**Heart Disease UCI Dataset** — UCI Machine Learning Repository (Cleveland Clinic)

- 303 patient records, 13 numeric features
- Binary target: 0 = No Heart Disease, 1 = Heart Disease
- Missing values: `ca` (4 rows), `thal` (2 rows) — imputed with column median
- Class balance: 54% No Disease / 46% Disease

---

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.11 |
| ML Framework | scikit-learn (ColumnTransformer + Pipeline) |
| Experiment Tracking | MLflow |
| API Framework | FastAPI + Uvicorn |
| Testing | Pytest + pytest-cov (34 tests) |
| Containerisation | Docker (multi-stage), Docker Compose |
| Orchestration | Kubernetes (Minikube) + Helm |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Linting | flake8 + black |

---

## Assignment Tasks Coverage

| # | Task | Marks | Key Files |
| --- | --- | --- | --- |
| 1 | Data Acquisition & EDA | 5 | `scripts/download_data.py`, `notebooks/01_eda.ipynb`, `screenshots/eda_*.png` |
| 2 | Feature Engineering & Model Development | 8 | `src/data_processing.py`, `src/train.py`, `notebooks/02_training.ipynb` |
| 3 | Experiment Tracking (MLflow) | 5 | `src/train.py`, `mlruns/` |
| 4 | Model Packaging & Reproducibility | 7 | `models/best_model.pkl`, `requirements.txt`, `environment.yml` |
| 5 | CI/CD Pipeline & Automated Testing | 8 | `.github/workflows/ci.yml`, `tests/` |
| 6 | Model Containerisation | 5 | `Dockerfile`, `api/app.py`, `docker-compose.yml` |
| 7 | Production Deployment | 7 | `k8s/`, `helm/` |
| 8 | Monitoring & Logging | 3 | `monitoring/`, `api/app.py` |
| 9 | Documentation & Reporting | 2 | `report/MLOps_Assignment_Report.md` |
