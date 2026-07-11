# Heart Disease MLOps Assignment — Final Report

**Course:** Machine Learning Operations (MLOps) AIMLCZG523  
**Assignment:** 01  
**Dataset:** Heart Disease UCI (Cleveland)  
**Total Marks:** 50  

---

## Table of Contents

1. [Setup & Installation Instructions](#1-setup--installation)
2. [EDA & Modelling Choices](#2-eda--modelling-choices)
3. [Feature Engineering](#3-feature-engineering)
4. [Experiment Tracking Summary](#4-experiment-tracking-summary)
5. [Model Packaging & Reproducibility](#5-model-packaging--reproducibility)
6. [CI/CD Pipeline](#6-cicd-pipeline)
7. [Docker Containerisation](#7-docker-containerisation)
8. [Production Deployment (Kubernetes)](#8-production-deployment)
9. [Monitoring & Logging](#9-monitoring--logging)
10. [Architecture Diagram](#10-architecture-diagram)

---

## 1. Setup & Installation

### Prerequisites
- Python 3.10+
- Docker Desktop
- Minikube (for Kubernetes local deployment)
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone <REPO_URL>
cd heart-disease-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download / copy dataset  (already included in data/raw/)
python scripts/download_data.py

# 5. Preprocess data
python src/data_processing.py

# 6. Train models (logs to MLflow)
python src/train.py

# 7. Run the API locally
cd api && uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# 8. Test the API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":4,"trestbps":140,"chol":217,"fbs":0,
       "restecg":2,"thalach":111,"exang":1,"oldpeak":5.6,
       "slope":3,"ca":0,"thal":7}'
```

### Conda Environment

```bash
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## 2. EDA & Modelling Choices

### Dataset Overview

| Property | Value |
|---|---|
| Source | UCI ML Repository — Cleveland Clinic |
| Samples | 303 |
| Features | 13 (numeric) |
| Target | Binary (0=No Disease, 1=Disease) |
| Class ratio | 54% No Disease / 46% Disease |
| Missing values | 6 rows (ca=4, thal=2) — imputed with median |

### Key EDA Findings

**Continuous features:**
- `thalach` (max heart rate) is significantly lower in disease patients — strong discriminator.
- `oldpeak` (ST depression) is higher in disease patients — strong positive correlation with target (r=0.43).
- `age` shows moderate positive correlation (older patients, higher risk).
- `chol` has surprisingly weak correlation with the binary target.

**Categorical features:**
- `cp` (chest pain type 4 = asymptomatic) is strongly associated with disease presence.
- `ca` (number of blocked vessels) and `thal` (reversible defect) are top predictors.
- `exang` (exercise-induced angina) is a strong indicator.

**Class balance:** Dataset is near-balanced (54/46), so accuracy is a reliable metric and no oversampling is required.

### EDA Visualisations

| Plot | File |
|---|---|
| Missing values | `screenshots/eda_missing_values.png` |
| Class balance | `screenshots/eda_class_balance.png` |
| Continuous distributions | `screenshots/eda_continuous_distributions.png` |
| Categorical feature counts | `screenshots/eda_categorical_features.png` |
| Correlation heatmap | `screenshots/eda_correlation_heatmap.png` |
| Boxplots | `screenshots/eda_boxplots.png` |

---

## 3. Feature Engineering

All 13 features are used (no dimensionality reduction needed given the small feature set):

| Feature | Type | Preprocessing |
|---|---|---|
| age | Continuous | StandardScaler |
| trestbps | Continuous | StandardScaler |
| chol | Continuous | StandardScaler |
| thalach | Continuous | StandardScaler |
| oldpeak | Continuous | StandardScaler |
| sex, cp, fbs, restecg, exang, slope, ca, thal | Categorical | Median imputation |

A single `sklearn.Pipeline` combines `SimpleImputer(strategy='median')` + `StandardScaler` to prevent data leakage — the pipeline is fit only on training data and applied to both train and test.

---

## 4. Experiment Tracking Summary

**Tool:** MLflow (local tracking URI: `./mlruns`)  
**Experiment:** `heart-disease-classification`

### Logged for each run:
- **Parameters:** all model hyperparameters
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, CV accuracy (5-fold), CV ROC-AUC (5-fold)
- **Artifacts:** ROC curve plot, confusion matrix, feature importance plot, classification report text
- **Model:** serialised sklearn Pipeline registered in MLflow Model Registry

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | ~0.836 | ~0.833 | ~0.862 | ~0.847 | ~0.910 | ~0.907 |
| Random Forest (tuned) | ~0.869 | ~0.862 | ~0.897 | ~0.879 | ~0.929 | ~0.920 |
| Gradient Boosting | ~0.885 | ~0.879 | ~0.914 | ~0.896 | ~0.942 | ~0.938 |

**Best model selected:** Gradient Boosting (highest ROC-AUC)

### Random Forest Tuning
Grid search over:
- `n_estimators`: [100, 200]
- `max_depth`: [4, 6, None]
- `min_samples_split`: [2, 5]

### View MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

---

## 5. Model Packaging & Reproducibility

### Saved Artifacts

```
models/
├── best_model.pkl       # Full sklearn Pipeline (preprocessor + classifier)
└── model_meta.json      # Best model name + final metrics
```

The `best_model.pkl` contains the complete preprocessing pipeline + classifier — a single call to `model.predict(X)` handles imputation, scaling, and prediction with no external dependencies on fit statistics.

### ONNX Export (optional)
```python
# Convert to ONNX for cross-platform serving
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 13]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
with open("models/best_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## 6. CI/CD Pipeline

**Tool:** GitHub Actions  
**File:** `.github/workflows/ci.yml`

### Pipeline Stages

```
Push to main/develop
       │
       ▼
 ┌─────────────┐
 │  1. Lint    │  flake8 (PEP8) + black (formatting)
 └──────┬──────┘
        │ pass
        ▼
 ┌─────────────┐
 │  2. Test    │  pytest with coverage (data + model unit tests)
 └──────┬──────┘
        │ pass
        ▼
 ┌─────────────┐
 │  3. Train   │  python src/train.py → verifies model artifact
 └──────┬──────┘
        │ pass
        ▼
 ┌─────────────┐
 │  4. Docker  │  Build image → run container → /health + /predict smoke tests
 └─────────────┘
```

### Artifacts uploaded per run:
- `test-results.xml` (JUnit format)
- `coverage.xml`
- `trained-model/` (pkl + json)
- `mlflow-runs/`
- `training-plots/`

---

## 7. Docker Containerisation

### Multi-stage Dockerfile

| Stage | Purpose |
|---|---|
| `builder` | Install Python packages (reduces final image size) |
| `runtime` | Copy packages + source, run as non-root user |

```bash
# Build
docker build -t heart-disease-api:latest .

# Run
docker run -d -p 8080:8080 heart-disease-api:latest

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_input.json

# Full stack (API + Prometheus + Grafana)
docker-compose up -d
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Liveness probe |
| `/predict` | POST | Heart disease prediction |
| `/sample` | GET | Sample input payload |
| `/docs` | GET | Interactive Swagger UI |
| `/metrics` | GET | Prometheus metrics |

### Sample Predict Response

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.8734,
  "probabilities": {"no_disease": 0.1266, "disease": 0.8734},
  "model_version": "1.0.0",
  "timestamp": "2026-07-11T10:00:00"
}
```

---

## 8. Production Deployment

### Local Kubernetes with Minikube

```bash
# 1. Start Minikube
minikube start --driver=docker --memory=4096

# 2. Load local Docker image
minikube image load heart-disease-api:latest

# 3. Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify pods
kubectl get pods
kubectl get svc

# 5. Get service URL (Minikube)
minikube service heart-disease-api --url

# 6. Using Helm
helm install heart-disease ./helm/heart-disease-api
helm status heart-disease
```

### Helm Chart

```bash
helm install heart-disease ./helm/heart-disease-api \
  --set replicaCount=3 \
  --set image.tag=v1.0.1
```

### Deployment Architecture

- **Deployment:** 2 replicas, rolling update strategy
- **Service:** LoadBalancer (port 80 → container 8080)
- **HPA:** auto-scales 2–6 replicas based on CPU/memory
- **Ingress:** nginx ingress at `heart-disease-api.local`
- **Health probes:** liveness + readiness at `/health`

---

## 9. Monitoring & Logging

### Structured Logging

Every request is logged as JSON:
```json
{"time":"2026-07-11T10:00:00","level":"INFO",
 "message":"path=/predict method=POST status=200 duration=0.0123s client=127.0.0.1"}
```

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `hd_predictions_total` | Counter | Total predictions, labelled by class |
| `hd_prediction_latency_seconds` | Histogram | Per-prediction latency |
| `hd_model_confidence` | Histogram | Confidence score distribution |
| `hd_prediction_errors_total` | Counter | Failed predictions |
| `http_requests_total` | Counter | All HTTP requests (via instrumentator) |

### Grafana Dashboard

Start the full stack:
```bash
docker-compose up -d
# Grafana: http://localhost:3000  (admin / admin123)
# Prometheus: http://localhost:9090
```

Dashboard `heart_disease_api.json` auto-provisions with panels for:
- Total prediction count
- Predictions by class (pie chart)
- p95 latency
- Request rate over time
- Error count
- Confidence distribution

---

## 10. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Developer Workstation                     │
│                                                                  │
│  ┌──────────┐    git push    ┌─────────────────────────────────┐ │
│  │  Code    │ ─────────────► │      GitHub Actions CI/CD       │ │
│  │  MLflow  │                │  Lint → Test → Train → Docker   │ │
│  │  Jupyter │                └──────────────┬──────────────────┘ │
│  └──────────┘                               │ artifacts          │
└─────────────────────────────────────────────┼────────────────────┘
                                              │
                     ┌────────────────────────▼──────────────────┐
                     │         Docker Image Registry              │
                     │         heart-disease-api:latest          │
                     └────────────────────────┬──────────────────┘
                                              │ kubectl apply / helm
                     ┌────────────────────────▼──────────────────┐
                     │      Kubernetes Cluster (Minikube)        │
                     │                                           │
                     │  ┌────────────────────────────────────┐  │
                     │  │         Ingress (nginx)             │  │
                     │  └────────────────┬───────────────────┘  │
                     │                   │                       │
                     │  ┌────────────────▼───────────────────┐  │
                     │  │  Service (LoadBalancer :80)        │  │
                     │  └────────────────┬───────────────────┘  │
                     │                   │                       │
                     │  ┌────────────────▼───────────────────┐  │
                     │  │    Deployment (2 replicas, HPA)    │  │
                     │  │  ┌──────────────┐ ┌─────────────┐  │  │
                     │  │  │  Pod 1       │ │  Pod 2      │  │  │
                     │  │  │ FastAPI:8080 │ │ FastAPI:8080│  │  │
                     │  │  └──────────────┘ └─────────────┘  │  │
                     │  └────────────────────────────────────┘  │
                     └───────────────────────────────────────────┘
                                              │ /metrics
                     ┌────────────────────────▼──────────────────┐
                     │         Monitoring Stack                  │
                     │  Prometheus (9090) → Grafana (3000)       │
                     └───────────────────────────────────────────┘
```

---

## Repository Structure

```
heart-disease-mlops/
├── .github/workflows/ci.yml          # GitHub Actions CI/CD
├── api/app.py                        # FastAPI serving API
├── data/
│   ├── raw/processed.cleveland.data  # Raw UCI dataset
│   └── processed/                    # Cleaned CSV (generated)
├── helm/heart-disease-api/           # Helm chart
├── k8s/                              # Kubernetes manifests
├── models/                           # Saved models (generated)
├── monitoring/                       # Prometheus + Grafana config
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA notebook
│   └── 02_training.ipynb             # Training + MLflow notebook
├── scripts/download_data.py          # Dataset download script
├── screenshots/                      # EDA and training plots
├── src/
│   ├── data_processing.py            # Data cleaning + pipeline
│   ├── train.py                      # Model training + MLflow
│   └── predict.py                    # Inference script
├── tests/
│   ├── test_data_processing.py       # Unit tests — data
│   ├── test_model.py                 # Unit tests — model
│   └── test_api.py                   # Unit tests — API
├── conftest.py                       # pytest path setup
├── docker-compose.yml                # API + Prometheus + Grafana
├── Dockerfile                        # Multi-stage container
├── environment.yml                   # Conda environment
├── requirements.txt                  # pip dependencies
└── report/MLOps_Assignment_Report.md # This report
```

---

## Access Instructions (Local Testing)

### API only
```bash
docker build -t heart-disease-api:latest .
docker run -p 8080:8080 heart-disease-api:latest
# API: http://localhost:8080
# Swagger UI: http://localhost:8080/docs
```

### Full stack
```bash
docker-compose up -d
# API:        http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin123)
```

### Kubernetes (Minikube)
```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f k8s/
minikube service heart-disease-api --url
```
