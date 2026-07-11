# Heart Disease MLOps Pipeline

End-to-end MLOps solution for heart disease prediction — MLOps Assignment 01, AIMLCZG523.

## Quick Start

```bash
pip install -r requirements.txt
python src/data_processing.py          # preprocess data
python src/train.py                    # train + log to MLflow
uvicorn api.app:app --port 8080        # serve API
```

## Full setup → see [report/MLOps_Assignment_Report.md](report/MLOps_Assignment_Report.md)

## Tasks covered

| # | Task | Marks |
|---|---|---|
| 1 | Data Acquisition & EDA | 5 |
| 2 | Feature Engineering & Model Development | 8 |
| 3 | Experiment Tracking (MLflow) | 5 |
| 4 | Model Packaging & Reproducibility | 7 |
| 5 | CI/CD Pipeline & Automated Testing | 8 |
| 6 | Model Containerisation (Docker/FastAPI) | 5 |
| 7 | Production Deployment (Kubernetes/Helm) | 7 |
| 8 | Monitoring & Logging (Prometheus/Grafana) | 3 |
| 9 | Documentation & Reporting | 2 |
