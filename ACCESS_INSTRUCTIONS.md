# API Access Instructions

## Local Testing (Docker — fastest)

### Prerequisites
- Docker Desktop installed and running

### Step 1 — Build the image
```bash
cd heart-disease-mlops

# Train the model first (if models/best_model.pkl does not exist)
pip install -r requirements.txt
python src/data_processing.py
python src/train.py

# Build Docker image
docker build -t heart-disease-api:latest .
```

### Step 2 — Run the container
```bash
docker run -d -p 8080:8080 --name hd-api heart-disease-api:latest
```

### Step 3 — Test endpoints

**Health check:**
```bash
curl http://localhost:8080/health
```

**Predict (sample patient with heart disease):**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_input.json
```

**Interactive Swagger UI:**  
Open http://localhost:8080/docs in your browser.

**Prometheus metrics:**  
http://localhost:8080/metrics

### Step 4 — Stop the container
```bash
docker stop hd-api && docker rm hd-api
```

---

## Full Stack (API + Prometheus + Grafana)

```bash
docker-compose up -d

# API:        http://localhost:8080
# Swagger UI: http://localhost:8080/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000   (login: admin / admin123)

# Stop everything
docker-compose down
```

---

## Local Kubernetes (Minikube)

### Prerequisites
- Minikube installed: https://minikube.sigs.k8s.io/docs/start/
- kubectl installed

```bash
# 1. Start cluster
minikube start --driver=docker --memory=4096

# 2. Load local image into Minikube
minikube image load heart-disease-api:latest

# 3. Apply all manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Watch pods come up
kubectl get pods -w

# 5. Get the service URL
minikube service heart-disease-api --url

# 6. Test (replace <URL> with output from step 5)
curl <URL>/health
curl -X POST <URL>/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_input.json
```

### Using Helm instead of raw manifests
```bash
helm install heart-disease ./helm/heart-disease-api
helm status heart-disease
minikube service heart-disease-api --url
```

---

## Running the API directly (no Docker)

```bash
pip install -r requirements.txt
python src/data_processing.py
python src/train.py

# Start API
PYTHONPATH=src:api uvicorn api.app:app --host 0.0.0.0 --port 8080

# Test
curl http://localhost:8080/health
```

---

## Sample API Request / Response

**Request:**
```json
{
  "age": 55.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 217.0,
  "fbs": 0.0, "restecg": 2.0, "thalach": 111.0, "exang": 1.0, "oldpeak": 5.6,
  "slope": 3.0, "ca": 0.0, "thal": 7.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.931,
  "probabilities": {
    "no_disease": 0.069,
    "disease": 0.931
  },
  "model_version": "1.0.0",
  "timestamp": "2026-07-11T15:18:48+00:00"
}
```
