"""
FastAPI application — Heart Disease Prediction API.
Tasks 6 & 8: /predict endpoint + Prometheus metrics + structured logging.
"""

import os
import sys
import logging
import pickle
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402
from prometheus_client import Counter, Histogram  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict as run_predict, SAMPLE_INPUT  # noqa: E402

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("heart_disease_api")

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment 01 — Predicts presence of heart disease from patient data.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Prometheus metrics ───────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

PREDICTION_COUNTER = Counter(
    "hd_predictions_total",
    "Total number of predictions made",
    ["prediction_class"],
)
PREDICTION_LATENCY = Histogram(
    "hd_prediction_latency_seconds",
    "Time spent on prediction",
)
MODEL_CONFIDENCE = Histogram(
    "hd_model_confidence",
    "Confidence score distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
REQUEST_ERRORS = Counter("hd_prediction_errors_total", "Total prediction errors")

# ─── Model loader ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"),
)
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_PATH}")
    return _model


# ─── Request / Response schemas ───────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=80, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression (oldpeak)")
    slope: float = Field(..., ge=1, le=3, description="Slope of peak exercise ST")
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thal: float = Field(..., ge=3, le=7, description="Thal (3=normal, 6=fixed, 7=reversible)")

    class Config:
        json_schema_extra = {"example": SAMPLE_INPUT}


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int
    label: str
    confidence: float
    probabilities: dict
    model_version: str = "1.0.0"
    timestamp: str


# ─── Middleware: request logging ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"path={request.url.path} method={request.method} "
        f"status={response.status_code} duration={duration:.4f}s "
        f"client={request.client.host if request.client else 'unknown'}"
    )
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Ops"],
    responses={503: {"description": "Model not ready"}},
)
def health():
    """Liveness probe — confirms the API is running and model is loaded."""
    try:
        get_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")


@app.get("/", tags=["Ops"])
def root():
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={500: {"description": "Prediction error"}},
)
def predict_endpoint(patient: PatientFeatures):
    """
    Accept patient health features and return heart disease prediction.

    - **prediction**: 0 = No Heart Disease, 1 = Heart Disease
    - **confidence**: probability of the predicted class
    - **probabilities**: full class probability breakdown
    """
    model = get_model()
    input_dict = patient.model_dump()

    start = time.time()
    try:
        result = run_predict(input_dict, model=model)
    except Exception as e:
        REQUEST_ERRORS.inc()
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    latency = time.time() - start

    PREDICTION_LATENCY.observe(latency)
    PREDICTION_COUNTER.labels(prediction_class=str(result["prediction"])).inc()
    MODEL_CONFIDENCE.observe(result["confidence"])

    logger.info(
        f"prediction={result['prediction']} confidence={result['confidence']:.4f} "
        f"latency={latency:.4f}s"
    )

    return PredictionResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/sample", tags=["Prediction"])
def sample_input():
    """Return a sample input payload for testing the /predict endpoint."""
    return SAMPLE_INPUT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False, log_level="info")
