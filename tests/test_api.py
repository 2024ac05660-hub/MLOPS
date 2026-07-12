"""
Unit tests for FastAPI prediction endpoint.
"""

import os
import sys
import pickle
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(scope="module")
def client():
    """Create a test client with a freshly trained model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from data_processing import (
        load_raw_data,
        clean_data,
        prepare_train_test,
        build_preprocessing_pipeline,
    )

    RAW_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "processed.cleveland.data"
    )
    df = clean_data(load_raw_data(RAW_PATH))
    X_train, _, y_train, _ = prepare_train_test(df)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump(pipeline, tmp)
    tmp.close()

    import app as app_module

    app_module.MODEL_PATH = tmp.name
    app_module._model = pipeline  # inject directly

    from app import app

    return TestClient(app)


VALID_PAYLOAD = {
    "age": 55.0,
    "sex": 1.0,
    "cp": 4.0,
    "trestbps": 140.0,
    "chol": 217.0,
    "fbs": 0.0,
    "restecg": 2.0,
    "thalach": 111.0,
    "exang": 1.0,
    "oldpeak": 5.6,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 7.0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_status(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_structure(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        body = r.json()
        assert "prediction" in body
        assert "label" in body
        assert "confidence" in body
        assert "probabilities" in body

    def test_predict_valid_class(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.json()["prediction"] in (0, 1)

    def test_predict_confidence_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        conf = r.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_missing_field_returns_422(self, client):
        incomplete = {"age": 55.0, "sex": 1.0}
        r = client.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_predict_invalid_type_returns_422(self, client):
        bad = dict(VALID_PAYLOAD)
        bad["age"] = "not_a_number"
        r = client.post("/predict", json=bad)
        assert r.status_code == 422
