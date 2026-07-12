"""
Unit tests for model training, evaluation, and inference.
"""

import os
import pickle
import sys
import tempfile

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import (  # noqa: E402
    build_preprocessing_pipeline,
    clean_data,
    load_raw_data,
    prepare_train_test,
)
from predict import predict, SAMPLE_INPUT  # noqa: E402

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "processed.cleveland.data")


@pytest.fixture(scope="module")
def dataset():
    df = clean_data(load_raw_data(RAW_PATH))
    return prepare_train_test(df)


@pytest.fixture(scope="module")
def trained_pipeline(dataset):
    """Train a lightweight LR pipeline for tests — avoids disk dependency."""
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = dataset
    preproc = build_preprocessing_pipeline()
    pipeline = Pipeline(
        [
            ("preprocessor", preproc),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


class TestModelPredictions:
    def test_predict_returns_valid_class(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        probas = pipeline.predict_proba(X_test)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_shape(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        probas = pipeline.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)

    def test_accuracy_above_threshold(self, trained_pipeline):
        from sklearn.metrics import accuracy_score

        pipeline, X_test, y_test = trained_pipeline
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        assert acc >= 0.70, f"Accuracy {acc:.3f} below minimum threshold 0.70"

    def test_roc_auc_above_threshold(self, trained_pipeline):
        from sklearn.metrics import roc_auc_score

        pipeline, X_test, y_test = trained_pipeline
        proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc >= 0.75, f"ROC-AUC {auc:.3f} below minimum threshold 0.75"


class TestModelSerialization:
    def test_pickle_save_load(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(pipeline, tmp)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            loaded_pipeline = pickle.load(f)
        os.unlink(tmp_path)
        original_preds = pipeline.predict(X_test)
        loaded_preds = loaded_pipeline.predict(X_test)
        assert np.array_equal(original_preds, loaded_preds)

    def test_pipeline_has_preprocessor(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_classifier(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        assert "classifier" in pipeline.named_steps


class TestInferencePipeline:
    def test_predict_function_returns_dict(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        assert isinstance(result, dict)

    def test_predict_function_keys(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_function_valid_prediction(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        assert result["prediction"] in (0, 1)

    def test_predict_function_confidence_range(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_function_label_matches_prediction(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        if result["prediction"] == 1:
            assert result["label"] == "Heart Disease"
        else:
            assert result["label"] == "No Heart Disease"

    def test_predict_probabilities_sum_to_one(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        result = predict(SAMPLE_INPUT, model=pipeline)
        total = result["probabilities"]["no_disease"] + result["probabilities"]["disease"]
        assert abs(total - 1.0) < 1e-4

    def test_predict_rejects_missing_features(self, trained_pipeline):
        """Partial input should raise KeyError (feature missing)."""
        pipeline, _, _ = trained_pipeline
        partial_input = {"age": 55.0, "sex": 1.0}
        with pytest.raises(KeyError):
            predict(partial_input, model=pipeline)
