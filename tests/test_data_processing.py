"""
Unit tests for data_processing module.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    CONTINUOUS_FEATURES,
    build_preprocessing_pipeline,
    clean_data,
    get_feature_target,
    load_raw_data,
    prepare_train_test,
)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "processed.cleveland.data")


@pytest.fixture
def raw_df():
    return load_raw_data(RAW_PATH)


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


class TestLoadRawData:
    def test_loads_correct_shape(self, raw_df):
        assert raw_df.shape[1] == 14

    def test_has_expected_columns(self, raw_df):
        assert list(raw_df.columns) == COLUMN_NAMES

    def test_row_count_reasonable(self, raw_df):
        assert len(raw_df) > 200

    def test_question_marks_become_nan(self, raw_df):
        # ca and thal have '?' values in Cleveland data
        assert raw_df["ca"].isnull().sum() + raw_df["thal"].isnull().sum() > 0


class TestCleanData:
    def test_target_is_binary(self, clean_df):
        assert set(clean_df["target"].unique()).issubset({0, 1})

    def test_no_missing_values(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_shape_preserved(self, raw_df, clean_df):
        assert raw_df.shape == clean_df.shape

    def test_both_classes_present(self, clean_df):
        assert len(clean_df["target"].unique()) == 2

    def test_ca_no_nan(self, clean_df):
        assert clean_df["ca"].isnull().sum() == 0

    def test_thal_no_nan(self, clean_df):
        assert clean_df["thal"].isnull().sum() == 0


class TestFeatureEngineering:
    def test_get_feature_target_shapes(self, clean_df):
        X, y = get_feature_target(clean_df)
        assert X.shape[1] == len(ALL_FEATURES)
        assert len(y) == len(clean_df)

    def test_all_features_present(self, clean_df):
        X, _ = get_feature_target(clean_df)
        for col in ALL_FEATURES:
            assert col in X.columns

    def test_feature_lists_non_overlapping(self):
        assert not set(CONTINUOUS_FEATURES) & set(CATEGORICAL_FEATURES)


class TestPreprocessingPipeline:
    def test_pipeline_fit_transform(self, clean_df):
        X, _ = get_feature_target(clean_df)
        pipeline = build_preprocessing_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape

    def test_pipeline_output_no_nan(self, clean_df):
        X, _ = get_feature_target(clean_df)
        pipeline = build_preprocessing_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_continuous_features_scaled(self, clean_df):
        X, _ = get_feature_target(clean_df)
        pipeline = build_preprocessing_pipeline()
        X_tr = pipeline.fit_transform(X)
        # After StandardScaler mean should be near 0 for continuous features
        cont_idx = [list(ALL_FEATURES).index(c) for c in CONTINUOUS_FEATURES]
        means = X_tr[:, cont_idx].mean(axis=0)
        assert all(abs(m) < 0.1 for m in means)


class TestTrainTestSplit:
    def test_split_shapes(self, clean_df):
        X_train, X_test, y_train, y_test = prepare_train_test(clean_df)
        total = len(clean_df)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) + len(y_test) == total

    def test_stratified_split(self, clean_df):
        X_train, X_test, y_train, y_test = prepare_train_test(clean_df)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        # Stratification should keep class ratios similar within 5%
        assert abs(train_ratio - test_ratio) < 0.05

    def test_no_data_leakage(self, clean_df):
        X_train, X_test, y_train, y_test = prepare_train_test(clean_df)
        # Indices should not overlap
        assert len(set(X_train.index) & set(X_test.index)) == 0
