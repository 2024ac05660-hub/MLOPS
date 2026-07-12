"""
Data loading, cleaning, and preprocessing pipeline for Heart Disease UCI dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "processed.cleveland.data"
)
PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "heart_disease_clean.csv"
)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values before imputation:")
        print(missing[missing > 0])
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(float)
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}  shape={df.shape}")


def load_processed(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_feature_target(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df["target"]
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Return a ColumnTransformer that applies:
    - Continuous features: median imputation + standard scaling
    - Categorical features: median imputation only (already numeric codes)
    """
    continuous_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, CONTINUOUS_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df_raw = load_raw_data()
    print(f"Raw shape: {df_raw.shape}")
    df_clean = clean_data(df_raw)
    save_processed(df_clean)
    print(df_clean.head())
    print("\nClass distribution:\n", df_clean["target"].value_counts())
