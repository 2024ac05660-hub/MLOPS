"""
Train, evaluate, and log ML models for Heart Disease prediction.
Tasks 2 & 3: Feature engineering, model development, MLflow experiment tracking.
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from data_processing import (
    load_processed, prepare_train_test, build_preprocessing_pipeline,
    PROCESSED_DATA_PATH, ALL_FEATURES,
)

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "heart-disease-classification"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def plot_roc_curve(model_name, y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#e74c3c", label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{model_name.replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_confusion_matrix(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_feature_importance(model_name, pipeline, feature_names):
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return None
    idx = np.argsort(importances)[::-1]
    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_names)), importances[idx], color="#3498db", edgecolor="white")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"Feature Importances — {model_name}", fontweight="bold")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"fi_{model_name.replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    return path


def train_and_log(model_name, classifier, params, X_train, X_test, y_train, y_test):
    """Train one model, log everything to MLflow, return (pipeline, metrics)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    preproc = build_preprocessing_pipeline()
    pipeline = Pipeline(
        [("preprocessor", preproc), ("classifier", classifier)],
        memory=None,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
        cv=cv, scoring=["accuracy", "roc_auc"], return_train_score=False,
    )
    metrics["cv_accuracy_mean"] = round(cv_results["test_accuracy"].mean(), 4)
    metrics["cv_roc_auc_mean"] = round(cv_results["test_roc_auc"].mean(), 4)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        roc_path = plot_roc_curve(model_name, y_test, y_proba)
        cm_path = plot_confusion_matrix(model_name, y_test, y_pred)
        fi_path = plot_feature_importance(model_name, pipeline, ALL_FEATURES)

        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_artifact(cm_path, artifact_path="plots")
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")

        mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                  registered_model_name=model_name.replace(" ", "_"))

        report = classification_report(y_test, y_pred,
                                       target_names=["No Disease", "Disease"])
        report_path = os.path.join(REPORTS_DIR, f"report_{model_name.replace(' ', '_')}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall   : {metrics['recall']}")
        print(f"  F1       : {metrics['f1']}")
        print(f"  ROC-AUC  : {metrics['roc_auc']}")
        print(f"  CV Acc   : {metrics['cv_accuracy_mean']}")
        print(f"  CV AUC   : {metrics['cv_roc_auc_mean']}")

    return pipeline, metrics


def tune_random_forest(X_train, y_train):
    """Grid-search best RF hyperparameters."""
    preproc = build_preprocessing_pipeline()
    base_pipeline = Pipeline(
        [("preprocessor", preproc),
         ("classifier", RandomForestClassifier(random_state=42))],
        memory=None,
    )
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6, None],
        "classifier__min_samples_split": [2, 5],
    }
    gs = GridSearchCV(base_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Best RF params: {gs.best_params_}  CV AUC={gs.best_score_:.4f}")
    return gs.best_params_, gs.best_estimator_


def save_best_model(pipeline, model_name="best_model"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved best model to {path}")
    return path


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_processed()
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # --- Model 1: Logistic Regression ---
    lr_params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
    lr_clf = LogisticRegression(**lr_params)
    lr_pipeline, lr_metrics = train_and_log(
        "Logistic Regression", lr_clf, lr_params, X_train, X_test, y_train, y_test
    )

    # --- Model 2: Random Forest (tuned) ---
    best_rf_params, _ = tune_random_forest(X_train, y_train)
    rf_clf = RandomForestClassifier(
        n_estimators=best_rf_params["classifier__n_estimators"],
        max_depth=best_rf_params["classifier__max_depth"],
        min_samples_split=best_rf_params["classifier__min_samples_split"],
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
    )
    rf_plain_params = {k.replace("classifier__", ""): v for k, v in best_rf_params.items()}
    rf_pipeline, rf_metrics = train_and_log(
        "Random Forest", rf_clf, rf_plain_params, X_train, X_test, y_train, y_test
    )

    # --- Model 3: Gradient Boosting ---
    gb_params = {
        "n_estimators": 150, "learning_rate": 0.1,
        "max_depth": 4, "random_state": 42,
    }
    gb_clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    gb_pipeline, gb_metrics = train_and_log(
        "Gradient Boosting", gb_clf, gb_params, X_train, X_test, y_train, y_test
    )

    # Select best model by ROC-AUC
    results = {
        "Logistic Regression": (lr_pipeline, lr_metrics),
        "Random Forest": (rf_pipeline, rf_metrics),
        "Gradient Boosting": (gb_pipeline, gb_metrics),
    }
    best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_pipeline, best_metrics = results[best_name]

    print(f"\n>>> Best model: {best_name}  ROC-AUC={best_metrics['roc_auc']}")
    save_best_model(best_pipeline, "best_model")

    meta = {"best_model": best_name, "metrics": best_metrics}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Training complete. Model and metadata saved.")


if __name__ == "__main__":
    main()
