"""
Inference script — load the saved best model and run predictions.
Task 4: Model packaging & reproducibility.
"""
import os
import json
import pickle
import argparse
import pandas as pd
from data_processing import ALL_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_meta.json")

SAMPLE_INPUT = {
    "age": 55.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 217.0,
    "fbs": 0.0, "restecg": 2.0, "thalach": 111.0, "exang": 1.0, "oldpeak": 5.6,
    "slope": 3.0, "ca": 0.0, "thal": 7.0,
}


def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(input_data: dict, model=None) -> dict:
    if model is None:
        model = load_model()
    df = pd.DataFrame([input_data])[ALL_FEATURES]
    prediction = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    confidence = float(round(proba[prediction], 4))
    return {
        "prediction": prediction,
        "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "confidence": confidence,
        "probabilities": {
            "no_disease": float(round(proba[0], 4)),
            "disease": float(round(proba[1], 4)),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction")
    parser.add_argument("--input", type=str, help="JSON string of patient features")
    args = parser.parse_args()

    model = load_model()
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        print(f"Loaded model: {meta.get('best_model', 'unknown')}")

    if args.input:
        input_data = json.loads(args.input)
    else:
        input_data = SAMPLE_INPUT
        print("No input provided, using sample patient data.")

    result = predict(input_data, model)
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
