"""
Download Heart Disease UCI Dataset from UCI ML Repository.
Usage:
    python scripts/download_data.py
"""
import os
import urllib.request

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed.cleveland.data")

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target",
]


def download():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        print(f"Dataset already exists at {OUTPUT_FILE}")
        return
    print(f"Downloading from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    download()
    print("Column order:", COLUMN_NAMES)
