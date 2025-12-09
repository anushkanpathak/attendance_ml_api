import os
import json
import joblib
import numpy as np

# Base directory = attendance_ml
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")

# Paths to saved model, scaler, and feature columns
MODEL_PATH = os.path.join(MODEL_DIR, "attendance_log_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# Load artifacts
log_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS = json.load(f)


def make_feature_vector(payload: dict):
    """
    Convert incoming JSON payload into a scaled feature vector
    expected by the model.

    payload example:
    {
      "overall_att": 0.68,
      "last7": 0.40,
      "last30": 0.55,
      "streak": 4,
      "trend": -0.12
    }
    """
    values = [payload.get(col, 0.0) for col in FEATURE_COLS]
    arr = np.array(values, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled


def predict_risk(payload: dict):
    """
    Returns (label, probability) where:
    label = 0 -> Safe
    label = 1 -> At Risk
    probability = probability of class 1 (At Risk)
    """
    X = make_feature_vector(payload)
    proba = log_model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)
    return label, float(proba)
