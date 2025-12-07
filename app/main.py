from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from src.config import load_config, PROJECT_ROOT

# -----------------------------
# Load config, model, preprocessor
# -----------------------------
cfg = load_config()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # this is /app/app inside container

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

NUM_COLS = cfg["features"]["numeric"]
CAT_COLS = cfg["features"]["categorical"]
ALL_COLS = NUM_COLS + CAT_COLS

app = FastAPI(
    title="Kidney CKD Prediction API",
    description="Predict CKD_Status using Kidney Function dataset model",
    version="1.0.0",
)


# -----------------------------
# Request schema
# -----------------------------
class KidneyInput(BaseModel):
    # numeric features
    Creatinine: float
    BUN: float
    GFR: float
    Urine_Output: float
    Age: float
    Protein_in_Urine: float
    Water_Intake: float

    # categorical features
    Diabetes: str      # e.g. "Yes" / "No"
    Hypertension: str  # e.g. "Yes" / "No"
    Medication: str    # e.g. "Yes" / "No" or medicine name


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Kidney CKD prediction API is running",
        "expected_features": ALL_COLS,
    }


@app.post("/predict")
def predict(input_data: KidneyInput):
    """
    Take one patient's data and return:
    - prediction (0/1)
    - probability_of_ckd (if model supports predict_proba)
    """

    # Convert Pydantic model to dict
    data_dict = input_data.dict()

    # Make sure columns are in exact order used in training
    df = pd.DataFrame([data_dict], columns=ALL_COLS)

    # Apply SAME preprocessing as training
    X_transformed = preprocessor.transform(df)

    # Predict class
    raw_pred = model.predict(X_transformed)[0]
    # Convert numpy int64 -> normal Python int
    pred_int = int(raw_pred)

    # Predict probability (if available)
    prob_val = None
    if hasattr(model, "predict_proba"):
        raw_prob = model.predict_proba(X_transformed)[0, 1]
        # Convert numpy float64 -> normal Python float
        prob_val = float(raw_prob)

    return {
        "prediction": pred_int,
        "probability_of_ckd": prob_val,
    }
