import joblib
import pandas as pd
from pathlib import Path


# Load the trained stroke model
MODEL_PATH = Path(__file__).parent / "stroke_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_stroke(patient_data):
    """
    Predict stroke risk for one patient.

    patient_data should be a dictionary containing
    the same input features used during model training.
    """

    patient_df = pd.DataFrame([patient_data])

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }