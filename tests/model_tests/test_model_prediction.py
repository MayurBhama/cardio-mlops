import os
import joblib
import pandas as pd
import pytest


def test_model_and_scaler_exist():
    """Ensure both model and scaler files are present."""
    if not os.path.exists("models/trained_models/best_model.pkl"):
        pytest.fail("Model file missing: models/trained_models/best_model.pkl")

    if not os.path.exists("models/trained_models/scaler.pkl"):
        pytest.fail("Scaler file missing: models/trained_models/scaler.pkl")


def test_prediction_valid():
    """Test if model can make a valid prediction on a sample input."""

    # Load model safely
    try:
        model = joblib.load("models/trained_models/best_model.pkl")
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

    # Load scaler safely
    try:
        scaler = joblib.load("models/trained_models/scaler.pkl")
    except Exception as e:
        pytest.fail(f"Failed to load scaler: {e}")

    # Sample input for prediction
    sample = pd.DataFrame([{
        "age_years": 45, "gender": 1, "height": 170, "weight": 82,
        "ap_hi": 145, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
        "smoke": 1, "alco": 0, "active": 0,
        "bmi": 28.3, "bmi_category": 2,
        "pulse_pressure": 50, "mean_arterial_pressure": 111.7,
        "bp_category": 2, "age_group": 2,
        "lifestyle_risk_score": 2, "metabolic_risk_score": 5,
        "combined_risk_score": 70
    }])

    # Scaling
    try:
        scaled = scaler.transform(sample)
    except Exception as e:
        pytest.fail(f"Scaler failed to transform sample: {e}")

    # Prediction
    try:
        pred = model.predict(scaled)[0]
    except Exception as e:
        pytest.fail(f"Model failed to predict: {e}")

    # Validate prediction result
    assert pred in [0, 1], f"Prediction out of valid range: {pred}"

    # Check probability output (important for clinical use)
    try:
        proba = model.predict_proba(scaled)[0, 1]
    except Exception:
        pytest.fail("Model does not support predict_proba()")

    assert 0.0 <= proba <= 1.0, f"Invalid predicted probability: {proba}"
