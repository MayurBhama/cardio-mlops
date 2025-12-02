import os
import joblib
import pytest


def test_model_file_exists():
    """Check if the trained model file exists."""
    if not os.path.exists("models/trained_models/best_model.pkl"):
        pytest.fail("Model file missing: models/trained_models/best_model.pkl")


def test_scaler_file_exists():
    """Check if the saved scaler file exists."""
    if not os.path.exists("models/trained_models/scaler.pkl"):
        pytest.fail("Scaler file missing: models/trained_models/scaler.pkl")


def test_model_loads_properly():
    """Ensure the model loads without errors and is usable."""
    try:
        model = joblib.load("models/trained_models/best_model.pkl")
    except Exception as e:
        pytest.fail(f"Model failed to load: {e}")

    # Basic sanity check: model should have a predict() method
    assert hasattr(model, "predict"), "Loaded model does not have a predict() method"

    # Scikit-learn models also typically have predict_proba
    assert hasattr(model, "predict_proba"), "Loaded model does not have a predict_proba() method"
