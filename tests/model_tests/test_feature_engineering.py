import os
import pandas as pd
import pytest


def test_feature_engineered_file_exists():
    """Ensure the featured dataset exists before running column checks."""
    if not os.path.exists("data/processed/cardio_featured.csv"):
        pytest.fail("The feature-engineered dataset is missing: data/processed/cardio_featured.csv")


def test_feature_engineering_output():
    """Verify that key engineered features exist in the processed dataset."""
    try:
        df = pd.read_csv("data/processed/cardio_featured.csv")
    except Exception as e:
        pytest.fail(f"Failed to load cardio_featured.csv: {e}")

    expected_features = [
        "bmi",
        "bmi_category",
        "pulse_pressure",
        "mean_arterial_pressure",
        "bp_category",
        "combined_risk_score"
    ]

    missing_features = [f for f in expected_features if f not in df.columns]

    assert len(missing_features) == 0, f"Missing engineered features: {missing_features}"
