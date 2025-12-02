import pandas as pd
import pytest
from utils.logger import logger

CLEANED_DATA_PATH = "data/processed/cardio_cleaned.csv"


def load_cleaned_data():
    """
    Utility loader with logger + safe exception handling.
    All tests call this function.
    """
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        logger.info("Loaded cleaned dataset successfully")
        return df
    except FileNotFoundError:
        logger.error(f"Cleaned data file not found at: {CLEANED_DATA_PATH}")
        pytest.fail(f"Missing file: {CLEANED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error loading cleaned data: {e}")
        pytest.fail(f"Unexpected error loading cleaned data: {e}")


def test_no_missing_values():
    df = load_cleaned_data()
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Cleaned data contains {missing} missing values"


def test_no_duplicates():
    df = load_cleaned_data()
    dup = df.duplicated().sum()
    assert dup == 0, f"Cleaned data contains {dup} duplicate rows"


def test_required_columns_present():
    df = load_cleaned_data()

    required_cols = [
        "age", "height", "weight",
        "ap_hi", "ap_lo",
        "cholesterol", "gluc",
        "smoke", "alco", "active",
        "cardio",
        "age_years"
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_valid_ranges():
    df = load_cleaned_data()

    assert (df["height"].between(120, 230)).all(), "Invalid height values detected"
    assert (df["weight"].between(30, 250)).all(), "Invalid weight values detected"
    assert (df["ap_hi"].between(80, 250)).all(), "Invalid systolic BP values detected"
    assert (df["ap_lo"].between(40, 150)).all(), "Invalid diastolic BP values detected"
    assert (df["ap_hi"] > df["ap_lo"]).all(), "BP anomaly: ap_hi <= ap_lo"


def test_categorical_values():
    df = load_cleaned_data()

    expected_values = {
        "gender": [1, 2],
        "cholesterol": [1, 2, 3],
        "gluc": [1, 2, 3],
        "smoke": [0, 1],
        "alco": [0, 1],
        "active": [0, 1]
    }

    for col, valid_vals in expected_values.items():
        assert df[col].isin(valid_vals).all(), f"{col} contains invalid values"


def test_age_years_range():
    df = load_cleaned_data()
    assert (df["age_years"].between(30, 80)).all(), "age_years contains unrealistic values"
