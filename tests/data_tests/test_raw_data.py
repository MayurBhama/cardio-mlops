import os
import pandas as pd
import pytest
from utils.logger import logger


RAW_DATA_PATH = "data/raw/cardio_train.csv"


def load_raw_data():
    """
    Utility loader for raw data with logging + controlled exception handling.
    All tests call this instead of reading directly.
    """
    try:
        if not os.path.exists(RAW_DATA_PATH):
            logger.error(f"Raw dataset not found: {RAW_DATA_PATH}")
            pytest.fail(f"Missing raw data file: {RAW_DATA_PATH}")

        df = pd.read_csv(RAW_DATA_PATH, sep=';')
        logger.info("Loaded raw dataset successfully")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        pytest.fail(f"CSV parsing error in raw dataset: {e}")

    except Exception as e:
        logger.error(f"Unexpected error loading raw dataset: {e}")
        pytest.fail(f"Unexpected error loading raw dataset: {e}")


def test_raw_data_exists():
    assert os.path.exists(RAW_DATA_PATH), "Raw data file missing!"


def test_raw_data_shape():
    df = load_raw_data()

    assert df.shape[0] > 1000, (
        f"Raw dataset seems unusually small: only {df.shape[0]} rows found"
    )

    assert df.shape[1] == 13, (
        f"Unexpected number of columns: expected 13 but found {df.shape[1]}"
    )


def test_raw_data_columns():
    df = load_raw_data()

    expected_cols = [
        "id", "age", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active", "cardio"
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column in raw dataset: {col}"


def test_no_completely_empty_columns():
    df = load_raw_data()

    empty_cols = df.columns[df.isnull().all()].tolist()
    assert len(empty_cols) == 0, f"Raw data contains empty columns: {empty_cols}"


def test_raw_data_has_minimal_quality():
    df = load_raw_data()

    # At least one unique value in each column
    for col in df.columns:
        assert df[col].nunique() > 1, f"Column {col} seems to contain only one repeated value"