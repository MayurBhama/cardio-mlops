"""
Data Ingestion Module
This module handles loading raw data and initial validation
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.exception import CustomException


class DataIngestion:
    """
    Class to handle data ingestion from various sources
    """

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize DataIngestion class
        
        Args:
            config_path: Path to data configuration file
        """
        try:
            self.config = self._load_config(config_path)
            logger.info("DataIngestion initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration file: {config_path}")
            raise CustomException(e, sys)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to the CSV file
        """
        try:
            logger.info(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path, sep=";")
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data file: {file_path}")
            raise CustomException(e, sys)

    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Extract dataset metadata and quality info
        """
        try:
            info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicates": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df) * 100),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,
            }

            logger.info("Data information extracted successfully")
            return info

        except Exception as e:
            logger.error("Error extracting data information")
            raise CustomException(e, sys)

    def validate_data_schema(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate schema according to config
        """
        try:
            errors = []

            expected_cols = list(self.config["columns"].values())
            missing_cols = set(expected_cols) - set(df.columns)

            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            for col in df.columns:
                if col in expected_cols:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        if col not in ["id"]:
                            errors.append(f"Column {col} must be numeric")

            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                errors.append(f"Empty columns found: {empty_cols}")

            is_valid = len(errors) == 0

            if is_valid:
                logger.info("Schema validation passed")
            else:
                logger.warning(f"Schema validation failed with {len(errors)} issues")
                for err in errors:
                    logger.warning(f" - {err}")

            return is_valid, errors

        except Exception as e:
            logger.error("Error validating schema")
            raise CustomException(e, sys)

    def initial_data_quality_check(self, df: pd.DataFrame) -> Dict:
        """
        Perform missing, duplicate, and categorical value checks
        """
        try:
            quality_report = {}

            missing_percent = (df.isnull().sum() / len(df) * 100)
            quality_report["missing_values"] = {
                "columns_with_missing": missing_percent[missing_percent > 0].to_dict(),
                "max_missing_percentage": missing_percent.max(),
                "passed": missing_percent.max()
                < self.config["quality_thresholds"]["max_missing_percentage"],
            }

            duplicate_count = df.duplicated().sum()
            duplicate_percent = (duplicate_count / len(df)) * 100
            quality_report["duplicates"] = {
                "count": int(duplicate_count),
                "percentage": duplicate_percent,
                "passed": duplicate_percent
                < (self.config["quality_thresholds"]["duplicate_threshold"] * 100),
            }

            categorical_checks = {}
            for col, valid_vals in self.config["categorical_values"].items():
                if col in df.columns:
                    unique_vals = df[col].unique()
                    invalid_vals = set(unique_vals) - set(valid_vals) - {np.nan}
                    categorical_checks[col] = {
                        "unique_values": unique_vals.tolist(),
                        "expected_values": valid_vals,
                        "invalid_values": list(invalid_vals),
                        "passed": len(invalid_vals) == 0,
                    }

            quality_report["categorical_validation"] = categorical_checks

            passed_checks = sum(
                [
                    quality_report["missing_values"]["passed"],
                    quality_report["duplicates"]["passed"],
                    sum([v["passed"] for v in categorical_checks.values()]),
                ]
            )

            total_checks = 2 + len(categorical_checks)
            quality_report["quality_score"] = round(
                (passed_checks / total_checks) * 100, 2
            )

            logger.info(f"Data quality score: {quality_report['quality_score']}%")
            return quality_report

        except Exception as e:
            logger.error("Error during data quality checks")
            raise CustomException(e, sys)

    def save_data(self, df: pd.DataFrame, output_path: str):
        """
        Save data to CSV
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving data to: {output_path}")
            raise CustomException(e, sys)


def main():
    """
    Demonstration usage
    """
    try:
        ingestion = DataIngestion()

        df = ingestion.load_data("data/raw/cardio_train.csv")

        info = ingestion.get_data_info(df)
        print("\n=== DATA INFORMATION ===")
        for k, v in info.items():
            print(k, ":", v)

        is_valid, errors = ingestion.validate_data_schema(df)
        print("\n=== SCHEMA VALIDATION ===")
        print("Valid:", is_valid)
        print("Errors:", errors)

        quality = ingestion.initial_data_quality_check(df)
        print("\n=== QUALITY REPORT ===")
        print(quality)

        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df = main()
