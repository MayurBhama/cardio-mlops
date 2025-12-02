"""
Data Cleaning Module
This module performs comprehensive data cleaning including:
- Outlier detection and handling
- Missing value imputation
- Data type corrections
- Invalid value handling
"""

import pandas as pd
import numpy as np
import yaml
import sys
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.exception import CustomException


class DataCleaner:
    """
    Comprehensive data cleaning class for cardiovascular dataset
    """

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize DataCleaner

        Args:
            config_path: Path to data configuration file
        """
        try:
            self.config = self._load_config(config_path)
            self.cleaning_report = {
                "initial_shape": None,
                "final_shape": None,
                "removed_rows": 0,
                "outliers_removed": {},
                "invalid_values_handled": {},
                "duplicates_removed": 0,
            }
            logger.info("DataCleaner initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {config_path}")
            raise CustomException(e, sys)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline
        """
        try:
            logger.info("Starting data cleaning pipeline...")
            self.cleaning_report["initial_shape"] = df.shape

            df_clean = df.copy()

            df_clean = self._remove_duplicates(df_clean)
            df_clean = self._convert_age_to_years(df_clean)
            df_clean = self._handle_invalid_categorical(df_clean)
            df_clean = self._remove_outliers(df_clean)
            df_clean = self._handle_blood_pressure_anomalies(df_clean)
            df_clean = self._handle_missing_values(df_clean)
            df_clean = self._ensure_data_types(df_clean)

            self.cleaning_report["final_shape"] = df_clean.shape
            self.cleaning_report["removed_rows"] = (
                self.cleaning_report["initial_shape"][0] - df_clean.shape[0]
            )

            logger.info(
                f"Data cleaning completed. Total removed rows: {self.cleaning_report['removed_rows']}"
            )

            return df_clean

        except Exception as e:
            logger.error("Error occurred during data cleaning")
            raise CustomException(e, sys)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        try:
            initial_len = len(df)
            df_clean = df.drop_duplicates()
            removed = initial_len - len(df_clean)
            self.cleaning_report["duplicates_removed"] = removed
            logger.info(f"Removed {removed} duplicate rows")
            return df_clean
        except Exception as e:
            raise CustomException(e, sys)

    def _convert_age_to_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert age from days to years"""
        try:
            df["age_years"] = (df["age"] / 365.25).round().astype(int)
            logger.info("Converted age from days to years")
            return df
        except Exception as e:
            logger.error("Error converting age to years")
            raise CustomException(e, sys)

    def _handle_invalid_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid categorical values"""
        try:
            for col, valid_values in self.config["categorical_values"].items():
                if col in df.columns:
                    initial_len = len(df)
                    df = df[df[col].isin(valid_values)]
                    removed = initial_len - len(df)

                    if removed > 0:
                        self.cleaning_report["invalid_values_handled"][col] = removed
                        logger.info(f"Removed {removed} invalid values in {col}")

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on valid ranges"""
        try:
            valid_ranges = self.config["valid_ranges"]
            initial_len = len(df)

            # Age
            if "age_years" in df.columns:
                low, high = valid_ranges["age_years"]
                mask = (df["age_years"] >= low) & (df["age_years"] <= high)
                removed = len(df) - mask.sum()
                df = df[mask]
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["age"] = removed

            # Height
            if "height" in df.columns:
                low, high = valid_ranges["height"]
                mask = (df["height"] >= low) & (df["height"] <= high)
                removed = len(df) - mask.sum()
                df = df[mask]
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["height"] = removed

            # Weight
            if "weight" in df.columns:
                low, high = valid_ranges["weight"]
                mask = (df["weight"] >= low) & (df["weight"] <= high)
                removed = len(df) - mask.sum()
                df = df[mask]
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["weight"] = removed

            # Systolic BP
            if "ap_hi" in df.columns:
                low, high = valid_ranges["ap_hi"]
                mask = (df["ap_hi"] >= low) & (df["ap_hi"] <= high)
                removed = len(df) - mask.sum()
                df = df[mask]
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["systolic_bp"] = removed

            # Diastolic BP
            if "ap_lo" in df.columns:
                low, high = valid_ranges["ap_lo"]
                mask = (df["ap_lo"] >= low) & (df["ap_lo"] <= high)
                removed = len(df) - mask.sum()
                df = df[mask]
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["diastolic_bp"] = removed

            total_removed = initial_len - len(df)
            logger.info(f"Total outliers removed: {total_removed}")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _handle_blood_pressure_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove cases where diastolic >= systolic"""
        try:
            if "ap_hi" in df.columns and "ap_lo" in df.columns:
                initial_len = len(df)
                df = df[df["ap_lo"] < df["ap_hi"]]
                removed = initial_len - len(df)
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["bp_anomalies"] = removed
                    logger.info(f"Removed {removed} BP anomalies")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values"""
        try:
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                initial = len(df)
                df = df.dropna()
                logger.info(f"Removed {initial - len(df)} rows with missing values")
            else:
                logger.info("No missing values found")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data types for numeric columns"""
        try:
            int_columns = [
                "id",
                "age",
                "gender",
                "height",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active",
                "cardio",
                "age_years",
            ]

            for col in int_columns:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            if "weight" in df.columns:
                df["weight"] = df["weight"].astype(float)

            logger.info("Data types ensured successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_cleaning_report(self) -> Dict:
        """Return cleaning summary report"""
        return self.cleaning_report

    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        try:
            stats_dict = {
                "numeric_stats": df.describe().to_dict(),
                "categorical_distribution": {},
                "target_distribution": df["cardio"].value_counts().to_dict()
                if "cardio" in df.columns
                else None,
                "gender_distribution": df["gender"].value_counts().to_dict()
                if "gender" in df.columns
                else None,
            }

            categorical_cols = ["cholesterol", "gluc", "smoke", "alco", "active"]

            for col in categorical_cols:
                if col in df.columns:
                    stats_dict["categorical_distribution"][col] = (
                        df[col].value_counts().to_dict()
                    )

            return stats_dict

        except Exception as e:
            raise CustomException(e, sys)


def main():
    """Run cleaning standalone for testing"""
    try:
        df = pd.read_csv("data/raw/cardio_train.csv", sep=";")
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)

        df_clean.to_csv("data/processed/cardio_cleaned.csv", index=False)
        logger.info("Cleaned data saved successfully")

        return df_clean
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_clean = main()