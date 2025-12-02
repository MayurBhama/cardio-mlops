"""
Feature Engineering Module
Creates new derived features to improve model performance:
1. BMI and body composition
2. Blood pressure derived metrics
3. Risk scores
4. Age-related features
5. Interaction features
"""

import pandas as pd
import numpy as np
import yaml
import sys
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class FeatureEngineer:
    """
    Feature engineering class for cardiovascular disease prediction
    """

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize FeatureEngineer
        """
        try:
            self.config = self._load_config(config_path)
            self.feature_config = self.config["feature_engineering"]
            self.new_features = []
            logger.info("FeatureEngineer initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {config_path}")
            raise CustomException(e, sys)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline
        """
        try:
            logger.info("Starting feature engineering...")
            df_feat = df.copy()

            if self.feature_config.get("create_bmi"):
                df_feat = self._create_bmi_features(df_feat)

            if self.feature_config.get("create_bp_features"):
                df_feat = self._create_bp_features(df_feat)

            if self.feature_config.get("create_age_groups"):
                df_feat = self._create_age_features(df_feat)

            if self.feature_config.get("create_risk_scores"):
                df_feat = self._create_risk_scores(df_feat)

            df_feat = self._create_interaction_features(df_feat)

            logger.info(
                f"Feature engineering completed. Total new features: {len(self.new_features)}"
            )
            logger.info(f"New features: {self.new_features}")

            return df_feat

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # BMI & Body Composition Features
    # -----------------------------------------------------------------------
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BMI-related engineered features"""
        try:
            logger.info("Creating BMI features...")

            df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
            df["bmi"] = df["bmi"].round(2)
            self.new_features.append("bmi")

            def categorize_bmi(bmi):
                if bmi < 18.5:
                    return 0
                elif bmi < 25:
                    return 1
                elif bmi < 30:
                    return 2
                return 3

            df["bmi_category"] = df["bmi"].apply(categorize_bmi)
            self.new_features.append("bmi_category")

            df["bsa"] = np.sqrt((df["height"] * df["weight"]) / 3600).round(2)
            self.new_features.append("bsa")

            df["weight_height_ratio"] = (df["weight"] / df["height"] * 100).round(2)
            self.new_features.append("weight_height_ratio")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # Blood Pressure Derived Features
    # -----------------------------------------------------------------------
    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BP-derived features"""
        try:
            logger.info("Creating blood pressure features...")

            df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
            self.new_features.append("pulse_pressure")

            df["mean_arterial_pressure"] = (
                df["ap_lo"] + (df["pulse_pressure"] / 3)
            ).round(1)
            self.new_features.append("mean_arterial_pressure")

            def categorize_bp(row):
                s, d = row["ap_hi"], row["ap_lo"]
                if s < 120 and d < 80:
                    return 0
                if s < 130 and d < 80:
                    return 1
                if s < 140 or d < 90:
                    return 2
                if s < 180 or d < 120:
                    return 3
                return 4

            df["bp_category"] = df.apply(categorize_bp, axis=1)
            self.new_features.append("bp_category")

            df["bp_ratio"] = (df["ap_hi"] / df["ap_lo"]).round(2)
            self.new_features.append("bp_ratio")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # Age Features
    # -----------------------------------------------------------------------
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Age groups and derived age metrics"""
        try:
            logger.info("Creating age features...")

            def categorize(age):
                if age < 40:
                    return 0
                if age < 50:
                    return 1
                if age < 60:
                    return 2
                return 3

            df["age_group"] = df["age_years"].apply(categorize)
            self.new_features.append("age_group")

            df["age_squared"] = df["age_years"] ** 2
            self.new_features.append("age_squared")

            return df
        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # Composite Risk Scores
    # -----------------------------------------------------------------------
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Risk scores combining multiple health indicators"""
        try:
            logger.info("Creating risk score features...")

            df["lifestyle_risk_score"] = (
                df["smoke"] + df["alco"] + (1 - df["active"])
            )
            self.new_features.append("lifestyle_risk_score")

            df["metabolic_risk_score"] = df["cholesterol"] + df["gluc"]
            self.new_features.append("metabolic_risk_score")

            df["combined_risk_score"] = (
                (df["bp_category"] / 4 * 30)
                + (df["bmi_category"] / 3 * 20)
                + (df["lifestyle_risk_score"] / 3 * 25)
                + (df["metabolic_risk_score"] / 6 * 25)
            ).round(1)
            self.new_features.append("combined_risk_score")

            df["age_adjusted_risk"] = (
                df["combined_risk_score"] * (1 + (df["age_years"] - 30) / 100)
            ).round(1)
            self.new_features.append("age_adjusted_risk")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # Interaction Features
    # -----------------------------------------------------------------------
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate second-order interaction features"""
        try:
            logger.info("Creating interaction features...")

            df["age_bmi_interaction"] = (df["age_years"] * df["bmi"]).round(2)
            self.new_features.append("age_bmi_interaction")

            df["age_bp_interaction"] = (
                df["age_years"] * df["mean_arterial_pressure"]
            ).round(2)
            self.new_features.append("age_bp_interaction")

            df["gender_bmi_interaction"] = df["gender"] * df["bmi_category"]
            self.new_features.append("gender_bmi_interaction")

            df["smoke_chol_interaction"] = df["smoke"] * df["cholesterol"]
            self.new_features.append("smoke_chol_interaction")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------------------
    # Utility Helpers
    # -----------------------------------------------------------------------
    def get_feature_importance_order(self) -> List[str]:
        """Return recommended feature order"""
        return [
            "age_years",
            "bp_category",
            "mean_arterial_pressure",
            "ap_hi",
            "ap_lo",
            "bmi",
            "bmi_category",
            "cholesterol",
            "gluc",
            "combined_risk_score",
            "age_adjusted_risk",
            "lifestyle_risk_score",
            "metabolic_risk_score",
            "pulse_pressure",
            "gender",
            "smoke",
            "age_bmi_interaction",
            "age_bp_interaction",
            "weight",
            "height",
            "active",
            "alco",
        ]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return descriptions of engineered features"""
        return {
            "bmi": "Body Mass Index",
            "bmi_category": "BMI category (0-3)",
            "bsa": "Body Surface Area",
            "weight_height_ratio": "Weight-to-height ratio",
            "pulse_pressure": "SBP - DBP",
            "mean_arterial_pressure": "MAP",
            "bp_category": "Blood pressure category (0-4)",
            "bp_ratio": "SBP/DBP ratio",
            "age_group": "Age category",
            "age_squared": "Age squared",
            "lifestyle_risk_score": "Smoking + alcohol + inactivity",
            "metabolic_risk_score": "Cholesterol + glucose",
            "combined_risk_score": "Composite risk score",
            "age_adjusted_risk": "Risk score adjusted for age",
            "age_bmi_interaction": "Age × BMI",
            "age_bp_interaction": "Age × BP",
            "gender_bmi_interaction": "Gender × BMI",
            "smoke_chol_interaction": "Smoking × Cholesterol",
        }


def main():
    """Standalone execution"""
    try:
        df = pd.read_csv("data/processed/cardio_cleaned.csv")
        engineer = FeatureEngineer()
        df_featured = engineer.create_all_features(df)
        df_featured.to_csv("data/processed/cardio_featured.csv", index=False)
        logger.info("Feature engineering completed and saved successfully")
        return df_featured
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_featured = main()
