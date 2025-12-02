"""
Model Training Module - FIXED VERSION
Trains multiple models and SAVES train/test splits for consistent evaluation
"""

import pandas as pd
import numpy as np
import yaml
import sys
import joblib
import mlflow
import mlflow.sklearn
import json

from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class ModelTrainer:
    """
    Comprehensive training class with MLflow logging & robust exception handling
    """

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        try:
            self.config = self._load_config(config_path)
            self.models = {}
            self.results = {}
            self.scalers = {}
            self.best_model = None
            self.best_model_name = None

            # Setup MLflow
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

            logger.info("ModelTrainer initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load model_config.yaml")
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Split -> Scale -> SAVE SPLITS -> Return train, val, test & scaler
        """
        try:
            logger.info("Preparing data for model training")

            features = [c for c in feature_columns if c in df.columns]
            if len(features) == 0:
                raise CustomException("No valid feature columns found", sys)

            X = df[features]
            y = df["cardio"]

            # Splits
            test_size = self.config["preprocessing"]["test_size"]
            val_size = self.config["preprocessing"]["validation_size"]
            random_state = self.config["preprocessing"]["random_state"]

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
            )

            logger.info(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

            # ============= CRITICAL FIX: SAVE RAW SPLITS BEFORE SCALING =============
            self._save_data_splits(X_train, y_train, X_val, y_val, X_test, y_test, features)
            # =========================================================================

            # Scaling
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            X_train_s = pd.DataFrame(X_train_s, columns=features)
            X_val_s = pd.DataFrame(X_val_s, columns=features)
            X_test_s = pd.DataFrame(X_test_s, columns=features)

            logger.info("Data preparation completed")

            return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    # NEW METHOD: Save data splits to CSV for consistent evaluation
    # -------------------------------------------------------------------
    def _save_data_splits(self, X_train, y_train, X_val, y_val, X_test, y_test, features):
        """
        CRITICAL: Save train/val/test splits as CSV files
        This ensures evaluation uses the SAME data as training!
        """
        try:
            logger.info("Saving data splits to CSV...")

            # Create directories
            Path("data/train").mkdir(parents=True, exist_ok=True)
            Path("data/validation").mkdir(parents=True, exist_ok=True)
            Path("data/test").mkdir(parents=True, exist_ok=True)

            # Save train set
            train_df = X_train.copy()
            train_df['cardio'] = y_train.values
            train_df.to_csv("data/train/train.csv", index=False)
            logger.info(f" Saved: data/train/train.csv ({train_df.shape})")

            # Save validation set
            val_df = X_val.copy()
            val_df['cardio'] = y_val.values
            val_df.to_csv("data/validation/validation.csv", index=False)
            logger.info(f" Saved: data/validation/validation.csv ({val_df.shape})")

            # Save test set
            test_df = X_test.copy()
            test_df['cardio'] = y_test.values
            test_df.to_csv("data/test/test.csv", index=False)
            logger.info(f" Saved: data/test/test.csv ({test_df.shape})")

            # Save feature names (CRITICAL for consistency)
            feature_metadata = {
                "features": features,
                "n_features": len(features),
                "saved_at": datetime.now().isoformat()
            }
            
            Path("models/trained_models").mkdir(parents=True, exist_ok=True)
            with open("models/trained_models/feature_names.json", 'w') as f:
                json.dump(feature_metadata, f, indent=4)
            logger.info(f" Saved: models/trained_models/feature_names.json ({len(features)} features)")

        except Exception as e:
            logger.error("Failed to save data splits!")
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def train_all_models(self, X_train, X_val, y_train, y_val):
        try:
            logger.info("Starting training of all models")

            self._train_logistic_regression(X_train, X_val, y_train, y_val)
            self._train_random_forest(X_train, X_val, y_train, y_val)
            self._train_gradient_boosting(X_train, X_val, y_train, y_val)
            self._train_xgboost(X_train, X_val, y_train, y_val)

            self._select_best_model()
            logger.info("All models trained successfully")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _evaluate_model(self, model, X_train, X_val, y_train, y_val) -> Dict:
        try:
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]

            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]

            return {
                "train_metrics": {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred),
                    "recall": recall_score(y_train, y_train_pred),
                    "f1_score": f1_score(y_train, y_train_pred),
                    "roc_auc": roc_auc_score(y_train, y_train_proba)
                },
                "val_metrics": {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "precision": precision_score(y_val, y_val_pred),
                    "recall": recall_score(y_val, y_val_pred),
                    "f1_score": f1_score(y_val, y_val_pred),
                    "roc_auc": roc_auc_score(y_val, y_val_proba)
                },
                "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
                "report": classification_report(y_val, y_val_pred)
            }

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _train_logistic_regression(self, X_train, X_val, y_train, y_val):
        try:
            params = self.config["models"]["logistic_regression"]

            with mlflow.start_run(run_name="Logistic_Regression"):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
                mlflow.log_params(params)
                mlflow.log_metrics(results["val_metrics"])
                mlflow.sklearn.log_model(model, "model")

                self.models["Logistic_Regression"] = model
                self.results["Logistic_Regression"] = results

                logger.info(f"Logistic Regression completed: AUC={results['val_metrics']['roc_auc']:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _train_random_forest(self, X_train, X_val, y_train, y_val):
        try:
            params = self.config["models"]["random_forest"]

            with mlflow.start_run(run_name="Random_Forest"):
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
                mlflow.log_params(params)
                mlflow.log_metrics(results["val_metrics"])

                feature_importance = dict(
                    zip(X_train.columns, model.feature_importances_)
                )
                mlflow.log_dict(feature_importance, "feature_importance.json")

                mlflow.sklearn.log_model(model, "model")

                self.models["Random_Forest"] = model
                self.results["Random_Forest"] = results

                logger.info(f"Random Forest completed: AUC={results['val_metrics']['roc_auc']:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _train_gradient_boosting(self, X_train, X_val, y_train, y_val):
        try:
            params = self.config["models"]["gradient_boosting"]

            with mlflow.start_run(run_name="Gradient_Boosting"):
                model = GradientBoostingClassifier(**params)
                model.fit(X_train, y_train)

                results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
                mlflow.log_params(params)
                mlflow.log_metrics(results["val_metrics"])
                mlflow.sklearn.log_model(model, "model")

                self.models["Gradient_Boosting"] = model
                self.results["Gradient_Boosting"] = results

                logger.info(f"Gradient Boosting completed: AUC={results['val_metrics']['roc_auc']:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _train_xgboost(self, X_train, X_val, y_train, y_val):
        try:
            params = self.config["models"]["xgboost"]

            with mlflow.start_run(run_name="XGBoost"):
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
                mlflow.log_params(params)
                mlflow.log_metrics(results["val_metrics"])
                mlflow.sklearn.log_model(model, "model")

                self.models["XGBoost"] = model
                self.results["XGBoost"] = results

                logger.info(f"XGBoost completed: AUC={results['val_metrics']['roc_auc']:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def _select_best_model(self):
        try:
            best_score = 0
            best_name = None

            for name, result in self.results.items():
                auc = result["val_metrics"]["roc_auc"]
                if auc > best_score:
                    best_score = auc
                    best_name = name

            self.best_model_name = best_name
            self.best_model = self.models[best_name]

            logger.info(f"Best Model Selected -> {best_name} (ROC-AUC={best_score:.4f})")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def evaluate_on_test(self, X_test, y_test) -> Dict:
        try:
            logger.info(f"Evaluating best model: {self.best_model_name}")

            model = self.best_model

            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
                "confusion_matrix": confusion_matrix(y_test, pred).tolist()
            }

            logger.info(f"Test evaluation completed: AUC={metrics['roc_auc']:.4f}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------
    def save_best_model(self, model_path: str, scaler_path: str):
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.best_model, model_path)
            joblib.dump(self.scalers.get("scaler"), scaler_path)

            logger.info(f"Best model saved at: {model_path}")
            logger.info(f"Scaler saved at: {scaler_path}")

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------------------
def main():
    try:
        df = pd.read_csv("data/processed/cardio_featured.csv")

        features = [
            "age_years", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc",
            "smoke", "alco", "active", "bmi", "bmi_category",
            "pulse_pressure", "mean_arterial_pressure",
            "bp_category", "age_group", "lifestyle_risk_score",
            "metabolic_risk_score", "combined_risk_score"
        ]

        trainer = ModelTrainer()

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = trainer.prepare_data(df, features)
        trainer.scalers["scaler"] = scaler

        trainer.train_all_models(X_train, X_val, y_train, y_val)

        test_results = trainer.evaluate_on_test(X_test, y_test)

        trainer.save_best_model(
            "models/trained_models/best_model.pkl",
            "models/trained_models/scaler.pkl"
        )

        logger.info("Training pipeline completed successfully")
        logger.info(f"\n{'='*80}")
        logger.info(" Data splits saved to: data/train, data/validation, data/test")
        logger.info(" Feature names saved to: models/trained_models/feature_names.json")
        logger.info(" Models saved to: models/trained_models/")
        logger.info(f"{'='*80}\n")
        
        return trainer

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()