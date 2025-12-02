"""
Model Evaluation Module - FIXED VERSION
Uses SAVED test data instead of creating new random splits
"""

import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path
from typing import Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class ModelEvaluator:
    """
    Comprehensive model evaluation class - USES SAVED TEST DATA
    """

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize ModelEvaluator by loading saved model & scaler
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("ModelEvaluator initialized. Model and scaler loaded.")
        except Exception as e:
            logger.error("Error loading model or scaler.")
            raise CustomException(e, sys)

    # =====================================================================
    # NEW METHOD: Load saved test data
    # =====================================================================
    def load_test_data(self) -> tuple:
        """
        CRITICAL FIX: Load the SAVED test data from training
        This ensures we evaluate on the SAME data the model was trained/validated on
        """
        try:
            test_path = Path("data/test/test.csv")
            
            if not test_path.exists():
                raise FileNotFoundError(
                    f"Test data not found at: {test_path}\n"
                    "Please run training first: python src/model_training/train.py"
                )
            
            logger.info(f"Loading saved test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            
            # Separate features and target
            if 'cardio' not in test_df.columns:
                raise ValueError("Target column 'cardio' not found in test data!")
            
            y_test = test_df['cardio']
            X_test = test_df.drop(columns=['cardio'])
            
            logger.info(f" Test data loaded: X={X_test.shape}, y={y_test.shape}")
            logger.info(f"  Class distribution: {y_test.value_counts().to_dict()}")
            
            # Verify feature consistency
            self._verify_features(X_test.columns.tolist())
            
            return X_test, y_test
            
        except Exception as e:
            logger.error("Failed to load test data!")
            raise CustomException(e, sys)

    # =====================================================================
    # NEW METHOD: Verify feature consistency
    # =====================================================================
    def _verify_features(self, test_features: list):
        """
        Verify that test data has the same features as training
        """
        try:
            feature_path = Path("models/trained_models/feature_names.json")
            
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    metadata = json.load(f)
                    saved_features = metadata.get('features', [])
                
                if set(saved_features) == set(test_features):
                    logger.info(f" Feature consistency verified ({len(saved_features)} features)")
                else:
                    missing = set(saved_features) - set(test_features)
                    extra = set(test_features) - set(saved_features)
                    
                    error_msg = "Feature mismatch detected!\n"
                    if missing:
                        error_msg += f"Missing features: {missing}\n"
                    if extra:
                        error_msg += f"Extra features: {extra}\n"
                    
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                logger.warning(f"Feature names file not found: {feature_path}")
                logger.warning("Cannot verify feature consistency!")
                
        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # Comprehensive Evaluation
    # =====================================================================
    def comprehensive_evaluation(self, X_test, y_test) -> Dict:
        """
        Perform all evaluations and reliability tests
        """
        try:
            logger.info("Starting comprehensive evaluation on SAVED test data")

            # Scale the test data
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

            results = {
                "basic_metrics": self._calculate_basic_metrics(y_test, y_pred, y_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
                "reliability_metrics": self._calculate_reliability_metrics(y_test, y_proba),
                "threshold_analysis": self._analyze_thresholds(y_test, y_proba)
            }

            logger.info("Comprehensive evaluation completed successfully.")
            return results

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # Basic Metrics
    # =====================================================================
    def _calculate_basic_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Calculate accuracy, precision, recall, F1, ROC-AUC, specificity"""
        try:
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, y_proba)),
                "specificity": float(self._calculate_specificity(y_true, y_pred))
            }
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def _calculate_specificity(self, y_true, y_pred) -> float:
        """TN / (TN + FP)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)
        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # Reliability / Calibration
    # =====================================================================
    def _calculate_reliability_metrics(self, y_true, y_proba) -> Dict:
        """
        Calibration, confidence, and stability metrics
        """
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            calibration_error = np.mean(np.abs(prob_true - prob_pred))

            high_conf_mask = (y_proba > 0.8) | (y_proba < 0.2)
            if high_conf_mask.sum() > 0:
                confident_accuracy = accuracy_score(
                    y_true[high_conf_mask],
                    (y_proba[high_conf_mask] > 0.5).astype(int)
                )
            else:
                confident_accuracy = 0

            return {
                "calibration_error": float(calibration_error),
                "high_confidence_accuracy": float(confident_accuracy),
                "high_confidence_percentage": float((high_conf_mask.sum() / len(y_true)) * 100),
                "mean_predicted_probability": float(y_proba.mean()),
                "std_predicted_probability": float(y_proba.std())
            }

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # Threshold Analysis
    # =====================================================================
    def _analyze_thresholds(self, y_true, y_proba) -> Dict:
        """Measure performance at various thresholds"""
        try:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            results = {}

            for t in thresholds:
                pred = (y_proba >= t).astype(int)
                results[f"threshold_{t}"] = {
                    "accuracy": float(accuracy_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0)),
                    "f1_score": float(f1_score(y_true, pred, zero_division=0))
                }

            return results

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # Single Prediction Report
    # =====================================================================
    def generate_prediction_report(self, features: pd.DataFrame) -> Dict:
        """
        Generate detailed prediction explanation for a single patient
        """
        try:
            scaled = self.scaler.transform(features)

            prediction = int(self.model.predict(scaled)[0])
            probability = float(self.model.predict_proba(scaled)[0, 1])

            risk = self._determine_risk_level(probability)
            confidence = self._calculate_confidence(probability)

            return {
                "prediction": prediction,
                "probability": probability,
                "risk_level": risk,
                "confidence": confidence,
                "interpretation": self._generate_interpretation(prediction, probability)
            }

        except Exception as e:
            raise CustomException(e, sys)

    # Risk level categories
    def _determine_risk_level(self, p: float) -> str:
        if p < 0.3:
            return "Low Risk"
        elif p < 0.5:
            return "Moderate Risk"
        elif p < 0.7:
            return "High Risk"
        return "Very High Risk"

    # Confidence calculation
    def _calculate_confidence(self, p: float) -> str:
        score = abs(p - 0.5) * 2
        if score > 0.8:
            return "Very High"
        elif score > 0.6:
            return "High"
        elif score > 0.4:
            return "Moderate"
        return "Low"

    # Text interpretation
    def _generate_interpretation(self, prediction: int, probability: float) -> str:
        if prediction == 0:
            return (
                f"The model indicates LOW likelihood of cardiovascular disease "
                f"(Risk Score: {probability*100:.1f}%). Maintain healthy habits."
            )
        else:
            return (
                f"The model indicates HIGH likelihood of cardiovascular disease "
                f"(Risk Score: {probability*100:.1f}%). Medical consultation recommended."
            )

    # =====================================================================
    # Feature Importance
    # =====================================================================
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Return feature importance for tree-based models"""
        try:
            if hasattr(self.model, "feature_importances_"):
                df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": self.model.feature_importances_
                }).sort_values("importance", ascending=False)

                logger.info("Feature importance extracted.")
                return df

            logger.warning("Model does not provide feature_importances_.")
            return None
        except Exception as e:
            raise CustomException(e, sys)


# ========================================================================
# Main Execution - FIXED VERSION
# ========================================================================
def main():
    try:
        logger.info("="*80)
        logger.info("MODEL EVALUATION - USING SAVED TEST DATA")
        logger.info("="*80)

        # Initialize evaluator
        evaluator = ModelEvaluator(
            "models/trained_models/best_model.pkl",
            "models/trained_models/scaler.pkl"
        )

        # Load SAVED test data (CRITICAL FIX!)
        X_test, y_test = evaluator.load_test_data()

        # Comprehensive evaluation
        results = evaluator.comprehensive_evaluation(X_test, y_test)

        # Print results
        print("\n" + "="*80)
        print("COMPREHENSIVE METRICS")
        print("="*80)
        metrics = results["basic_metrics"]
        print(f"{'Accuracy:':<20} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"{'Precision:':<20} {metrics['precision']:.4f}")
        print(f"{'Recall:':<20} {metrics['recall']:.4f}")
        print(f"{'F1-Score:':<20} {metrics['f1_score']:.4f}")
        print(f"{'ROC-AUC:':<20} {metrics['roc_auc']:.4f}")
        print(f"{'Specificity:':<20} {metrics['specificity']:.4f}")

        # Confusion Matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = results["confusion_matrix"]
        print(f"TN: {cm[0][0]:5d}  |  FP: {cm[0][1]:5d}")
        print(f"FN: {cm[1][0]:5d}  |  TP: {cm[1][1]:5d}")

        # Sample prediction
        print("\n" + "="*80)
        print("SAMPLE PREDICTION REPORT")
        print("="*80)
        sample_report = evaluator.generate_prediction_report(X_test.iloc[0:1])
        for key, value in sample_report.items():
            print(f"{key}: {value}")

        print("\n" + "="*80)
        print(" EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

        return evaluator, results

    except Exception as e:
        logger.error("Evaluation failed!")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()