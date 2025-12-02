"""
Model Evaluation Module
This module provides comprehensive model evaluation and reliability assessment
"""

import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    Focus on reliability and interpretability
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize ModelEvaluator
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
    
    def comprehensive_evaluation(self, X_test, y_test) -> Dict:
        """
        Perform comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        evaluation_results = {
            'basic_metrics': self._calculate_basic_metrics(y_test, y_pred, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'reliability_metrics': self._calculate_reliability_metrics(y_test, y_proba),
            'threshold_analysis': self._analyze_thresholds(y_test, y_proba)
        }
        
        logger.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def _calculate_basic_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Calculate basic classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
    
    def _calculate_specificity(self, y_true, y_pred) -> float:
        """Calculate specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    
    def _calculate_reliability_metrics(self, y_true, y_proba) -> Dict:
        """
        Calculate metrics related to prediction reliability
        """
        # Calibration metrics
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        # Confidence analysis
        high_confidence_mask = (y_proba > 0.8) | (y_proba < 0.2)
        high_confidence_accuracy = accuracy_score(
            y_true[high_confidence_mask],
            (y_proba[high_confidence_mask] > 0.5).astype(int)
        ) if high_confidence_mask.sum() > 0 else 0
        
        return {
            'calibration_error': calibration_error,
            'high_confidence_accuracy': high_confidence_accuracy,
            'high_confidence_percentage': (high_confidence_mask.sum() / len(y_true)) * 100,
            'mean_predicted_probability': y_proba.mean(),
            'std_predicted_probability': y_proba.std()
        }
    
    def _analyze_thresholds(self, y_true, y_proba) -> Dict:
        """
        Analyze performance at different probability thresholds
        """
        thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_results = {}
        
        for threshold in thresholds_to_test:
            y_pred_threshold = (y_proba >= threshold).astype(int)
            threshold_results[f'threshold_{threshold}'] = {
                'accuracy': accuracy_score(y_true, y_pred_threshold),
                'precision': precision_score(y_true, y_pred_threshold),
                'recall': recall_score(y_true, y_pred_threshold),
                'f1_score': f1_score(y_true, y_pred_threshold)
            }
        
        return threshold_results
    
    def generate_prediction_report(self, features: pd.DataFrame) -> Dict:
        """
        Generate detailed prediction report for a single patient
        
        Args:
            features: Patient features (single row DataFrame)
            
        Returns:
            Dictionary with prediction details
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Determine risk level
        risk_level = self._determine_risk_level(probability)
        
        # Get confidence
        confidence = self._calculate_confidence(probability)
        
        report = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'confidence': confidence,
            'interpretation': self._generate_interpretation(prediction, probability, risk_level)
        }
        
        return report
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.5:
            return "Moderate Risk"
        elif probability < 0.7:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate prediction confidence"""
        confidence_score = abs(probability - 0.5) * 2  # 0 to 1 scale
        
        if confidence_score > 0.8:
            return "Very High"
        elif confidence_score > 0.6:
            return "High"
        elif confidence_score > 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_interpretation(self, prediction: int, probability: float, risk_level: str) -> str:
        """Generate human-readable interpretation"""
        if prediction == 0:
            interpretation = f"Based on the provided health parameters, the model predicts a LOW likelihood of cardiovascular disease (Risk Score: {probability*100:.1f}%). "
            interpretation += "However, maintaining a healthy lifestyle is always recommended."
        else:
            interpretation = f"Based on the provided health parameters, the model predicts a HIGH likelihood of cardiovascular disease (Risk Score: {probability*100:.1f}%). "
            interpretation += "Please consult with a healthcare professional for proper medical evaluation and guidance."
        
        return interpretation
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance if model supports it
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature importance calculated")
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return None


def main():
    """
    Main function to demonstrate model evaluation
    """
    # Load test data
    df = pd.read_csv("data/processed/cardio_featured.csv")
    
    # Define features
    feature_columns = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'bmi', 'bmi_category', 'pulse_pressure', 
        'mean_arterial_pressure', 'bp_category',
        'age_group', 'lifestyle_risk_score', 
        'metabolic_risk_score', 'combined_risk_score'
    ]
    
    X = df[feature_columns]
    y = df['cardio']
    
    # Take last 20% as test set
    test_size = int(len(df) * 0.2)
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        "models/trained_models/best_model.pkl",
        "models/trained_models/scaler.pkl"
    )
    
    # Comprehensive evaluation
    results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    print("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
    print(f"Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"Precision: {results['basic_metrics']['precision']:.4f}")
    print(f"Recall: {results['basic_metrics']['recall']:.4f}")
    print(f"F1-Score: {results['basic_metrics']['f1_score']:.4f}")
    print(f"ROC-AUC: {results['basic_metrics']['roc_auc']:.4f}")
    
    print("\n=== RELIABILITY METRICS ===")
    print(f"Calibration Error: {results['reliability_metrics']['calibration_error']:.4f}")
    print(f"High Confidence Accuracy: {results['reliability_metrics']['high_confidence_accuracy']:.4f}")
    
    # Test single prediction
    sample_patient = X_test.iloc[0:1]
    report = evaluator.generate_prediction_report(sample_patient)
    
    print("\n=== SAMPLE PREDICTION REPORT ===")
    print(f"Prediction: {'Cardiovascular Disease' if report['prediction'] == 1 else 'No Disease'}")
    print(f"Probability: {report['probability']:.2%}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Confidence: {report['confidence']}")
    print(f"Interpretation: {report['interpretation']}")
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()