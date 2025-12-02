"""
Model Training Module
This module trains multiple models and selects the best one
Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training class with MLflow tracking
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize ModelTrainer
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        logger.info("ModelTrainer initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_data(self, df: pd.DataFrame, 
                     feature_columns: List[str]) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame with features
            feature_columns: List of feature column names to use
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        """
        logger.info("Preparing data for training...")
        
        # Separate features and target
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        y = df['cardio'].copy()
        
        logger.info(f"Using {len(available_features)} features for training")
        
        # Split data: 70% train, 15% validation, 15% test
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['validation_size']
        random_state = self.config['preprocessing']['random_state']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Train set size: {X_train.shape}")
        logger.info(f"Validation set size: {X_val.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=available_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features)
        
        logger.info("Data preparation completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    
    def train_all_models(self, X_train, X_val, y_train, y_val):
        """
        Train all configured models
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
        """
        logger.info("=" * 60)
        logger.info("Starting model training...")
        logger.info("=" * 60)
        
        # 1. Logistic Regression
        self._train_logistic_regression(X_train, X_val, y_train, y_val)
        
        # 2. Random Forest
        self._train_random_forest(X_train, X_val, y_train, y_val)
        
        # 3. Gradient Boosting
        self._train_gradient_boosting(X_train, X_val, y_train, y_val)
        
        # 4. XGBoost
        self._train_xgboost(X_train, X_val, y_train, y_val)
        
        # Select best model
        self._select_best_model()
        
        logger.info("=" * 60)
        logger.info("Model training completed")
        logger.info("=" * 60)
    
    def _train_logistic_regression(self, X_train, X_val, y_train, y_val):
        """Train Logistic Regression model"""
        logger.info("\nTraining Logistic Regression...")
        
        with mlflow.start_run(run_name="Logistic_Regression"):
            # Get parameters
            params = self.config['models']['logistic_regression']
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
            
            # Log metrics
            mlflow.log_metrics(results['val_metrics'])
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Logistic_Regression'] = model
            self.results['Logistic_Regression'] = results
            
            logger.info(f"Logistic Regression - Val Accuracy: {results['val_metrics']['accuracy']:.4f}")
            logger.info(f"Logistic Regression - Val ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")
    
    def _train_random_forest(self, X_train, X_val, y_train, y_val):
        """Train Random Forest model"""
        logger.info("\nTraining Random Forest...")
        
        with mlflow.start_run(run_name="Random_Forest"):
            # Get parameters
            params = self.config['models']['random_forest']
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
            
            # Log metrics
            mlflow.log_metrics(results['val_metrics'])
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Random_Forest'] = model
            self.results['Random_Forest'] = results
            
            logger.info(f"Random Forest - Val Accuracy: {results['val_metrics']['accuracy']:.4f}")
            logger.info(f"Random Forest - Val ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")
    
    def _train_gradient_boosting(self, X_train, X_val, y_train, y_val):
        """Train Gradient Boosting model"""
        logger.info("\nTraining Gradient Boosting...")
        
        with mlflow.start_run(run_name="Gradient_Boosting"):
            # Get parameters
            params = self.config['models']['gradient_boosting']
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
            
            # Log metrics
            mlflow.log_metrics(results['val_metrics'])
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Gradient_Boosting'] = model
            self.results['Gradient_Boosting'] = results
            
            logger.info(f"Gradient Boosting - Val Accuracy: {results['val_metrics']['accuracy']:.4f}")
            logger.info(f"Gradient Boosting - Val ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")
    
    def _train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost model"""
        logger.info("\nTraining XGBoost...")
        
        with mlflow.start_run(run_name="XGBoost"):
            # Get parameters
            params = self.config['models']['xgboost']
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
            
            # Evaluate
            results = self._evaluate_model(model, X_train, X_val, y_train, y_val)
            
            # Log metrics
            mlflow.log_metrics(results['val_metrics'])
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['XGBoost'] = model
            self.results['XGBoost'] = results
            
            logger.info(f"XGBoost - Val Accuracy: {results['val_metrics']['accuracy']:.4f}")
            logger.info(f"XGBoost - Val ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")
    
    def _evaluate_model(self, model, X_train, X_val, y_train, y_val) -> Dict:
        """
        Evaluate model on training and validation sets
        
        Args:
            model: Trained model
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        results = {
            'train_metrics': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
                'f1_score': f1_score(y_train, y_train_pred),
                'roc_auc': roc_auc_score(y_train, y_train_proba)
            },
            'val_metrics': {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1_score': f1_score(y_val, y_val_pred),
                'roc_auc': roc_auc_score(y_val, y_val_proba)
            },
            'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist(),
            'classification_report': classification_report(y_val, y_val_pred)
        }
        
        return results
    
    def _select_best_model(self):
        """Select the best model based on validation ROC-AUC"""
        best_score = 0
        best_name = None
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        for name, results in self.results.items():
            val_roc_auc = results['val_metrics']['roc_auc']
            val_accuracy = results['val_metrics']['accuracy']
            
            logger.info(f"{name}:")
            logger.info(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
            logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
            logger.info(f"  Validation F1-Score: {results['val_metrics']['f1_score']:.4f}")
            
            if val_roc_auc > best_score:
                best_score = val_roc_auc
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info("=" * 60)
        logger.info(f"BEST MODEL: {best_name} (ROC-AUC: {best_score:.4f})")
        logger.info("=" * 60)
    
    def evaluate_on_test(self, X_test, y_test) -> Dict:
        """
        Evaluate best model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with test metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"TESTING BEST MODEL: {self.best_model_name}")
        logger.info("=" * 60)
        
        # Predictions
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        test_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_results['precision']:.4f}")
        logger.info(f"Test Recall: {test_results['recall']:.4f}")
        logger.info(f"Test F1-Score: {test_results['f1_score']:.4f}")
        logger.info(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{test_results['confusion_matrix']}")
        
        return test_results
    
    def save_best_model(self, model_path: str, scaler_path: str):
        """
        Save best model and scaler
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        # Create directories
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scalers.get('scaler'), scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")


def main():
    """
    Main function to demonstrate model training
    """
    # Load featured data
    df = pd.read_csv("data/processed/cardio_featured.csv")
    
    # Define features to use
    feature_columns = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'bmi', 'bmi_category', 'pulse_pressure', 
        'mean_arterial_pressure', 'bp_category',
        'age_group', 'lifestyle_risk_score', 
        'metabolic_risk_score', 'combined_risk_score'
    ]
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = trainer.prepare_data(df, feature_columns)
    trainer.scalers['scaler'] = scaler
    
    # Train all models
    trainer.train_all_models(X_train, X_val, y_train, y_val)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test(X_test, y_test)
    
    # Save best model
    trainer.save_best_model(
        "models/trained_models/best_model.pkl",
        "models/trained_models/scaler.pkl"
    )
    
    print("\n Training completed successfully!")
    return trainer


if __name__ == "__main__":
    trainer = main()