"""
Feature Engineering Module
This module creates new features from existing ones to improve model performance
Focus areas:
1. BMI and body composition features
2. Blood pressure features
3. Risk score features
4. Age-related features
5. Interaction features
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for cardiovascular disease prediction
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize FeatureEngineer
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self._load_config(config_path)
        self.feature_config = self.config['feature_engineering']
        self.new_features = []
        logger.info("FeatureEngineer initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Starting feature engineering...")
        df_feat = df.copy()
        
        # 1. BMI and body composition features
        if self.feature_config['create_bmi']:
            df_feat = self._create_bmi_features(df_feat)
        
        # 2. Blood pressure features
        if self.feature_config['create_bp_features']:
            df_feat = self._create_bp_features(df_feat)
        
        # 3. Age-related features
        if self.feature_config['create_age_groups']:
            df_feat = self._create_age_features(df_feat)
        
        # 4. Risk score features
        if self.feature_config['create_risk_scores']:
            df_feat = self._create_risk_scores(df_feat)
        
        # 5. Interaction features
        df_feat = self._create_interaction_features(df_feat)
        
        logger.info(f"Feature engineering completed. Created {len(self.new_features)} new features")
        logger.info(f"New features: {self.new_features}")
        
        return df_feat
    
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create BMI and related body composition features
        
        BMI = weight (kg) / (height (m))^2
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with BMI features
        """
        logger.info("Creating BMI features...")
        
        # Calculate BMI
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['bmi'] = df['bmi'].round(2)
        self.new_features.append('bmi')
        
        # BMI categories (WHO classification)
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 0  # Underweight
            elif bmi < 25:
                return 1  # Normal
            elif bmi < 30:
                return 2  # Overweight
            else:
                return 3  # Obese
        
        df['bmi_category'] = df['bmi'].apply(categorize_bmi)
        self.new_features.append('bmi_category')
        
        # Body surface area (BSA) - Mosteller formula
        df['bsa'] = np.sqrt((df['height'] * df['weight']) / 3600)
        df['bsa'] = df['bsa'].round(2)
        self.new_features.append('bsa')
        
        # Weight-to-height ratio
        df['weight_height_ratio'] = (df['weight'] / df['height'] * 100).round(2)
        self.new_features.append('weight_height_ratio')
        
        logger.info("BMI features created successfully")
        return df
    
    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create blood pressure related features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with BP features
        """
        logger.info("Creating blood pressure features...")
        
        # Pulse pressure (difference between systolic and diastolic)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        self.new_features.append('pulse_pressure')
        
        # Mean arterial pressure (MAP)
        # MAP = DBP + (1/3)(SBP - DBP)
        df['mean_arterial_pressure'] = df['ap_lo'] + (df['pulse_pressure'] / 3)
        df['mean_arterial_pressure'] = df['mean_arterial_pressure'].round(1)
        self.new_features.append('mean_arterial_pressure')
        
        # Blood pressure category (AHA guidelines)
        def categorize_bp(row):
            systolic = row['ap_hi']
            diastolic = row['ap_lo']
            
            if systolic < 120 and diastolic < 80:
                return 0  # Normal
            elif systolic < 130 and diastolic < 80:
                return 1  # Elevated
            elif systolic < 140 or diastolic < 90:
                return 2  # Hypertension Stage 1
            elif systolic < 180 or diastolic < 120:
                return 3  # Hypertension Stage 2
            else:
                return 4  # Hypertensive Crisis
        
        df['bp_category'] = df.apply(categorize_bp, axis=1)
        self.new_features.append('bp_category')
        
        # BP ratio (systolic/diastolic)
        df['bp_ratio'] = (df['ap_hi'] / df['ap_lo']).round(2)
        self.new_features.append('bp_ratio')
        
        logger.info("Blood pressure features created successfully")
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age features
        """
        logger.info("Creating age features...")
        
        # Age groups
        def categorize_age(age):
            if age < 40:
                return 0  # Young
            elif age < 50:
                return 1  # Middle-aged
            elif age < 60:
                return 2  # Senior
            else:
                return 3  # Elderly
        
        df['age_group'] = df['age_years'].apply(categorize_age)
        self.new_features.append('age_group')
        
        # Age squared (for non-linear relationships)
        df['age_squared'] = df['age_years'] ** 2
        self.new_features.append('age_squared')
        
        logger.info("Age features created successfully")
        return df
    
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk score features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk scores
        """
        logger.info("Creating risk score features...")
        
        # Lifestyle risk score (0-3 based on smoking, alcohol, inactivity)
        df['lifestyle_risk_score'] = (
            df['smoke'] + 
            df['alco'] + 
            (1 - df['active'])  # Inactivity is a risk
        )
        self.new_features.append('lifestyle_risk_score')
        
        # Metabolic risk score (cholesterol + glucose levels)
        # Higher values indicate more risk
        df['metabolic_risk_score'] = df['cholesterol'] + df['gluc']
        self.new_features.append('metabolic_risk_score')
        
        # Combined risk score (normalized 0-100)
        # Includes: BP category, BMI category, lifestyle, metabolic
        df['combined_risk_score'] = (
            (df['bp_category'] / 4 * 30) +  # BP contributes 30%
            (df['bmi_category'] / 3 * 20) +  # BMI contributes 20%
            (df['lifestyle_risk_score'] / 3 * 25) +  # Lifestyle contributes 25%
            (df['metabolic_risk_score'] / 6 * 25)  # Metabolic contributes 25%
        ).round(1)
        self.new_features.append('combined_risk_score')
        
        # Age-adjusted risk (risk increases with age)
        df['age_adjusted_risk'] = (
            df['combined_risk_score'] * (1 + (df['age_years'] - 30) / 100)
        ).round(1)
        self.new_features.append('age_adjusted_risk')
        
        logger.info("Risk score features created successfully")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Age × BMI interaction
        df['age_bmi_interaction'] = (df['age_years'] * df['bmi']).round(2)
        self.new_features.append('age_bmi_interaction')
        
        # Age × Blood Pressure interaction
        df['age_bp_interaction'] = (df['age_years'] * df['mean_arterial_pressure']).round(2)
        self.new_features.append('age_bp_interaction')
        
        # Gender × BMI interaction (different risk profiles)
        df['gender_bmi_interaction'] = df['gender'] * df['bmi_category']
        self.new_features.append('gender_bmi_interaction')
        
        # Smoking × Cholesterol interaction
        df['smoke_chol_interaction'] = df['smoke'] * df['cholesterol']
        self.new_features.append('smoke_chol_interaction')
        
        logger.info("Interaction features created successfully")
        return df
    
    def get_feature_importance_order(self) -> List[str]:
        """
        Get recommended order of features by expected importance
        
        Returns:
            List of feature names
        """
        # Based on medical literature and domain knowledge
        important_features = [
            'age_years',
            'bp_category',
            'mean_arterial_pressure',
            'ap_hi',
            'ap_lo',
            'bmi',
            'bmi_category',
            'cholesterol',
            'gluc',
            'combined_risk_score',
            'age_adjusted_risk',
            'lifestyle_risk_score',
            'metabolic_risk_score',
            'pulse_pressure',
            'gender',
            'smoke',
            'age_bmi_interaction',
            'age_bp_interaction',
            'weight',
            'height',
            'active',
            'alco'
        ]
        return important_features
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all engineered features
        
        Returns:
            Dictionary with feature descriptions
        """
        descriptions = {
            'bmi': 'Body Mass Index - weight/height²',
            'bmi_category': 'BMI category (0:Underweight, 1:Normal, 2:Overweight, 3:Obese)',
            'bsa': 'Body Surface Area (Mosteller formula)',
            'weight_height_ratio': 'Weight to height ratio',
            'pulse_pressure': 'Difference between systolic and diastolic BP',
            'mean_arterial_pressure': 'Average arterial pressure during one cardiac cycle',
            'bp_category': 'BP category (0:Normal, 1:Elevated, 2:Stage1, 3:Stage2, 4:Crisis)',
            'bp_ratio': 'Systolic to diastolic BP ratio',
            'age_group': 'Age category (0:Young<40, 1:Middle 40-50, 2:Senior 50-60, 3:Elderly 60+)',
            'age_squared': 'Age squared for non-linear relationships',
            'lifestyle_risk_score': 'Combined lifestyle risk (smoking + alcohol + inactivity)',
            'metabolic_risk_score': 'Combined metabolic risk (cholesterol + glucose)',
            'combined_risk_score': 'Overall risk score (0-100)',
            'age_adjusted_risk': 'Risk score adjusted for age',
            'age_bmi_interaction': 'Interaction between age and BMI',
            'age_bp_interaction': 'Interaction between age and blood pressure',
            'gender_bmi_interaction': 'Interaction between gender and BMI',
            'smoke_chol_interaction': 'Interaction between smoking and cholesterol'
        }
        return descriptions


def main():
    """
    Main function to demonstrate feature engineering
    """
    # Load cleaned data
    df = pd.read_csv("data/processed/cardio_cleaned.csv")
    print(f"Initial data shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create features
    df_featured = engineer.create_all_features(df)
    print(f"\nData shape after feature engineering: {df_featured.shape}")
    print(f"New features created: {engineer.new_features}")
    
    # Display sample of new features
    print("\n=== SAMPLE OF NEW FEATURES ===")
    new_feature_cols = ['age_years', 'bmi', 'bmi_category', 'mean_arterial_pressure', 
                        'bp_category', 'combined_risk_score', 'cardio']
    print(df_featured[new_feature_cols].head(10))
    
    # Get feature descriptions
    descriptions = engineer.get_feature_descriptions()
    print("\n=== FEATURE DESCRIPTIONS ===")
    for feature, desc in list(descriptions.items())[:5]:
        print(f"{feature}: {desc}")
    
    # Save featured data
    df_featured.to_csv("data/processed/cardio_featured.csv", index=False)
    print("\nFeatured data saved to data/processed/cardio_featured.csv")
    
    return df_featured


if __name__ == "__main__":
    df_featured = main()