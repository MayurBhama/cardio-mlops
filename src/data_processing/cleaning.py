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
import logging
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.config = self._load_config(config_path)
        self.cleaning_report = {
            'initial_shape': None,
            'final_shape': None,
            'removed_rows': 0,
            'outliers_removed': {},
            'invalid_values_handled': {},
            'duplicates_removed': 0
        }
        logger.info("DataCleaner initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        self.cleaning_report['initial_shape'] = df.shape
        
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 2: Convert age from days to years
        df_clean = self._convert_age_to_years(df_clean)
        
        # Step 3: Handle invalid categorical values
        df_clean = self._handle_invalid_categorical(df_clean)
        
        # Step 4: Remove physiological outliers
        df_clean = self._remove_outliers(df_clean)
        
        # Step 5: Handle blood pressure anomalies
        df_clean = self._handle_blood_pressure_anomalies(df_clean)
        
        # Step 6: Handle missing values (if any)
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 7: Ensure data types
        df_clean = self._ensure_data_types(df_clean)
        
        self.cleaning_report['final_shape'] = df_clean.shape
        self.cleaning_report['removed_rows'] = self.cleaning_report['initial_shape'][0] - df_clean.shape[0]
        
        logger.info(f"Data cleaning completed. Removed {self.cleaning_report['removed_rows']} rows")
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_len = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_len - len(df_clean)
        
        self.cleaning_report['duplicates_removed'] = removed
        logger.info(f"Removed {removed} duplicate rows")
        
        return df_clean
    
    def _convert_age_to_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert age from days to years and create age_years column
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age in years
        """
        df['age_years'] = (df['age'] / 365.25).round().astype(int)
        logger.info("Converted age from days to years")
        return df
    
    def _handle_invalid_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle invalid values in categorical columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid categorical values
        """
        for col, valid_values in self.config['categorical_values'].items():
            if col in df.columns:
                initial_len = len(df)
                df = df[df[col].isin(valid_values)]
                removed = initial_len - len(df)
                
                if removed > 0:
                    self.cleaning_report['invalid_values_handled'][col] = removed
                    logger.info(f"Removed {removed} rows with invalid {col} values")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers based on valid ranges defined in config
        Uses domain knowledge for cardiovascular data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without outliers
        """
        valid_ranges = self.config['valid_ranges']
        initial_len = len(df)
        
        # Age outliers (using years now)
        if 'age_years' in df.columns:
            age_min, age_max = valid_ranges['age_years']
            mask = (df['age_years'] >= age_min) & (df['age_years'] <= age_max)
            removed = len(df) - mask.sum()
            df = df[mask]
            if removed > 0:
                self.cleaning_report['outliers_removed']['age'] = removed
                logger.info(f"Removed {removed} age outliers")
        
        # Height outliers
        if 'height' in df.columns:
            height_min, height_max = valid_ranges['height']
            mask = (df['height'] >= height_min) & (df['height'] <= height_max)
            removed = len(df) - mask.sum()
            df = df[mask]
            if removed > 0:
                self.cleaning_report['outliers_removed']['height'] = removed
                logger.info(f"Removed {removed} height outliers")
        
        # Weight outliers
        if 'weight' in df.columns:
            weight_min, weight_max = valid_ranges['weight']
            mask = (df['weight'] >= weight_min) & (df['weight'] <= weight_max)
            removed = len(df) - mask.sum()
            df = df[mask]
            if removed > 0:
                self.cleaning_report['outliers_removed']['weight'] = removed
                logger.info(f"Removed {removed} weight outliers")
        
        # Blood pressure outliers
        if 'ap_hi' in df.columns:
            ap_hi_min, ap_hi_max = valid_ranges['ap_hi']
            mask = (df['ap_hi'] >= ap_hi_min) & (df['ap_hi'] <= ap_hi_max)
            removed = len(df) - mask.sum()
            df = df[mask]
            if removed > 0:
                self.cleaning_report['outliers_removed']['systolic_bp'] = removed
                logger.info(f"Removed {removed} systolic BP outliers")
        
        if 'ap_lo' in df.columns:
            ap_lo_min, ap_lo_max = valid_ranges['ap_lo']
            mask = (df['ap_lo'] >= ap_lo_min) & (df['ap_lo'] <= ap_lo_max)
            removed = len(df) - mask.sum()
            df = df[mask]
            if removed > 0:
                self.cleaning_report['outliers_removed']['diastolic_bp'] = removed
                logger.info(f"Removed {removed} diastolic BP outliers")
        
        total_removed = initial_len - len(df)
        logger.info(f"Total outliers removed: {total_removed}")
        
        return df
    
    def _handle_blood_pressure_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle cases where diastolic BP >= systolic BP (medical impossibility)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without BP anomalies
        """
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            initial_len = len(df)
            
            # Remove rows where diastolic >= systolic
            df = df[df['ap_lo'] < df['ap_hi']]
            
            removed = initial_len - len(df)
            if removed > 0:
                self.cleaning_report['outliers_removed']['bp_anomalies'] = removed
                logger.info(f"Removed {removed} rows with BP anomalies (diastolic >= systolic)")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values if any exist
        Strategy: For this dataset, we remove rows with missing values
        as the dataset is large enough
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without missing values
        """
        initial_len = len(df)
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            df = df.dropna()
            removed = initial_len - len(df)
            logger.info(f"Removed {removed} rows with missing values")
        else:
            logger.info("No missing values found")
        
        return df
    
    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns have correct data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with correct data types
        """
        # Integer columns
        int_columns = ['id', 'age', 'gender', 'height', 'ap_hi', 'ap_lo',
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'age_years']
        
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Float columns
        if 'weight' in df.columns:
            df['weight'] = df['weight'].astype(float)
        
        logger.info("Data types ensured")
        return df
    
    def get_cleaning_report(self) -> Dict:
        """
        Get detailed cleaning report
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.cleaning_report
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive statistics of cleaned data
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'numeric_stats': df.describe().to_dict(),
            'categorical_distribution': {},
            'target_distribution': df['cardio'].value_counts().to_dict() if 'cardio' in df.columns else None,
            'gender_distribution': df['gender'].value_counts().to_dict() if 'gender' in df.columns else None
        }
        
        # Categorical features distribution
        categorical_cols = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
        for col in categorical_cols:
            if col in df.columns:
                stats_dict['categorical_distribution'][col] = df[col].value_counts().to_dict()
        
        return stats_dict


def main():
    """
    Main function to demonstrate data cleaning
    """
    # Load data (assuming it's been ingested)
    df = pd.read_csv("data/raw/cardio_train.csv", sep=';')
    print(f"Initial data shape: {df.shape}")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean data
    df_clean = cleaner.clean_data(df)
    print(f"\nCleaned data shape: {df_clean.shape}")
    
    # Get cleaning report
    report = cleaner.get_cleaning_report()
    print("\n=== CLEANING REPORT ===")
    print(f"Initial shape: {report['initial_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"Total removed rows: {report['removed_rows']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Outliers removed: {report['outliers_removed']}")
    
    # Get statistics
    stats = cleaner.get_data_statistics(df_clean)
    print(f"\n=== TARGET DISTRIBUTION ===")
    print(f"Class distribution: {stats['target_distribution']}")
    
    # Save cleaned data
    df_clean.to_csv("data/processed/cardio_cleaned.csv", index=False)
    print("\nCleaned data saved to data/processed/cardio_cleaned.csv")
    
    return df_clean


if __name__ == "__main__":
    df_clean = main()