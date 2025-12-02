"""
Data Ingestion Module
This module handles loading raw data and initial validation
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.config = self._load_config(config_path)
        logger.info("DataIngestion initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            
            # Read CSV file with separator detection
            df = pd.read_csv(file_path, sep=';')
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        logger.info("Data information extracted successfully")
        return info
    
    def validate_data_schema(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate if the data matches expected schema
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if required columns exist
        expected_columns = list(self.config['columns'].values())
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        # Check data types (basic check)
        for col in df.columns:
            if col in expected_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    if col not in ['id']:  # ID can be string sometimes
                        errors.append(f"Column {col} is not numeric")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            errors.append(f"Empty columns found: {empty_cols}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data schema validation passed")
        else:
            logger.warning(f"Data schema validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    def initial_data_quality_check(self, df: pd.DataFrame) -> Dict:
        """
        Perform initial data quality checks
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality check results
        """
        quality_report = {}
        
        # 1. Check missing values
        missing_percent = (df.isnull().sum() / len(df) * 100)
        quality_report['missing_values'] = {
            'columns_with_missing': missing_percent[missing_percent > 0].to_dict(),
            'max_missing_percentage': missing_percent.max(),
            'passed': missing_percent.max() < self.config['quality_thresholds']['max_missing_percentage']
        }
        
        # 2. Check duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percent = (duplicate_count / len(df)) * 100
        quality_report['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': duplicate_percent,
            'passed': duplicate_percent < (self.config['quality_thresholds']['duplicate_threshold'] * 100)
        }
        
        # 3. Check categorical values
        categorical_checks = {}
        for col, valid_values in self.config['categorical_values'].items():
            if col in df.columns:
                unique_values = df[col].unique()
                invalid_values = set(unique_values) - set(valid_values) - {np.nan}
                categorical_checks[col] = {
                    'unique_values': unique_values.tolist(),
                    'expected_values': valid_values,
                    'invalid_values': list(invalid_values),
                    'passed': len(invalid_values) == 0
                }
        
        quality_report['categorical_validation'] = categorical_checks
        
        # 4. Overall quality score
        passed_checks = sum([
            quality_report['missing_values']['passed'],
            quality_report['duplicates']['passed'],
            sum([v['passed'] for v in categorical_checks.values()])
        ])
        total_checks = 2 + len(categorical_checks)
        quality_report['quality_score'] = (passed_checks / total_checks) * 100
        
        logger.info(f"Data quality score: {quality_report['quality_score']:.2f}%")
        
        return quality_report
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            output_path: Path where to save the file
        """
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise


def main():
    """
    Main function to demonstrate data ingestion
    """
    # Initialize ingestion
    ingestion = DataIngestion()
    
    # Load data
    df = ingestion.load_data("data/raw/cardio_train.csv")
    
    # Get data info
    info = ingestion.get_data_info(df)
    print("\n=== DATA INFORMATION ===")
    print(f"Shape: {info['shape']}")
    print(f"Duplicates: {info['duplicates']} ({info['duplicate_percentage']:.2f}%)")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    
    # Validate schema
    is_valid, errors = ingestion.validate_data_schema(df)
    print(f"\n=== SCHEMA VALIDATION ===")
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Quality check
    quality_report = ingestion.initial_data_quality_check(df)
    print(f"\n=== QUALITY CHECK ===")
    print(f"Quality Score: {quality_report['quality_score']:.2f}%")
    
    return df


if __name__ == "__main__":
    df = main()