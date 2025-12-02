import pandas as pd
from src.data_processing.cleaning import DataCleaner

def run_stage():
    df = pd.read_csv("data/processed/cardio_ingested.csv")
    cleaner = DataCleaner("configs/data_config.yaml")
    df_clean = cleaner.clean_data(df)
    df_clean.to_csv("data/processed/cardio_cleaned.csv", index=False)

if __name__ == "__main__":
    run_stage()