import pandas as pd
from src.data_ingestion.ingestion import DataIngestion

def run_stage():
    ingestion = DataIngestion("configs/data_config.yaml")
    df = ingestion.load_data("data/raw/cardio_train.csv")
    ingestion.save_data(df, "data/processed/cardio_ingested.csv")

if __name__ == "__main__":
    run_stage()