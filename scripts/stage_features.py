import pandas as pd
from src.feature_engineering.features import FeatureEngineer

def run_stage():
    df = pd.read_csv("data/processed/cardio_cleaned.csv")
    fe = FeatureEngineer("configs/model_config.yaml")
    df_feat = fe.create_all_features(df)
    df_feat.to_csv("data/processed/cardio_featured.csv", index=False)

if __name__ == "__main__":
    run_stage()