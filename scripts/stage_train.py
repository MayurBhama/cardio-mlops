import pandas as pd
from src.model_training.train import ModelTrainer

def run_stage():
    df = pd.read_csv("data/processed/cardio_featured.csv")

    feature_columns = [
        'age_years','gender','height','weight','ap_hi','ap_lo','cholesterol',
        'gluc','smoke','alco','active','bmi','bmi_category','pulse_pressure',
        'mean_arterial_pressure','bp_category','age_group','lifestyle_risk_score',
        'metabolic_risk_score','combined_risk_score'
    ]

    trainer = ModelTrainer("configs/model_config.yaml")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = trainer.prepare_data(df, feature_columns)
    trainer.scalers['scaler'] = scaler

    trainer.train_all_models(X_train, X_val, y_train, y_val)

    trainer.save_best_model(
        "models/trained_models/best_model.pkl",
        "models/trained_models/scaler.pkl"
    )

if __name__ == "__main__":
    run_stage()