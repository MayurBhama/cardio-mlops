# debug_evaluation_predictions.py
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 1) Load featured data
df = pd.read_csv("data/processed/cardio_featured.csv")

feature_columns = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi", "bmi_category", "pulse_pressure",
    "mean_arterial_pressure", "bp_category",
    "age_group", "lifestyle_risk_score",
    "metabolic_risk_score", "combined_risk_score",
]

X = df[feature_columns]
y = df["cardio"]

# this reproduces what your current evaluate.py does (last 20% slice)
test_size = int(len(df) * 0.2)
X_test = X.iloc[-test_size:]
y_test = y.iloc[-test_size:]

# 2) Load model + scaler
model = joblib.load("models/trained_models/best_model.pkl")
scaler = joblib.load("models/trained_models/scaler.pkl")

# 3) Scale + predict
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("y_test distribution:", y_test.value_counts().to_dict())
print("y_pred distribution:", {v: int((y_pred == v).sum()) for v in [0, 1]})

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nMin/Max probabilities:", float(y_proba.min()), float(y_proba.max()))