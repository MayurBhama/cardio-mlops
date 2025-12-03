"""
FastAPI Inference Service for Cardio ML Project

- Accepts human-friendly inputs (age, height, BP, etc.)
- Internally recreates all engineered features
- Uses trained model + scaler
- Returns:
    - prediction (0/1)
    - probability
    - risk level
    - human-readable summary
    - detailed advice
    - heart health score
    - cholesterol / glucose / BP interpretation
"""

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

from utils.logger import logger
from utils.exception import CustomException


# =====================================================
# 1. Load Model and Scaler
# =====================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model_artifacts/best_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "model_artifacts/scaler.pkl")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    logger.info("FastAPI: Model and scaler loaded successfully.")

except Exception as e:
    logger.error("FastAPI: Failed to load model or scaler.")
    raise CustomException(e, sys)


FEATURE_COLUMNS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi", "bmi_category", "pulse_pressure",
    "mean_arterial_pressure", "bp_category",
    "age_group", "lifestyle_risk_score",
    "metabolic_risk_score", "combined_risk_score",
]


# =====================================================
# 2. Pydantic Model (User Input)
# =====================================================

class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    height: float = Field(..., ge=120, le=220)
    weight: float = Field(..., ge=30, le=250)
    ap_hi: int = Field(..., ge=60, le=250)
    ap_lo: int = Field(..., ge=30, le=200)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)


# =====================================================
# 3. Helper Functions
# =====================================================

def gender_to_int(gender: str) -> int:
    gender = gender.strip().lower()
    return 2 if gender in ["male", "m"] else 1


def categorize_bmi(bmi: float) -> int:
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    return 3


def categorize_bp(ap_hi: int, ap_lo: int) -> int:
    if ap_hi < 120 and ap_lo < 80:
        return 0
    elif ap_hi < 130 and ap_lo < 80:
        return 1
    elif ap_hi < 140 or ap_lo < 90:
        return 2
    elif ap_hi < 180 or ap_lo < 120:
        return 3
    return 4


def categorize_age(age_years: int) -> int:
    if age_years < 40:
        return 0
    elif age_years < 50:
        return 1
    elif age_years < 60:
        return 2
    return 3


# =====================================================
# Interpretation Helpers
# =====================================================

def interpret_bp_category(bp_cat: int) -> str:
    mapping = {
        0: "Normal blood pressure.",
        1: "Elevated blood pressure. Monitor regularly.",
        2: "Hypertension Stage 1.",
        3: "Hypertension Stage 2.",
        4: "Hypertensive Crisis. Seek immediate medical care."
    }
    return mapping.get(bp_cat, "Unknown BP status.")


def interpret_cholesterol(level: int) -> str:
    mapping = {
        1: "Cholesterol is normal.",
        2: "Cholesterol is above normal. Reduce fried and fatty foods.",
        3: "Very high cholesterol. Medical review recommended."
    }
    return mapping.get(level, "Unknown cholesterol level.")


def interpret_glucose(level: int) -> str:
    mapping = {
        1: "Glucose is normal.",
        2: "Glucose above normal. Possible prediabetes.",
        3: "High glucose. Possible diabetes. Medical tests recommended."
    }
    return mapping.get(level, "Unknown glucose level.")


# =====================================================
# 4. Feature Builder
# =====================================================

def build_feature_row(p: PatientInput) -> pd.DataFrame:
    try:
        age_years = p.age
        gender_int = gender_to_int(p.gender)

        bmi = round(p.weight / ((p.height / 100) ** 2), 2)
        bmi_cat = categorize_bmi(bmi)

        pulse_pressure = p.ap_hi - p.ap_lo
        mean_arterial_pressure = round(p.ap_lo + pulse_pressure / 3, 1)
        bp_category = categorize_bp(p.ap_hi, p.ap_lo)

        age_group = categorize_age(age_years)

        lifestyle_risk_score = p.smoke + p.alco + (1 - p.active)
        metabolic_risk_score = p.cholesterol + p.gluc

        combined_risk_score = round(
            (bp_category / 4 * 30)
            + (bmi_cat / 3 * 20)
            + (lifestyle_risk_score / 3 * 25)
            + (metabolic_risk_score / 6 * 25),
            1
        )

        row = {
            "age_years": age_years,
            "gender": gender_int,
            "height": p.height,
            "weight": p.weight,
            "ap_hi": p.ap_hi,
            "ap_lo": p.ap_lo,
            "cholesterol": p.cholesterol,
            "gluc": p.gluc,
            "smoke": p.smoke,
            "alco": p.alco,
            "active": p.active,
            "bmi": bmi,
            "bmi_category": bmi_cat,
            "pulse_pressure": pulse_pressure,
            "mean_arterial_pressure": mean_arterial_pressure,
            "bp_category": bp_category,
            "age_group": age_group,
            "lifestyle_risk_score": lifestyle_risk_score,
            "metabolic_risk_score": metabolic_risk_score,
            "combined_risk_score": combined_risk_score,
        }

        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    except Exception as e:
        logger.error("Failed to build feature row.")
        raise CustomException(e, sys)


# =====================================================
# 5. Advice Generator
# =====================================================

def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.5:
        return "Moderate"
    elif prob < 0.7:
        return "High"
    return "Very High"


def generate_advice(p: PatientInput, bmi: float, bp_cat: int, prob: float):
    messages = []

    if bmi >= 25:
        messages.append("BMI indicates overweight or obesity. Adopt calorie deficit and walk daily.")
    elif bmi < 18.5:
        messages.append("BMI is below normal. Improve calorie intake and maintain balanced diet.")

    if bp_cat >= 2:
        messages.append("Blood pressure is high. Reduce salt intake and avoid fried foods.")

    if p.cholesterol > 1:
        messages.append("Cholesterol elevated. Reduce oily foods and increase fiber.")

    if p.gluc > 1:
        messages.append("Blood sugar elevated. Reduce sugar and refined carbs.")

    if p.smoke == 1:
        messages.append("Smoking increases heart risk. Consider quitting.")

    if p.alco == 1:
        messages.append("Alcohol increases BP. Reduce intake.")

    if p.active == 0:
        messages.append("Low activity increases risk. Aim for 30 minutes of walking daily.")

    risk_level = get_risk_level(prob)

    if risk_level in ["High", "Very High"]:
        messages.append("Cardiovascular risk is high. Medical consultation recommended.")
    elif risk_level == "Moderate":
        messages.append("Moderate risk. Small lifestyle changes can reduce risk.")
    else:
        messages.append("Risk is low. Maintain healthy habits.")

    summary = f"Estimated {prob*100:.1f}% chance of cardiovascular disease. Category: {risk_level}."

    return {
        "risk_level": risk_level,
        "summary": summary,
        "recommendations": messages
    }


# =====================================================
# 6. FastAPI App and Prediction Endpoint (with Hypotension Handling)
# =====================================================

app = FastAPI(
    title="CardioPredict API",
    description="API for cardiovascular disease prediction",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "CardioPredict API running", "status": "OK"}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        logger.info(f"Prediction request: {patient.dict()}")

        # ---------------------------------------------------
        # 🔥 HYPOTENSION CHECK (model not trained for low BP)
        # ---------------------------------------------------
        if patient.ap_hi < 90 or patient.ap_lo < 60:
            return {
                "warning": "Low blood pressure detected (<90/60). Model was not trained on hypotension cases. Prediction may not be reliable.",
                "prediction": None,
                "probability": None,
                "risk_level": "Not Applicable",
                "summary": "BP below normal physiological range. Clinical evaluation recommended.",
                "bp_interpretation": "Hypotension detected. Seek medical evaluation.",
                "cholesterol_interpretation": interpret_cholesterol(patient.cholesterol),
                "glucose_interpretation": interpret_glucose(patient.gluc),
                "risk_factors": {
                    "bmi": round(patient.weight / ((patient.height / 100) ** 2), 2),
                    "bmi_category": categorize_bmi(round(patient.weight / ((patient.height / 100) ** 2), 2)),
                    "blood_pressure_category": "Hypotension",
                    "lifestyle_score": patient.smoke + patient.alco + (1 - patient.active),
                    "metabolic_score": patient.cholesterol + patient.gluc,
                    "combined_risk_score": None
                },
                "recommendations": [
                    "Your blood pressure is significantly low. This may cause dizziness, weakness, or fainting.",
                    "Increase water intake and avoid sudden standing.",
                    "Consult a doctor before relying on predictive models."
                ]
            }

        # ---------------------------------------------------
        # 🔥 Normal prediction flow
        # ---------------------------------------------------

        features_df = build_feature_row(patient)
        scaled = scaler.transform(features_df)

        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0, 1])

        bmi = float(features_df["bmi"].iloc[0])
        bp_cat = int(features_df["bp_category"].iloc[0])
        combined_score = float(features_df["combined_risk_score"].iloc[0])

        advice = generate_advice(patient, bmi, bp_cat, prob)

        heart_health_score = round(100 - combined_score, 1)
        heart_health_status = (
            "Excellent" if heart_health_score >= 80 else
            "Good" if heart_health_score >= 60 else
            "Moderate" if heart_health_score >= 40 else
            "Poor" if heart_health_score >= 20 else
            "Very Poor"
        )

        response = {
            "prediction": pred,
            "probability": prob,
            "risk_level": advice["risk_level"],
            "summary": advice["summary"],

            "heart_health": {
                "score": heart_health_score,
                "status": heart_health_status,
                "interpretation": (
                    "Heart health is excellent." if heart_health_score >= 80 else
                    "Good heart health but improvements possible." if heart_health_score >= 60 else
                    "Heart health moderate. Needs attention." if heart_health_score >= 40 else
                    "Heart health poor. Significant lifestyle changes needed." if heart_health_score >= 20 else
                    "Very poor heart health. Medical intervention recommended."
                )
            },

            "bp_interpretation": interpret_bp_category(bp_cat),
            "cholesterol_interpretation": interpret_cholesterol(patient.cholesterol),
            "glucose_interpretation": interpret_glucose(patient.gluc),

            "risk_factors": {
                "bmi": bmi,
                "bmi_category": int(features_df["bmi_category"].iloc[0]),
                "blood_pressure_category": bp_cat,
                "lifestyle_score": int(features_df["lifestyle_risk_score"].iloc[0]),
                "metabolic_score": int(features_df["metabolic_risk_score"].iloc[0]),
                "combined_risk_score": combined_score
            },

            "recommendations": advice["recommendations"]
        }

        logger.info("Prediction successful.")
        return response

    except Exception as e:
        logger.error("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(CustomException(e, sys)))
