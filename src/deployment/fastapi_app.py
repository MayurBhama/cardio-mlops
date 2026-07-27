"""
FastAPI Inference Service for Cardio ML Project
Updated for Render Deployment + XGBoost Fix + Correct Paths

- Loads model safely using absolute paths
- Recreates ALL engineered features exactly like training
- Handles hypotension safely
- Returns prediction + risk + detailed interpretation
"""

import pandas as pd
import joblib
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import xgboost  # IMPORTANT: required for joblib to load XGBClassifier

from utils.logger import logger
from utils.exception import CustomException


# =====================================================
# RESOLVE ROOT DIRECTORY (RENDER SAFE) & LOAD MODEL
# =====================================================

try:
    ROOT_DIR = Path(__file__).resolve().parents[2]

    MODEL_PATH = ROOT_DIR / "models" / "trained_models" / "best_model.pkl"
    SCALER_PATH = ROOT_DIR / "models" / "trained_models" / "scaler.pkl"

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"FastAPI: Loaded model from {MODEL_PATH}")
    logger.info(f"FastAPI: Loaded scaler from {SCALER_PATH}")

except Exception as e:
    import traceback
    from datetime import datetime
    import os
    
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"error_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    with open(log_path, "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
        
    logger.error(f"FastAPI Initialization Error. Traceback written to {log_filename}")
    raise e


# =====================================================
# FEATURE COLUMNS — MUST MATCH TRAINING EXACTLY
# =====================================================

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
# INPUT MODEL
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
# HELPER FUNCTIONS
# =====================================================

def gender_to_int(g: str):
    g = g.lower().strip()
    return 2 if g in ["male", "m"] else 1


def categorize_bmi(b):
    if b < 18.5: return 0
    if b < 25: return 1
    if b < 30: return 2
    return 3


def categorize_bp(sys_bp, dia_bp):
    # NHS Blood Pressure Categories
    # Normal: 90/60 to 120/80
    # High Normal: 120/80 to 140/90
    # Stage 1 (Hypertension): 140/90 to 160/100
    # Stage 2: 160/100 to 180/120
    # Stage 3 (Severe/Crisis): >180 or >120
    if sys_bp <= 120 and dia_bp <= 80: 
        return 0  # Normal
    if sys_bp < 140 and dia_bp < 90: 
        return 1  # High Normal
    if sys_bp < 160 and dia_bp < 100: 
        return 2  # Stage 1 Hypertension
    if sys_bp < 180 and dia_bp < 120: 
        return 3  # Stage 2 Hypertension
    return 4  # Stage 3 Severe Hypertension


def categorize_age(a):
    if a < 40: return 0
    if a < 50: return 1
    if a < 60: return 2
    return 3


# =====================================================
# INTERPRETATION HELPERS
# =====================================================

def interpret_bp_category(bp_cat):
    # NHS Blood Pressure terminology
    mapping = {
        0: "Normal blood pressure.",
        1: "High normal blood pressure.",
        2: "High blood pressure (Stage 1).",
        3: "High blood pressure (Stage 2).",
        4: "Severe hypertension. Seek immediate care."
    }
    return mapping.get(bp_cat, "Unknown BP status.")


def interpret_cholesterol(level):
    mapping = {
        1: "Cholesterol is normal.",
        2: "Cholesterol is above normal. Reduce oily foods.",
        3: "Very high cholesterol. Medical review recommended."
    }
    return mapping.get(level, "Unknown cholesterol level.")


def interpret_glucose(level):
    mapping = {
        1: "Glucose levels within normal range.",
        2: "Glucose above normal. Possible impaired glucose regulation.",
        3: "Elevated glucose levels. Consider glucose tolerance testing."
    }
    return mapping.get(level, "Unknown glucose level.")


# =====================================================
# BUILD FEATURE ROW (MATCH TRAINING)
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
            + (metabolic_risk_score / 6 * 25), 1
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
        raise CustomException(e, sys)


# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(prob):
    if prob < 0.3: return "Low"
    if prob < 0.5: return "Moderate"
    if prob < 0.7: return "High"
    return "Very High"


# =====================================================
# ADVICE GENERATOR
# =====================================================

def generate_advice(p, bmi, bp_cat, prob):
    msgs = []

    if bmi >= 25:
        msgs.append("BMI is high. Begin calorie deficit + daily walking.")
    elif bmi < 18.5:
        msgs.append("BMI low. Increase healthy calorie intake.")

    if bp_cat >= 2:
        msgs.append("High blood pressure detected. Reduce salt and fried foods.")

    if p.cholesterol > 1:
        msgs.append("Cholesterol elevated. Increase fiber, avoid fried foods.")

    if p.gluc > 1:
        msgs.append("High sugar levels. Reduce sugar + refined carbs.")

    if p.smoke:
        msgs.append("Smoking increases cardiac risk. Consider quitting.")

    if p.alco:
        msgs.append("Alcohol raises BP. Reduce intake.")

    if not p.active:
        msgs.append("Increase activity — 30 minutes walking daily.")

    risk = get_risk_level(prob)

    if risk in ["High", "Very High"]:
        msgs.append("High cardiovascular risk. Medical consultation recommended.")
    elif risk == "Moderate":
        msgs.append("Moderate risk. Lifestyle changes recommended.")
    else:
        msgs.append("Low risk. Maintain healthy habits.")

    return {
        "risk_level": risk,
        "summary": f"Estimated risk: {prob*100:.1f}% ({risk}).",
        "recommendations": msgs
    }


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="CardioPredict API",
    version="1.0.0",
    description="Heart disease prediction API"
)


@app.get("/")
def root():
    return {"message": "CardioPredict API running", "status": "OK"}


# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
def predict(patient: PatientInput):

    # ------------------ Hypotension handling ------------------
    if patient.ap_hi < 90 or patient.ap_lo < 60:
        return {
            "warning": "Hypotension (<90/60). Model not trained for low BP.",
            "prediction": None,
            "risk_level": "Not Applicable",
            "summary": "BP too low for reliable prediction."
        }

    try:
        logger.info(f"Prediction request: {patient.dict()}")

        df = build_feature_row(patient)
        scaled = scaler.transform(df)

        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1])

        bmi = float(df["bmi"].iloc[0])
        bp_cat = int(df["bp_category"].iloc[0])
        combined_score = float(df["combined_risk_score"].iloc[0])

        advice = generate_advice(patient, bmi, bp_cat, prob)

        # Heart Health Score
        heart_health_score = round(100 - combined_score, 1)

        heart_status = (
            "Excellent" if heart_health_score >= 80 else
            "Good" if heart_health_score >= 60 else
            "Moderate" if heart_health_score >= 40 else
            "Poor" if heart_health_score >= 20 else
            "Very Poor"
        )

        return {
            "prediction": pred,
            "probability": prob,
            "risk_level": advice["risk_level"],
            "summary": advice["summary"],

            "heart_health": {
                "score": heart_health_score,
                "status": heart_status
            },

            "bp_interpretation": interpret_bp_category(bp_cat),
            "cholesterol_interpretation": interpret_cholesterol(patient.cholesterol),
            "glucose_interpretation": interpret_glucose(patient.gluc),

            "risk_factors": {
                "bmi": bmi,
                "bmi_category": int(df["bmi_category"].iloc[0]),
                "blood_pressure_category": bp_cat,
                "combined_risk_score": combined_score
            },

            "recommendations": advice["recommendations"]
        }

    except Exception as e:
        logger.error("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(CustomException(e, sys)))
