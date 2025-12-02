"""
FastAPI Deployment Module
This module serves the ML model via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CardioPredict API",
    description="API for Cardiovascular Disease Prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load("models/trained_models/best_model.pkl")
    scaler = joblib.load("models/trained_models/scaler.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None


class PatientInput(BaseModel):
    """
    Input schema for patient data
    All inputs are user-friendly and in standard units
    """
    age: int = Field(..., ge=30, le=70, description="Age in years (30-70)")
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    height: float = Field(..., ge=140, le=210, description="Height in cm (140-210)")
    weight: float = Field(..., ge=40, le=200, description="Weight in kg (40-200)")
    systolic_bp: int = Field(..., ge=90, le=200, description="Systolic Blood Pressure (90-200 mmHg)")
    diastolic_bp: int = Field(..., ge=60, le=130, description="Diastolic Blood Pressure (60-130 mmHg)")
    cholesterol: str = Field(..., description="Cholesterol level: 'normal', 'above_normal', 'well_above_normal'")
    glucose: str = Field(..., description="Glucose level: 'normal', 'above_normal', 'well_above_normal'")
    smoking: bool = Field(..., description="Does the patient smoke?")
    alcohol: bool = Field(..., description="Does the patient consume alcohol?")
    physical_activity: bool = Field(..., description="Is the patient physically active?")
    
    @validator('gender')
    def validate_gender(cls, v):
        v = v.lower()
        if v not in ['male', 'female']:
            raise ValueError('Gender must be either "male" or "female"')
        return v
    
    @validator('cholesterol', 'glucose')
    def validate_levels(cls, v):
        v = v.lower()
        if v not in ['normal', 'above_normal', 'well_above_normal']:
            raise ValueError('Level must be "normal", "above_normal", or "well_above_normal"')
        return v
    
    @validator('systolic_bp', 'diastolic_bp')
    def validate_bp_relationship(cls, v, values):
        """Ensure systolic BP > diastolic BP"""
        if 'systolic_bp' in values and 'diastolic_bp' in values:
            if values['systolic_bp'] <= values['diastolic_bp']:
                raise ValueError('Systolic BP must be greater than Diastolic BP')
        return v


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: str
    probability: float
    risk_level: str
    confidence: str
    risk_score: float
    interpretation: str
    recommendations: list
    health_metrics: Dict
    timestamp: str


def preprocess_input(patient: PatientInput) -> pd.DataFrame:
    """
    Convert user-friendly input to model-ready features
    
    Args:
        patient: PatientInput object
        
    Returns:
        DataFrame with all required features
    """
    # Convert categorical inputs to numeric
    gender_map = {'female': 1, 'male': 2}
    level_map = {'normal': 1, 'above_normal': 2, 'well_above_normal': 3}
    
    # Create base features
    data = {
        'age_years': patient.age,
        'gender': gender_map[patient.gender],
        'height': patient.height,
        'weight': patient.weight,
        'ap_hi': patient.systolic_bp,
        'ap_lo': patient.diastolic_bp,
        'cholesterol': level_map[patient.cholesterol],
        'gluc': level_map[patient.glucose],
        'smoke': int(patient.smoking),
        'alco': int(patient.alcohol),
        'active': int(patient.physical_activity)
    }
    
    # Create engineered features
    # BMI features
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    data['bmi_category'] = 0 if data['bmi'] < 18.5 else (1 if data['bmi'] < 25 else (2 if data['bmi'] < 30 else 3))
    
    # Blood pressure features
    data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    data['mean_arterial_pressure'] = data['ap_lo'] + (data['pulse_pressure'] / 3)
    
    # BP category
    if data['ap_hi'] < 120 and data['ap_lo'] < 80:
        data['bp_category'] = 0
    elif data['ap_hi'] < 130 and data['ap_lo'] < 80:
        data['bp_category'] = 1
    elif data['ap_hi'] < 140 or data['ap_lo'] < 90:
        data['bp_category'] = 2
    elif data['ap_hi'] < 180 or data['ap_lo'] < 120:
        data['bp_category'] = 3
    else:
        data['bp_category'] = 4
    
    # Age group
    if data['age_years'] < 40:
        data['age_group'] = 0
    elif data['age_years'] < 50:
        data['age_group'] = 1
    elif data['age_years'] < 60:
        data['age_group'] = 2
    else:
        data['age_group'] = 3
    
    # Risk scores
    data['lifestyle_risk_score'] = data['smoke'] + data['alco'] + (1 - data['active'])
    data['metabolic_risk_score'] = data['cholesterol'] + data['gluc']
    data['combined_risk_score'] = (
        (data['bp_category'] / 4 * 30) +
        (data['bmi_category'] / 3 * 20) +
        (data['lifestyle_risk_score'] / 3 * 25) +
        (data['metabolic_risk_score'] / 6 * 25)
    )
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure correct column order
    feature_order = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'bmi', 'bmi_category', 'pulse_pressure', 
        'mean_arterial_pressure', 'bp_category',
        'age_group', 'lifestyle_risk_score', 
        'metabolic_risk_score', 'combined_risk_score'
    ]
    
    return df[feature_order]


def generate_recommendations(patient: PatientInput, risk_level: str, health_metrics: Dict) -> list:
    """
    Generate personalized health recommendations
    
    Args:
        patient: PatientInput object
        risk_level: Calculated risk level
        health_metrics: Health metrics dictionary
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Blood pressure recommendations
    if patient.systolic_bp > 130 or patient.diastolic_bp > 80:
        recommendations.append({
            "category": "Blood Pressure",
            "priority": "High",
            "recommendation": "Your blood pressure is elevated. Consider reducing salt intake, managing stress, and consulting with your doctor about blood pressure management."
        })
    
    # BMI recommendations
    bmi = health_metrics['bmi']
    if bmi > 25:
        recommendations.append({
            "category": "Weight Management",
            "priority": "High" if bmi > 30 else "Medium",
            "recommendation": f"Your BMI is {bmi:.1f}, which is {'overweight' if bmi < 30 else 'in the obese range'}. Consider a balanced diet and regular exercise to achieve a healthy weight."
        })
    
    # Lifestyle recommendations
    if patient.smoking:
        recommendations.append({
            "category": "Smoking Cessation",
            "priority": "Critical",
            "recommendation": "Smoking significantly increases cardiovascular disease risk. Consider smoking cessation programs and consult your healthcare provider for support."
        })
    
    if not patient.physical_activity:
        recommendations.append({
            "category": "Physical Activity",
            "priority": "High",
            "recommendation": "Regular physical activity is crucial for heart health. Aim for at least 150 minutes of moderate aerobic activity per week."
        })
    
    # Cholesterol recommendations
    if patient.cholesterol in ['above_normal', 'well_above_normal']:
        recommendations.append({
            "category": "Cholesterol Management",
            "priority": "High",
            "recommendation": "Your cholesterol levels are elevated. Consider a heart-healthy diet low in saturated fats and consult with your doctor about cholesterol management."
        })
    
    # Glucose recommendations
    if patient.glucose in ['above_normal', 'well_above_normal']:
        recommendations.append({
            "category": "Blood Sugar Control",
            "priority": "High",
            "recommendation": "Your glucose levels are elevated. Monitor your blood sugar regularly and consult with your healthcare provider about diabetes risk assessment."
        })
    
    # General recommendation
    recommendations.append({
        "category": "Regular Check-ups",
        "priority": "Medium",
        "recommendation": "Schedule regular health check-ups and cardiac screenings, especially if you have multiple risk factors."
    })
    
    return recommendations


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to CardioPredict API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_cardiovascular_disease(patient: PatientInput):
    """
    Predict cardiovascular disease risk
    
    Args:
        patient: Patient input data
        
    Returns:
        Prediction response with detailed analysis
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Preprocess input
        features = preprocess_input(patient)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.5:
            risk_level = "Moderate Risk"
        elif probability < 0.7:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        # Calculate confidence
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.8:
            confidence = "Very High"
        elif confidence_score > 0.6:
            confidence = "High"
        elif confidence_score > 0.4:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        # Generate interpretation
        if prediction == 0:
            interpretation = f"Based on your health parameters, the model predicts a LOW likelihood of cardiovascular disease. Your overall risk score is {probability*100:.1f}%. Continue maintaining a healthy lifestyle!"
        else:
            interpretation = f"Based on your health parameters, the model predicts a HIGH likelihood of cardiovascular disease. Your overall risk score is {probability*100:.1f}%. Please consult with a healthcare professional for proper evaluation."
        
        # Calculate health metrics
        health_metrics = {
            'bmi': round(float(features['bmi'].values[0]), 2),
            'bmi_status': 'Underweight' if features['bmi_category'].values[0] == 0 else ('Normal' if features['bmi_category'].values[0] == 1 else ('Overweight' if features['bmi_category'].values[0] == 2 else 'Obese')),
            'blood_pressure_status': 'Normal' if features['bp_category'].values[0] == 0 else ('Elevated' if features['bp_category'].values[0] == 1 else ('Stage 1 Hypertension' if features['bp_category'].values[0] == 2 else ('Stage 2 Hypertension' if features['bp_category'].values[0] == 3 else 'Hypertensive Crisis'))),
            'pulse_pressure': int(features['pulse_pressure'].values[0]),
            'mean_arterial_pressure': round(float(features['mean_arterial_pressure'].values[0]), 1)
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(patient, risk_level, health_metrics)
        
        # Create response
        response = PredictionResponse(
            prediction="Cardiovascular Disease Detected" if prediction == 1 else "No Cardiovascular Disease",
            probability=round(float(probability), 4),
            risk_level=risk_level,
            confidence=confidence,
            risk_score=round(float(probability * 100), 2),
            interpretation=interpretation,
            recommendations=recommendations,
            health_metrics=health_metrics,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {response.prediction}, Probability: {response.probability}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features_count": len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else "Unknown",
        "last_updated": "2024-01-01",  # Update this with actual date
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)