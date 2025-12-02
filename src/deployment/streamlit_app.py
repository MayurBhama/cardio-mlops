"""
Streamlit Web Interface for CardioPredict
User-friendly interface for cardiovascular disease prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="CardioPredict - Heart Disease Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E63946;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #457B9D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #E63946;
    }
    .recommendation-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler (cached)"""
    try:
        # Note: Ensure these paths match your actual model location
        model = joblib.load("models/trained_models/best_model.pkl")
        scaler = joblib.load("models/trained_models/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def calculate_features(age, gender, height, weight, systolic_bp, diastolic_bp,
                       cholesterol, glucose, smoking, alcohol, physical_activity):
    """Calculate all required features from inputs"""
    
    # Base features
    data = {
        'age_years': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': systolic_bp,
        'ap_lo': diastolic_bp,
        'cholesterol': cholesterol,
        'gluc': glucose,
        'smoke': smoking,
        'alco': alcohol,
        'active': physical_activity
    }
    
    # BMI and body features
    data['bmi'] = weight / ((height / 100) ** 2)
    
    if data['bmi'] < 18.5:
        data['bmi_category'] = 0
    elif data['bmi'] < 25:
        data['bmi_category'] = 1
    elif data['bmi'] < 30:
        data['bmi_category'] = 2
    else:
        data['bmi_category'] = 3
        
    # Blood pressure features
    data['pulse_pressure'] = systolic_bp - diastolic_bp
    data['mean_arterial_pressure'] = diastolic_bp + (data['pulse_pressure'] / 3)
    
    # BP category
    if systolic_bp < 120 and diastolic_bp < 80:
        data['bp_category'] = 0
    elif systolic_bp < 130 and diastolic_bp < 80:
        data['bp_category'] = 1
    elif systolic_bp < 140 or diastolic_bp < 90:
        data['bp_category'] = 2
    elif systolic_bp < 180 or diastolic_bp < 120:
        data['bp_category'] = 3
    else:
        data['bp_category'] = 4
        
    # Age group
    if age < 40:
        data['age_group'] = 0
    elif age < 50:
        data['age_group'] = 1
    elif age < 60:
        data['age_group'] = 2
    else:
        data['age_group'] = 3
        
    # Risk scores
    data['lifestyle_risk_score'] = smoking + alcohol + (1 - physical_activity)
    data['metabolic_risk_score'] = cholesterol + glucose
    data['combined_risk_score'] = (
        (data['bp_category'] / 4 * 30) +
        (data['bmi_category'] / 3 * 20) +
        (data['lifestyle_risk_score'] / 3 * 25) +
        (data['metabolic_risk_score'] / 6 * 25)
    )
    
    return data

def create_gauge_chart(value, title):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 50], 'color': '#FFD700'},
                {'range': [50, 70], 'color': '#FFA500'},
                {'range': [70, 100], 'color': '#FF6347'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def generate_recommendations(bmi, bp_category, cholesterol, glucose, smoking,
                             alcohol, physical_activity):
    """Generate personalized recommendations"""
    recommendations = []
    
    # BMI recommendations
    if bmi > 25:
        priority = "High Priority" if bmi > 30 else "Medium Priority"
        recommendations.append({
            "title": "Weight Management",
            "priority": priority,
            "text": f"Your BMI is {bmi:.1f}. Consider consulting a nutritionist for a personalized diet plan."
        })
        
    # Blood pressure recommendations
    if bp_category >= 2:
        recommendations.append({
            "title": "Blood Pressure Control",
            "priority": "High Priority",
            "text": "Your blood pressure is elevated. Reduce sodium intake, manage stress, and consult your doctor."
        })
        
    # Lifestyle recommendations
    if smoking == 1:
        recommendations.append({
            "title": "Smoking Cessation",
            "priority": "Critical",
            "text": "Smoking dramatically increases heart disease risk. Seek professional help to quit."
        })
        
    if physical_activity == 0:
        recommendations.append({
            "title": "Physical Activity",
            "priority": "Medium Priority",
            "text": "Aim for 150 minutes of moderate aerobic activity per week."
        })
        
    if cholesterol >= 2:
        recommendations.append({
            "title": "Cholesterol Management",
            "priority": "Medium Priority",
            "text": "Follow a heart-healthy diet low in saturated fats. Consider regular check-ups."
        })
        
    if glucose >= 2:
        recommendations.append({
            "title": "Blood Sugar Control",
            "priority": "Medium Priority",
            "text": "Monitor blood sugar regularly and consult your doctor about diabetes screening."
        })
        
    return recommendations

def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">CardioPredict</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Cardiovascular Disease Risk Assessment</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("Model not loaded. Please ensure model files are in the correct directory.")
        return
        
    # Sidebar for inputs
    st.sidebar.header("Patient Information")
    
    # Demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age (years)", 30, 70, 45, help="Your current age")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender_code = 1 if gender == "Female" else 2
    
    # Physical Measurements
    st.sidebar.subheader("Physical Measurements")
    height = st.sidebar.number_input("Height (cm)", 140, 210, 170, help="Your height in centimeters")
    weight = st.sidebar.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.5, help="Your weight in kilograms")
    
    # Blood Pressure
    st.sidebar.subheader("Blood Pressure")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        systolic_bp = st.number_input("Systolic", 90, 200, 120, help="Upper BP value")
    with col2:
        diastolic_bp = st.number_input("Diastolic", 60, 130, 80, help="Lower BP value")
        
    # Validate BP
    if systolic_bp <= diastolic_bp:
        st.sidebar.error("Systolic BP must be greater than Diastolic BP")
        
    # Lab Results
    st.sidebar.subheader("Lab Results")
    cholesterol = st.sidebar.select_slider(
        "Cholesterol Level",
        options=["Normal", "Above Normal", "Well Above Normal"],
        value="Normal"
    )
    cholesterol_code = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
    
    glucose = st.sidebar.select_slider(
        "Glucose Level",
        options=["Normal", "Above Normal", "Well Above Normal"],
        value="Normal"
    )
    glucose_code = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[glucose]
    
    # Lifestyle
    st.sidebar.subheader("Lifestyle Factors")
    smoking = st.sidebar.checkbox("Do you smoke?")
    alcohol = st.sidebar.checkbox("Do you consume alcohol regularly?")
    physical_activity = st.sidebar.checkbox("Are you physically active?")
    
    # Convert to codes
    smoking_code = 1 if smoking else 0
    alcohol_code = 1 if alcohol else 0
    activity_code = 1 if physical_activity else 0
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("Analyze Risk", use_container_width=True)
    
    # Main content area
    if predict_button and systolic_bp > diastolic_bp:
        
        # Calculate features
        features_dict = calculate_features(
            age, gender_code, height, weight, systolic_bp, diastolic_bp,
            cholesterol_code, glucose_code, smoking_code, alcohol_code, activity_code
        )
        
        # Create DataFrame
        feature_order = [
            'age_years', 'gender', 'height', 'weight',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active',
            'bmi', 'bmi_category', 'pulse_pressure', 
            'mean_arterial_pressure', 'bp_category',
            'age_group', 'lifestyle_risk_score', 
            'metabolic_risk_score', 'combined_risk_score'
        ]
        
        features_df = pd.DataFrame([features_dict])[feature_order]
        
        # Scale and predict
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        risk_score = probability * 100
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low Risk"
            risk_color = "green"
        elif risk_score < 50:
            risk_level = "Moderate Risk"
            risk_color = "orange"
        elif risk_score < 70:
            risk_level = "High Risk"
            risk_color = "red"
        else:
            risk_level = "Very High Risk"
            risk_color = "darkred"
            
        # Display results
        st.success("Analysis Complete!")
        
        # Risk Score Gauge
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col2:
            st.plotly_chart(create_gauge_chart(risk_score, "Cardiovascular Risk Score"), use_container_width=True)
            
        # Risk Level
        st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{risk_level}</h2>", unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("---")
        st.subheader("Detailed Analysis")
        
        if prediction == 0:
            interpretation = f"""
            Based on your health parameters, the AI model predicts a **{risk_level}** of cardiovascular disease. 
            Your overall risk score is **{risk_score:.1f}%**.
            
            This is encouraging! However, continue maintaining healthy habits and regular check-ups.
            """
        else:
            interpretation = f"""
            Based on your health parameters, the AI model predicts a **{risk_level}** of cardiovascular disease. 
            Your overall risk score is **{risk_score:.1f}%**.
            
            Important: This prediction suggests elevated risk. Please consult with a healthcare professional 
            for proper medical evaluation and guidance. This tool is not a substitute for professional medical advice.
            """
            
        st.info(interpretation)
        
        # Health Metrics
        st.markdown("---")
        st.subheader("Your Health Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        bmi = features_dict['bmi']
        bmi_status = ['Underweight', 'Normal', 'Overweight', 'Obese'][features_dict['bmi_category']]
        
        bp_status = ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN', 'Crisis'][features_dict['bp_category']]
        
        with col1:
            st.metric("BMI", f"{bmi:.1f}", bmi_status)
            
        with col2:
            st.metric("BP Status", bp_status, f"{systolic_bp}/{diastolic_bp}")
            
        with col3:
            st.metric("Pulse Pressure", f"{features_dict['pulse_pressure']} mmHg")
            
        with col4:
            st.metric("MAP", f"{features_dict['mean_arterial_pressure']:.1f} mmHg")
            
        # Recommendations
        st.markdown("---")
        st.subheader("Personalized Recommendations")
        
        recommendations = generate_recommendations(
            bmi, features_dict['bp_category'], cholesterol_code, glucose_code,
            smoking_code, alcohol_code, activity_code
        )
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec['title']} - {rec['priority']}</h4>
                    <p>{rec['text']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Great job! You're maintaining healthy habits. Keep it up!")
            
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>Medical Disclaimer:</strong> This tool uses machine learning to assess cardiovascular disease risk 
            based on the provided health parameters. It is intended for educational and informational purposes only and 
            should NOT replace professional medical advice, diagnosis, or treatment. Always consult with qualified 
            healthcare professionals for medical decisions.
        </div>
        """, unsafe_allow_html=True)
        
    elif not predict_button:
        # Welcome screen
        st.info("Please fill in your health information in the sidebar and click 'Analyze Risk' to get started.")
        
        # About section
        st.markdown("---")
        st.subheader("About CardioPredict")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is CardioPredict?**
            
            CardioPredict is an AI-powered tool that assesses your risk of cardiovascular disease 
            based on various health parameters including:
            
            - Demographics (age, gender)
            - Physical measurements (height, weight)
            - Blood pressure readings
            - Lab results (cholesterol, glucose)
            - Lifestyle factors (smoking, alcohol, physical activity)
            """)
            
        with col2:
            st.markdown("""
            **How does it work?**
            
            1. **Data Collection**: Enter your health information
            2. **Feature Engineering**: The system calculates additional health metrics (BMI, blood pressure categories, risk scores)
            3. **AI Prediction**: A machine learning model analyzes all parameters
            4. **Risk Assessment**: You receive a comprehensive risk analysis with personalized recommendations
            
            The model was trained on 70,000+ patient records and achieves high accuracy in risk prediction.
            """)

if __name__ == "__main__":
    main()