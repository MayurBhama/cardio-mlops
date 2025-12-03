import streamlit as st
import requests


API_URL = "https://cardiopredict-heart-disease-risk.onrender.com/predict"


def send_request(payload):
    """Send request to FastAPI backend"""
    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            return {"error": response.text}

        return response.json()

    except Exception as e:
        return {"error": str(e)}


def main():

    st.set_page_config(page_title="CardioPredict", layout="wide")

    st.title("CardioPredict – Heart Disease Risk Assessment")
    st.write("Provide your health details for a **personalized cardiovascular risk report**.")

    # --------------------------
    # USER INPUT FORM
    # --------------------------
    with st.form("input_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["male", "female"])
            height = st.number_input("Height (cm)", 120, 220, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 75)

        with c2:
            ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", 80, 250, 135)
            ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", 40, 200, 85)
            cholesterol = st.selectbox("Cholesterol Level (1 Normal, 2 High, 3 Very High)", [1, 2, 3])
            gluc = st.selectbox("Glucose Level (1 Normal, 2 High, 3 Very High)", [1, 2, 3])
            smoke = st.selectbox("Do you smoke?", [0, 1])
            alco = st.selectbox("Do you consume alcohol?", [0, 1])
            active = st.selectbox("Physically Active?", [0, 1])

        submitted = st.form_submit_button("Predict Heart Risk")

    # --------------------------
    # CALL API
    # --------------------------
    if submitted:

        payload = {
            "age": age, "gender": gender, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc,
            "smoke": smoke, "alco": alco, "active": active
        }

        output = send_request(payload)

        # API communication error
        if "error" in output:
            st.error(f"Backend Error: {output['error']}")
            return

        # ==================================================
        # SPECIAL CASE — HYPOTENSION HANDLING
        # ==================================================
        if output.get("warning") is not None:
            st.error(output["warning"])

            st.subheader("Summary")
            st.info(output["summary"])

            st.subheader("Blood Pressure Interpretation")
            st.warning(output["bp_interpretation"])

            st.subheader("Other Indicators")
            st.write(f"**Cholesterol:** {output['cholesterol_interpretation']}")
            st.write(f"**Glucose:** {output['glucose_interpretation']}")

            st.subheader("Risk Markers")
            st.json(output["risk_factors"])

            st.subheader("Recommendations")
            for r in output["recommendations"]:
                st.markdown(f"- {r}")

            st.stop()

        # ==================================================
        # NORMAL RESULT
        # ==================================================
        st.subheader("Prediction Summary")
        colA, colB = st.columns(2)

        with colA:
            st.metric("Prediction", "Disease" if output["prediction"] == 1 else "No Disease")
            st.metric("Disease Probability", f"{output['probability']*100:.1f}%")
            st.metric("Risk Classification", output["risk_level"])

        with colB:
            rf = output["risk_factors"]
            st.metric("Body Mass Index (BMI)", rf["bmi"])
            st.metric("Blood Pressure Category", rf["blood_pressure_category"])
            st.metric("Combined Risk Score", rf["combined_risk_score"])

        # HEART HEALTH SCORE
        if "heart_health" in output:
            st.write("---")
            st.subheader("Heart Health Score")

            heart = output["heart_health"]

            st.metric("Heart Health Score (0–100)", heart["score"])

            st.info(
                f"Heart Health Status: {heart['status']}\n\n"
                f"Interpretation: {heart['interpretation']}"
            )

        # SUMMARY
        st.write("---")
        st.subheader("Detailed Summary")
        st.info(output["summary"])

        # INTERPRETATIONS
        st.write("---")
        st.subheader("Clinical Interpretations")
        st.markdown(f"**Blood Pressure:** {output['bp_interpretation']}")
        st.markdown(f"**Cholesterol:** {output['cholesterol_interpretation']}")
        st.markdown(f"**Glucose:** {output['glucose_interpretation']}")

        # RISK MARKERS
        st.write("---")
        st.subheader("Key Risk Markers")
        st.json(output["risk_factors"])

        # RECOMMENDATIONS
        st.write("---")
        st.subheader("Personalized Recommendations")
        for r in output["recommendations"]:
            st.markdown(f"- {r}")


if __name__ == "__main__":
    main()
