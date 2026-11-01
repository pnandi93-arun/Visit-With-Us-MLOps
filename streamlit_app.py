
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Visit with Us | Wellness Package", page_icon="✈️", layout="centered")
st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package – Customer Propensity")
st.write("Estimate historical package-purchase propensity from customer attributes. This is a propensity estimate, not a guarantee of purchase or a Wellness-specific trained outcome.")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
if not MODEL_PATH.exists():
    st.error("Model file not found. Ensure models/best_model.joblib is committed to the repository.")
    st.stop()
bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
threshold = float(bundle.get("threshold", 0.5))
features = bundle["feature_columns"]

with st.form("customer_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of people visiting", min_value=1, max_value=20, value=2)
    preferred_star = st.selectbox("Preferred property star", [3.0, 4.0, 5.0], index=1)
    marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    trips = st.number_input("Number of trips per year", min_value=0.0, max_value=50.0, value=2.0, step=1.0)
    passport = st.selectbox("Has passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car = st.selectbox("Owns a car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    children = st.number_input("Children below age 5 visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", min_value=0.0, max_value=500000.0, value=25000.0, step=1000.0)
    submitted = st.form_submit_button("Predict propensity")

if submitted:
    row = {
      "Age":float(age), "CityTier":int(city_tier), "Occupation":occupation,
      "Gender":gender, "NumberOfPersonVisiting":int(persons),
      "PreferredPropertyStar":float(preferred_star), "MaritalStatus":marital,
      "NumberOfTrips":float(trips), "Passport":int(passport), "OwnCar":int(own_car),
      "NumberOfChildrenVisiting":int(children), "Designation":designation,
      "MonthlyIncome":float(income)
    }
    input_df = pd.DataFrame([{col: row.get(col, None) for col in features}])
    probability = float(model.predict_proba(input_df)[:,1][0])
    prediction = int(probability >= threshold)
    st.metric("Estimated purchase propensity", f"{probability:.1%}")
    if prediction:
        st.success("Higher propensity segment – consider a relevant, permission-based wellness offer.")
    else:
        st.info("Lower propensity at the current threshold. This does not mean the customer will never purchase.")
    st.caption(f"Model: {bundle.get('model_name','saved model')} | Decision threshold: {threshold:.2f}")
