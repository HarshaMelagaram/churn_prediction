import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model_pipeline = joblib.load('model/churn_model_pipeline.pkl')

# Streamlit UI
st.title('Customer Churn Prediction')

st.sidebar.header('User Input Features')

def user_input():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.sidebar.slider('Monthly Charges', 0.0, 200.0, 70.0)
    total_charges = st.sidebar.slider('Total Charges', 0.0, 10000.0, 1400.0)

    data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'PaperlessBilling': paperless_billing,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return data

input_data = user_input()

# Predict
if st.button('Predict Churn'):
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

    st.write(f"### Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"### Probability of Churn: {prediction_proba:.2%}")
