import streamlit as st
import numpy as np
import joblib

# Load the saved model (Random Forest)
model = joblib.load(r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_1\best_rf_model.joblib')

# Streamlit UI
st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")
st.title("Boston Housing Price Predictor")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Features")
    rm = st.number_input("Number of Rooms (RM)", min_value=0.0, step=0.1, value=6.0, help="Average number of rooms per house")
    lstat = st.number_input("Poverty Level (% LSTAT)", min_value=0.0, max_value=100.0, step=0.1, value=12.0, help="Percentage of population below poverty line")
    ptratio = st.number_input("Student-Teacher Ratio (PTRATIO)", min_value=0.0, step=0.1, value=18.0, help="Student-to-teacher ratio in nearby schools")

# Prediction
if st.button("Predict Price", key="predict_button", help="Click to predict the house price"):
    client_data = np.array([[rm, lstat, ptratio]])
    predicted_price = model.predict(client_data)[0]
    st.success(f"Predicted Price: **${predicted_price:,.2f}**")