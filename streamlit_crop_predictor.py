import os
import streamlit as st

st.write("DEBUG: Files in directory:", os.listdir())
st.write("DEBUG: Current working directory:", os.getcwd())

import os
import pandas as pd
import streamlit as st
import pickle

# App title and instructions
st.title("AI Crop Predictor Dashboard")
st.write("Upload your dataset, view the data, and try out crop predictions interactively!")

# ---------------------------
# 1) Load data
# ---------------------------
DATAFILE = "crop_recommendation.csv"
if not os.path.isfile(DATAFILE):
    st.error(f"Dataset file '{DATAFILE}' not found in the current directory.")
    st.stop()

data = pd.read_csv(DATAFILE)

# 2) Rename columns if needed
column_map = {
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Potassium": "K",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "pH_Value": "ph",
    "Rainfall": "rainfall",
    "Crop": "label"
}
data = data.rename(columns=column_map)
required_columns = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}
missing = required_columns - set(data.columns)
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

# ---------------------------
# 3) Display DataFrame
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(100))  # Show first 100 for quick view

# ---------------------------
# 4) User Input Section
# ---------------------------
st.subheader("Enter Values to Predict Crop")
with st.form("prediction_form"):
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0)
    submit = st.form_submit_button("Predict Crop")

# ---------------------------
# 5) Load trained model and predict
# ---------------------------
MODELFILE = "crop_model.pkl"
if not os.path.isfile(MODELFILE):
    st.warning(f"No trained model found (expected {MODELFILE}). Please train your model and place the file here.")
else:
    with open(MODELFILE, "rb") as f:
        model = pickle.load(f)
    # Prediction handler
    if submit:
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        pred = model.predict(input_df)[0]
        st.success(f"Recommended Crop: {pred}")

# ---------------------------
# 6) Optionally, display some statistics
# ---------------------------
st.subheader("Basic Dataset Statistics")
st.write(data.describe())
