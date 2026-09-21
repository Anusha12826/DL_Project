
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 California House Price Prediction")

st.write(
    "Enter housing details to estimate the median "
    "house value using our trained ANN model."
)

st.info(
    "This predicts a housing area's median value, "
    "not a guaranteed price for an individual house."
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "california_housing_ann.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"

@st.cache_resource
def load_files():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

try:
    model, scaler, features = load_files()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

st.header("Enter Housing Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income ($10,000 units)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        house_age = st.number_input(
            "House Age",
            min_value=1.0,
            max_value=100.0,
            value=20.0
        )

        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=0.1,
            value=6.0
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.1,
            value=1.0
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=1.0,
            value=800.0
        )

        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=0.1,
            value=3.0
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=34.05,
            format="%.2f"
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.25,
            format="%.2f"
        )

    submitted = st.form_submit_button(
        "Predict House Value",
        use_container_width=True
    )

if submitted:

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    input_data = input_data[features]

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(
        scaled_data,
        verbose=0
    )[0][0]

    predicted_dollars = float(prediction) * 100000

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Median House Value",
        f"${predicted_dollars:,.0f}"
    )

    st.success(
        f"Predicted value: ${predicted_dollars:,.0f}"
    )

    st.caption(
        "This is a model estimate, not a professional "
        "appraisal or guaranteed market price."
    )

st.divider()
st.caption("Deep Learning Project | Team 8 | ANN")
