import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
URL = "https://api.open-meteo.com/v1/forecast"

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")

@st.cache_resource
def load_model():
    with (Path(__file__).parent / "model.pkl").open("rb") as f:
        return pickle.load(f)

st.title("🍦 Ice Cream Sales Predictor")
st.write("Now using **temperature AND chance of rain** to predict ice cream sales.")
st.caption("Class project using simulated sales data. Predictions are illustrative, not business forecasts.")

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date", value=date.today() + timedelta(days=1),
    min_value=date.today(), max_value=date.today() + timedelta(days=15),
)

if st.button("Predict sales", type="primary"):
    latitude, longitude = CITIES[city]
    try:
        with st.spinner("Getting the weather forecast..."):
            response = requests.get(
                URL,
                params={
                    "latitude": latitude, "longitude": longitude,
                    "daily": "temperature_2m_max,precipitation_probability_max",
                    "temperature_unit": "fahrenheit", "timezone": "auto",
                    "start_date": str(selected_date), "end_date": str(selected_date),
                },
                timeout=30,
            )
            response.raise_for_status()
            daily = response.json()["daily"]
            temperature = daily["temperature_2m_max"][0]
            rain_probability = daily["precipitation_probability_max"][0]
            if temperature is None or rain_probability is None:
                st.error("Forecast data is missing for this date. Please try another date.")
                st.stop()
            features = pd.DataFrame({
                "temperature": [temperature],
                "rain_probability": [rain_probability],
            })
            prediction = max(0, float(load_model().predict(features)[0]))
    except (requests.RequestException, KeyError, IndexError, ValueError, OSError) as exc:
        st.error("Could not retrieve the forecast or load the model. Please try again later.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast high", f"{temperature:.0f} °F")
    c2.metric("Chance of rain", f"{rain_probability:.0f}%")
    c3.metric("Predicted sales", f"{prediction:.0f} cones")

    if not 50 <= temperature <= 100:
        st.warning("Temperature is outside the model's training range (50–100 °F). Use caution.")
    if prediction > 300:
        st.success("Busy day expected: consider stocking extra cones.")
    elif prediction > 200:
        st.info("Moderate sales expected: plan normal inventory.")
    else:
        st.info("Slower day expected: consider a promotion or smaller inventory.")
    st.caption("Rain chance is a forecast probability, not the amount of rainfall.")
