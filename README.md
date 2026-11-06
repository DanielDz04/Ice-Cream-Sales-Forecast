# HWK 3: Ice Cream Sales Predictor 🍦

This updates the Week 3 classroom app by adding **chance of rain (%)** as a second predictor alongside the daily high temperature (°F).

## What changed
1. `train_model.py` generates synthetic example sales using both predictors, trains a linear regression model, evaluates it on a held-out test set, and saves `model.pkl`.
2. `app.py` requests **both** `temperature_2m_max` and `precipitation_probability_max` from the Open-Meteo API, then uses both inputs in the saved model.
3. The app displays the temperature, rain chance, predicted sales, and a simple inventory suggestion.

## Deploy
1. Upload `app.py`, `model.pkl`, and `requirements.txt` to the **root of a public GitHub repository**. You can also include the training script and notebook to show your work.
2. Visit https://share.streamlit.io/ and choose **Create app**. Connect the repository and choose `app.py` as the entry point.
3. Wait for the deployment to finish, open the public app URL, and test a city/date.
4. Submit **the public app URL** in Canvas before the deadline.

To retrain locally, install the requirements and run `python train_model.py`. This overwrites `model.pkl`.

**Limitations:** Sales are synthetic for learning purposes, not real business data. Rain chance is a probability of precipitation, not rainfall volume. Weather forecasts can be missing or unavailable. The model was trained on temperatures of 50–100 °F.
