"""Train a two-feature educational ice cream sales model on synthetic data."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 400
temperature = rng.uniform(50, 100, n)
rain_probability = rng.uniform(0, 100, n)
sales = np.maximum(0, 5 * temperature - 100 - 1.2 * rain_probability + rng.normal(0, 25, n))
df = pd.DataFrame({"temperature": temperature, "rain_probability": rain_probability, "sales": sales})
features = ["temperature", "rain_probability"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df["sales"], test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("Test MAE:", round(mean_absolute_error(y_test, model.predict(X_test)), 1), "cones")
print("Test R²:", round(r2_score(y_test, model.predict(X_test)), 3))
print("Coefficients:", dict(zip(features, model.coef_)))
with (Path(__file__).parent / "model.pkl").open("wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
