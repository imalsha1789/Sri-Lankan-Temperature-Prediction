
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor


# 1. Load dataset
data = pd.read_csv("dataset/cleaned_weather.csv")


# 2. Date features
data["date"] = pd.to_datetime(data["date"])

data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_year"] = data["date"].dt.day_of_year


# Sort data chronologically
data = data.sort_values(["date", "city"]).reset_index(drop=True)


# 3. Input features and target
features = [
    "city", "latitude", "longitude",
    "temp_min_c", "precipitation_mm",
    "wind_speed_max_kmh", "humidity_mean_pct",
    "year", "month", "day", "day_of_year"
]

X = data[features]
y = data["temp_max_c"]


# 4. Chronological split by unique dates
unique_dates = np.sort(data["date"].unique())

split_index = int(len(unique_dates) * 0.8)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_mask = data["date"].isin(train_dates)
test_mask = data["date"].isin(test_dates)

X_train = X.loc[train_mask].reset_index(drop=True)
X_test = X.loc[test_mask].reset_index(drop=True)

y_train = y.loc[train_mask].reset_index(drop=True)
y_test = y.loc[test_mask].reset_index(drop=True)


# 5. CatBoost model
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    thread_count=-1
)


# 6. Train
print("Training CatBoost model...")

model.fit(
    X_train,
    y_train,
    cat_features=["city"]
)


# 7. Predict
predictions = model.predict(X_test)


# 8. Evaluate
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n===== CATBOOST RESULTS =====")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))


# 9. Save model
joblib.dump(model, "catboost_model.pkl")

print("\nCatBoost model saved successfully!")
