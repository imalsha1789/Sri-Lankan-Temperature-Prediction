
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned dataset
data = pd.read_csv("dataset/cleaned_weather.csv")

# 2. Convert date into datetime
data["date"] = pd.to_datetime(data["date"])

# 3. Create date-based features
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_year"] = data["date"].dt.day_of_year

# 4. Select input features and target
features = [
    "city",
    "latitude",
    "longitude",
    "temp_min_c",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "humidity_mean_pct",
    "year",
    "month",
    "day",
    "day_of_year"
]

X = data[features]
y = data["temp_max_c"]

# 5. Split data chronologically by unique dates
unique_dates = np.sort(data["date"].unique())

split_index = int(len(unique_dates) * 0.8)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_mask = data["date"].isin(train_dates)
test_mask = data["date"].isin(test_dates)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# 6. Prepare categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown="ignore"), ["city"])
    ],
    remainder="passthrough"
)

# 7. Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 8. Train model
print("Training Random Forest model...")
model.fit(X_train, y_train)

# 9. Make predictions
predictions = model.predict(X_test)

# 10. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# 11. Save trained model
joblib.dump(model, "random_forest_model.pkl")

print("\nModel saved successfully!")
