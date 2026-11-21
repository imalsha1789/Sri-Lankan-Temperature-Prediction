
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# 3. Features and target
features = [
    "city", "latitude", "longitude",
    "temp_min_c", "precipitation_mm",
    "wind_speed_max_kmh", "humidity_mean_pct",
    "year", "month", "day", "day_of_year"
]

X = data[features]
y = data["temp_max_c"]


# 4. Chronological train-test split by unique dates
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


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(handle_unknown="ignore"),
            ["city"]
        ),
        (
            "numeric",
            StandardScaler(),
            [
                "latitude", "longitude", "temp_min_c",
                "precipitation_mm", "wind_speed_max_kmh",
                "humidity_mean_pct", "year", "month",
                "day", "day_of_year"
            ]
        )
    ]
)


# 6. Random Forest model
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ))
])


# 7. ANN model
ann_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=42
    ))
])


# 8. Hybrid model (Random Forest + ANN)
hybrid_model = VotingRegressor([
    ("random_forest", rf_pipeline),
    ("ann", ann_pipeline)
])


# 9. Train
print("Training Hybrid Model...")

hybrid_model.fit(X_train, y_train)


# 10. Predictions
predictions = hybrid_model.predict(X_test)


# 11. Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n===== HYBRID MODEL RESULTS =====")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))


# 12. Save model
joblib.dump(hybrid_model, "hybrid_model.pkl")

print("\nHybrid model saved successfully!")
