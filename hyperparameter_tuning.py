
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# 1. Load dataset
data = pd.read_csv("dataset/cleaned_weather.csv")

data["date"] = pd.to_datetime(data["date"])

# Sort records chronologically
data = data.sort_values(["date", "city"]).reset_index(drop=True)


# 2. Create date features
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_year"] = data["date"].dt.day_of_year


# 3. Define features and target
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

train_date_values = data.loc[train_mask, "date"].reset_index(drop=True)


# 5. Date-based cross-validation
tscv = TimeSeriesSplit(n_splits=3)

cv_splits = []

for train_idx, valid_idx in tscv.split(train_dates):

    fold_train_dates = train_dates[train_idx]
    fold_valid_dates = train_dates[valid_idx]

    fold_train_idx = np.flatnonzero(
        train_date_values.isin(fold_train_dates).to_numpy()
    )

    fold_valid_idx = np.flatnonzero(
        train_date_values.isin(fold_valid_dates).to_numpy()
    )

    cv_splits.append((fold_train_idx, fold_valid_idx))


# 6. Preprocessing
def create_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "city",
                OneHotEncoder(handle_unknown="ignore"),
                ["city"]
            )
        ],
        remainder="passthrough"
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])


# 7. Models and parameter grids
models = {

    "Random Forest": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 20],
            "regressor__min_samples_split": [2, 5]
        },
        "filename": "tuned_random_forest_model.pkl"
    },

    "XGBoost": {
        "model": XGBRegressor(
            random_state=42,
            n_jobs=1,
            verbosity=0
        ),
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [3, 6],
            "regressor__learning_rate": [0.05, 0.1]
        },
        "filename": "tuned_xgboost_model.pkl"
    },

    "LightGBM": {
        "model": LGBMRegressor(
            random_state=42,
            n_jobs=1,
            verbosity=-1
        ),
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [-1, 10],
            "regressor__learning_rate": [0.05, 0.1]
        },
        "filename": "tuned_lightgbm_model.pkl"
    },

    "CatBoost": {
        "model": CatBoostRegressor(
            random_seed=42,
            verbose=0,
            thread_count=1,
            allow_writing_files=False
        ),
        "params": {
            "regressor__iterations": [200, 400],
            "regressor__depth": [4, 6],
            "regressor__learning_rate": [0.05, 0.1]
        },
        "filename": "tuned_catboost_model.pkl"
    }
}


# 8. Hyperparameter tuning and evaluation
results = []

for model_name, config in models.items():

    print("\n" + "=" * 55)
    print(f"Starting {model_name} hyperparameter tuning...")
    print("=" * 55)

    pipeline = create_pipeline(config["model"])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Evaluate on untouched test set
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nBest Parameters:", search.best_params_)
    print("Best CV MAE:", round(-search.best_score_, 3))
    print("Test MAE:", round(mae, 3))
    print("Test RMSE:", round(rmse, 3))
    print("Test R2:", round(r2, 3))

    # Save tuned model
    joblib.dump(best_model, config["filename"])

    print("Saved:", config["filename"])

    results.append({
        "Model": model_name,
        "Best CV MAE": round(-search.best_score_, 3),
        "Test MAE": round(mae, 3),
        "Test RMSE": round(rmse, 3),
        "Test R2": round(r2, 3),
        "Best Parameters": search.best_params_
    })


# 9. Final comparison
results_df = pd.DataFrame(results)

print("\n\n========== HYPERPARAMETER TUNING COMPARISON ==========")
print(
    results_df[
        ["Model", "Best CV MAE", "Test MAE", "Test RMSE", "Test R2"]
    ].to_string(index=False)
)

print("\nAll model tuning processes completed!")
