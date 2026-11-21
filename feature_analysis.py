
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# 1. Load trained models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")


# 2. Random Forest feature importance
rf_feature_names = (
    rf_model.named_steps["preprocessor"].get_feature_names_out()
)

rf_importance = (
    rf_model.named_steps["regressor"].feature_importances_
)

rf_results = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importance
}).sort_values(
    by="Importance",
    ascending=False
)

# Select Top 10 features
rf_top10 = rf_results.head(10).sort_values(
    by="Importance",
    ascending=True
)


# 3. XGBoost feature importance
xgb_feature_names = (
    xgb_model.named_steps["preprocessor"].get_feature_names_out()
)

xgb_importance = (
    xgb_model.named_steps["regressor"].feature_importances_
)

xgb_results = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance": xgb_importance
}).sort_values(
    by="Importance",
    ascending=False
)

# Select Top 10 features
xgb_top10 = xgb_results.head(10).sort_values(
    by="Importance",
    ascending=True
)


# 4. Display results
print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
print(rf_results.to_string(index=False))

print("\n===== XGBOOST FEATURE IMPORTANCE =====")
print(xgb_results.to_string(index=False))


# 5. Random Forest graph
plt.figure(figsize=(10, 6))

plt.barh(
    rf_top10["Feature"],
    rf_top10["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest - Top 10 Feature Importance")

plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=300)
plt.close()


# 6. XGBoost graph
plt.figure(figsize=(10, 6))

plt.barh(
    xgb_top10["Feature"],
    xgb_top10["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("XGBoost - Top 10 Feature Importance")

plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=300)
plt.close()


# 7. Save feature importance results
rf_results.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

xgb_results.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)


print("\nFeature importance graphs saved successfully!")
print("Feature importance CSV files saved successfully!")
