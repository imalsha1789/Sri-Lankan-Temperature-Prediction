
import pandas as pd
import matplotlib.pyplot as plt


# 1. Model evaluation results
results = {
    "Model": [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Stacking",
        "Hybrid (RF + ANN)",
        "Tuned Random Forest",
        "Tuned XGBoost",
        "Tuned LightGBM",
        "Tuned CatBoost"
    ],

    "MAE": [
        0.839,
        0.818,
        0.819,
        0.856,
        0.814,
        0.827,
        0.837,
        0.815,
        0.819,
        0.822
    ],

    "RMSE": [
        1.093,
        1.063,
        1.062,
        1.108,
        1.053,
        1.079,
        1.090,
        1.057,
        1.062,
        1.067
    ],

    "R2": [
        0.892,
        0.898,
        0.898,
        0.889,
        0.900,
        0.895,
        0.892,
        0.899,
        0.898,
        0.897
    ]
}


# 2. Create DataFrame
df = pd.DataFrame(results)


# 3. Display comparison table
print("\n===== MODEL COMPARISON =====")

print(df.to_string(index=False))


# 4. Identify best models
best_mae = df.loc[df["MAE"].idxmin()]
best_rmse = df.loc[df["RMSE"].idxmin()]
best_r2 = df.loc[df["R2"].idxmax()]

print("\n===== BEST MODEL RESULTS =====")

print("Best MAE Model:", best_mae["Model"])
print("Best RMSE Model:", best_rmse["Model"])
print("Best R2 Model:", best_r2["Model"])


# 5. MAE Comparison
plt.figure(figsize=(12, 6))

plt.bar(df["Model"], df["MAE"])

plt.title("Model Comparison - MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("MAE (Lower is Better)")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("model_comparison_mae.png")
plt.show()


# 6. RMSE Comparison
plt.figure(figsize=(12, 6))

plt.bar(df["Model"], df["RMSE"])

plt.title("Model Comparison - RMSE")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE (Lower is Better)")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("model_comparison_rmse.png")
plt.show()


# 7. R2 Comparison
plt.figure(figsize=(12, 6))

plt.bar(df["Model"], df["R2"])

plt.title("Model Comparison - R2 Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 (Higher is Better)")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("model_comparison_r2.png")
plt.show()


# 8. Save comparison results
df.to_csv("model_comparison_results.csv", index=False)

print("\nComparison graphs saved successfully!")
print("Comparison results saved successfully!")
