
import pandas as pd

# Load dataset
data = pd.read_csv("dataset/weather.csv")

# Convert date column into datetime format
data["date"] = pd.to_datetime(data["date"])

# Remove duplicate records
data = data.drop_duplicates()

# Remove rows with missing values
data = data.dropna()

# Save cleaned dataset
data.to_csv("dataset/cleaned_weather.csv", index=False)

print("Data cleaning completed successfully!")
print("Final dataset shape:", data.shape)
print("Missing values:", data.isnull().sum().sum())
print("Duplicate rows:", data.duplicated().sum())
