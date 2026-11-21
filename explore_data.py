import pandas as pd

data = pd.read_csv("dataset/weather.csv")

print("===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== DATASET SIZE =====")
print("Rows and Columns:", data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns.tolist())

print("\n===== DATA TYPES =====")
print(data.dtypes)

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(data.duplicated().sum())

print("\n===== STATISTICS =====")
print(data.describe())