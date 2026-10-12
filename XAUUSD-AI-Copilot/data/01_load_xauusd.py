import pandas as pd

# XAUUSD historical data
file_path = r"data\raw\xauusd-m15-bid-2025-03-19-2025-03-20.csv"

# Load data
df = pd.read_csv(file_path)

print("\n===== XAUUSD DATA =====")
print(df.head())

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== ROW COUNT =====")
print(len(df))

# Convert timestamp from milliseconds to datetime
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

# Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

print("\n===== FIRST 5 CANDLES =====")
print(df[["datetime", "open", "high", "low", "close"]].head())

print("\n===== LAST 5 CANDLES =====")
print(df[["datetime", "open", "high", "low", "close"]].tail())

# Data quality checks
print("\n===== DATA QUALITY =====")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\n===== PRICE RANGE =====")
print("Highest price:", df["high"].max())
print("Lowest price :", df["low"].min())