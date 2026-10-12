import sys
from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

SOURCE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(SOURCE_DIR))


# ==========================================
# DATA FILE
# ==========================================

DATA_DIR = SOURCE_DIR / "data" / "raw"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )

print("\nCSV files found:")

for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file.name}")

# Select the largest CSV as the research dataset
file_path = max(
    csv_files,
    key=lambda file: file.stat().st_size
)

print("\nSelected dataset:")
print(file_path.name)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(file_path)


# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n==============================================")
print("       XAUUSD DATA VALIDATION")
print("==============================================")

print("\nFile:", file_path.name)

print("\nColumns:")
print(df.columns.tolist())

print("\nRows:")
print(len(df))


# ==========================================
# DATETIME
# ==========================================

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ==========================================
# DATE RANGE
# ==========================================

print("\nStart:")
print(df["datetime"].iloc[0])

print("\nEnd:")
print(df["datetime"].iloc[-1])


# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing values:")

print(
    df[
        ["timestamp", "open", "high", "low", "close"]
    ].isnull().sum()
)


# ==========================================
# DUPLICATES
# ==========================================

print("\nDuplicate timestamps:")

print(
    df["timestamp"].duplicated().sum()
)


# ==========================================
# OHLC VALIDATION
# ==========================================

invalid_ohlc = df[
    (df["high"] < df["open"])
    | (df["high"] < df["close"])
    | (df["low"] > df["open"])
    | (df["low"] > df["close"])
]


print("\nInvalid OHLC rows:")
print(len(invalid_ohlc))


# ==========================================
# PRICE RANGE
# ==========================================

print("\nPrice range:")

print(
    "Highest high:",
    df["high"].max()
)

print(
    "Lowest low :",
    df["low"].min()
)


# ==========================================
# TIME GAPS
# ==========================================

time_diff = (
    df["datetime"]
    .diff()
    .dropna()
)

expected_interval = pd.Timedelta(minutes=15)

unexpected_gaps = time_diff[
    time_diff != expected_interval
]


print("\nUnexpected time intervals:")
print(len(unexpected_gaps))

if len(unexpected_gaps) > 0:

    print("\nSample gaps:")

    print(
        time_diff[
            time_diff != expected_interval
        ].head(10)
    )


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n==============================================")
print("              VALIDATION SUMMARY")
print("==============================================")

print(f"Rows              : {len(df):,}")
print(f"Start             : {df['datetime'].iloc[0]}")
print(f"End               : {df['datetime'].iloc[-1]}")
print(
    f"Duplicate times   : "
    f"{df['timestamp'].duplicated().sum()}"
)
print(
    f"Invalid OHLC rows : "
    f"{len(invalid_ohlc)}"
)
print(
    f"Unexpected gaps   : "
    f"{len(unexpected_gaps)}"
)

print("==============================================")