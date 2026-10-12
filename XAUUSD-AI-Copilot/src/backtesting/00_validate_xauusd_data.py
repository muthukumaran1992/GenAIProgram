import sys
from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))


# ==========================================
# DATA FILE
# ==========================================

DATA_DIR = PROJECT_DIR / "data" / "raw"

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
df.columns = df.columns.str.strip().str.lower()

time_column = "timestamp" if "timestamp" in df.columns else "date"
required_columns = {time_column, "open", "high", "low", "close"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in {file_path.name}: "
        f"{sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError(f"No market data found in {file_path.name}.")


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

if time_column == "timestamp":
    df["datetime"] = pd.to_datetime(df[time_column], unit="ms")
else:
    df["datetime"] = pd.to_datetime(df[time_column])

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
        [time_column, "open", "high", "low", "close"]
    ].isnull().sum()
)


# ==========================================
# DUPLICATES
# ==========================================

print("\nDuplicate timestamps:")

print(
    df[time_column].duplicated().sum()
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

# ==========================================
# GAP DETAILS
# ==========================================

if len(unexpected_gaps) > 0:

    gap_rows = df.loc[
        unexpected_gaps.index
    ].copy()

    gap_rows["gap"] = unexpected_gaps.values

    print("\n==============================================")
    print("              GAP ANALYSIS")
    print("==============================================")

    print("\nLargest gaps:")

    print(
        gap_rows[
            ["datetime", "gap"]
        ]
        .sort_values(
            "gap",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nGap distribution:")

    print(
        gap_rows["gap"]
        .value_counts()
        .head(20)
    )

    # ==========================================
# CLASSIFY GAPS
# ==========================================

if len(unexpected_gaps) > 0:

    gap_details = []

    for idx, gap in unexpected_gaps.items():

        current_time = df.loc[idx, "datetime"]
        previous_time = df.loc[idx - 1, "datetime"]

        gap_details.append({
            "previous": previous_time,
            "current": current_time,
            "gap": gap,
            "previous_day": previous_time.day_name(),
            "current_day": current_time.day_name()
        })

    gap_df = pd.DataFrame(gap_details)

    print("\n==============================================")
    print("              GAP CLASSIFICATION")
    print("==============================================")

    print("\nLargest gaps:")

    print(
        gap_df
        .sort_values("gap", ascending=False)
        .head(20)
        .to_string(index=False)
    )


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
    f"{df[time_column].duplicated().sum()}"
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