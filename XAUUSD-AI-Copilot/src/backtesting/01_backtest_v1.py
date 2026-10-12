import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "XAUUSD_m15_20240101_20260930.csv"
)

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
# 2. PREPARE DATETIME
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
# 3. BASIC VALIDATION
# ==========================================

# ==========================================
# 4. TECHNICAL INDICATORS
# ==========================================

df = add_indicators(df)


# ==========================================
# 5. REPORT
# ==========================================

print("\n==============================================")
print("       XAUUSD BACKTESTING ENGINE V1")
print("==============================================")

print(
    f"\nCandles           : {len(df):,}"
)

print(
    f"Start             : "
    f"{df['datetime'].iloc[0]}"
)

print(
    f"End               : "
    f"{df['datetime'].iloc[-1]}"
)

print("\nLatest candle:")

print(
    df.iloc[-1][
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ]
)

print("\nLatest indicators:")

print(
    df.iloc[-1][
        [
            "ema20",
            "ema50",
            "rsi14",
            "atr14"
        ]
    ]
)