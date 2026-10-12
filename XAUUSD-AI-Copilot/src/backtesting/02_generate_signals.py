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
# 2. INDICATORS
# ==========================================

df = add_indicators(df)


# ==========================================
# 3. GENERATE SIGNALS
# ==========================================

df["signal"] = "WAIT"


for i in range(50, len(df) - 1):

    current = df.iloc[i]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    # --------------------------------------
    # MARKET TREND
    # --------------------------------------

    if price > ema20 and ema20 > ema50:

        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS"


    # --------------------------------------
    # PREVIOUS 20-CANDLE RANGE
    # --------------------------------------

    previous_candles = df.iloc[i - 20:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()


    # --------------------------------------
    # CURRENT CANDLE
    # --------------------------------------

    bullish_candle = (
        current["close"] > current["open"]
    )

    bearish_candle = (
        current["close"] < current["open"]
    )


    # --------------------------------------
    # BREAKOUT
    # --------------------------------------

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
    )


    # --------------------------------------
    # PULLBACK
    # --------------------------------------

    bullish_pullback = (
        trend == "BULLISH"
        and current["low"] <= ema20
        and price > ema20
        and bullish_candle
        and rsi < 70
    )

    bearish_pullback = (
        trend == "BEARISH"
        and current["high"] >= ema20
        and price < ema20
        and bearish_candle
        and rsi > 30
    )


    # --------------------------------------
    # FINAL SIGNAL
    # --------------------------------------

    if bullish_breakout or bullish_pullback:

        df.loc[i, "signal"] = "BUY"

    elif bearish_breakout or bearish_pullback:

        df.loc[i, "signal"] = "SELL"


# ==========================================
# 4. SIGNAL SUMMARY
# ==========================================

print("\n==============================================")
print("          XAUUSD SIGNAL GENERATOR")
print("==============================================")

print("\nSignal counts:")

print(
    df["signal"]
    .value_counts()
)


print("\nFirst signals:")

print(
    df[
        df["signal"] != "WAIT"
    ][
        [
            "datetime",
            "close",
            "ema20",
            "ema50",
            "rsi14",
            "signal"
        ]
    ]
    .head(10)
    .to_string(index=False)
)