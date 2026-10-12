from pathlib import Path

import pandas as pd
from technical_indicators import add_indicators

# ==========================================
# LOAD DATA
# ==========================================

file_path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "xauusd-m15-bid-2025-03-19-2025-03-20.csv"
)

df = pd.read_csv(file_path)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ==========================================
# CALCULATE INDICATORS
# ==========================================

df = add_indicators(df)


# ==========================================
# DISPLAY
# ==========================================

latest = df.iloc[-1]

print("\n========================================")
print("       CENTRALIZED INDICATORS")
print("========================================")

print(f"\nPrice  : {latest['close']:.3f}")
print(f"EMA20  : {latest['ema20']:.3f}")
print(f"EMA50  : {latest['ema50']:.3f}")
print(f"RSI14  : {latest['rsi14']:.2f}")
print(f"ATR14  : {latest['atr14']:.3f}")

print("\n========================================")