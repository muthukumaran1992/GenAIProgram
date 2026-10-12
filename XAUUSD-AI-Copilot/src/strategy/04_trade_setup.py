import pandas as pd


# ============================================
# Load XAUUSD data
# ============================================

file_path = r"data\raw\xauusd-m15-bid-2025-03-19-2025-03-20.csv"

df = pd.read_csv(file_path)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ============================================
# EMA 20 / EMA 50
# ============================================

df["ema20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["ema50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()


# ============================================
# RSI 14
# ============================================

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

df["rsi14"] = 100 - (
    100 / (1 + rs)
)


# ============================================
# ATR 14
# ============================================

previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]

tr2 = (
    df["high"] - previous_close
).abs()

tr3 = (
    df["low"] - previous_close
).abs()

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["atr14"] = true_range.rolling(14).mean()


# ============================================
# Support / Resistance
# ============================================

lookback = 20

df["support"] = (
    df["low"]
    .rolling(lookback)
    .min()
)

df["resistance"] = (
    df["high"]
    .rolling(lookback)
    .max()
)


# ============================================
# Current values
# ============================================

latest = df.iloc[-1]

price = latest["close"]
ema20 = latest["ema20"]
ema50 = latest["ema50"]
rsi = latest["rsi14"]
atr = latest["atr14"]

support = latest["support"]
resistance = latest["resistance"]


# ============================================
# Trend
# ============================================

if price > ema20 and ema20 > ema50:
    trend = "BULLISH"

elif price < ema20 and ema20 < ema50:
    trend = "BEARISH"

else:
    trend = "SIDEWAYS"


# ============================================
# Momentum
# ============================================

if rsi >= 70:
    momentum = "OVERBOUGHT"

elif rsi >= 55:
    momentum = "BULLISH"

elif rsi >= 45:
    momentum = "NEUTRAL"

elif rsi >= 30:
    momentum = "BEARISH"

else:
    momentum = "OVERSOLD"


# ============================================
# Display
# ============================================

print("\n========================================")
print("       XAUUSD TRADE SETUP")
print("========================================")

print(f"Price       : {price:.3f}")
print(f"EMA20       : {ema20:.3f}")
print(f"EMA50       : {ema50:.3f}")
print(f"RSI14       : {rsi:.2f}")
print(f"ATR14       : {atr:.3f}")

print(f"\nSupport     : {support:.3f}")
print(f"Resistance  : {resistance:.3f}")

print(f"\nTrend       : {trend}")
print(f"Momentum    : {momentum}")

print("========================================")
