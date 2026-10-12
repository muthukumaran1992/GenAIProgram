import pandas as pd

from market_state import determine_market_state


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
# Market State
# ============================================

state = determine_market_state(df)


# ============================================
# Display
# ============================================

print("\n========================================")
print("       XAUUSD MARKET STATE")
print("========================================")

print(
    f"Price          : {state['close']:.3f}"
)

print(
    f"EMA20          : {state['ema20']:.3f}"
)

print(
    f"EMA50          : {state['ema50']:.3f}"
)

print(
    f"RSI14          : {state['rsi14']:.2f}"
)

print(
    f"ATR14          : {state['atr14']:.3f}"
)

print(
    f"Trend          : {state['trend']}"
)

print(
    f"Momentum       : {state['momentum']}"
)

print(
    f"Price Position : {state['price_position']}"
)

print(
    f"Market State   : {state['market_state']}"
)

print("========================================")