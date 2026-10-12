import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

file_path = r"data\raw\xauusd-m15-bid-2025-03-19-2025-03-20.csv"

df = pd.read_csv(file_path)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ==========================================
# INDICATORS
# ==========================================

df["ema20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["ema50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()


# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

df["rsi14"] = 100 - (
    100 / (1 + rs)
)


# ATR
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


# ==========================================
# SUPPORT / RESISTANCE
# ==========================================

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


# ==========================================
# LATEST MARKET DATA
# ==========================================

latest = df.iloc[-1]

price = latest["close"]
atr = latest["atr14"]
support = latest["support"]
resistance = latest["resistance"]

ema20 = latest["ema20"]
ema50 = latest["ema50"]
rsi = latest["rsi14"]


# ==========================================
# HYPOTHETICAL TRADE
# ==========================================

entry = price

# For this learning exercise:
stop_loss = entry - (1.5 * atr)

risk = entry - stop_loss

# Target based on 2:1 R:R
take_profit = entry + (2 * risk)

reward = take_profit - entry

risk_reward = reward / risk


# ==========================================
# OUTPUT
# ==========================================

print("\n========================================")
print("       XAUUSD RISK / REWARD ENGINE")
print("========================================")

print(f"Current Price : {price:.3f}")
print(f"ATR14         : {atr:.3f}")

print(f"\nSupport       : {support:.3f}")
print(f"Resistance    : {resistance:.3f}")

print("\n--- HYPOTHETICAL BUY SETUP ---")

print(f"Entry         : {entry:.3f}")
print(f"Stop Loss     : {stop_loss:.3f}")
print(f"Take Profit   : {take_profit:.3f}")

print(f"\nRisk          : {risk:.3f}")
print(f"Reward        : {reward:.3f}")