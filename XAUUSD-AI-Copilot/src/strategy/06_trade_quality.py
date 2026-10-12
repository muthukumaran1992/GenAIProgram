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
tr2 = (df["high"] - previous_close).abs()
tr3 = (df["low"] - previous_close).abs()

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
    df["low"].rolling(lookback).min()
)

df["resistance"] = (
    df["high"].rolling(lookback).max()
)


# ==========================================
# CURRENT MARKET DATA
# ==========================================

latest = df.iloc[-1]

price = latest["close"]
ema20 = latest["ema20"]
ema50 = latest["ema50"]
rsi = latest["rsi14"]
atr = latest["atr14"]

support = latest["support"]
resistance = latest["resistance"]


# ==========================================
# TRADE SETUP
# ==========================================

entry = price

stop_loss = entry - (1.5 * atr)

risk = entry - stop_loss

take_profit = entry + (2 * risk)

reward = take_profit - entry

risk_reward = reward / risk


# ==========================================
# SCORE COMPONENTS
# ==========================================

score = 0

print("\n========================================")
print("       XAUUSD TRADE QUALITY ENGINE")
print("========================================")


# ------------------------------------------
# 1. TREND
# ------------------------------------------

if price > ema20 and ema20 > ema50:

    trend = "BULLISH"
    score += 25

elif price < ema20 and ema20 < ema50:

    trend = "BEARISH"
    score += 25

else:

    trend = "SIDEWAYS"
    score += 10


# ------------------------------------------
# 2. MOMENTUM
# ------------------------------------------

if 55 <= rsi < 70:

    momentum = "BULLISH"
    score += 20

elif 45 <= rsi < 55:

    momentum = "NEUTRAL"
    score += 10

elif rsi >= 70:

    momentum = "OVERBOUGHT"
    score += 5

elif rsi <= 30:

    momentum = "OVERSOLD"
    score += 5

else:

    momentum = "BEARISH"
    score += 5


# ------------------------------------------
# 3. ENTRY LOCATION
# ------------------------------------------

distance_from_resistance = resistance - entry

if distance_from_resistance > atr:

    entry_quality = "GOOD"
    score += 20

elif distance_from_resistance > 0:

    entry_quality = "NEAR_RESISTANCE"
    score += 5

else:

    entry_quality = "BREAKOUT"
    score += 10


# ------------------------------------------
# 4. RISK / REWARD
# ------------------------------------------

if risk_reward >= 2:

    rr_quality = "GOOD"
    score += 20

elif risk_reward >= 1.5:

    rr_quality = "ACCEPTABLE"
    score += 15

elif risk_reward >= 1:

    rr_quality = "WEAK"
    score += 5

else:

    rr_quality = "POOR"


# ------------------------------------------
# 5. SUPPORT / RESISTANCE
# ------------------------------------------

if entry > support and entry < resistance:

    structure = "INSIDE_RANGE"
    score += 10

elif entry >= resistance:

    structure = "BREAKOUT_ZONE"
    score += 10

else:

    structure = "BELOW_SUPPORT"


# ==========================================
# FINAL DECISION
# ==========================================

if score >= 75:

    decision = "BUY"

elif score >= 55:

    decision = "WAIT / CONFIRM"

else:

    decision = "NO TRADE"


# ==========================================
# OUTPUT
# ==========================================

print(f"\nPrice              : {price:.3f}")
print(f"EMA20              : {ema20:.3f}")
print(f"EMA50              : {ema50:.3f}")
print(f"RSI14              : {rsi:.2f}")
print(f"ATR14              : {atr:.3f}")

print(f"\nSupport             : {support:.3f}")
print(f"Resistance          : {resistance:.3f}")

print(f"\nTrend               : {trend}")
print(f"Momentum            : {momentum}")
print(f"Entry Quality       : {entry_quality}")
print(f"Structure           : {structure}")
print(f"R:R Quality         : {rr_quality}")

print("\n--- TRADE PARAMETERS ---")

print(f"Entry               : {entry:.3f}")
print(f"Stop Loss           : {stop_loss:.3f}")
print(f"Take Profit         : {take_profit:.3f}")
print(f"Risk                : {risk:.3f}")
print(f"Reward              : {reward:.3f}")
print(f"Risk/Reward         : 1 : {risk_reward:.2f}")

print("\n--- FINAL SCORE ---")

print(f"Trade Quality Score : {score}/95")
print(f"Decision             : {decision}")

print("========================================")