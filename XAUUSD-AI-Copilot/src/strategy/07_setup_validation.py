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
# DETERMINE TREND
# ==========================================

if price > ema20 and ema20 > ema50:

    trend = "BULLISH"

elif price < ema20 and ema20 < ema50:

    trend = "BEARISH"

else:

    trend = "SIDEWAYS"


# ==========================================
# DETERMINE MOMENTUM
# ==========================================

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


# ==========================================
# RESISTANCE DISTANCE
# ==========================================

distance_to_resistance = resistance - price

distance_to_support = price - support


# ==========================================
# SETUP VALIDATION
# ==========================================

valid_setup = True
reasons = []


# ------------------------------------------
# RULE 1 — Trend
# ------------------------------------------

if trend == "SIDEWAYS":

    valid_setup = False

    reasons.append(
        "No clear directional trend"
    )


# ------------------------------------------
# RULE 2 — BUY momentum
# ------------------------------------------

if trend == "BULLISH" and rsi >= 70:

    reasons.append(
        "Bullish trend but momentum is overbought"
    )


# ------------------------------------------
# RULE 3 — Near resistance
# ------------------------------------------

if distance_to_resistance < atr:

    valid_setup = False

    reasons.append(
        "Entry is too close to resistance"
    )


# ------------------------------------------
# RULE 4 — Risk / Reward
# ------------------------------------------

entry = price

stop_loss = entry - (1.5 * atr)

risk = entry - stop_loss

take_profit = entry + (2 * risk)

reward = take_profit - entry

risk_reward = reward / risk


if risk_reward < 1.5:

    valid_setup = False

    reasons.append(
        "Risk/reward is below minimum threshold"
    )


# ==========================================
# FINAL DECISION
# ==========================================

if valid_setup:

    decision = "VALID SETUP"

else:

    decision = "WAIT / NO TRADE"


# ==========================================
# OUTPUT
# ==========================================

print("\n========================================")
print("       XAUUSD SETUP VALIDATION")
print("========================================")

print(f"\nPrice              : {price:.3f}")
print(f"EMA20              : {ema20:.3f}")
print(f"EMA50              : {ema50:.3f}")

print(f"\nRSI14              : {rsi:.2f}")
print(f"ATR14              : {atr:.3f}")

print(f"\nSupport             : {support:.3f}")
print(f"Resistance          : {resistance:.3f}")

print(f"\nTrend               : {trend}")
print(f"Momentum            : {momentum}")

print("\n--- SETUP ANALYSIS ---")

print(
    f"Distance to Resistance : "
    f"{distance_to_resistance:.3f}"
)

print(
    f"Distance to Support    : "
    f"{distance_to_support:.3f}"
)

print(f"Risk/Reward            : 1 : {risk_reward:.2f}")

print("\n--- VALIDATION ---")

for reason in reasons:

    print("⚠️", reason)

print(f"\nFINAL DECISION         : {decision}")

print("========================================")