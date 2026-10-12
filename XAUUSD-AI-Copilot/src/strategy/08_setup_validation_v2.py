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
# EMA
# ==========================================

df["ema20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["ema50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()


# ==========================================
# RSI
# ==========================================

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

df["rsi14"] = 100 - (
    100 / (1 + rs)
)


# ==========================================
# ATR
# ==========================================

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
# PREVIOUS SUPPORT / RESISTANCE
# ==========================================

lookback = 20

df["support"] = (
    df["low"]
    .rolling(lookback)
    .min()
    .shift(1)
)

df["resistance"] = (
    df["high"]
    .rolling(lookback)
    .max()
    .shift(1)
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
# TREND
# ==========================================

if price > ema20 and ema20 > ema50:

    trend = "BULLISH"

elif price < ema20 and ema20 < ema50:

    trend = "BEARISH"

else:

    trend = "SIDEWAYS"


# ==========================================
# MOMENTUM
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
# DISTANCES
# ==========================================

distance_to_resistance = resistance - price

distance_to_support = price - support


# ==========================================
# BREAKOUT STATUS
# ==========================================

if price > resistance:

    breakout_status = "ABOVE_RESISTANCE"

elif price >= resistance - (0.5 * atr):

    breakout_status = "APPROACHING_RESISTANCE"

else:

    breakout_status = "BELOW_RESISTANCE"


# ==========================================
# HYPOTHETICAL BUY SETUP
# ==========================================

entry = price

stop_loss = entry - (1.5 * atr)

risk = entry - stop_loss

take_profit = entry + (2 * risk)

reward = take_profit - entry

risk_reward = reward / risk


# ==========================================
# VALIDATION
# ==========================================

valid_setup = True

reasons = []


# ------------------------------------------
# Rule 1 — Trend
# ------------------------------------------

if trend == "SIDEWAYS":

    valid_setup = False

    reasons.append(
        "No clear directional trend"
    )


# ------------------------------------------
# Rule 2 — Overbought
# ------------------------------------------

if trend == "BULLISH" and rsi >= 70:

    reasons.append(
        "Bullish trend but momentum is overbought"
    )


# ------------------------------------------
# Rule 3 — Resistance
# ------------------------------------------

if breakout_status == "APPROACHING_RESISTANCE":

    valid_setup = False

    reasons.append(
        "Price is approaching previous resistance"
    )


if breakout_status == "ABOVE_RESISTANCE":

    reasons.append(
        "Price is above previous resistance"
    )


# ------------------------------------------
# Rule 4 — R:R
# ------------------------------------------

if risk_reward < 1.5:

    valid_setup = False

    reasons.append(
        "Risk/reward below minimum threshold"
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
print("     XAUUSD SETUP VALIDATION V2")
print("========================================")

print(f"\nPrice              : {price:.3f}")
print(f"EMA20              : {ema20:.3f}")
print(f"EMA50              : {ema50:.3f}")

print(f"\nRSI14              : {rsi:.2f}")
print(f"ATR14              : {atr:.3f}")

print(f"\nPrevious Support   : {support:.3f}")
print(f"Previous Resistance: {resistance:.3f}")

print(f"\nTrend              : {trend}")
print(f"Momentum           : {momentum}")

print("\n--- STRUCTURE ---")

print(
    f"Distance to Resistance : "
    f"{distance_to_resistance:.3f}"
)

print(
    f"Distance to Support    : "
    f"{distance_to_support:.3f}"
)

print(
    f"Breakout Status        : "
    f"{breakout_status}"
)

print("\n--- TRADE PARAMETERS ---")

print(f"Entry                  : {entry:.3f}")
print(f"Stop Loss              : {stop_loss:.3f}")
print(f"Take Profit            : {take_profit:.3f}")
print(f"Risk                   : {risk:.3f}")
print(f"Reward                 : {reward:.3f}")
print(f"Risk/Reward            : 1 : {risk_reward:.2f}")

print("\n--- VALIDATION ---")

for reason in reasons:

    print("⚠️", reason)

print(f"\nFINAL DECISION         : {decision}")

print("========================================")
