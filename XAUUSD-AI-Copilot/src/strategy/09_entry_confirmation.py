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
# CURRENT + PREVIOUS CANDLE
# ==========================================

latest = df.iloc[-1]
previous = df.iloc[-2]

price = latest["close"]
previous_close_price = previous["close"]

high = latest["high"]
low = latest["low"]

previous_high = previous["high"]
previous_low = previous["low"]

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
# CANDLE DIRECTION
# ==========================================

if price > latest["open"]:

    candle_direction = "BULLISH"

elif price < latest["open"]:

    candle_direction = "BEARISH"

else:

    candle_direction = "NEUTRAL"


# ==========================================
# BREAKOUT CONFIRMATION
# ==========================================

breakout_confirmed = False

if (
    price > resistance
    and previous_close_price <= resistance
):

    breakout_confirmed = True


# ==========================================
# PULLBACK CONFIRMATION
# ==========================================

pullback_confirmed = False

# Price touched / approached EMA20
ema_distance = abs(price - ema20)

if (
    trend == "BULLISH"
    and candle_direction == "BULLISH"
    and price > ema20
    and ema_distance <= atr
):

    pullback_confirmed = True


# ==========================================
# ENTRY SIGNAL
# ==========================================

if breakout_confirmed:

    setup_type = "BREAKOUT BUY"

    entry_signal = "BUY"

elif pullback_confirmed:

    setup_type = "PULLBACK BUY"

    entry_signal = "BUY"

else:

    setup_type = "NO CONFIRMED SETUP"

    entry_signal = "WAIT"


# ==========================================
# OUTPUT
# ==========================================

print("\n========================================")
print("       XAUUSD ENTRY CONFIRMATION")
print("========================================")

print(f"\nPrice              : {price:.3f}")

print(f"EMA20              : {ema20:.3f}")
print(f"EMA50              : {ema50:.3f}")

print(f"RSI14              : {rsi:.2f}")
print(f"ATR14              : {atr:.3f}")

print(f"\nSupport             : {support:.3f}")
print(f"Resistance          : {resistance:.3f}")

print(f"\nTrend               : {trend}")

print(
    f"Candle Direction   : "
    f"{candle_direction}"
)

print("\n--- CONFIRMATION ---")

print(
    f"Breakout Confirmed : "
    f"{breakout_confirmed}"
)

print(
    f"Pullback Confirmed : "
    f"{pullback_confirmed}"
)

print(
    f"\nSetup Type         : "
    f"{setup_type}"
)

print(
    f"Entry Signal       : "
    f"{entry_signal}"
)

print("========================================")