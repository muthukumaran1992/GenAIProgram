import pandas as pd

# ============================================
# 1. Load XAUUSD historical data
# ============================================

file_path = r"data\raw\xauusd-m15-bid-2025-03-19-2025-03-20.csv"

df = pd.read_csv(file_path)

# ============================================
# 2. Convert timestamp to datetime
# ============================================

df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

# Sort by time
df = df.sort_values("datetime").reset_index(drop=True)

# ============================================
# 3. Calculate EMA 20 and EMA 50
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
# 4. Display results
# ============================================

print("\n===== XAUUSD EMA ANALYSIS =====")

print(
    df[
        [
            "datetime",
            "close",
            "ema20",
            "ema50"
        ]
    ].tail(20)
)

# ============================================
# 5. Determine basic trend
# ============================================

if df["ema20"].iloc[-1] > df["ema50"].iloc[-1]:
    trend = "BULLISH"
else:
    trend = "BEARISH"


# ============================================
# Calculate RSI 14
# ============================================

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss

df["rsi14"] = 100 - (100 / (1 + rs))

# ============================================
# Calculate ATR 14
# ============================================

previous_close = df["close"].shift(1)

# True Range components
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - previous_close).abs()
tr3 = (df["low"] - previous_close).abs()

# True Range = maximum of the three
df["true_range"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

# ATR 14
df["atr14"] = df["true_range"].rolling(window=14).mean()


print("\n===== XAUUSD TECHNICAL ANALYSIS =====")

print(
    df[
        [
            "datetime",
            "close",
            "ema20",
            "ema50",
            "rsi14",
            "atr14"
        ]
    ].tail(20)
)