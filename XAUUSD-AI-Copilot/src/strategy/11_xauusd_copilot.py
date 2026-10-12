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

file_path = PROJECT_DIR / "data" / "raw" / "xauusd-m15-bid-2025-03-19-2025-03-20.csv"

df = pd.read_csv(file_path)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ==========================================
# 2. TECHNICAL INDICATORS
# ==========================================

df = add_indicators(df)

if df.empty:
    raise ValueError(f"No market data found in {file_path}")

latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest
price = float(latest["close"])
ema20 = float(latest["ema20"])
ema50 = float(latest["ema50"])
rsi = float(latest["rsi14"])
atr = float(latest["atr14"])

if price > ema20 and ema20 > ema50:
    trend = "BULLISH"

elif price < ema20 and ema20 < ema50:
    trend = "BEARISH"

else:
    trend = "SIDEWAYS"

if rsi >= 70:
    momentum = "OVERBOUGHT"
elif rsi <= 30:
    momentum = "OVERSOLD"
else:
    momentum = "NEUTRAL"

recent_candles = df.iloc[max(0, len(df) - 21):-1]
if recent_candles.empty:
    recent_candles = df.tail(1)

support = float(recent_candles["low"].min())
resistance = float(recent_candles["high"].max())

if latest["high"] > previous["high"]:
    high_structure = "HIGHER HIGH"
elif latest["high"] < previous["high"]:
    high_structure = "LOWER HIGH"
else:
    high_structure = "EQUAL HIGH"

if latest["low"] > previous["low"]:
    low_structure = "HIGHER LOW"
elif latest["low"] < previous["low"]:
    low_structure = "LOWER LOW"
else:
    low_structure = "EQUAL LOW"

if high_structure == "HIGHER HIGH" and low_structure == "HIGHER LOW":
    market_structure = "BULLISH"
elif high_structure == "LOWER HIGH" and low_structure == "LOWER LOW":
    market_structure = "BEARISH"
else:
    market_structure = "MIXED"

if latest["close"] > latest["open"]:
    candle_direction = "BULLISH"
elif latest["close"] < latest["open"]:
    candle_direction = "BEARISH"
else:
    candle_direction = "DOJI"

bullish_breakout = (
    price > resistance
    and previous["close"] <= resistance
)

bearish_breakout = (
    price < support
    and previous["close"] >= support
)

breakout_confirmed = bullish_breakout or bearish_breakout

pullback_confirmed = (
    (trend == "BULLISH" and latest["low"] <= ema20 and price > ema20)
    or
    (trend == "BEARISH" and latest["high"] >= ema20 and price < ema20)
)

if breakout_confirmed:
    setup_type = "BREAKOUT"
elif pullback_confirmed:
    setup_type = "PULLBACK"
else:
    setup_type = "NO CONFIRMED SETUP"

direction = trend if trend in {"BULLISH", "BEARISH"} else market_structure
if direction == "BULLISH":
    stop_loss = price - (1.5 * atr)
    risk = price - stop_loss
    take_profit = price + (2 * risk)
    decision = "BUY" if setup_type != "NO CONFIRMED SETUP" else "WAIT"
elif direction == "BEARISH":
    stop_loss = price + (1.5 * atr)
    risk = stop_loss - price
    take_profit = price - (2 * risk)
    decision = "SELL" if setup_type != "NO CONFIRMED SETUP" else "WAIT"
else:
    stop_loss = price - (1.5 * atr)
    risk = price - stop_loss
    take_profit = price + (2 * risk)
    decision = "WAIT"

entry = price
reward = abs(take_profit - entry)
risk_reward = reward / risk if risk else 0.0

print("\n")
print("================================================")
print("             XAUUSD AI TRADING COPILOT")
print("================================================")

print(f"\nAnalysis Time      : {latest['datetime']}")

print("\n--- MARKET ---")

print(f"Price              : {price:.3f}")
print(f"EMA20              : {ema20:.3f}")
print(f"EMA50              : {ema50:.3f}")

print(f"RSI14              : {rsi:.2f}")
print(f"ATR14              : {atr:.3f}")

print(f"\nTrend              : {trend}")
print(f"Momentum           : {momentum}")

print(f"\nSupport            : {support:.3f}")
print(f"Resistance         : {resistance:.3f}")

print("\n--- MARKET STRUCTURE ---")

print(f"High Structure     : {high_structure}")
print(f"Low Structure      : {low_structure}")
print(f"Market Structure   : {market_structure}")

print("\n--- ENTRY ANALYSIS ---")

print(f"Candle Direction   : {candle_direction}")

print(
    f"Breakout Confirmed : "
    f"{breakout_confirmed}"
)

print(
    f"Pullback Confirmed : "
    f"{pullback_confirmed}"
)

print(f"Setup Type         : {setup_type}")

print("\n--- RISK MANAGEMENT ---")

print(f"Entry              : {entry:.3f}")
print(f"Stop Loss          : {stop_loss:.3f}")
print(f"Take Profit        : {take_profit:.3f}")

print(f"Risk               : {risk:.3f}")
print(f"Reward             : {reward:.3f}")

print(
    f"Risk/Reward        : "
    f"1 : {risk_reward:.2f}"
)

print("\n================================================")
print(f"              FINAL DECISION : {decision}")
print("================================================")