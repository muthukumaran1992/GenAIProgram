import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))


# ============================================================
# LOAD DATA
# ============================================================

# IMPORTANT:
# This script recreates the V1 trades using the same logic
# from 03_trade_simulator.py.

from indicators.technical_indicators import add_indicators


file_path = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "XAUUSD_m15_20240101_20260930.csv"
)

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()

time_column = "timestamp" if "timestamp" in df.columns else "date"
required_columns = {time_column, "open", "high", "low", "close"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in {file_path.name}: "
        f"{sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError(f"No market data found in {file_path.name}.")

if time_column == "timestamp":
    df["datetime"] = pd.to_datetime(
        df[time_column],
        unit="ms",
        errors="coerce",
    )
else:
    df["datetime"] = pd.to_datetime(
        df[time_column],
        errors="coerce",
    )

df = df.dropna(subset=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

if df.empty:
    raise ValueError(
        f"No valid datetimes found in {file_path.name}."
    )

df = add_indicators(df)


# ============================================================
# SIGNAL GENERATION
# ============================================================

df["signal"] = "WAIT"

for i in range(50, len(df) - 1):

    current = df.iloc[i]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    # ----------------------------
    # Market trend
    # ----------------------------

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    # ----------------------------
    # Support / resistance
    # ----------------------------

    previous_candles = df.iloc[i - 20:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # ----------------------------
    # Breakout
    # ----------------------------

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
    )

    # ----------------------------
    # Pullback
    # ----------------------------

    bullish_pullback = (
        trend == "BULLISH"
        and current["low"] <= ema20
        and price > ema20
        and bullish_candle
        and rsi < 70
    )

    bearish_pullback = (
        trend == "BEARISH"
        and current["high"] >= ema20
        and price < ema20
        and bearish_candle
        and rsi > 30
    )

    # ----------------------------
    # Final signal
    # ----------------------------

    if bullish_breakout or bullish_pullback:

        df.loc[i, "signal"] = "BUY"

    elif bearish_breakout or bearish_pullback:

        df.loc[i, "signal"] = "SELL"


# ============================================================
# TRADE SIMULATION
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 2.0

trades = []

i = 50

while i < len(df) - 1:

    signal = df.loc[i, "signal"]

    if signal == "WAIT":

        i += 1
        continue

    # ----------------------------
    # Entry
    # ----------------------------

    entry_index = i + 1

    entry_candle = df.iloc[entry_index]

    entry_time = entry_candle["datetime"]
    entry_price = float(entry_candle["open"])

    atr = float(df.loc[i, "atr14"])

    risk = ATR_STOP_MULTIPLIER * atr

    # ----------------------------
    # Stop / target
    # ----------------------------

    if signal == "BUY":

        stop_loss = entry_price - risk
        take_profit = entry_price + (
            risk * REWARD_MULTIPLIER
        )

    else:

        stop_loss = entry_price + risk
        take_profit = entry_price - (
            risk * REWARD_MULTIPLIER
        )

    # ----------------------------
    # Find exit
    # ----------------------------

    exit_index = None
    exit_price = None
    result = None
    r_multiple = None

    for j in range(entry_index, len(df)):

        future = df.iloc[j]

        high = float(future["high"])
        low = float(future["low"])

        if signal == "BUY":

            stop_hit = low <= stop_loss
            target_hit = high >= take_profit

            # Conservative assumption:
            # if both are hit in same candle,
            # assume SL happened first.

            if stop_hit:

                exit_index = j
                exit_price = stop_loss
                result = "LOSS"
                r_multiple = -1.0
                break

            elif target_hit:

                exit_index = j
                exit_price = take_profit
                result = "WIN"
                r_multiple = REWARD_MULTIPLIER
                break

        else:

            stop_hit = high >= stop_loss
            target_hit = low <= take_profit

            if stop_hit:

                exit_index = j
                exit_price = stop_loss
                result = "LOSS"
                r_multiple = -1.0
                break

            elif target_hit:

                exit_index = j
                exit_price = take_profit
                result = "WIN"
                r_multiple = REWARD_MULTIPLIER
                break

    # ----------------------------
    # Open at end
    # ----------------------------

    if exit_index is None:

        exit_index = len(df) - 1

        exit_price = float(
            df.iloc[-1]["close"]
        )

        if signal == "BUY":

            pnl = exit_price - entry_price

        else:

            pnl = entry_price - exit_price

        r_multiple = pnl / risk

        result = "OPEN_AT_END"

    # ----------------------------
    # Store trade
    # ----------------------------

    trades.append(
        {
            "signal_time": df.loc[i, "datetime"],
            "entry_time": entry_time,
            "exit_time": df.loc[exit_index, "datetime"],
            "direction": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_price": exit_price,
            "atr": atr,
            "risk": risk,
            "result": result,
            "r_multiple": r_multiple,
        }
    )

    # Important:
    # Only one position at a time.

    i = exit_index + 1


trades_df = pd.DataFrame(trades)


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("             XAUUSD V1 PERFORMANCE ANALYSIS")
print("=" * 60)


if trades_df.empty:

    print("No trades generated.")
    sys.exit()


# ============================================================
# BASIC STATISTICS
# ============================================================

resolved = trades_df[
    trades_df["result"].isin(["WIN", "LOSS"])
].copy()

wins = resolved[
    resolved["result"] == "WIN"
]

losses = resolved[
    resolved["result"] == "LOSS"
]

open_trades = trades_df[
    trades_df["result"] == "OPEN_AT_END"
]


total_trades = len(trades_df)

resolved_trades = len(resolved)

win_count = len(wins)

loss_count = len(losses)


win_rate = (
    win_count / resolved_trades * 100
    if resolved_trades > 0
    else 0
)

loss_rate = (
    loss_count / resolved_trades * 100
    if resolved_trades > 0
    else 0
)


total_r = resolved["r_multiple"].sum()

average_r = (
    resolved["r_multiple"].mean()
    if resolved_trades > 0
    else 0
)


average_win = (
    wins["r_multiple"].mean()
    if not wins.empty
    else 0
)

average_loss = (
    losses["r_multiple"].mean()
    if not losses.empty
    else 0
)


gross_profit = wins["r_multiple"].sum()

gross_loss = abs(
    losses["r_multiple"].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else float("inf")
)


# ============================================================
# PRINT CORE METRICS
# ============================================================

print("\nCORE METRICS")
print("-" * 60)

print(f"Total trades       : {total_trades:,}")
print(f"Resolved trades    : {resolved_trades:,}")
print(f"Wins               : {win_count:,}")
print(f"Losses             : {loss_count:,}")
print(f"Open at end        : {len(open_trades):,}")

print(f"\nWin rate           : {win_rate:.2f}%")
print(f"Loss rate          : {loss_rate:.2f}%")

print(f"\nTotal R            : {total_r:.2f}R")
print(f"Average R/trade    : {average_r:.4f}R")

print(f"\nAverage win        : {average_win:.2f}R")
print(f"Average loss       : {average_loss:.2f}R")

print(f"\nProfit factor      : {profit_factor:.3f}")


# ============================================================
# EXPECTANCY
# ============================================================

print("\nEXPECTANCY")
print("-" * 60)

win_probability = win_rate / 100

loss_probability = loss_rate / 100

expectancy = (
    win_probability * average_win
    + loss_probability * average_loss
)

print(f"Win probability    : {win_probability:.4f}")
print(f"Loss probability   : {loss_probability:.4f}")

print(f"Expectancy         : {expectancy:.4f}R/trade")


# ============================================================
# HOLDING TIME
# ============================================================

trades_df["holding_time"] = (
    trades_df["exit_time"]
    - trades_df["entry_time"]
)

resolved["holding_time"] = (
    resolved["exit_time"]
    - resolved["entry_time"]
)

holding_hours = (
    resolved["holding_time"]
    .dt.total_seconds()
    / 3600
)

print("\nHOLDING TIME")
print("-" * 60)

print(
    f"Average holding    : "
    f"{holding_hours.mean():.2f} hours"
)

print(
    f"Median holding     : "
    f"{holding_hours.median():.2f} hours"
)

print(
    f"Maximum holding    : "
    f"{holding_hours.max():.2f} hours"
)


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

resolved["equity_r"] = (
    resolved["r_multiple"].cumsum()
)

resolved["peak"] = (
    resolved["equity_r"].cummax()
)

resolved["drawdown"] = (
    resolved["equity_r"]
    - resolved["peak"]
)

max_drawdown = (
    resolved["drawdown"].min()
)

print("\nDRAWDOWN")
print("-" * 60)

print(
    f"Final equity       : "
    f"{resolved['equity_r'].iloc[-1]:.2f}R"
)

print(
    f"Maximum drawdown   : "
    f"{max_drawdown:.2f}R"
)


# ============================================================
# CONSECUTIVE LOSSES
# ============================================================

max_consecutive_losses = 0
current_losses = 0

for result in resolved["result"]:

    if result == "LOSS":

        current_losses += 1

        max_consecutive_losses = max(
            max_consecutive_losses,
            current_losses
        )

    else:

        current_losses = 0


print("\nLOSING STREAK")
print("-" * 60)

print(
    f"Maximum consecutive losses : "
    f"{max_consecutive_losses}"
)


# ============================================================
# BUY vs SELL
# ============================================================

print("\nBUY vs SELL")
print("-" * 60)

for direction in ["BUY", "SELL"]:

    subset = resolved[
        resolved["direction"] == direction
    ]

    if subset.empty:
        continue

    direction_wins = (
        subset["result"] == "WIN"
    ).sum()

    direction_win_rate = (
        direction_wins
        / len(subset)
        * 100
    )

    direction_r = (
        subset["r_multiple"].sum()
    )

    direction_expectancy = (
        subset["r_multiple"].mean()
    )

    print(f"\n{direction}")

    print(
        f"Trades             : {len(subset):,}"
    )

    print(
        f"Win rate           : "
        f"{direction_win_rate:.2f}%"
    )

    print(
        f"Total R            : "
        f"{direction_r:.2f}R"
    )

    print(
        f"Average R          : "
        f"{direction_expectancy:.4f}R"
    )


# ============================================================
# YEARLY PERFORMANCE
# ============================================================

resolved["year"] = (
    resolved["entry_time"].dt.year
)

print("\nYEARLY PERFORMANCE")
print("-" * 60)

yearly = (
    resolved
    .groupby("year")
    .agg(
        trades=("r_multiple", "count"),
        total_r=("r_multiple", "sum"),
        average_r=("r_multiple", "mean"),
        wins=("result", lambda x:
              (x == "WIN").sum()),
        losses=("result", lambda x:
                (x == "LOSS").sum())
    )
)

yearly["win_rate"] = (
    yearly["wins"]
    / yearly["trades"]
    * 100
)

print(
    yearly.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# MONTHLY PERFORMANCE
# ============================================================

resolved["month"] = (
    resolved["entry_time"]
    .dt.to_period("M")
)

monthly = (
    resolved
    .groupby("month")["r_multiple"]
    .agg(
        trades="count",
        total_r="sum",
        average_r="mean"
    )
)

print("\nMONTHLY PERFORMANCE")
print("-" * 60)

print(
    monthly
    .sort_values("total_r")
    .head(10)
    .to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nBest 10 months:")

print(
    monthly
    .sort_values("total_r", ascending=False)
    .head(10)
    .to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("                 V1 SUMMARY")
print("=" * 60)

print(
    f"Total R            : {total_r:.2f}R"
)

print(
    f"Win rate           : {win_rate:.2f}%"
)

print(
    f"Profit factor      : {profit_factor:.3f}"
)

print(
    f"Expectancy         : {expectancy:.4f}R"
)

print(
    f"Max drawdown       : {max_drawdown:.2f}R"
)

print(
    f"Max losing streak  : "
    f"{max_consecutive_losses}"
)

print("=" * 60)