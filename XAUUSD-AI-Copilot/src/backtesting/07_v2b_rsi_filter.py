import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators


# ============================================================
# CONFIGURATION
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 2.0

# V2-B RSI FILTER
RSI_MIN = 50
RSI_MAX = 70

file_path = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "XAUUSD_m15_20240101_20260930.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

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
df["setup_type"] = "NONE"
df["trend"] = "UNKNOWN"


for i in range(50, len(df) - 1):

    current = df.iloc[i]
    previous = df.iloc[i - 1]

    price = current["close"]
    previous_close = previous["close"]

    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if price > ema20 and ema20 > ema50:

        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS"

    df.loc[i, "trend"] = trend

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    previous_candles = df.iloc[i - 20:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()

    # --------------------------------------------------------
    # CANDLE DIRECTION
    # --------------------------------------------------------

    bullish_candle = (
        current["close"] > current["open"]
    )

    bearish_candle = (
        current["close"] < current["open"]
    )

    # ========================================================
    # V2-A BREAKOUT LOGIC
    # ========================================================

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
        and previous_close <= resistance
        and bullish_candle
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
        and previous_close >= support
        and bearish_candle
    )

    # ========================================================
    # V1 PULLBACK LOGIC
    # ========================================================

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

    # ========================================================
    # BASE SIGNAL
    # ========================================================

    base_signal = None
    base_setup = None

    if bullish_breakout:

        base_signal = "BUY"
        base_setup = "BREAKOUT"

    elif bearish_breakout:

        base_signal = "SELL"
        base_setup = "BREAKOUT"

    elif bullish_pullback:

        base_signal = "BUY"
        base_setup = "PULLBACK"

    elif bearish_pullback:

        base_signal = "SELL"
        base_setup = "PULLBACK"

    # ========================================================
    # V2-B RSI FILTER
    #
    # Only allow the signal if RSI is between 50 and 70.
    # ========================================================

    rsi_filter_passed = (
        RSI_MIN <= rsi <= RSI_MAX
    )

    if (
        base_signal is not None
        and rsi_filter_passed
    ):

        df.loc[i, "signal"] = base_signal
        df.loc[i, "setup_type"] = base_setup


# ============================================================
# TRADE SIMULATION
# ============================================================

trades = []

i = 50

while i < len(df) - 1:

    signal = df.loc[i, "signal"]

    if signal == "WAIT":

        i += 1
        continue

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    entry_index = i + 1

    entry_candle = df.iloc[entry_index]

    entry_time = entry_candle["datetime"]

    entry_price = float(
        entry_candle["open"]
    )

    atr = float(
        df.loc[i, "atr14"]
    )

    risk = (
        ATR_STOP_MULTIPLIER * atr
    )

    # --------------------------------------------------------
    # STOP / TARGET
    # --------------------------------------------------------

    if signal == "BUY":

        stop_loss = (
            entry_price - risk
        )

        take_profit = (
            entry_price
            + risk * REWARD_MULTIPLIER
        )

    else:

        stop_loss = (
            entry_price + risk
        )

        take_profit = (
            entry_price
            - risk * REWARD_MULTIPLIER
        )

    # --------------------------------------------------------
    # FIND EXIT
    # --------------------------------------------------------

    exit_index = None
    exit_price = None
    result = None
    r_multiple = None

    for j in range(
        entry_index,
        len(df)
    ):

        future = df.iloc[j]

        high = float(future["high"])
        low = float(future["low"])

        if signal == "BUY":

            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

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

            stop_hit = (
                high >= stop_loss
            )

            target_hit = (
                low <= take_profit
            )

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

    # --------------------------------------------------------
    # OPEN AT END
    # --------------------------------------------------------

    if exit_index is None:

        exit_index = len(df) - 1

        exit_price = float(
            df.iloc[-1]["close"]
        )

        if signal == "BUY":

            pnl = (
                exit_price
                - entry_price
            )

        else:

            pnl = (
                entry_price
                - exit_price
            )

        r_multiple = (
            pnl / risk
        )

        result = "OPEN_AT_END"

    # --------------------------------------------------------
    # SAVE TRADE
    # --------------------------------------------------------

    trades.append(
        {
            "signal_time":
                df.loc[i, "datetime"],

            "entry_time":
                entry_time,

            "exit_time":
                df.loc[exit_index, "datetime"],

            "direction":
                signal,

            "setup_type":
                df.loc[i, "setup_type"],

            "trend":
                df.loc[i, "trend"],

            "rsi":
                float(df.loc[i, "rsi14"]),

            "atr":
                atr,

            "entry_price":
                entry_price,

            "stop_loss":
                stop_loss,

            "take_profit":
                take_profit,

            "exit_price":
                exit_price,

            "result":
                result,

            "r_multiple":
                r_multiple
        }
    )

    i = exit_index + 1


trades_df = pd.DataFrame(trades)


# ============================================================
# PERFORMANCE
# ============================================================

resolved = trades_df[
    trades_df["result"].isin(
        ["WIN", "LOSS"]
    )
].copy()

wins = (
    resolved["result"] == "WIN"
).sum()

losses = (
    resolved["result"] == "LOSS"
).sum()

resolved_count = len(resolved)

win_rate = (
    wins / resolved_count * 100
    if resolved_count
    else 0
)

total_r = (
    resolved["r_multiple"].sum()
)

average_r = (
    resolved["r_multiple"].mean()
    if resolved_count
    else 0
)

gross_profit = resolved.loc[
    resolved["r_multiple"] > 0,
    "r_multiple"
].sum()

gross_loss = abs(
    resolved.loc[
        resolved["r_multiple"] < 0,
        "r_multiple"
    ].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else float("inf")
)


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

if not resolved.empty:

    equity = (
        resolved["r_multiple"]
        .cumsum()
    )

    peak = equity.cummax()

    drawdown = (
        equity - peak
    )

    max_drawdown = drawdown.min()

else:

    max_drawdown = 0


# ============================================================
# BUY / SELL STATISTICS
# ============================================================

def get_direction_stats(direction):

    subset = resolved[
        resolved["direction"] == direction
    ]

    if subset.empty:

        return 0, 0, 0, 0

    count = len(subset)

    wins_count = (
        subset["result"] == "WIN"
    ).sum()

    wr = (
        wins_count / count * 100
    )

    total = (
        subset["r_multiple"].sum()
    )

    avg = (
        subset["r_multiple"].mean()
    )

    return count, wr, total, avg


buy = get_direction_stats("BUY")
sell = get_direction_stats("SELL")


# ============================================================
# SETUP STATISTICS
# ============================================================

def get_setup_stats(setup):

    subset = resolved[
        resolved["setup_type"] == setup
    ]

    if subset.empty:

        return 0, 0, 0, 0

    count = len(subset)

    wins_count = (
        subset["result"] == "WIN"
    ).sum()

    wr = (
        wins_count / count * 100
    )

    total = (
        subset["r_multiple"].sum()
    )

    avg = (
        subset["r_multiple"].mean()
    )

    return count, wr, total, avg


breakout = get_setup_stats(
    "BREAKOUT"
)

pullback = get_setup_stats(
    "PULLBACK"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("                 XAUUSD V2-B RSI FILTER")
print("=" * 75)

print("\nCONFIGURATION")
print("-" * 75)

print(
    f"RSI range          : "
    f"{RSI_MIN} - {RSI_MAX}"
)

print(
    f"ATR stop multiplier: "
    f"{ATR_STOP_MULTIPLIER}"
)

print(
    f"Reward multiplier  : "
    f"{REWARD_MULTIPLIER}"
)


print("\nOVERALL")
print("-" * 75)

print(
    f"Total trades       : "
    f"{len(trades_df):,}"
)

print(
    f"Resolved trades    : "
    f"{resolved_count:,}"
)

print(
    f"Wins               : "
    f"{wins:,}"
)

print(
    f"Losses             : "
    f"{losses:,}"
)

print(
    f"Open at end        : "
    f"{len(trades_df) - resolved_count:,}"
)

print(
    f"\nWin rate           : "
    f"{win_rate:.2f}%"
)

print(
    f"Total R            : "
    f"{total_r:.2f}R"
)

print(
    f"Average R/trade    : "
    f"{average_r:.4f}R"
)

print(
    f"Profit factor      : "
    f"{profit_factor:.3f}"
)

print(
    f"Maximum drawdown   : "
    f"{max_drawdown:.2f}R"
)


print("\nDIRECTION")
print("-" * 75)

print(
    f"BUY                : "
    f"Trades={buy[0]:,} "
    f"WinRate={buy[1]:.2f}% "
    f"TotalR={buy[2]:.2f} "
    f"AvgR={buy[3]:.4f}"
)

print(
    f"SELL               : "
    f"Trades={sell[0]:,} "
    f"WinRate={sell[1]:.2f}% "
    f"TotalR={sell[2]:.2f} "
    f"AvgR={sell[3]:.4f}"
)


print("\nSETUP TYPE")
print("-" * 75)

print(
    f"BREAKOUT           : "
    f"Trades={breakout[0]:,} "
    f"WinRate={breakout[1]:.2f}% "
    f"TotalR={breakout[2]:.2f} "
    f"AvgR={breakout[3]:.4f}"
)

print(
    f"PULLBACK           : "
    f"Trades={pullback[0]:,} "
    f"WinRate={pullback[1]:.2f}% "
    f"TotalR={pullback[2]:.2f} "
    f"AvgR={pullback[3]:.4f}"
)


# ============================================================
# SETUP + DIRECTION
# ============================================================

print("\nSETUP + DIRECTION")
print("-" * 75)

for setup in [
    "BREAKOUT",
    "PULLBACK"
]:

    for direction in [
        "BUY",
        "SELL"
    ]:

        subset = resolved[
            (resolved["setup_type"] == setup)
            &
            (resolved["direction"] == direction)
        ]

        if subset.empty:
            continue

        count = len(subset)

        wins_count = (
            subset["result"] == "WIN"
        ).sum()

        wr = (
            wins_count / count * 100
        )

        total = (
            subset["r_multiple"].sum()
        )

        avg = (
            subset["r_multiple"].mean()
        )

        print(
            f"{direction} {setup:<12} : "
            f"Trades={count:>5} "
            f"WinRate={wr:>6.2f}% "
            f"TotalR={total:>8.2f} "
            f"AvgR={avg:>7.4f}"
        )


# ============================================================
# V1 / V2-A / V2-B
# ============================================================

print("\nSTRATEGY COMPARISON")
print("-" * 75)

print(
    "                       V1       V2-A       V2-B"
)

print(
    f"Trades             "
    f"{3318:>8,} "
    f"{3316:>10,} "
    f"{resolved_count:>10,}"
)

print(
    f"Win rate           "
    f"{34.36:>8.2f}% "
    f"{34.32:>10.2f}% "
    f"{win_rate:>10.2f}%"
)

print(
    f"Total R            "
    f"{102.00:>8.2f} "
    f"{98.00:>10.2f} "
    f"{total_r:>10.2f}"
)

print(
    f"Average R          "
    f"{0.0307:>8.4f} "
    f"{0.0296:>10.4f} "
    f"{average_r:>10.4f}"
)

print(
    f"Profit factor      "
    f"{1.047:>8.3f} "
    f"{1.045:>10.3f} "
    f"{profit_factor:>10.3f}"
)


print("\n")
print("=" * 75)
print("                 V2-B TEST COMPLETE")
print("=" * 75)

print(
    "\nOnly the RSI filter was added to V2-A."
)

print(
    "All other strategy and execution assumptions "
    "remain unchanged."
)

print("=" * 75)