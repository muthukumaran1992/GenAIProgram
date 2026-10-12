import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators


# ============================================================
# CONFIGURATION
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 2.0

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
    #
    # IMPORTANT:
    # Only candles BEFORE the signal candle are used.
    # This avoids look-ahead bias.
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
    # V2-A IMPROVED BREAKOUT
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
    # PULLBACK
    #
    # IMPORTANT:
    # Pullback logic is IDENTICAL to V1.
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
    # FINAL SIGNAL
    # ========================================================

    if bullish_breakout:

        df.loc[i, "signal"] = "BUY"
        df.loc[i, "setup_type"] = "BREAKOUT"

    elif bearish_breakout:

        df.loc[i, "signal"] = "SELL"
        df.loc[i, "setup_type"] = "BREAKOUT"

    elif bullish_pullback:

        df.loc[i, "signal"] = "BUY"
        df.loc[i, "setup_type"] = "PULLBACK"

    elif bearish_pullback:

        df.loc[i, "signal"] = "SELL"
        df.loc[i, "setup_type"] = "PULLBACK"


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
    # STOP LOSS / TAKE PROFIT
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

        high = float(
            future["high"]
        )

        low = float(
            future["low"]
        )

        if signal == "BUY":

            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

            # Conservative assumption:
            # SL is assumed first if both are hit.

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
    # STORE TRADE
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
                float(
                    df.loc[i, "rsi14"]
                ),

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

    # Move after completed trade

    i = exit_index + 1


trades_df = pd.DataFrame(
    trades
)


# ============================================================
# BASIC RESULTS
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
# BUY / SELL
# ============================================================

def direction_stats(
    direction
):

    subset = resolved[
        resolved["direction"] == direction
    ]

    if subset.empty:

        return (
            0,
            0,
            0,
            0
        )

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

    return (
        count,
        wr,
        total,
        avg
    )


buy_stats = direction_stats(
    "BUY"
)

sell_stats = direction_stats(
    "SELL"
)


# ============================================================
# SETUP STATS
# ============================================================

def setup_stats(
    setup
):

    subset = resolved[
        resolved["setup_type"] == setup
    ]

    if subset.empty:

        return (
            0,
            0,
            0,
            0
        )

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

    return (
        count,
        wr,
        total,
        avg
    )


breakout_stats = setup_stats(
    "BREAKOUT"
)

pullback_stats = setup_stats(
    "PULLBACK"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print(
    "              XAUUSD V2-A BREAKOUT TEST"
)
print("=" * 70)

print("\nOVERALL")
print("-" * 70)

print(
    f"Total trades       : {len(trades_df):,}"
)

print(
    f"Resolved trades    : {resolved_count:,}"
)

print(
    f"Wins               : {wins:,}"
)

print(
    f"Losses             : {losses:,}"
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


print("\nDIRECTION")
print("-" * 70)

print(
    f"BUY                : "
    f"Trades={buy_stats[0]:,} "
    f"WinRate={buy_stats[1]:.2f}% "
    f"TotalR={buy_stats[2]:.2f} "
    f"AvgR={buy_stats[3]:.4f}"
)

print(
    f"SELL               : "
    f"Trades={sell_stats[0]:,} "
    f"WinRate={sell_stats[1]:.2f}% "
    f"TotalR={sell_stats[2]:.2f} "
    f"AvgR={sell_stats[3]:.4f}"
)


print("\nSETUP TYPE")
print("-" * 70)

print(
    f"BREAKOUT           : "
    f"Trades={breakout_stats[0]:,} "
    f"WinRate={breakout_stats[1]:.2f}% "
    f"TotalR={breakout_stats[2]:.2f} "
    f"AvgR={breakout_stats[3]:.4f}"
)

print(
    f"PULLBACK           : "
    f"Trades={pullback_stats[0]:,} "
    f"WinRate={pullback_stats[1]:.2f}% "
    f"TotalR={pullback_stats[2]:.2f} "
    f"AvgR={pullback_stats[3]:.4f}"
)


# ============================================================
# SETUP + DIRECTION
# ============================================================

print("\nSETUP + DIRECTION")
print("-" * 70)

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
# COMPARISON WITH V1
# ============================================================

print("\nV1 vs V2-A")
print("-" * 70)

print(
    "                         V1          V2-A"
)

print(
    f"Trades              "
    f"{3318:>8,}   "
    f"{resolved_count:>10,}"
)

print(
    f"Win rate            "
    f"{34.36:>8.2f}%   "
    f"{win_rate:>9.2f}%"
)

print(
    f"Total R             "
    f"{102.00:>8.2f}   "
    f"{total_r:>10.2f}"
)

print(
    f"Average R           "
    f"{0.0307:>8.4f}   "
    f"{average_r:>10.4f}"
)

print(
    f"Profit factor       "
    f"{1.047:>8.3f}   "
    f"{profit_factor:>10.3f}"
)


print("\n")
print("=" * 70)
print(
    "              V2-A TEST COMPLETE"
)
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "Only the breakout confirmation logic "
    "was changed."
)

print(
    "Pullback logic, ATR risk, 2:1 target, "
    "and execution assumptions remain unchanged."
)

print("=" * 70)