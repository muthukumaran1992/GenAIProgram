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

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if price > ema20 and ema20 > ema50:

        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS"

    df.loc[i, "trend"] = trend

    # --------------------------------------------------------
    # Support / resistance
    # --------------------------------------------------------

    previous_candles = df.iloc[i - 20:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()

    bullish_candle = (
        current["close"] > current["open"]
    )

    bearish_candle = (
        current["close"] < current["open"]
    )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
    )

    # --------------------------------------------------------
    # Pullback
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

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
    # Entry
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
    # Stop / target
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
    # Exit
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
    # Open at dataset end
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
    # Trade record
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

    # Move to next available trade

    i = exit_index + 1


trades_df = pd.DataFrame(
    trades
)


# ============================================================
# FILTER RESOLVED TRADES
# ============================================================

resolved = trades_df[
    trades_df["result"].isin(
        ["WIN", "LOSS"]
    )
].copy()


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_group(
    data,
    group_name
):

    if data.empty:

        print(
            f"\n{group_name}"
        )

        print(
            "No trades."
        )

        return

    trades_count = len(data)

    wins = (
        data["result"] == "WIN"
    ).sum()

    losses = (
        data["result"] == "LOSS"
    ).sum()

    win_rate = (
        wins / trades_count * 100
    )

    total_r = (
        data["r_multiple"].sum()
    )

    average_r = (
        data["r_multiple"].mean()
    )

    gross_profit = data.loc[
        data["r_multiple"] > 0,
        "r_multiple"
    ].sum()

    gross_loss = abs(
        data.loc[
            data["r_multiple"] < 0,
            "r_multiple"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    print(
        f"{group_name:<25}"
        f" Trades={trades_count:>5} "
        f"WinRate={win_rate:>6.2f}% "
        f"TotalR={total_r:>8.2f} "
        f"AvgR={average_r:>7.4f} "
        f"PF={profit_factor:>6.3f}"
    )


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 95)
print(
    "                 XAUUSD V1 DIAGNOSTIC ANALYSIS"
)
print("=" * 95)


# ============================================================
# OVERALL
# ============================================================

print("\nOVERALL")
print("-" * 95)

analyze_group(
    resolved,
    "ALL TRADES"
)


# ============================================================
# SETUP TYPE
# ============================================================

print("\nSETUP TYPE")
print("-" * 95)

for setup in [
    "BREAKOUT",
    "PULLBACK"
]:

    subset = resolved[
        resolved["setup_type"] == setup
    ]

    analyze_group(
        subset,
        setup
    )


# ============================================================
# DIRECTION
# ============================================================

print("\nDIRECTION")
print("-" * 95)

for direction in [
    "BUY",
    "SELL"
]:

    subset = resolved[
        resolved["direction"] == direction
    ]

    analyze_group(
        subset,
        direction
    )


# ============================================================
# SETUP + DIRECTION
# ============================================================

print("\nSETUP + DIRECTION")
print("-" * 95)

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

        analyze_group(
            subset,
            f"{direction} {setup}"
        )


# ============================================================
# TREND
# ============================================================

print("\nTREND")
print("-" * 95)

for trend in [
    "BULLISH",
    "BEARISH",
    "SIDEWAYS"
]:

    subset = resolved[
        resolved["trend"] == trend
    ]

    analyze_group(
        subset,
        trend
    )


# ============================================================
# RSI BUCKETS
# ============================================================

resolved["rsi_zone"] = pd.cut(
    resolved["rsi"],
    bins=[
        0,
        40,
        50,
        60,
        70,
        100
    ],
    labels=[
        "<40",
        "40-50",
        "50-60",
        "60-70",
        ">70"
    ],
    include_lowest=True
)

print("\nRSI ZONES")
print("-" * 95)

for zone in [
    "<40",
    "40-50",
    "50-60",
    "60-70",
    ">70"
]:

    subset = resolved[
        resolved["rsi_zone"] == zone
    ]

    analyze_group(
        subset,
        f"RSI {zone}"
    )


# ============================================================
# ATR VOLATILITY BUCKETS
# ============================================================

resolved["atr_zone"] = pd.qcut(
    resolved["atr"],
    q=4,
    labels=[
        "LOW",
        "MEDIUM-LOW",
        "MEDIUM-HIGH",
        "HIGH"
    ],
    duplicates="drop"
)

print("\nATR VOLATILITY ZONES")
print("-" * 95)

for zone in [
    "LOW",
    "MEDIUM-LOW",
    "MEDIUM-HIGH",
    "HIGH"
]:

    subset = resolved[
        resolved["atr_zone"] == zone
    ]

    analyze_group(
        subset,
        f"ATR {zone}"
    )


# ============================================================
# SESSION
# ============================================================

# The dataset timestamp is treated as UTC here.
# We will explicitly examine the timestamp distribution
# before changing this assumption.

resolved["hour_utc"] = (
    resolved["entry_time"].dt.hour
)


def get_session(hour):

    if 0 <= hour < 7:

        return "ASIA"

    elif 7 <= hour < 12:

        return "LONDON"

    elif 12 <= hour < 16:

        return "LONDON_NY"

    elif 16 <= hour < 21:

        return "NEW_YORK"

    else:

        return "OTHER"


resolved["session"] = (
    resolved["hour_utc"]
    .apply(get_session)
)


print("\nTRADING SESSION")
print("-" * 95)

for session in [
    "ASIA",
    "LONDON",
    "LONDON_NY",
    "NEW_YORK",
    "OTHER"
]:

    subset = resolved[
        resolved["session"] == session
    ]

    analyze_group(
        subset,
        session
    )


# ============================================================
# YEAR + DIRECTION
# ============================================================

print("\nYEAR + DIRECTION")
print("-" * 95)

resolved["year"] = (
    resolved["entry_time"].dt.year
)

for year in sorted(
    resolved["year"].unique()
):

    for direction in [
        "BUY",
        "SELL"
    ]:

        subset = resolved[
            (resolved["year"] == year)
            &
            (resolved["direction"] == direction)
        ]

        analyze_group(
            subset,
            f"{year} {direction}"
        )


# ============================================================
# YEAR + SETUP
# ============================================================

print("\nYEAR + SETUP")
print("-" * 95)

for year in sorted(
    resolved["year"].unique()
):

    for setup in [
        "BREAKOUT",
        "PULLBACK"
    ]:

        subset = resolved[
            (resolved["year"] == year)
            &
            (resolved["setup_type"] == setup)
        ]

        analyze_group(
            subset,
            f"{year} {setup}"
        )


# ============================================================
# BEST / WORST TRADES
# ============================================================

print("\nBEST 10 TRADES")
print("-" * 95)

print(
    resolved[
        [
            "entry_time",
            "direction",
            "setup_type",
            "trend",
            "rsi",
            "atr",
            "result",
            "r_multiple"
        ]
    ]
    .sort_values(
        "r_multiple",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


print("\nWORST 10 TRADES")
print("-" * 95)

print(
    resolved[
        [
            "entry_time",
            "direction",
            "setup_type",
            "trend",
            "rsi",
            "atr",
            "result",
            "r_multiple"
        ]
    ]
    .sort_values(
        "r_multiple"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 95)
print(
    "DIAGNOSTIC ANALYSIS COMPLETE"
)
print("=" * 95)

print(
    "\nDo NOT modify V1 yet."
)

print(
    "Use these results to identify which"
    " conditions are helping or hurting performance."
)

print("=" * 95)