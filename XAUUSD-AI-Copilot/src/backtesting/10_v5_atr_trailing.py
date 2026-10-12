import sys
import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators

# ============================================================
# XAUUSD V5
# V2-B ENTRY + ATR PROFIT-FOLLOWING EXIT
# ============================================================

DATA_FILE = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\raw\XAUUSD_m15_20240101_20260930.csv"
)

OUTPUT_FILE = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\processed\v5_atr_trailing_trades.csv"
)

# ============================================================
# V2-B ENTRY CONFIGURATION - DO NOT CHANGE
# ============================================================

RSI_MIN = 50
RSI_MAX = 70

ATR_INITIAL_STOP_MULTIPLIER = 1.5

LOOKBACK = 20
START_INDEX = 50

# ============================================================
# V5 EXIT CONFIGURATION
# ============================================================

TRAIL_START_R = 1.0

TRAIL_ATR_MULTIPLIER = 1.5

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("XAUUSD V5 - ATR PROFIT-FOLLOWING EXIT")
print("=" * 75)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():
    raise ValueError("Invalid dates found.")

df = df.sort_values("date").reset_index(drop=True)

print(f"\nRows : {len(df):,}")
print(f"Start: {df['date'].min()}")
print(f"End  : {df['date'].max()}")


# ============================================================
# INDICATORS
# ============================================================

df = add_indicators(df)


# ============================================================
# V2-B SIGNAL GENERATION
# ============================================================

df["signal"] = "WAIT"
df["setup_type"] = None
df["trend"] = None

for i in range(START_INDEX, len(df) - 1):

    current = df.iloc[i]
    previous = df.iloc[i - 1]

    price = current["close"]

    ema20 = current["ema20"]
    ema50 = current["ema50"]

    rsi = current["rsi14"]

    # --------------------------------------------------------
    # V2-B TREND
    # --------------------------------------------------------

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    df.loc[i, "trend"] = trend

    # --------------------------------------------------------
    # V2-B RSI FILTER
    # --------------------------------------------------------

    if not (RSI_MIN <= rsi <= RSI_MAX):
        continue

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    previous_candles = df.iloc[i - LOOKBACK:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()

    previous_close = previous["close"]

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # --------------------------------------------------------
    # V2-A BREAKOUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # V1 PULLBACK
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
    # SIGNAL
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
# BACKTEST
# ============================================================

trades = []

i = START_INDEX

while i < len(df) - 1:

    signal = df.iloc[i]["signal"]

    if signal not in ["BUY", "SELL"]:
        i += 1
        continue

    # --------------------------------------------------------
    # ENTRY = NEXT CANDLE OPEN
    # --------------------------------------------------------

    entry_index = i + 1

    if entry_index >= len(df):
        break

    entry_candle = df.iloc[entry_index]

    entry = entry_candle["open"]

    signal_atr = df.iloc[i]["atr14"]

    if pd.isna(signal_atr) or signal_atr <= 0:
        i += 1
        continue

    # --------------------------------------------------------
    # INITIAL RISK
    # --------------------------------------------------------

    initial_risk = (
        ATR_INITIAL_STOP_MULTIPLIER * signal_atr
    )

    if signal == "BUY":

        initial_stop = entry - initial_risk

    else:

        initial_stop = entry + initial_risk

    current_stop = initial_stop

    trailing_active = False

    max_favorable_price = entry
    min_favorable_price = entry

    exit_index = None
    exit_price = None
    result = None

    max_r_reached = 0.0

    # --------------------------------------------------------
    # MANAGE TRADE
    # --------------------------------------------------------

    for j in range(entry_index, len(df)):

        candle = df.iloc[j]

        candle_high = candle["high"]
        candle_low = candle["low"]

        current_atr = candle["atr14"]

        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = signal_atr

        # ====================================================
        # BUY
        # ====================================================

        if signal == "BUY":

            # ------------------------------------------------
            # Track maximum favorable price
            # ------------------------------------------------

            max_favorable_price = max(
                max_favorable_price,
                candle_high
            )

            # ------------------------------------------------
            # Calculate maximum favorable R
            # ------------------------------------------------

            favorable_r = (
                max_favorable_price - entry
            ) / initial_risk

            max_r_reached = max(
                max_r_reached,
                favorable_r
            )

            # ------------------------------------------------
            # Activate trailing after +1R
            # ------------------------------------------------

            if (
                not trailing_active
                and favorable_r >= TRAIL_START_R
            ):
                trailing_active = True

                # Move stop to breakeven
                current_stop = max(
                    current_stop,
                    entry
                )

            # ------------------------------------------------
            # ATR trailing
            # ------------------------------------------------

            if trailing_active:

                atr_stop = (
                    max_favorable_price
                    - TRAIL_ATR_MULTIPLIER * current_atr
                )

                current_stop = max(
                    current_stop,
                    atr_stop,
                    entry
                )

            # ------------------------------------------------
            # Check SL
            # ------------------------------------------------

            if candle_low <= current_stop:

                exit_index = j
                exit_price = current_stop

                if exit_price > entry:
                    result = "WIN"
                else:
                    result = "LOSS"

                break

        # ====================================================
        # SELL
        # ====================================================

        else:

            # ------------------------------------------------
            # Track minimum favorable price
            # ------------------------------------------------

            min_favorable_price = min(
                min_favorable_price,
                candle_low
            )

            # ------------------------------------------------
            # Maximum favorable R
            # ------------------------------------------------

            favorable_r = (
                entry - min_favorable_price
            ) / initial_risk

            max_r_reached = max(
                max_r_reached,
                favorable_r
            )

            # ------------------------------------------------
            # Activate trailing after +1R
            # ------------------------------------------------

            if (
                not trailing_active
                and favorable_r >= TRAIL_START_R
            ):
                trailing_active = True

                # Move stop to breakeven
                current_stop = min(
                    current_stop,
                    entry
                )

            # ------------------------------------------------
            # ATR trailing
            # ------------------------------------------------

            if trailing_active:

                atr_stop = (
                    min_favorable_price
                    + TRAIL_ATR_MULTIPLIER * current_atr
                )

                current_stop = min(
                    current_stop,
                    atr_stop,
                    entry
                )

            # ------------------------------------------------
            # Check SL
            # ------------------------------------------------

            if candle_high >= current_stop:

                exit_index = j
                exit_price = current_stop

                if exit_price < entry:
                    result = "WIN"
                else:
                    result = "LOSS"

                break

    # ========================================================
    # OPEN AT END
    # ========================================================

    if exit_index is None:

        exit_index = len(df) - 1

        exit_price = df.iloc[-1]["close"]

        result = "OPEN_AT_END"

    # ========================================================
    # R MULTIPLE
    # ========================================================

    if signal == "BUY":

        r_multiple = (
            exit_price - entry
        ) / initial_risk

    else:

        r_multiple = (
            entry - exit_price
        ) / initial_risk

    # ========================================================
    # HOLDING TIME
    # ========================================================

    holding_hours = (
        df.iloc[exit_index]["date"]
        - entry_candle["date"]
    ).total_seconds() / 3600

    # ========================================================
    # SAVE TRADE
    # ========================================================

    trades.append({

        "signal_time": df.iloc[i]["date"],

        "entry_time": entry_candle["date"],

        "exit_time": df.iloc[exit_index]["date"],

        "year": df.iloc[i]["date"].year,

        "direction": signal,

        "setup_type": df.iloc[i]["setup_type"],

        "trend": df.iloc[i]["trend"],

        "rsi": df.iloc[i]["rsi14"],

        "signal_atr": signal_atr,

        "entry": entry,

        "initial_stop": initial_stop,

        "final_stop": current_stop,

        "exit_price": exit_price,

        "trailing_activated": trailing_active,

        "max_r_reached": max_r_reached,

        "result": result,

        "r_multiple": r_multiple,

        "holding_hours": holding_hours
    })

    # --------------------------------------------------------
    # Resume after trade closes
    # --------------------------------------------------------

    i = exit_index + 1


trades = pd.DataFrame(trades)

resolved = trades[
    trades["result"].isin(["WIN", "LOSS"])
].copy()


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_performance(data):

    if data.empty:
        return None

    wins = data[
        data["result"] == "WIN"
    ]

    losses = data[
        data["result"] == "LOSS"
    ]

    gross_profit = wins["r_multiple"].sum()

    gross_loss = abs(
        losses["r_multiple"].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = data["r_multiple"].cumsum()

    drawdown = (
        equity - equity.cummax()
    )

    return {
        "trades": len(data),

        "win_rate":
            len(wins) / len(data) * 100,

        "total_r":
            data["r_multiple"].sum(),

        "avg_r":
            data["r_multiple"].mean(),

        "profit_factor":
            profit_factor,

        "max_drawdown":
            drawdown.min(),

        "avg_winner":
            wins["r_multiple"].mean()
            if not wins.empty else 0,

        "avg_loser":
            losses["r_multiple"].mean()
            if not losses.empty else 0,

        "avg_holding":
            data["holding_hours"].mean(),

        "median_holding":
            data["holding_hours"].median(),

        "avg_max_r":
            data["max_r_reached"].mean()
    }


# ============================================================
# OVERALL
# ============================================================

print("\n")
print("=" * 75)
print("V5 OVERALL PERFORMANCE")
print("=" * 75)

metrics = calculate_performance(resolved)

for key, value in metrics.items():

    if key == "profit_factor":
        print(f"{key:<20}: {value:.3f}")

    elif key in [
        "win_rate"
    ]:
        print(f"{key:<20}: {value:.2f}%")

    elif key in [
        "total_r",
        "avg_r",
        "max_drawdown",
        "avg_winner",
        "avg_loser",
        "avg_max_r"
    ]:
        print(f"{key:<20}: {value:.4f}R")

    else:
        print(f"{key:<20}: {value:.2f}")


# ============================================================
# YEAR ANALYSIS
# ============================================================

print("\n")
print("=" * 75)
print("YEAR ANALYSIS")
print("=" * 75)

print(
    f"{'Year':<10}"
    f"{'Trades':>10}"
    f"{'Win%':>10}"
    f"{'Total R':>12}"
    f"{'Avg R':>12}"
    f"{'PF':>10}"
)

print("-" * 75)

for year in sorted(resolved["year"].unique()):

    subset = resolved[
        resolved["year"] == year
    ]

    m = calculate_performance(subset)

    print(
        f"{year:<10}"
        f"{m['trades']:>10}"
        f"{m['win_rate']:>9.2f}%"
        f"{m['total_r']:>11.2f}R"
        f"{m['avg_r']:>11.4f}"
        f"{m['profit_factor']:>10.3f}"
    )


# ============================================================
# DIRECTION
# ============================================================

print("\n")
print("=" * 75)
print("DIRECTION")
print("=" * 75)

for direction in ["BUY", "SELL"]:

    subset = resolved[
        resolved["direction"] == direction
    ]

    if subset.empty:
        continue

    m = calculate_performance(subset)

    print(
        f"{direction}: "
        f"Trades={m['trades']} | "
        f"Win={m['win_rate']:.2f}% | "
        f"R={m['total_r']:.2f}R | "
        f"AvgR={m['avg_r']:.4f} | "
        f"PF={m['profit_factor']:.3f}"
    )


# ============================================================
# SETUP
# ============================================================

print("\n")
print("=" * 75)
print("SETUP")
print("=" * 75)

for setup in ["BREAKOUT", "PULLBACK"]:

    subset = resolved[
        resolved["setup_type"] == setup
    ]

    if subset.empty:
        continue

    m = calculate_performance(subset)

    print(
        f"{setup}: "
        f"Trades={m['trades']} | "
        f"Win={m['win_rate']:.2f}% | "
        f"R={m['total_r']:.2f}R | "
        f"AvgR={m['avg_r']:.4f} | "
        f"PF={m['profit_factor']:.3f}"
    )


# ============================================================
# 2024-2025 VS 2026
# ============================================================

print("\n")
print("=" * 75)
print("2024-2025 VS 2026")
print("=" * 75)

for label, start, end in [
    ("2024-2025", 2024, 2025),
    ("2026", 2026, 2026)
]:

    subset = resolved[
        (resolved["year"] >= start)
        & (resolved["year"] <= end)
    ]

    m = calculate_performance(subset)

    print(
        f"{label}: "
        f"Trades={m['trades']} | "
        f"Win={m['win_rate']:.2f}% | "
        f"R={m['total_r']:.2f}R | "
        f"AvgR={m['avg_r']:.4f} | "
        f"PF={m['profit_factor']:.3f} | "
        f"MaxDD={m['max_drawdown']:.2f}R"
    )


# ============================================================
# TRAILING STATISTICS
# ============================================================

print("\n")
print("=" * 75)
print("TRAILING STATISTICS")
print("=" * 75)

activated = resolved[
    resolved["trailing_activated"] == True
]

print(
    f"Trades reaching +{TRAIL_START_R:.1f}R : "
    f"{len(activated)}"
)

print(
    f"Percentage reaching +{TRAIL_START_R:.1f}R : "
    f"{len(activated) / len(resolved) * 100:.2f}%"
)

print(
    f"Average maximum R reached : "
    f"{resolved['max_r_reached'].mean():.3f}R"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

trades.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("Trades saved to:")
print(OUTPUT_FILE)