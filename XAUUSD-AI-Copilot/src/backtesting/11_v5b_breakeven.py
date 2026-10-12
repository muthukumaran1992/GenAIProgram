import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot"

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "XAUUSD_m15_20240101_20260930.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "v5b_breakeven_trades.csv"
)

# Allow imports from project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.indicators.technical_indicators import add_indicators


# ============================================================
# PARAMETERS
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_R = 2.0

# V2-B RSI filter
RSI_MIN = 50
RSI_MAX = 70

LOOKBACK = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V5-B: BREAKEVEN + FIXED 1:2R BACKTEST")
print("=" * 70)

print("\nLoading data...")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

# Robust datetime handling
df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError("Invalid dates found in dataset.")

df = df.sort_values("date").reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Start: {df['date'].min()}")
print(f"End:   {df['date'].max()}")


# ============================================================
# VALIDATE OHLC
# ============================================================

required_columns = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# INDICATORS
# ============================================================

print("\nCalculating indicators...")

df = add_indicators(df)

print("Indicators calculated:")
print("EMA20")
print("EMA50")
print("RSI14")
print("ATR14")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def determine_trend(row):
    price = row["close"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    if price > ema20 and ema20 > ema50:
        return "BULLISH"

    elif price < ema20 and ema20 < ema50:
        return "BEARISH"

    return "SIDEWAYS"


def get_support_resistance(index):

    # IMPORTANT:
    # Only previous candles are used.
    # Current candle is NOT included.
    start = max(0, index - LOOKBACK)

    previous = df.iloc[start:index]

    if len(previous) < LOOKBACK:
        return np.nan, np.nan

    resistance = previous["high"].max()
    support = previous["low"].min()

    return support, resistance


# ============================================================
# GENERATE V2-B SIGNAL
# ============================================================

def generate_signal(index):

    current = df.iloc[index]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    if pd.isna(ema20) or pd.isna(ema50):
        return "WAIT", None

    if pd.isna(rsi):
        return "WAIT", None

    # V2-B RSI filter
    if rsi < RSI_MIN or rsi > RSI_MAX:
        return "WAIT", None

    trend = determine_trend(current)

    support, resistance = get_support_resistance(index)

    if pd.isna(support) or pd.isna(resistance):
        return "WAIT", None

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if (
        trend == "BULLISH"
        and price > resistance
        and bullish_candle
    ):
        return "BUY", "BREAKOUT"

    if (
        trend == "BEARISH"
        and price < support
        and bearish_candle
    ):
        return "SELL", "BREAKOUT"

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    bullish_pullback = (
        trend == "BULLISH"
        and current["low"] <= ema20
        and price > ema20
        and bullish_candle
    )

    if bullish_pullback:
        return "BUY", "PULLBACK"

    bearish_pullback = (
        trend == "BEARISH"
        and current["high"] >= ema20
        and price < ema20
        and bearish_candle
    )

    if bearish_pullback:
        return "SELL", "PULLBACK"

    return "WAIT", None


# ============================================================
# BACKTEST
# ============================================================

print("\nRunning V5-B backtest...")
print()
print("Rules:")
print(f"Initial SL : {ATR_STOP_MULTIPLIER} ATR")
print("Breakeven  : +1R")
print(f"Take Profit: +{REWARD_R}R")
print("Trailing   : None")
print("Entry      : Next candle OPEN")
print("SL priority: SL checked before TP")
print()


trades = []

i = LOOKBACK + 50

while i < len(df) - 1:

    signal, setup = generate_signal(i)

    if signal == "WAIT":
        i += 1
        continue

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    entry_index = i + 1

    if entry_index >= len(df):
        break

    entry_row = df.iloc[entry_index]

    entry_time = entry_row["date"]
    entry_price = entry_row["open"]

    atr = df.iloc[i]["atr14"]

    if pd.isna(atr) or atr <= 0:
        i += 1
        continue

    risk_points = atr * ATR_STOP_MULTIPLIER

    if signal == "BUY":

        stop_loss = entry_price - risk_points
        take_profit = entry_price + (
            risk_points * REWARD_R
        )

        breakeven_trigger = entry_price + risk_points

    else:

        stop_loss = entry_price + risk_points
        take_profit = entry_price - (
            risk_points * REWARD_R
        )

        breakeven_trigger = entry_price - risk_points

    # Keep original SL for calculations
    initial_stop = stop_loss

    # --------------------------------------------------------
    # TRADE MANAGEMENT
    # --------------------------------------------------------

    breakeven_hit = False
    exit_price = None
    exit_time = None
    exit_reason = None

    max_favorable_price = entry_price
    max_r_reached = 0.0

    j = entry_index

    while j < len(df):

        candle = df.iloc[j]

        candle_high = candle["high"]
        candle_low = candle["low"]

        # ----------------------------------------------------
        # UPDATE MAX FAVORABLE EXCURSION
        # ----------------------------------------------------

        if signal == "BUY":

            favorable_move = candle_high - entry_price

            max_favorable_price = max(
                max_favorable_price,
                candle_high
            )

        else:

            favorable_move = entry_price - candle_low

            max_favorable_price = min(
                max_favorable_price,
                candle_low
            )

        current_max_r = favorable_move / risk_points

        max_r_reached = max(
            max_r_reached,
            current_max_r
        )

        # ----------------------------------------------------
        # MOVE STOP TO BREAKEVEN AFTER +1R
        # ----------------------------------------------------

        if not breakeven_hit:

            if signal == "BUY" and candle_high >= breakeven_trigger:

                breakeven_hit = True
                stop_loss = entry_price

            elif signal == "SELL" and candle_low <= breakeven_trigger:

                breakeven_hit = True
                stop_loss = entry_price

        # ----------------------------------------------------
        # EXIT LOGIC
        # ----------------------------------------------------

        if signal == "BUY":

            # IMPORTANT:
            # If both SL and TP occur in same candle,
            # assume SL happens first (conservative).

            if candle_low <= stop_loss:

                exit_price = stop_loss
                exit_time = candle["date"]

                if breakeven_hit and stop_loss == entry_price:
                    exit_reason = "BREAKEVEN"
                else:
                    exit_reason = "STOP_LOSS"

                break

            if candle_high >= take_profit:

                exit_price = take_profit
                exit_time = candle["date"]
                exit_reason = "TAKE_PROFIT"

                break

        else:

            if candle_high >= stop_loss:

                exit_price = stop_loss
                exit_time = candle["date"]

                if breakeven_hit and stop_loss == entry_price:
                    exit_reason = "BREAKEVEN"
                else:
                    exit_reason = "STOP_LOSS"

                break

            if candle_low <= take_profit:

                exit_price = take_profit
                exit_time = candle["date"]
                exit_reason = "TAKE_PROFIT"

                break

        j += 1

    # --------------------------------------------------------
    # OPEN POSITION AT END OF DATA
    # --------------------------------------------------------

    if exit_price is None:

        final_row = df.iloc[-1]

        exit_price = final_row["close"]
        exit_time = final_row["date"]
        exit_reason = "OPEN_AT_END"

    # --------------------------------------------------------
    # CALCULATE R RESULT
    # --------------------------------------------------------

    if signal == "BUY":

        pnl_points = exit_price - entry_price

    else:

        pnl_points = entry_price - exit_price

    result_r = pnl_points / risk_points

    # --------------------------------------------------------
    # RECORD TRADE
    # --------------------------------------------------------

    trades.append({
        "signal_time": df.iloc[i]["date"],
        "entry_time": entry_time,
        "exit_time": exit_time,

        "signal": signal,
        "setup": setup,

        "entry_price": entry_price,
        "initial_stop_loss": initial_stop,
        "final_stop_loss": stop_loss,
        "take_profit": take_profit,

        "atr14": atr,
        "risk_points": risk_points,

        "breakeven_trigger": breakeven_trigger,
        "breakeven_hit": breakeven_hit,

        "exit_price": exit_price,
        "exit_reason": exit_reason,

        "result_r": result_r,
        "max_r_reached": max_r_reached,

        "holding_hours": (
            exit_time - entry_time
        ).total_seconds() / 3600
    })

    # --------------------------------------------------------
    # ONE POSITION AT A TIME
    # Move to candle after exit
    # --------------------------------------------------------

    i = max(j + 1, i + 1)


# ============================================================
# CREATE TRADE DATAFRAME
# ============================================================

trades_df = pd.DataFrame(trades)

if trades_df.empty:
    raise ValueError("No trades generated.")


# ============================================================
# STATISTICS
# ============================================================

total_trades = len(trades_df)

wins = trades_df[
    trades_df["result_r"] > 0
]

losses = trades_df[
    trades_df["result_r"] < 0
]

breakeven_trades = trades_df[
    trades_df["result_r"] == 0
]

win_rate = (
    len(wins) / total_trades * 100
)

total_r = trades_df["result_r"].sum()

avg_r = trades_df["result_r"].mean()

gross_profit = wins["result_r"].sum()

gross_loss = abs(
    losses["result_r"].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else np.inf
)

avg_winner = (
    wins["result_r"].mean()
    if len(wins) > 0
    else 0
)

avg_loser = (
    losses["result_r"].mean()
    if len(losses) > 0
    else 0
)

avg_holding = trades_df[
    "holding_hours"
].mean()

median_holding = trades_df[
    "holding_hours"
].median()

avg_max_r = trades_df[
    "max_r_reached"
].mean()


# ============================================================
# MAX DRAWDOWN
# ============================================================

equity = trades_df["result_r"].cumsum()

peak = equity.cummax()

drawdown = equity - peak

max_drawdown = drawdown.min()


# ============================================================
# MAX LOSING STREAK
# ============================================================

max_losing_streak = 0
current_losing_streak = 0

for result in trades_df["result_r"]:

    if result < 0:

        current_losing_streak += 1

        max_losing_streak = max(
            max_losing_streak,
            current_losing_streak
        )

    else:

        current_losing_streak = 0


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("V5-B RESULTS")
print("=" * 70)

print(f"Total trades       : {total_trades:,}")
print(f"Wins               : {len(wins):,}")
print(f"Losses             : {len(losses):,}")
print(f"Breakeven          : {len(breakeven_trades):,}")

print(f"Win rate           : {win_rate:.2f}%")
print(f"Total R            : {total_r:.2f}R")
print(f"Average R/trade    : {avg_r:.4f}R")
print(f"Profit Factor      : {profit_factor:.3f}")

print(f"Average winner     : {avg_winner:.4f}R")
print(f"Average loser      : {avg_loser:.4f}R")

print(f"Max Drawdown       : {max_drawdown:.2f}R")
print(f"Max Losing Streak  : {max_losing_streak}")

print(f"Avg Holding        : {avg_holding:.2f} hours")
print(f"Median Holding     : {median_holding:.2f} hours")

print(f"Avg Max R Reached  : {avg_max_r:.4f}R")


# ============================================================
# EXIT REASON ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("EXIT REASONS")
print("=" * 70)

exit_stats = (
    trades_df
    .groupby("exit_reason")
    .agg(
        trades=("result_r", "count"),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean")
    )
    .sort_values("trades", ascending=False)
)

print(exit_stats.to_string())


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("DIRECTION")
print("=" * 70)

direction_stats = (
    trades_df
    .groupby("signal")
    .agg(
        trades=("result_r", "count"),
        win_rate=("result_r",
                  lambda x: (x > 0).mean() * 100),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean")
    )
)

print(direction_stats.to_string())


# ============================================================
# SETUP ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SETUP")
print("=" * 70)

setup_stats = (
    trades_df
    .groupby("setup")
    .agg(
        trades=("result_r", "count"),
        win_rate=("result_r",
                  lambda x: (x > 0).mean() * 100),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean")
    )
)

print(setup_stats.to_string())


# ============================================================
# YEAR ANALYSIS
# ============================================================

trades_df["year"] = pd.to_datetime(
    trades_df["entry_time"]
).dt.year

print("\n")
print("=" * 70)
print("YEAR")
print("=" * 70)

year_stats = (
    trades_df
    .groupby("year")
    .agg(
        trades=("result_r", "count"),
        win_rate=("result_r",
                  lambda x: (x > 0).mean() * 100),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean")
    )
)

print(year_stats.to_string())


# ============================================================
# 2024-25 vs 2026
# ============================================================

trades_df["period"] = np.where(
    trades_df["year"] <= 2025,
    "2024-2025",
    "2026"
)

print("\n")
print("=" * 70)
print("2024-2025 vs 2026")
print("=" * 70)

period_stats = (
    trades_df
    .groupby("period")
    .agg(
        trades=("result_r", "count"),
        win_rate=("result_r",
                  lambda x: (x > 0).mean() * 100),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean")
    )
)

print(period_stats.to_string())


# ============================================================
# BREAKEVEN ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("BREAKEVEN ANALYSIS")
print("=" * 70)

be_count = trades_df["breakeven_hit"].sum()

be_pct = (
    be_count / total_trades * 100
)

print(
    f"Trades reaching +1R : {be_count:,} "
    f"({be_pct:.2f}%)"
)

print(
    f"Trades reaching TP   : "
    f"{(trades_df['exit_reason'] == 'TAKE_PROFIT').sum():,}"
)

print(
    f"Trades stopped       : "
    f"{(trades_df['exit_reason'] == 'STOP_LOSS').sum():,}"
)

print(
    f"Breakeven exits      : "
    f"{(trades_df['exit_reason'] == 'BREAKEVEN').sum():,}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

trades_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Saved trades to:")
print(OUTPUT_FILE)

print("\nV5-B backtest completed successfully.")