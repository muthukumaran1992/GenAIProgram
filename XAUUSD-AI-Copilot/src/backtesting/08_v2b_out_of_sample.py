import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\raw\XAUUSD_m15_20240101_20260930.csv"
)

RSI_MIN = 50
RSI_MAX = 70

ATR_STOP_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 2.0

LOOKBACK = 20
START_INDEX = 50


# ============================================================
# ROBUST DATETIME PARSER
# ============================================================

def parse_datetime(series):

    # First try normal datetime strings
    parsed = pd.to_datetime(
        series,
        errors="coerce"
    )

    valid_ratio = parsed.notna().mean()

    if valid_ratio >= 0.95:
        return parsed

    # If numeric timestamps are present, determine scale
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() >= 0.95:

        median_value = numeric.dropna().abs().median()

        if median_value > 1e12:
            unit = "ms"
        elif median_value > 1e9:
            unit = "s"
        else:
            raise ValueError(
                "Timestamp values could not be safely identified "
                "as Unix seconds or milliseconds."
            )

        parsed = pd.to_datetime(
            numeric,
            unit=unit,
            errors="coerce"
        )

    if parsed.isna().any():
        bad_count = parsed.isna().sum()
        raise ValueError(
            f"Unable to parse {bad_count} timestamps."
        )

    return parsed


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("XAUUSD V2-B OUT-OF-SAMPLE TEST")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:")
print(df.columns.tolist())

time_column = "timestamp" if "timestamp" in df.columns else "date"
required_columns = {
    time_column,
    "open",
    "high",
    "low",
    "close",
    "volume",
}
missing = sorted(required_columns.difference(df.columns))

if missing:
    raise ValueError(f"Missing columns in {DATA_FILE.name}: {missing}")

if df.empty:
    raise ValueError(f"No market data found in {DATA_FILE.name}.")

df["date"] = parse_datetime(df[time_column])

df = df.sort_values("date").reset_index(drop=True)

if df.empty:
    raise ValueError(f"No valid market data found in {DATA_FILE.name}.")

print("\nDATA")
print(f"Rows        : {len(df):,}")
print(f"Start       : {df['date'].min()}")
print(f"End         : {df['date'].max()}")


# ============================================================
# INDICATORS
# ============================================================

df = add_indicators(df)


# ============================================================
# GENERATE V2-B SIGNALS
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
    # V1 TREND
    # --------------------------------------------------------

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    df.loc[i, "trend"] = trend

    # --------------------------------------------------------
    # RSI FILTER - V2-B
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

    if bullish_breakout or bullish_pullback:

        df.loc[i, "signal"] = "BUY"

        if bullish_breakout:
            df.loc[i, "setup_type"] = "BREAKOUT"
        else:
            df.loc[i, "setup_type"] = "PULLBACK"

    elif bearish_breakout or bearish_pullback:

        df.loc[i, "signal"] = "SELL"

        if bearish_breakout:
            df.loc[i, "setup_type"] = "BREAKOUT"
        else:
            df.loc[i, "setup_type"] = "PULLBACK"


# ============================================================
# SIMULATOR
# ============================================================

def run_backtest(data, start_date, end_date):

    trades = []

    i = START_INDEX

    while i < len(data) - 1:

        signal_time = data.iloc[i]["date"]

        # Only evaluate trades inside requested period
        if signal_time < start_date:
            i += 1
            continue

        if signal_time > end_date:
            break

        signal = data.iloc[i]["signal"]

        if signal not in ["BUY", "SELL"]:
            i += 1
            continue

        # ----------------------------------------------------
        # NEXT CANDLE ENTRY
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(data):
            break

        entry_candle = data.iloc[entry_index]

        entry = entry_candle["open"]

        atr = data.iloc[i]["atr14"]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        risk = ATR_STOP_MULTIPLIER * atr
        reward = REWARD_MULTIPLIER * risk

        if signal == "BUY":

            stop_loss = entry - risk
            take_profit = entry + reward

        else:

            stop_loss = entry + risk
            take_profit = entry - reward

        # ----------------------------------------------------
        # FIND EXIT
        # ----------------------------------------------------

        exit_index = None
        exit_price = None
        result = None

        for j in range(entry_index, len(data)):

            candle = data.iloc[j]

            high = candle["high"]
            low = candle["low"]

            if signal == "BUY":

                hit_sl = low <= stop_loss
                hit_tp = high >= take_profit

                # Conservative assumption
                if hit_sl:
                    exit_price = stop_loss
                    result = "LOSS"
                    exit_index = j
                    break

                elif hit_tp:
                    exit_price = take_profit
                    result = "WIN"
                    exit_index = j
                    break

            else:

                hit_sl = high >= stop_loss
                hit_tp = low <= take_profit

                # Conservative assumption
                if hit_sl:
                    exit_price = stop_loss
                    result = "LOSS"
                    exit_index = j
                    break

                elif hit_tp:
                    exit_price = take_profit
                    result = "WIN"
                    exit_index = j
                    break

        # ----------------------------------------------------
        # OPEN AT END
        # ----------------------------------------------------

        if exit_index is None:

            exit_index = len(data) - 1
            exit_price = data.iloc[-1]["close"]
            result = "OPEN_AT_END"

        # ----------------------------------------------------
        # R MULTIPLE
        # ----------------------------------------------------

        if signal == "BUY":
            actual_r = (exit_price - entry) / risk
        else:
            actual_r = (entry - exit_price) / risk

        holding_hours = (
            data.iloc[exit_index]["date"]
            - entry_candle["date"]
        ).total_seconds() / 3600

        trades.append({
            "signal_time": signal_time,
            "entry_time": entry_candle["date"],
            "exit_time": data.iloc[exit_index]["date"],
            "direction": signal,
            "setup_type": data.iloc[i]["setup_type"],
            "trend": data.iloc[i]["trend"],
            "rsi": data.iloc[i]["rsi14"],
            "atr": atr,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_price": exit_price,
            "result": result,
            "r_multiple": actual_r,
            "holding_hours": holding_hours
        })

        # Resume after trade closes
        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE
# ============================================================

def performance_report(trades, label):

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    if trades.empty:
        print("No trades.")
        return

    resolved = trades[
        trades["result"].isin(["WIN", "LOSS"])
    ]

    wins = resolved[resolved["result"] == "WIN"]
    losses = resolved[resolved["result"] == "LOSS"]

    total_r = resolved["r_multiple"].sum()

    avg_r = (
        resolved["r_multiple"].mean()
        if len(resolved)
        else 0
    )

    win_rate = (
        len(wins) / len(resolved) * 100
        if len(resolved)
        else 0
    )

    gross_profit = wins["r_multiple"].sum()

    gross_loss = abs(losses["r_multiple"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # --------------------------------------------------------
    # EQUITY / DRAWDOWN
    # --------------------------------------------------------

    equity = resolved["r_multiple"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # MAX LOSING STREAK
    # --------------------------------------------------------

    max_losing_streak = 0
    current_streak = 0

    for result in resolved["result"]:

        if result == "LOSS":
            current_streak += 1
            max_losing_streak = max(
                max_losing_streak,
                current_streak
            )
        else:
            current_streak = 0

    print(f"Total trades       : {len(trades):,}")
    print(f"Resolved trades    : {len(resolved):,}")
    print(f"Wins               : {len(wins):,}")
    print(f"Losses             : {len(losses):,}")
    print(f"Open at end        : {len(trades) - len(resolved):,}")

    print()
    print(f"Win rate           : {win_rate:.2f}%")
    print(f"Total R            : {total_r:.2f}R")
    print(f"Average R/trade    : {avg_r:.4f}R")
    print(f"Profit factor      : {profit_factor:.3f}")
    print(f"Maximum drawdown   : {max_drawdown:.2f}R")
    print(f"Max losing streak  : {max_losing_streak}")

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    print("\nDIRECTION")

    for direction in ["BUY", "SELL"]:

        subset = resolved[
            resolved["direction"] == direction
        ]

        if subset.empty:
            continue

        subset_wins = subset[
            subset["result"] == "WIN"
        ]

        subset_losses = subset[
            subset["result"] == "LOSS"
        ]

        gp = subset_wins["r_multiple"].sum()
        gl = abs(subset_losses["r_multiple"].sum())

        pf = gp / gl if gl > 0 else np.inf

        print(
            f"{direction}: "
            f"Trades {len(subset)}, "
            f"WinRate "
            f"{len(subset_wins) / len(subset) * 100:.2f}%, "
            f"TotalR {subset['r_multiple'].sum():.2f}R, "
            f"AvgR {subset['r_multiple'].mean():.4f}, "
            f"PF {pf:.3f}"
        )

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    print("\nSETUP TYPE")

    for setup in ["BREAKOUT", "PULLBACK"]:

        subset = resolved[
            resolved["setup_type"] == setup
        ]

        if subset.empty:
            continue

        wins_setup = subset[
            subset["result"] == "WIN"
        ]

        losses_setup = subset[
            subset["result"] == "LOSS"
        ]

        gp = wins_setup["r_multiple"].sum()
        gl = abs(losses_setup["r_multiple"].sum())

        pf = gp / gl if gl > 0 else np.inf

        print(
            f"{setup}: "
            f"Trades {len(subset)}, "
            f"WinRate "
            f"{len(wins_setup) / len(subset) * 100:.2f}%, "
            f"TotalR {subset['r_multiple'].sum():.2f}R, "
            f"AvgR {subset['r_multiple'].mean():.4f}, "
            f"PF {pf:.3f}"
        )


# ============================================================
# DEVELOPMENT / IN-SAMPLE
# ============================================================

development_trades = run_backtest(
    df,
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2025-12-31 23:59:59")
)

performance_report(
    development_trades,
    "DEVELOPMENT / IN-SAMPLE — 2024-2025"
)


# ============================================================
# OUT-OF-SAMPLE
# ============================================================

oos_trades = run_backtest(
    df,
    pd.Timestamp("2026-01-01"),
    pd.Timestamp("2026-09-29 23:59:59")
)

performance_report(
    oos_trades,
    "OUT-OF-SAMPLE — 2026"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\processed\v2b_oos_trades.csv"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

oos_trades.to_csv(
    output_file,
    index=False
)

print("\n")
print(f"OOS trades saved to:")
print(output_file)