import sys
import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators

# ============================================================
# CONFIG
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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():
    raise ValueError("Invalid dates found.")

df = df.sort_values("date").reset_index(drop=True)


df = add_indicators(df)


# ============================================================
# GENERATE FROZEN V2-B SIGNALS
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

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    df.loc[i, "trend"] = trend

    # V2-B RSI filter
    if not (RSI_MIN <= rsi <= RSI_MAX):
        continue

    previous_candles = df.iloc[i - LOOKBACK:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()

    previous_close = previous["close"]

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # V2-A breakout confirmation
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

    # V1 pullback
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
# WALK-FORWARD TEST
# ============================================================

windows = [

    (
        "WF1",
        "2024-01-01",
        "2024-06-30 23:59:59",
        "2024-07-01",
        "2024-12-31 23:59:59"
    ),

    (
        "WF2",
        "2024-07-01",
        "2024-12-31 23:59:59",
        "2025-01-01",
        "2025-06-30 23:59:59"
    ),

    (
        "WF3",
        "2025-01-01",
        "2025-06-30 23:59:59",
        "2025-07-01",
        "2025-12-31 23:59:59"
    ),

    (
        "WF4",
        "2025-07-01",
        "2025-12-31 23:59:59",
        "2026-01-01",
        "2026-06-30 23:59:59"
    ),

    (
        "WF5",
        "2026-01-01",
        "2026-06-30 23:59:59",
        "2026-07-01",
        "2026-09-29 23:59:59"
    )
]


def run_period(start_date, end_date):

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    trades = []

    i = START_INDEX

    while i < len(df) - 1:

        signal_time = df.iloc[i]["date"]

        if signal_time < start_date:
            i += 1
            continue

        if signal_time > end_date:
            break

        signal = df.iloc[i]["signal"]

        if signal not in ["BUY", "SELL"]:
            i += 1
            continue

        entry_index = i + 1

        if entry_index >= len(df):
            break

        entry_candle = df.iloc[entry_index]

        entry = entry_candle["open"]

        atr = df.iloc[i]["atr14"]

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

        exit_index = None
        exit_price = None
        result = None

        for j in range(entry_index, len(df)):

            candle = df.iloc[j]

            if signal == "BUY":

                hit_sl = candle["low"] <= stop_loss
                hit_tp = candle["high"] >= take_profit

                # Conservative assumption
                if hit_sl:
                    exit_index = j
                    exit_price = stop_loss
                    result = "LOSS"
                    break

                elif hit_tp:
                    exit_index = j
                    exit_price = take_profit
                    result = "WIN"
                    break

            else:

                hit_sl = candle["high"] >= stop_loss
                hit_tp = candle["low"] <= take_profit

                if hit_sl:
                    exit_index = j
                    exit_price = stop_loss
                    result = "LOSS"
                    break

                elif hit_tp:
                    exit_index = j
                    exit_price = take_profit
                    result = "WIN"
                    break

        if exit_index is None:
            exit_index = len(df) - 1
            exit_price = df.iloc[-1]["close"]
            result = "OPEN_AT_END"

        if signal == "BUY":
            r_multiple = (exit_price - entry) / risk
        else:
            r_multiple = (entry - exit_price) / risk

        trades.append({
            "signal_time": signal_time,
            "entry_time": entry_candle["date"],
            "exit_time": df.iloc[exit_index]["date"],
            "direction": signal,
            "setup_type": df.iloc[i]["setup_type"],
            "result": result,
            "r_multiple": r_multiple
        })

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(trades):

    if trades.empty:
        return {
            "trades": 0,
            "resolved": 0,
            "win_rate": 0,
            "total_r": 0,
            "avg_r": 0,
            "profit_factor": 0,
            "max_dd": 0
        }

    resolved = trades[
        trades["result"].isin(["WIN", "LOSS"])
    ].copy()

    if resolved.empty:
        return {
            "trades": len(trades),
            "resolved": 0,
            "win_rate": 0,
            "total_r": 0,
            "avg_r": 0,
            "profit_factor": 0,
            "max_dd": 0
        }

    wins = resolved[
        resolved["result"] == "WIN"
    ]

    losses = resolved[
        resolved["result"] == "LOSS"
    ]

    total_r = resolved["r_multiple"].sum()

    avg_r = resolved["r_multiple"].mean()

    win_rate = len(wins) / len(resolved) * 100

    gross_profit = wins["r_multiple"].sum()
    gross_loss = abs(losses["r_multiple"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = resolved["r_multiple"].cumsum()

    drawdown = equity - equity.cummax()

    max_dd = drawdown.min()

    return {
        "trades": len(trades),
        "resolved": len(resolved),
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_dd": max_dd
    }


# ============================================================
# RUN WALK-FORWARD
# ============================================================

results = []

print("=" * 80)
print("XAUUSD V3 WALK-FORWARD VALIDATION")
print("=" * 80)

for name, train_start, train_end, test_start, test_end in windows:

    # --------------------------------------------------------
    # TRAINING PERIOD
    # --------------------------------------------------------

    train_trades = run_period(
        train_start,
        train_end
    )

    train_metrics = calculate_metrics(
        train_trades
    )

    # --------------------------------------------------------
    # FORWARD TEST
    # --------------------------------------------------------

    test_trades = run_period(
        test_start,
        test_end
    )

    test_metrics = calculate_metrics(
        test_trades
    )

    results.append({
        "window": name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        **{
            f"train_{k}": v
            for k, v in train_metrics.items()
        },
        **{
            f"test_{k}": v
            for k, v in test_metrics.items()
        }
    })

    print("\n" + "-" * 80)
    print(name)

    print(
        f"TRAIN : "
        f"{train_start} → {train_end}"
    )

    print(
        f"TEST  : "
        f"{test_start} → {test_end}"
    )

    print("\nTRAINING")
    print(
        f"Trades={train_metrics['trades']} | "
        f"Win={train_metrics['win_rate']:.2f}% | "
        f"R={train_metrics['total_r']:.2f} | "
        f"PF={train_metrics['profit_factor']:.3f}"
    )

    print("\nFORWARD TEST")
    print(
        f"Trades={test_metrics['trades']} | "
        f"Win={test_metrics['win_rate']:.2f}% | "
        f"R={test_metrics['total_r']:.2f} | "
        f"AvgR={test_metrics['avg_r']:.4f} | "
        f"PF={test_metrics['profit_factor']:.3f} | "
        f"MaxDD={test_metrics['max_dd']:.2f}R"
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("WALK-FORWARD SUMMARY")
print("=" * 80)

print(
    results_df[
        [
            "window",
            "test_trades",
            "test_win_rate",
            "test_total_r",
            "test_avg_r",
            "test_profit_factor",
            "test_max_dd"
        ]
    ].to_string(index=False)
)


# ============================================================
# AGGREGATED FORWARD PERFORMANCE
# ============================================================

forward_r = results_df["test_total_r"].sum()
forward_trades = results_df["test_trades"].sum()

print("\n")
print("AGGREGATED FORWARD TEST")
print("-" * 40)

print(f"Total forward trades : {forward_trades}")
print(f"Total forward R      : {forward_r:.2f}R")

if forward_trades > 0:
    print(
        f"Average R/trade     : "
        f"{forward_r / forward_trades:.4f}R"
    )

positive_windows = (
    results_df["test_total_r"] > 0
).sum()

print(
    f"Positive windows     : "
    f"{positive_windows}/{len(results_df)}"
)


# ============================================================
# SAVE
# ============================================================

output_file = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\processed\v3_walk_forward_results.csv"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)