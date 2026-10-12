
import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# V2-B 2026 FORENSIC ANALYSIS
# ============================================================
# Purpose:
#   Investigate WHY the frozen V2-B strategy deteriorated in 2026.
#
# IMPORTANT:
#   - V2-B entry logic is NOT changed.
#   - Fixed 1.5 ATR SL and fixed 2R TP are NOT changed.
#   - No new trading filter is optimized here.
#   - This is diagnostic analysis only.
# ============================================================

PROJECT_ROOT = r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot"

DATA_FILE = os.path.join(
    PROJECT_ROOT, "data", "raw",
    "XAUUSD_m15_20240101_20260930.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "v6_2026_forensic_trades.csv"
)

REPORT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "v6_2026_forensic_report.txt"
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.indicators.technical_indicators import add_indicators


# ============================================================
# FROZEN V2-B PARAMETERS
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_R = 2.0
RSI_MIN = 50
RSI_MAX = 70
LOOKBACK = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("V2-B 2026 FORENSIC ANALYSIS")
print("=" * 78)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError("Invalid dates found.")

df = df.sort_values("date").reset_index(drop=True)

df = add_indicators(df)

# Extra diagnostic features
df["ema_spread"] = df["ema20"] - df["ema50"]
df["ema_spread_pct"] = (
    df["ema_spread"] / df["close"] * 100
)
df["atr_pct"] = (
    df["atr14"] / df["close"] * 100
)
df["candle_range"] = df["high"] - df["low"]
df["body_size"] = (df["close"] - df["open"]).abs()
df["body_pct_range"] = np.where(
    df["candle_range"] > 0,
    df["body_size"] / df["candle_range"] * 100,
    0
)


# ============================================================
# V2-B SIGNAL LOGIC
# ============================================================

def determine_trend(row):
    if row["close"] > row["ema20"] and row["ema20"] > row["ema50"]:
        return "BULLISH"
    elif row["close"] < row["ema20"] and row["ema20"] < row["ema50"]:
        return "BEARISH"
    return "SIDEWAYS"


def get_support_resistance(index):
    start = max(0, index - LOOKBACK)
    previous = df.iloc[start:index]

    if len(previous) < LOOKBACK:
        return np.nan, np.nan

    return previous["low"].min(), previous["high"].max()


def generate_v2b_signal(index):
    current = df.iloc[index]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    if any(pd.isna(x) for x in [ema20, ema50, rsi]):
        return "WAIT", None

    # Frozen V2-B RSI filter
    if rsi < RSI_MIN or rsi > RSI_MAX:
        return "WAIT", None

    trend = determine_trend(current)
    support, resistance = get_support_resistance(index)

    if pd.isna(support) or pd.isna(resistance):
        return "WAIT", None

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # Previous candle for breakout confirmation
    previous_close = df.iloc[index - 1]["close"]

    # Frozen V2-A/V2-B breakout confirmation
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

    if bullish_breakout:
        return "BUY", "BREAKOUT"

    if bearish_breakout:
        return "SELL", "BREAKOUT"

    # Frozen V2-B pullback logic
    bullish_pullback = (
        trend == "BULLISH"
        and current["low"] <= ema20
        and price > ema20
        and bullish_candle
    )

    bearish_pullback = (
        trend == "BEARISH"
        and current["high"] >= ema20
        and price < ema20
        and bearish_candle
    )

    if bullish_pullback:
        return "BUY", "PULLBACK"

    if bearish_pullback:
        return "SELL", "PULLBACK"

    return "WAIT", None


# ============================================================
# BACKTEST FROZEN V2-B
# ============================================================

trades = []

i = LOOKBACK + 50

while i < len(df) - 1:

    signal, setup = generate_v2b_signal(i)

    if signal == "WAIT":
        i += 1
        continue

    entry_index = i + 1

    if entry_index >= len(df):
        break

    signal_row = df.iloc[i]
    entry_row = df.iloc[entry_index]

    entry_price = entry_row["open"]
    entry_time = entry_row["date"]

    atr = signal_row["atr14"]

    if pd.isna(atr) or atr <= 0:
        i += 1
        continue

    risk = atr * ATR_STOP_MULTIPLIER

    if signal == "BUY":
        stop = entry_price - risk
        target = entry_price + risk * REWARD_R
    else:
        stop = entry_price + risk
        target = entry_price - risk * REWARD_R

    exit_price = None
    exit_time = None
    exit_reason = None

    j = entry_index

    max_favorable = 0.0
    max_adverse = 0.0

    while j < len(df):

        candle = df.iloc[j]

        if signal == "BUY":

            favorable = candle["high"] - entry_price
            adverse = entry_price - candle["low"]

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            # Conservative SL-first rule
            if candle["low"] <= stop:
                exit_price = stop
                exit_reason = "STOP_LOSS"
                exit_time = candle["date"]
                break

            if candle["high"] >= target:
                exit_price = target
                exit_reason = "TAKE_PROFIT"
                exit_time = candle["date"]
                break

        else:

            favorable = entry_price - candle["low"]
            adverse = candle["high"] - entry_price

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            if candle["high"] >= stop:
                exit_price = stop
                exit_reason = "STOP_LOSS"
                exit_time = candle["date"]
                break

            if candle["low"] <= target:
                exit_price = target
                exit_reason = "TAKE_PROFIT"
                exit_time = candle["date"]
                break

        j += 1

    if exit_price is None:
        final = df.iloc[-1]
        exit_price = final["close"]
        exit_time = final["date"]
        exit_reason = "OPEN_AT_END"

    if signal == "BUY":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    result_r = pnl / risk

    # Entry-time diagnostics
    trend = determine_trend(signal_row)

    support, resistance = get_support_resistance(i)

    distance_ema20 = (
        entry_price - signal_row["ema20"]
    )

    distance_ema20_atr = (
        distance_ema20 / atr
        if atr > 0 else np.nan
    )

    ema_spread = signal_row["ema_spread"]
    ema_spread_atr = (
        abs(ema_spread) / atr
        if atr > 0 else np.nan
    )

    trades.append({
        "signal_time": signal_row["date"],
        "entry_time": entry_time,
        "exit_time": exit_time,

        "year": entry_time.year,
        "month": entry_time.month,
        "month_name": entry_time.strftime("%Y-%m"),

        "signal": signal,
        "setup": setup,
        "trend": trend,

        "entry_price": entry_price,
        "atr14": atr,
        "atr_pct": signal_row["atr_pct"],

        "ema20": signal_row["ema20"],
        "ema50": signal_row["ema50"],
        "ema_spread": ema_spread,
        "ema_spread_pct": signal_row["ema_spread_pct"],
        "ema_spread_atr": ema_spread_atr,

        "rsi14": signal_row["rsi14"],

        "distance_ema20": distance_ema20,
        "distance_ema20_atr": distance_ema20_atr,

        "candle_range": signal_row["candle_range"],
        "body_size": signal_row["body_size"],
        "body_pct_range": signal_row["body_pct_range"],

        "risk_points": risk,
        "stop_loss": stop,
        "take_profit": target,

        "support": support,
        "resistance": resistance,

        "exit_price": exit_price,
        "exit_reason": exit_reason,

        "result_r": result_r,
        "max_favorable_r": max_favorable / risk,
        "max_adverse_r": max_adverse / risk,

        "holding_hours": (
            exit_time - entry_time
        ).total_seconds() / 3600
    })

    i = max(j + 1, i + 1)


trades_df = pd.DataFrame(trades)

if trades_df.empty:
    raise ValueError("No trades generated.")


# ============================================================
# CREATE PERIOD LABELS
# ============================================================

trades_df["period"] = np.where(
    trades_df["year"] <= 2025,
    "2024-2025",
    "2026"
)

trades_df["rsi_band"] = pd.cut(
    trades_df["rsi14"],
    bins=[49.999, 55, 60, 65, 70.001],
    labels=["50-55", "55-60", "60-65", "65-70"]
)

# ATR quartiles calculated from entry trades
atr_q = trades_df["atr14"].quantile([0.25, 0.50, 0.75]).values

trades_df["atr_regime"] = pd.cut(
    trades_df["atr14"],
    bins=[
        -np.inf,
        atr_q[0],
        atr_q[1],
        atr_q[2],
        np.inf
    ],
    labels=[
        "LOW",
        "MEDIUM-LOW",
        "MEDIUM-HIGH",
        "HIGH"
    ]
)

# EMA trend-strength buckets
trades_df["trend_strength"] = pd.cut(
    trades_df["ema_spread_atr"],
    bins=[-np.inf, 0.25, 0.5, 1.0, np.inf],
    labels=[
        "WEAK <0.25 ATR",
        "0.25-0.50 ATR",
        "0.50-1.00 ATR",
        ">1.00 ATR"
    ]
)

# Entry distance from EMA20
trades_df["ema_distance_regime"] = pd.cut(
    trades_df["distance_ema20_atr"].abs(),
    bins=[-np.inf, 0.25, 0.5, 1.0, np.inf],
    labels=[
        "<0.25 ATR",
        "0.25-0.50 ATR",
        "0.50-1.00 ATR",
        ">1.00 ATR"
    ]
)


# ============================================================
# STATISTICS HELPERS
# ============================================================

def stats(data):
    if len(data) == 0:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "total_r": 0,
            "avg_r": np.nan,
            "profit_factor": np.nan,
            "avg_mfe_r": np.nan,
            "avg_mae_r": np.nan
        }

    wins = data[data["result_r"] > 0]
    losses = data[data["result_r"] < 0]

    gp = wins["result_r"].sum()
    gl = abs(losses["result_r"].sum())

    pf = gp / gl if gl > 0 else np.inf

    return {
        "trades": len(data),
        "win_rate": (data["result_r"] > 0).mean() * 100,
        "total_r": data["result_r"].sum(),
        "avg_r": data["result_r"].mean(),
        "profit_factor": pf,
        "avg_mfe_r": data["max_favorable_r"].mean(),
        "avg_mae_r": data["max_adverse_r"].mean()
    }


def grouped_stats(data, column):
    rows = []

    for key, group in data.groupby(column, observed=True):
        row = {column: key}
        row.update(stats(group))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAX DRAWDOWN
# ============================================================

trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

equity = trades_df["result_r"].cumsum()
peak = equity.cummax()
drawdown = equity - peak

trades_df["equity_r"] = equity
trades_df["drawdown_r"] = drawdown


# ============================================================
# REPORT
# ============================================================

lines = []

def report(text=""):
    print(text)
    lines.append(text)


report("=" * 78)
report("V2-B 2026 FORENSIC REPORT")
report("=" * 78)

overall = stats(trades_df)

report("\nOVERALL")
report(pd.DataFrame([overall]).to_string(index=False))

report("\nPERIOD")
report(grouped_stats(trades_df, "period").to_string(index=False))

report("\nMONTH")
report(grouped_stats(trades_df, "month_name").to_string(index=False))

report("\nDIRECTION")
report(grouped_stats(trades_df, "signal").to_string(index=False))

report("\nSETUP")
report(grouped_stats(trades_df, "setup").to_string(index=False))

report("\nTREND")
report(grouped_stats(trades_df, "trend").to_string(index=False))

report("\nRSI BAND")
report(grouped_stats(trades_df, "rsi_band").to_string(index=False))

report("\nATR REGIME")
report(grouped_stats(trades_df, "atr_regime").to_string(index=False))

report("\nTREND STRENGTH")
report(grouped_stats(trades_df, "trend_strength").to_string(index=False))

report("\nDISTANCE FROM EMA20")
report(
    grouped_stats(
        trades_df,
        "ema_distance_regime"
    ).to_string(index=False)
)

# 2026-specific slices
df2026 = trades_df[trades_df["year"] == 2026].copy()
df_pre2026 = trades_df[trades_df["year"] <= 2025].copy()

report("\n" + "=" * 78)
report("2026 vs 2024-2025 COMPARISON")
report("=" * 78)

comparison_rows = []

for name, data in [
    ("2024-2025", df_pre2026),
    ("2026", df2026)
]:
    s = stats(data)
    s["period"] = name
    comparison_rows.append(s)

comparison = pd.DataFrame(comparison_rows)

report(comparison.to_string(index=False))

report("\n2026 MONTHLY BREAKDOWN")
report(
    grouped_stats(
        df2026,
        "month_name"
    ).to_string(index=False)
)

report("\n2026 DIRECTION")
report(
    grouped_stats(
        df2026,
        "signal"
    ).to_string(index=False)
)

report("\n2026 SETUP")
report(
    grouped_stats(
        df2026,
        "setup"
    ).to_string(index=False)
)

report("\n2026 RSI")
report(
    grouped_stats(
        df2026,
        "rsi_band"
    ).to_string(index=False)
)

report("\n2026 ATR")
report(
    grouped_stats(
        df2026,
        "atr_regime"
    ).to_string(index=False)
)

report("\n2026 TREND STRENGTH")
report(
    grouped_stats(
        df2026,
        "trend_strength"
    ).to_string(index=False)
)

report("\n2026 EMA20 DISTANCE")
report(
    grouped_stats(
        df2026,
        "ema_distance_regime"
    ).to_string(index=False)
)

# Best/worst months
monthly = grouped_stats(df2026, "month_name")

if not monthly.empty:
    best_month = monthly.loc[
        monthly["total_r"].idxmax()
    ]

    worst_month = monthly.loc[
        monthly["total_r"].idxmin()
    ]

    report("\n2026 BEST MONTH")
    report(best_month.to_string())

    report("\n2026 WORST MONTH")
    report(worst_month.to_string())

# MFE/MAE
report("\nMFE / MAE")
report(
    pd.DataFrame([
        {
            "period": "2024-2025",
            "avg_MFE_R": df_pre2026["max_favorable_r"].mean(),
            "median_MFE_R": df_pre2026["max_favorable_r"].median(),
            "avg_MAE_R": df_pre2026["max_adverse_r"].mean(),
            "median_MAE_R": df_pre2026["max_adverse_r"].median()
        },
        {
            "period": "2026",
            "avg_MFE_R": df2026["max_favorable_r"].mean(),
            "median_MFE_R": df2026["max_favorable_r"].median(),
            "avg_MAE_R": df2026["max_adverse_r"].mean(),
            "median_MAE_R": df2026["max_adverse_r"].median()
        }
    ]).to_string(index=False)
)

# Exit reasons
report("\nEXIT REASONS")
report(
    trades_df.groupby(
        ["period", "exit_reason"]
    )["result_r"].agg(
        trades="count",
        total_r="sum",
        avg_r="mean"
    ).to_string()
)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

trades_df.to_csv(
    OUTPUT_FILE,
    index=False
)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

report("\n" + "=" * 78)
report("FILES SAVED")
report("=" * 78)
report(OUTPUT_FILE)
report(REPORT_FILE)

report("\nAnalysis complete.")
