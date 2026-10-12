"""
XAUUSD AI Copilot - V7 Breakout-Only Backtest & Walk-Forward

Strategy frozen for this experiment:
- Breakout only; NO pullbacks
- RSI: 50 <= RSI14 < 70
- BUY: bullish breakout above prior 20-candle resistance
- SELL: bearish breakout below prior 20-candle support
- Entry: next available candle OPEN
- SL: 1.5 * ATR14
- TP: fixed 2R
- One position at a time
- Same-candle SL/TP: SL first
- No trailing / breakeven
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.technical_indicators import add_indicators

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "XAUUSD_m15_20240101_20260930.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LOOKBACK = 20
RSI_LOW = 50
RSI_HIGH = 70
ATR_STOP_MULT = 1.5
RR = 2.0


def add_levels(df):
    df = df.copy()
    # Prior candles only — no look-ahead.
    df["resistance"] = df["high"].rolling(LOOKBACK).max().shift(1)
    df["support"] = df["low"].rolling(LOOKBACK).min().shift(1)
    return df


def get_breakout_signal(df, i):
    c = df.iloc[i]
    p = df.iloc[i - 1]

    required = ["ema20", "ema50", "rsi14", "atr14", "resistance", "support"]
    if any(pd.isna(c[x]) for x in required):
        return None

    price = c["close"]

    if price > c["ema20"] > c["ema50"]:
        trend = "BULLISH"
    elif price < c["ema20"] < c["ema50"]:
        trend = "BEARISH"
    else:
        return None

    if not (RSI_LOW <= c["rsi14"] < RSI_HIGH):
        return None

    bullish = c["close"] > c["open"]
    bearish = c["close"] < c["open"]

    # Confirmed breakout: current close crosses the prior range.
    bullish_breakout = (
        trend == "BULLISH"
        and price > c["resistance"]
        and p["close"] <= c["resistance"]
        and bullish
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < c["support"]
        and p["close"] >= c["support"]
        and bearish
    )

    if bullish_breakout:
        return "BUY"

    if bearish_breakout:
        return "SELL"

    return None


def simulate_trade(df, signal_i, signal):
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None

    s = df.iloc[signal_i]
    e = df.iloc[entry_i]

    atr = float(s["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    entry = float(e["open"])
    risk = ATR_STOP_MULT * atr

    if signal == "BUY":
        stop = entry - risk
        target = entry + RR * risk
    else:
        stop = entry + risk
        target = entry - RR * risk

    mfe = 0.0
    mae = 0.0

    for j in range(entry_i, len(df)):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if signal == "BUY":
            mfe = max(mfe, (high - entry) / risk)
            mae = max(mae, (entry - low) / risk)
            hit_sl = low <= stop
            hit_tp = high >= target
        else:
            mfe = max(mfe, (entry - low) / risk)
            mae = max(mae, (high - entry) / risk)
            hit_sl = high >= stop
            hit_tp = low <= target

        # Conservative assumption when both happen in one candle.
        if hit_sl:
            result_r = -1.0
            exit_reason = "SL"
            exit_i = j
            break

        if hit_tp:
            result_r = RR
            exit_reason = "TP"
            exit_i = j
            break
    else:
        exit_i = len(df) - 1
        exit_price = float(df.iloc[exit_i]["close"])

        if signal == "BUY":
            result_r = (exit_price - entry) / risk
        else:
            result_r = (entry - exit_price) / risk

        exit_reason = "OPEN_AT_END"

    return {
        "signal_time": s["date"],
        "entry_time": e["date"],
        "exit_time": df.iloc[exit_i]["date"],
        "signal": signal,
        "entry_price": entry,
        "atr14": atr,
        "risk_points": risk,
        "stop_loss": stop,
        "take_profit": target,
        "rsi14": float(s["rsi14"]),
        "ema20": float(s["ema20"]),
        "ema50": float(s["ema50"]),
        "resistance": float(s["resistance"]),
        "support": float(s["support"]),
        "result_r": result_r,
        "max_favorable_r": mfe,
        "max_adverse_r": mae,
        "exit_reason": exit_reason,
    }


def build_trades(df, start=None, end=None):
    trades = []
    i = LOOKBACK + 55

    while i < len(df) - 1:
        signal_time = df.iloc[i]["date"]

        if start is not None and signal_time < start:
            i += 1
            continue

        if end is not None and signal_time > end:
            break

        signal = get_breakout_signal(df, i)

        if signal is None:
            i += 1
            continue

        trade = simulate_trade(df, i, signal)

        if trade is None:
            i += 1
            continue

        trades.append(trade)

        # One position at a time.
        exit_time = trade["exit_time"]
        i += 1

        while i < len(df) and df.iloc[i]["date"] <= exit_time:
            i += 1

    return pd.DataFrame(trades)


def stats(df):
    if len(df) == 0:
        return dict(
            trades=0, win_rate=np.nan, total_r=0.0,
            avg_r=np.nan, pf=np.nan, max_dd=np.nan,
            max_losing_streak=0
        )

    r = df["result_r"]
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())

    equity = r.cumsum()
    drawdown = equity - equity.cummax()

    streak = 0
    max_streak = 0
    for value in r:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return dict(
        trades=len(df),
        win_rate=(r > 0).mean() * 100,
        total_r=r.sum(),
        avg_r=r.mean(),
        pf=wins / losses if losses else np.inf,
        max_dd=drawdown.min(),
        max_losing_streak=max_streak,
    )


def show(label, df):
    s = stats(df)
    print(
        f"{label:<34} "
        f"Trades={s['trades']:4d} "
        f"WR={s['win_rate']:6.2f}% "
        f"R={s['total_r']:8.2f} "
        f"AvgR={s['avg_r']:7.4f} "
        f"PF={s['pf']:6.3f} "
        f"MaxDD={s['max_dd']:7.2f} "
        f"MaxLossStreak={s['max_losing_streak']:2d}"
    )


def monthly_report(trades):
    if trades.empty:
        return

    x = trades.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"])
    x["year"] = x["signal_time"].dt.year
    x["month"] = x["signal_time"].dt.month

    print("\nMONTHLY RESULTS")
    print("-" * 90)

    table = (
        x.groupby(["year", "month"])
        .agg(
            trades=("result_r", "size"),
            win_rate=("result_r", lambda z: (z > 0).mean() * 100),
            total_r=("result_r", "sum"),
            avg_r=("result_r", "mean"),
        )
        .reset_index()
    )

    print(table.to_string(index=False))


def direction_report(trades):
    print("\nDIRECTION RESULTS")
    print("-" * 90)

    table = (
        trades.groupby("signal")
        .agg(
            trades=("result_r", "size"),
            win_rate=("result_r", lambda z: (z > 0).mean() * 100),
            total_r=("result_r", "sum"),
            avg_r=("result_r", "mean"),
        )
        .reset_index()
    )

    print(table.to_string(index=False))


def walk_forward_report(df):
    """
    Fixed V7 breakout rules.
    No parameter optimization.
    Each forward period is unseen.
    """

    windows = [
        ("WF1", "2024-01-01", "2024-06-30 23:59:59",
               "2024-07-01", "2024-12-31 23:59:59"),
        ("WF2", "2024-07-01", "2024-12-31 23:59:59",
               "2025-01-01", "2025-06-30 23:59:59"),
        ("WF3", "2025-01-01", "2025-06-30 23:59:59",
               "2025-07-01", "2025-12-31 23:59:59"),
        ("WF4", "2025-07-01", "2025-12-31 23:59:59",
               "2026-01-01", "2026-06-30 23:59:59"),
        ("WF5", "2026-01-01", "2026-06-30 23:59:59",
               "2026-07-01", "2026-09-29 23:59:59"),
    ]

    print("\n" + "=" * 90)
    print("BREAKOUT-ONLY WALK-FORWARD VALIDATION")
    print("=" * 90)

    all_forward = []

    for name, train_start, train_end, test_start, test_end in windows:
        train_start = pd.Timestamp(train_start)
        train_end = pd.Timestamp(train_end)
        test_start = pd.Timestamp(test_start)
        test_end = pd.Timestamp(test_end)

        # Training is informational only here because the strategy is frozen.
        train = build_trades(df, train_start, train_end)
        test = build_trades(df, test_start, test_end)

        print(f"\n{name}")
        show("Training", train)
        show("Forward test", test)

        if not test.empty:
            test = test.copy()
            test["window"] = name
            all_forward.append(test)

    if all_forward:
        combined = pd.concat(all_forward, ignore_index=True)

        print("\nAGGREGATED FORWARD PERFORMANCE")
        show("All forward windows", combined)

        positive = 0
        for name in combined["window"].unique():
            if combined.loc[
                combined["window"] == name, "result_r"
            ].sum() > 0:
                positive += 1

        print(
            f"Positive forward windows: {positive}/"
            f"{combined['window'].nunique()}"
        )

        return combined

    return pd.DataFrame()


def main():
    print("=" * 90)
    print("XAUUSD V7 BREAKOUT-ONLY BACKTEST")
    print("=" * 90)

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    df = add_indicators(df)
    df = add_levels(df)

    print(f"Rows: {len(df):,}")
    print(f"Start: {df.date.min()}")
    print(f"End:   {df.date.max()}")

    print("\nBuilding BREAKOUT-ONLY trades...")
    trades = build_trades(df)

    if trades.empty:
        raise RuntimeError("No breakout trades generated.")

    trades["signal_time"] = pd.to_datetime(trades["signal_time"])
    trades["year"] = trades["signal_time"].dt.year

    print(f"Generated trades: {len(trades)}")

    print("\n" + "=" * 90)
    print("BASELINE RESULTS")
    print("=" * 90)

    show("All data", trades)

    for year in sorted(trades["year"].unique()):
        show(str(year), trades[trades.year == year])

    show(
        "2024-2025",
        trades[trades.year.isin([2024, 2025])]
    )

    show(
        "2026",
        trades[trades.year == 2026]
    )

    direction_report(trades)
    monthly_report(trades)

    # Exit breakdown
    print("\nEXIT RESULTS")
    print("-" * 90)
    exit_table = (
        trades.groupby("exit_reason")
        .agg(
            trades=("result_r", "size"),
            total_r=("result_r", "sum"),
            avg_r=("result_r", "mean"),
        )
        .reset_index()
    )
    print(exit_table.to_string(index=False))

    forward = walk_forward_report(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "v7_breakout_only_trades.csv"
    trades.to_csv(output, index=False)

    if not forward.empty:
        wf_output = OUTPUT_DIR / "v7_breakout_walk_forward.csv"
        forward.to_csv(wf_output, index=False)
        print("\nSaved walk-forward:", wf_output)

    print("\nSaved trades:", output)

    print("\n" + "=" * 90)
    print("STRATEGY IS FROZEN FOR THIS EXPERIMENT")
    print("=" * 90)
    print("BREAKOUT ONLY | RSI 50-70 | SL 1.5 ATR | TP 2R")
    print("NO PULLBACK | NO TRAILING | NO BREAKEVEN")
    print("\nDo not change parameters until the validation results are reviewed.")


if __name__ == "__main__":
    main()
