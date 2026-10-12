"""
XAUUSD AI Copilot - V8 Asymmetric RSI Breakout Study

BUY breakout:
    RSI 50-70

SELL breakout candidates:
    30-40
    35-45
    40-50
    30-50

Everything else is unchanged:
- Previous 20-candle support/resistance
- Confirmed close breakout
- Entry next candle OPEN
- SL 1.5 ATR14
- TP fixed 2R
- One position at a time
- Same-candle SL/TP -> SL first

Purpose:
Determine whether bearish breakout trades need a different RSI
range than bullish breakouts.

This is a research experiment. No SELL RSI range is adopted
until walk-forward results are reviewed.
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
BUY_RSI = (50, 70)
SELL_RSI_RANGES = {
    "SELL_30_40": (30, 40),
    "SELL_35_45": (35, 45),
    "SELL_40_50": (40, 50),
    "SELL_30_50": (30, 50),
}

ATR_STOP_MULT = 1.5
RR = 2.0


def add_levels(df):
    df = df.copy()
    df["resistance"] = df["high"].rolling(LOOKBACK).max().shift(1)
    df["support"] = df["low"].rolling(LOOKBACK).min().shift(1)
    return df


def get_signal(df, i, sell_rsi):
    c = df.iloc[i]
    p = df.iloc[i - 1]

    required = [
        "ema20", "ema50", "rsi14", "atr14",
        "resistance", "support"
    ]

    if any(pd.isna(c[x]) for x in required):
        return None

    price = c["close"]

    # BUY regime
    if price > c["ema20"] > c["ema50"]:
        bullish = c["close"] > c["open"]

        if BUY_RSI[0] <= c["rsi14"] < BUY_RSI[1]:
            if (
                price > c["resistance"]
                and p["close"] <= c["resistance"]
                and bullish
            ):
                return "BUY"

    # SELL regime
    if price < c["ema20"] < c["ema50"]:
        bearish = c["close"] < c["open"]

        if sell_rsi[0] <= c["rsi14"] < sell_rsi[1]:
            if (
                price < c["support"]
                and p["close"] >= c["support"]
                and bearish
            ):
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

        # Conservative same-candle rule.
        if hit_sl:
            return {
                "signal_time": s["date"],
                "entry_time": e["date"],
                "exit_time": df.iloc[j]["date"],
                "signal": signal,
                "entry_price": entry,
                "atr14": atr,
                "risk_points": risk,
                "stop_loss": stop,
                "take_profit": target,
                "rsi14": float(s["rsi14"]),
                "ema20": float(s["ema20"]),
                "ema50": float(s["ema50"]),
                "result_r": -1.0,
                "max_favorable_r": mfe,
                "max_adverse_r": mae,
                "exit_reason": "SL",
            }

        if hit_tp:
            return {
                "signal_time": s["date"],
                "entry_time": e["date"],
                "exit_time": df.iloc[j]["date"],
                "signal": signal,
                "entry_price": entry,
                "atr14": atr,
                "risk_points": risk,
                "stop_loss": stop,
                "take_profit": target,
                "rsi14": float(s["rsi14"]),
                "ema20": float(s["ema20"]),
                "ema50": float(s["ema50"]),
                "result_r": RR,
                "max_favorable_r": mfe,
                "max_adverse_r": mae,
                "exit_reason": "TP",
            }

    exit_i = len(df) - 1
    exit_price = float(df.iloc[exit_i]["close"])

    if signal == "BUY":
        result_r = (exit_price - entry) / risk
    else:
        result_r = (entry - exit_price) / risk

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
        "result_r": result_r,
        "max_favorable_r": mfe,
        "max_adverse_r": mae,
        "exit_reason": "OPEN_AT_END",
    }


def build_trades(df, sell_rsi, start=None, end=None):
    trades = []

    i = LOOKBACK + 55

    while i < len(df) - 1:

        signal_time = df.iloc[i]["date"]

        if start is not None and signal_time < start:
            i += 1
            continue

        if end is not None and signal_time > end:
            break

        signal = get_signal(df, i, sell_rsi)

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

        while (
            i < len(df)
            and df.iloc[i]["date"] <= exit_time
        ):
            i += 1

    return pd.DataFrame(trades)


def statistics(df):
    if df.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "total_r": 0,
            "avg_r": np.nan,
            "pf": np.nan,
            "max_dd": np.nan,
            "max_loss_streak": 0,
        }

    r = df["result_r"]

    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())

    equity = r.cumsum()
    dd = equity - equity.cummax()

    streak = 0
    max_streak = 0

    for value in r:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trades": len(df),
        "win_rate": (r > 0).mean() * 100,
        "total_r": r.sum(),
        "avg_r": r.mean(),
        "pf": gross_profit / gross_loss if gross_loss else np.inf,
        "max_dd": dd.min(),
        "max_loss_streak": max_streak,
    }


def show(label, df):
    s = statistics(df)

    print(
        f"{label:<34} "
        f"Trades={s['trades']:4d} "
        f"WR={s['win_rate']:6.2f}% "
        f"R={s['total_r']:8.2f} "
        f"AvgR={s['avg_r']:7.4f} "
        f"PF={s['pf']:6.3f} "
        f"MaxDD={s['max_dd']:7.2f} "
        f"MaxLossStreak={s['max_loss_streak']:2d}"
    )


def sell_monthly_report(trades):
    if trades.empty:
        return

    x = trades.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"])
    x["year"] = x["signal_time"].dt.year
    x["month"] = x["signal_time"].dt.month

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


def walk_forward(df, sell_rsi):
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

    forward = []

    for name, tr_start, tr_end, te_start, te_end in windows:

        train = build_trades(
            df,
            sell_rsi,
            pd.Timestamp(tr_start),
            pd.Timestamp(tr_end),
        )

        test = build_trades(
            df,
            sell_rsi,
            pd.Timestamp(te_start),
            pd.Timestamp(te_end),
        )

        print(f"\n{name}")
        show("Training", train)
        show("Forward test", test)

        if not test.empty:
            test = test.copy()
            test["window"] = name
            forward.append(test)

    if not forward:
        return pd.DataFrame()

    combined = pd.concat(forward, ignore_index=True)

    print("\nAGGREGATED FORWARD")
    show("All forward windows", combined)

    return combined


def main():

    print("=" * 90)
    print("XAUUSD V8 ASYMMETRIC RSI BREAKOUT STUDY")
    print("=" * 90)

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values("date").reset_index(drop=True)

    df = add_indicators(df)
    df = add_levels(df)

    print(f"Rows: {len(df):,}")

    all_results = []
    all_forward = []

    for name, sell_rsi in SELL_RSI_RANGES.items():

        print("\n" + "=" * 90)
        print(f"{name}: SELL RSI {sell_rsi[0]}-{sell_rsi[1]}")
        print("=" * 90)

        trades = build_trades(df, sell_rsi)

        if trades.empty:
            print("No trades generated.")
            continue

        trades["signal_time"] = pd.to_datetime(
            trades["signal_time"]
        )

        trades["year"] = trades.signal_time.dt.year
        trades["month"] = trades.signal_time.dt.month

        buy = trades[trades.signal == "BUY"]
        sell = trades[trades.signal == "SELL"]

        print("\nALL")
        show("All trades", trades)

        print("\nBUY")
        show("BUY", buy)

        print("\nSELL")
        show("SELL", sell)

        print("\nSELL by year")
        for year in sorted(sell.year.unique()):
            show(
                f"SELL {year}",
                sell[sell.year == year]
            )

        print("\nSELL monthly")
        sell_monthly_report(sell)

        print("\nWALK-FORWARD")
        wf = walk_forward(df, sell_rsi)

        all_results.append(
            trades.assign(rsi_variant=name)
        )

        if not wf.empty:
            all_forward.append(
                wf.assign(rsi_variant=name)
            )

    print("\n" + "=" * 90)
    print("SELL RSI COMPARISON")
    print("=" * 90)

    comparison = []

    for trades in all_results:

        sell = trades[trades.signal == "SELL"]

        s = statistics(sell)

        comparison.append({
            "rsi_variant": trades.rsi_variant.iloc[0],
            "sell_trades": s["trades"],
            "sell_win_rate": s["win_rate"],
            "sell_total_r": s["total_r"],
            "sell_avg_r": s["avg_r"],
            "sell_pf": s["pf"],
            "sell_max_dd": s["max_dd"],
        })

    comparison = pd.DataFrame(comparison)

    print(comparison.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades_output = OUTPUT_DIR / "v8_asymmetric_rsi_breakout_trades.csv"
    comparison_output = OUTPUT_DIR / "v8_sell_rsi_comparison.csv"

    pd.concat(all_results, ignore_index=True).to_csv(
        trades_output,
        index=False
    )

    comparison.to_csv(
        comparison_output,
        index=False
    )

    if all_forward:
        wf_output = OUTPUT_DIR / "v8_sell_rsi_walk_forward.csv"

        pd.concat(
            all_forward,
            ignore_index=True
        ).to_csv(
            wf_output,
            index=False
        )

        print("\nSaved walk-forward:", wf_output)

    print("\nSaved trades:", trades_output)
    print("Saved comparison:", comparison_output)

    print("\n" + "=" * 90)
    print("IMPORTANT")
    print("=" * 90)
    print("BUY RSI remains 50-70.")
    print("SELL RSI is being tested independently.")
    print("SL remains 1.5 ATR.")
    print("TP remains 2R.")
    print("No RSI range is adopted until walk-forward results are reviewed.")


if __name__ == "__main__":
    main()
