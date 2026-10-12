"""
V9 — XAUUSD BUY Breakout Transaction-Cost & Slippage Validation

Frozen strategy:
    BUY only
    Breakout above previous 20-candle resistance
    RSI 50-70
    Entry = next candle open
    SL = 1.5 ATR14
    TP = 2R
    One position at a time
    No pullbacks
    No trailing stop
    No breakeven

Important:
This V9 test applies costs to the already-defined gross trade result.
It is a robustness/sensitivity test, not a broker-execution simulator.
Because the source data is M15 OHLC, exact bid/ask path inside a candle is
not available. Therefore spread/slippage are modeled as round-trip price
costs converted into R.

Edit COST_SCENARIOS below to match your broker once actual spread/slippage
data is available.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Allow imports when this script is run from src/backtesting
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.technical_indicators import add_indicators


RAW_FILE = PROJECT_ROOT / "data" / "raw" / "XAUUSD_m15_20240101_20260930.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Price units for XAUUSD.
# These are deliberately sensitivity scenarios, NOT claims about your broker.
# Round-trip slippage = entry slippage + exit slippage.
COST_SCENARIOS = [
    {"name": "ZERO_COST",       "spread": 0.00, "slippage_each_side": 0.00},
    {"name": "LOW_COST",        "spread": 0.20, "slippage_each_side": 0.05},
    {"name": "MODERATE_COST",   "spread": 0.50, "slippage_each_side": 0.10},
    {"name": "HIGH_COST",       "spread": 1.00, "slippage_each_side": 0.25},
    {"name": "VERY_HIGH_COST",  "spread": 1.50, "slippage_each_side": 0.50},
    {"name": "STRESS_TEST",     "spread": 2.00, "slippage_each_side": 1.00},
]


def load_data():
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Robust datetime handling for string / Unix seconds / Unix milliseconds.
    raw_date = df["date"]

    if pd.api.types.is_numeric_dtype(raw_date):
        sample = pd.to_numeric(raw_date, errors="coerce").dropna()
        if sample.empty:
            raise ValueError("Date column contains no usable numeric values.")

        median_value = sample.abs().median()

        if median_value >= 1e11:
            unit = "ms"
        elif median_value >= 1e9:
            unit = "s"
        else:
            raise ValueError(
                "Numeric timestamp format is ambiguous/unrecognized. "
                f"Median absolute value={median_value}"
            )

        df["date"] = pd.to_datetime(raw_date, unit=unit, errors="coerce")
    else:
        df["date"] = pd.to_datetime(raw_date, errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return add_indicators(df)


def generate_frozen_trades(df):
    """
    Recreates the frozen V7 BUY-only breakout strategy.

    Signal candle:
        close > previous 20-candle resistance
        previous close <= resistance
        bullish candle
        RSI 50-70
        bullish trend

    Entry:
        next candle OPEN

    Risk:
        1.5 ATR

    Reward:
        2R

    Same-candle SL/TP:
        conservative assumption: SL first
    """

    trades = []
    position = None

    for i in range(20, len(df) - 1):
        current = df.iloc[i]

        # Manage an existing trade using the current candle.
        if position is not None:
            high = current["high"]
            low = current["low"]

            # Conservative same-candle assumption: SL first.
            if low <= position["sl"]:
                exit_price = position["sl"]
                exit_reason = "SL"
                r_result = -1.0

                trades.append({
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": current["date"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "atr": position["atr"],
                    "risk_price": position["risk_price"],
                    "r_result": r_result,
                    "exit_reason": exit_reason,
                    "setup": "BREAKOUT",
                })
                position = None
                continue

            if high >= position["tp"]:
                exit_price = position["tp"]
                exit_reason = "TP"
                r_result = 2.0

                trades.append({
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": current["date"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "atr": position["atr"],
                    "risk_price": position["risk_price"],
                    "r_result": r_result,
                    "exit_reason": exit_reason,
                    "setup": "BREAKOUT",
                })
                position = None
                continue

        if position is not None:
            continue

        # Previous 20 candles only — avoids look-ahead.
        lookback = df.iloc[i - 20:i]
        resistance = lookback["high"].max()

        price = current["close"]
        previous_close = df.iloc[i - 1]["close"]

        bullish_candle = current["close"] > current["open"]

        trend_bullish = (
            price > current["ema20"]
            and current["ema20"] > current["ema50"]
        )

        breakout = (
            trend_bullish
            and price > resistance
            and previous_close <= resistance
            and bullish_candle
            and 50 <= current["rsi14"] <= 70
        )

        if not breakout:
            continue

        entry_bar = df.iloc[i + 1]

        entry_price = entry_bar["open"]
        atr = current["atr14"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        risk_price = 1.5 * atr
        sl = entry_price - risk_price
        tp = entry_price + 2.0 * risk_price

        position = {
            "signal_date": current["date"],
            "entry_date": entry_bar["date"],
            "entry_price": entry_price,
            "atr": atr,
            "risk_price": risk_price,
            "sl": sl,
            "tp": tp,
        }

    # If still open at the end, mark it separately.
    if position is not None:
        final_close = df.iloc[-1]["close"]
        r_result = (final_close - position["entry_price"]) / position["risk_price"]

        trades.append({
            "signal_date": position["signal_date"],
            "entry_date": position["entry_date"],
            "exit_date": df.iloc[-1]["date"],
            "entry_price": position["entry_price"],
            "exit_price": final_close,
            "atr": position["atr"],
            "risk_price": position["risk_price"],
            "r_result": r_result,
            "exit_reason": "OPEN_AT_END",
            "setup": "BREAKOUT",
        })

    return pd.DataFrame(trades)


def add_costs(trades, spread, slippage_each_side):
    result = trades.copy()

    # Round-trip price cost:
    # spread + entry slippage + exit slippage
    round_trip_price_cost = spread + (2.0 * slippage_each_side)

    # Convert price cost to R using each trade's actual initial risk.
    result["cost_price"] = round_trip_price_cost
    result["cost_r"] = result["cost_price"] / result["risk_price"]
    result["net_r"] = result["r_result"] - result["cost_r"]

    return result


def summarize(df, label):
    resolved = df[df["exit_reason"] != "OPEN_AT_END"].copy()

    wins = resolved[resolved["net_r"] > 0]
    losses = resolved[resolved["net_r"] < 0]

    gross_profit = wins["net_r"].sum()
    gross_loss = abs(losses["net_r"].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = len(wins) / len(resolved) if len(resolved) else 0
    total_r = df["net_r"].sum()
    avg_r = df["net_r"].mean()

    # Equity curve and max drawdown in R.
    equity = df["net_r"].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min() if len(drawdown) else 0

    # Losing streak.
    max_losing_streak = 0
    current_streak = 0
    for value in resolved["net_r"]:
        if value < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    return {
        "scenario": label,
        "trades": len(df),
        "resolved": len(resolved),
        "win_rate_pct": round(win_rate * 100, 2),
        "total_net_R": round(total_r, 2),
        "avg_net_R": round(avg_r, 4),
        "profit_factor": round(pf, 3),
        "max_drawdown_R": round(max_dd, 2),
        "max_losing_streak": max_losing_streak,
    }


def main():
    print("=" * 70)
    print("V9 — TRANSACTION COST & SLIPPAGE VALIDATION")
    print("=" * 70)

    df = load_data()

    print(f"Data: {RAW_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")

    trades = generate_frozen_trades(df)

    if trades.empty:
        raise RuntimeError("No trades generated. Check the frozen strategy logic.")

    print(f"\nFrozen strategy generated {len(trades):,} trades.")

    gross_summary = summarize(trades.assign(net_r=trades["r_result"]), "ZERO_COST_BASELINE")
    print("\nBASELINE")
    for k, v in gross_summary.items():
        print(f"{k:22}: {v}")

    all_results = []
    all_trade_results = []

    for scenario in COST_SCENARIOS:
        tested = add_costs(
            trades,
            spread=scenario["spread"],
            slippage_each_side=scenario["slippage_each_side"],
        )

        summary = summarize(tested, scenario["name"])
        summary["spread_price"] = scenario["spread"]
        summary["slippage_each_side"] = scenario["slippage_each_side"]
        summary["round_trip_cost_price"] = (
            scenario["spread"] + 2 * scenario["slippage_each_side"]
        )

        all_results.append(summary)

        tested["scenario"] = scenario["name"]
        all_trade_results.append(tested)

    comparison = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("V9 COST SENSITIVITY")
    print("=" * 70)
    print(
        comparison[
            [
                "scenario",
                "spread_price",
                "slippage_each_side",
                "total_net_R",
                "avg_net_R",
                "profit_factor",
                "max_drawdown_R",
                "win_rate_pct",
            ]
        ].to_string(index=False)
    )

    comparison_file = OUTPUT_DIR / "v9_transaction_cost_comparison.csv"
    trades_file = OUTPUT_DIR / "v9_transaction_cost_trades.csv"

    comparison.to_csv(comparison_file, index=False)
    pd.concat(all_trade_results, ignore_index=True).to_csv(trades_file, index=False)

    print("\nSaved:")
    print(comparison_file)
    print(trades_file)

    print("\nInterpretation:")
    print("- ZERO_COST is the frozen V7 baseline.")
    print("- Costs reduce expectancy because each trade pays a round-trip cost.")
    print("- Do not select the 'best' scenario; the purpose is robustness.")
    print("- If realistic broker costs make PF < 1, the strategy is not viable.")
    print("- If the strategy remains positive across plausible costs, proceed")
    print("  to V9 risk sizing + Monte Carlo validation.")
    print("- Exact broker spread/slippage should replace these sensitivity values.")


if __name__ == "__main__":
    main()
