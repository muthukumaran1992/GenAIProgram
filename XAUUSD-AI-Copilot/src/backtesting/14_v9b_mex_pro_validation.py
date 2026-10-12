"""
V9-B — MEX Atlantic Pro Cost Calibration

Purpose
-------
Validate the frozen XAUUSD BUY-breakout strategy using configurable
MEX Atlantic Pro execution-cost assumptions.

FROZEN STRATEGY
---------------
BUY breakout only
RSI 50-70
Previous 20-candle resistance
Entry = next candle open
SL = 1.5 ATR14
TP = 2R
One position at a time
No pullbacks
No trailing
No breakeven
SELL disabled

COST MODEL
----------
The script separates:
1. Spread
2. Slippage
3. Commission (default 0 for Pro)
4. Overnight swap

IMPORTANT
---------
The supplied MEX information gives account-level spread information,
but not a verified historical XAUUSD spread series or exact XAUUSD
swap value for every day. Therefore:
- spread/slippage are configurable scenarios
- commission defaults to zero for Pro
- swap defaults to zero until the actual MT5 XAUUSD swap specification
  is supplied

Do NOT treat the output as exact broker execution until actual MT5
spread/swap data is available.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.technical_indicators import add_indicators

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "XAUUSD_m15_20240101_20260930.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# MEX PRO CALIBRATION SCENARIOS
# ------------------------------------------------------------
# These are sensitivity assumptions, not claims about actual
# historical execution.
#
# spread = total quoted bid/ask spread in XAUUSD price units
# slippage_each_side = adverse price movement on entry and exit
#
COST_SCENARIOS = [
    {
        "name": "PRO_TIGHT",
        "spread": 0.02,
        "slippage_each_side": 0.00,
    },
    {
        "name": "PRO_GOOD",
        "spread": 0.05,
        "slippage_each_side": 0.02,
    },
    {
        "name": "PRO_NORMAL",
        "spread": 0.10,
        "slippage_each_side": 0.03,
    },
    {
        "name": "PRO_CONSERVATIVE",
        "spread": 0.20,
        "slippage_each_side": 0.05,
    },
    {
        "name": "PRO_STRESS",
        "spread": 0.50,
        "slippage_each_side": 0.10,
    },
]

# Pro account: supplied information says commission-free.
COMMISSION_PER_LOT_USD = 0.0

# ------------------------------------------------------------
# SWAP
# ------------------------------------------------------------
# Keep this at zero until the actual XAUUSD swap from MT5 is known.
#
# Set this to the actual USD swap cost per lot per rollover night
# when available.
SWAP_COST_USD_PER_LOT_PER_NIGHT = 0.0

# For now, we report overnight trades separately.
# This is safer than inventing a swap rate.
# The strategy trades M15 and can hold beyond one calendar day.


def parse_dates(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            raise ValueError("No usable numeric timestamps.")

        median_value = values.abs().median()

        if median_value >= 1e11:
            unit = "ms"
        elif median_value >= 1e9:
            unit = "s"
        else:
            raise ValueError(
                f"Unrecognized numeric timestamp scale: {median_value}"
            )

        return pd.to_datetime(series, unit=unit, errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def load_data():
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = parse_dates(df["date"])

    if df["date"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return add_indicators(df)


def generate_frozen_trades(df):
    trades = []
    position = None

    for i in range(20, len(df) - 1):

        current = df.iloc[i]

        # ----------------------------------------------------
        # Manage open position
        # ----------------------------------------------------
        if position is not None:

            high = current["high"]
            low = current["low"]

            # Conservative assumption:
            # if SL and TP are both touched in the same M15 candle,
            # assume SL happened first.
            if low <= position["sl"]:

                exit_price = position["sl"]

                trades.append({
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": current["date"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "atr": position["atr"],
                    "risk_price": position["risk_price"],
                    "gross_r": -1.0,
                    "exit_reason": "SL",
                })

                position = None
                continue

            if high >= position["tp"]:

                exit_price = position["tp"]

                trades.append({
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": current["date"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "atr": position["atr"],
                    "risk_price": position["risk_price"],
                    "gross_r": 2.0,
                    "exit_reason": "TP",
                })

                position = None
                continue

        if position is not None:
            continue

        # ----------------------------------------------------
        # Frozen BUY breakout signal
        # ----------------------------------------------------
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

        atr = current["atr14"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_price = entry_bar["open"]
        risk_price = 1.5 * atr

        position = {
            "signal_date": current["date"],
            "entry_date": entry_bar["date"],
            "entry_price": entry_price,
            "atr": atr,
            "risk_price": risk_price,
            "sl": entry_price - risk_price,
            "tp": entry_price + 2.0 * risk_price,
        }

    # --------------------------------------------------------
    # Open position at dataset end
    # --------------------------------------------------------
    if position is not None:

        final_close = df.iloc[-1]["close"]

        gross_r = (
            final_close - position["entry_price"]
        ) / position["risk_price"]

        trades.append({
            "signal_date": position["signal_date"],
            "entry_date": position["entry_date"],
            "exit_date": df.iloc[-1]["date"],
            "entry_price": position["entry_price"],
            "exit_price": final_close,
            "atr": position["atr"],
            "risk_price": position["risk_price"],
            "gross_r": gross_r,
            "exit_reason": "OPEN_AT_END",
        })

    return pd.DataFrame(trades)


def calculate_overnight_nights(entry_date, exit_date):
    """
    Approximate number of calendar rollover nights crossed.

    This is intentionally conservative/simple.
    It is NOT the broker's exact rollover-time calculation.

    Once MT5 swap specifications and rollover time are known,
    this function should be replaced with broker-specific logic.
    """
    entry_date = pd.Timestamp(entry_date)
    exit_date = pd.Timestamp(exit_date)

    if exit_date.date() <= entry_date.date():
        return 0

    return max(0, (exit_date.date() - entry_date.date()).days)


def apply_costs(trades, spread, slippage_each_side):
    result = trades.copy()

    # Round-trip price cost.
    # For a BUY:
    # entry is adversely affected by spread/slippage,
    # and exit is adversely affected by slippage.
    #
    # This model deliberately uses a symmetric sensitivity model
    # rather than pretending to know the exact historical bid/ask path.
    round_trip_price_cost = (
        spread + 2.0 * slippage_each_side
    )

    result["spread_price_cost"] = spread
    result["slippage_price_cost"] = 2.0 * slippage_each_side

    result["execution_cost_price"] = round_trip_price_cost

    # Convert price cost into R using actual initial risk.
    result["execution_cost_r"] = (
        result["execution_cost_price"] /
        result["risk_price"]
    )

    # Commission is zero for supplied Pro account assumptions.
    result["commission_usd"] = COMMISSION_PER_LOT_USD
    result["commission_r"] = 0.0

    # Overnight calculation.
    result["overnight_nights"] = result.apply(
        lambda row: calculate_overnight_nights(
            row["entry_date"],
            row["exit_date"],
        ),
        axis=1,
    )

    result["swap_usd"] = (
        result["overnight_nights"]
        * SWAP_COST_USD_PER_LOT_PER_NIGHT
    )

    # Swap-to-R conversion is left at zero when swap rate is unknown.
    # We report the overnight exposure separately.
    result["swap_r"] = 0.0

    result["net_r"] = (
        result["gross_r"]
        - result["execution_cost_r"]
        - result["commission_r"]
        - result["swap_r"]
    )

    return result


def summarize(trades, scenario):
    resolved = trades[
        trades["exit_reason"] != "OPEN_AT_END"
    ].copy()

    wins = resolved[resolved["net_r"] > 0]
    losses = resolved[resolved["net_r"] < 0]

    gross_profit = wins["net_r"].sum()
    gross_loss = abs(losses["net_r"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    win_rate = (
        len(wins) / len(resolved)
        if len(resolved)
        else 0
    )

    equity = trades["net_r"].cumsum()
    drawdown = equity - equity.cummax()

    max_dd = (
        drawdown.min()
        if len(drawdown)
        else 0
    )

    max_losing_streak = 0
    current_streak = 0

    for value in resolved["net_r"]:

        if value < 0:
            current_streak += 1
            max_losing_streak = max(
                max_losing_streak,
                current_streak,
            )
        else:
            current_streak = 0

    overnight_trades = int(
        (trades["overnight_nights"] > 0).sum()
    )

    total_overnight_nights = int(
        trades["overnight_nights"].sum()
    )

    return {
        "scenario": scenario["name"],
        "spread_price": scenario["spread"],
        "slippage_each_side": scenario["slippage_each_side"],
        "commission_per_lot_usd": COMMISSION_PER_LOT_USD,
        "trades": len(trades),
        "resolved": len(resolved),
        "overnight_trades": overnight_trades,
        "overnight_nights": total_overnight_nights,
        "win_rate_pct": round(win_rate * 100, 2),
        "total_net_R": round(trades["net_r"].sum(), 2),
        "avg_net_R": round(trades["net_r"].mean(), 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_R": round(max_dd, 2),
        "max_losing_streak": max_losing_streak,
    }


def main():

    print("=" * 75)
    print("V9-B — MEX ATLANTIC PRO COST CALIBRATION")
    print("=" * 75)

    df = load_data()

    print(f"Data: {RAW_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")

    trades = generate_frozen_trades(df)

    if trades.empty:
        raise RuntimeError(
            "No trades generated. Check frozen strategy."
        )

    print(
        f"\nFrozen BUY-breakout strategy generated "
        f"{len(trades):,} trades."
    )

    print("\nBroker assumptions:")
    print("Account      : MEX Atlantic Pro")
    print("Commission   : $0 per supplied account information")
    print(
        f"Swap/lot/night: ${SWAP_COST_USD_PER_LOT_PER_NIGHT:.2f} "
        "(placeholder until actual MT5 value is supplied)"
    )

    results = []
    trade_results = []

    for scenario in COST_SCENARIOS:

        tested = apply_costs(
            trades,
            spread=scenario["spread"],
            slippage_each_side=scenario["slippage_each_side"],
        )

        summary = summarize(
            tested,
            scenario,
        )

        results.append(summary)

        tested["scenario"] = scenario["name"]

        trade_results.append(tested)

    comparison = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("V9-B MEX PRO COST SENSITIVITY")
    print("=" * 75)

    columns = [
        "scenario",
        "spread_price",
        "slippage_each_side",
        "total_net_R",
        "avg_net_R",
        "profit_factor",
        "max_drawdown_R",
        "max_losing_streak",
        "overnight_trades",
        "overnight_nights",
    ]

    print(
        comparison[columns].to_string(
            index=False
        )
    )

    comparison_file = (
        OUTPUT_DIR /
        "v9b_mex_pro_cost_comparison.csv"
    )

    trades_file = (
        OUTPUT_DIR /
        "v9b_mex_pro_cost_trades.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    pd.concat(
        trade_results,
        ignore_index=True,
    ).to_csv(
        trades_file,
        index=False,
    )

    print("\nSaved:")
    print(comparison_file)
    print(trades_file)

    print("\n" + "=" * 75)
    print("IMPORTANT")
    print("=" * 75)

    print(
        "1. Spread/slippage are sensitivity assumptions, "
        "not historical MEX execution data."
    )

    print(
        "2. Commission is set to zero based on the supplied "
        "Pro account information."
    )

    print(
        "3. Overnight trades are counted, but swap is zero "
        "until the actual MT5 XAUUSD swap rate is supplied."
    )

    print(
        "4. Do not treat the result as live-trading profitability "
        "until actual broker execution data is calibrated."
    )

    print(
        "5. If net PF remains >1 across realistic scenarios, "
        "the next phase is Monte Carlo + position sizing."
    )


if __name__ == "__main__":
    main()
