import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# XAUUSD V2-B — EARLY FOLLOW-THROUGH ANALYSIS
# ============================================================
#
# Purpose:
#   Determine whether March 2026 failed because trades did not
#   show enough favorable movement soon after entry.
#
# IMPORTANT:
#   This is DIAGNOSTIC ONLY.
#   V2-B rules are NOT changed.
#   Fixed 1.5 ATR SL and 2R TP remain unchanged.
#
# Tests:
#   15 min
#   30 min
#   1 hour
#   2 hours
#   3 hours
#
# For each period we calculate:
#   - Maximum favorable excursion
#   - Maximum adverse excursion
#   - Percentage reaching +0.25R
#   - Percentage reaching +0.50R
#   - Percentage reaching +0.75R
#   - Percentage reaching +1.00R
#
# Comparison:
#   2024-2025
#   2026 excluding March
#   March 2026
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot"
)

DATA_FILE = BASE_DIR / "data" / "raw" / "XAUUSD_m15_20240101_20260930.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "v2b_early_followthrough_analysis.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

ATR_MULTIPLIER = 1.5
RR = 2.0

# 15-minute candle intervals
HORIZONS = {
    "15M": 1,
    "30M": 2,
    "1H": 4,
    "2H": 8,
    "3H": 12,
}


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading XAUUSD data...")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

print("Columns:", list(df.columns))

# Robust timestamp handling
df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError("Some timestamps could not be parsed.")

df = df.sort_values("date").reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Start: {df['date'].iloc[0]}")
print(f"End  : {df['date'].iloc[-1]}")


# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------

print("\nCalculating indicators...")

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1 / 14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1 / 14,
    adjust=False
).mean()

rs = avg_gain / avg_loss

df["rsi14"] = 100 - (100 / (1 + rs))

previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = (df["high"] - previous_close).abs()
tr3 = (df["low"] - previous_close).abs()

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["atr14"] = true_range.ewm(
    alpha=1 / 14,
    adjust=False
).mean()

df["ema20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["ema50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()


# ------------------------------------------------------------
# V2-B SIGNAL GENERATION
# ------------------------------------------------------------

print("\nGenerating V2-B signals...")

signals = []

LOOKBACK = 20

for i in range(LOOKBACK + 1, len(df) - 1):

    current = df.iloc[i]
    previous = df.iloc[i - 1]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]
    atr = current["atr14"]

    if pd.isna(atr) or pd.isna(rsi):
        continue

    trend = "SIDEWAYS"

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    # --------------------------------------------------------
    # Previous 20 candles
    # --------------------------------------------------------

    lookback = df.iloc[i - LOOKBACK:i]

    resistance = lookback["high"].max()
    support = lookback["low"].min()

    bullish_candle = current["close"] > current["open"]
    bearish_candle = current["close"] < current["open"]

    # --------------------------------------------------------
    # V2-B BREAKOUT
    # --------------------------------------------------------

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
        and previous["close"] <= resistance
        and bullish_candle
        and 50 <= rsi <= 70
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
        and previous["close"] >= support
        and bearish_candle
        and 50 <= rsi <= 70
    )

    # --------------------------------------------------------
    # V2-B PULLBACK
    # --------------------------------------------------------

    bullish_pullback = (
        trend == "BULLISH"
        and current["low"] <= ema20
        and price > ema20
        and bullish_candle
        and 50 <= rsi <= 70
    )

    bearish_pullback = (
        trend == "BEARISH"
        and current["high"] >= ema20
        and price < ema20
        and bearish_candle
        and 50 <= rsi <= 70
    )

    signal = None
    setup = None

    if bullish_breakout:
        signal = "BUY"
        setup = "BREAKOUT"

    elif bearish_breakout:
        signal = "SELL"
        setup = "BREAKOUT"

    elif bullish_pullback:
        signal = "BUY"
        setup = "PULLBACK"

    elif bearish_pullback:
        signal = "SELL"
        setup = "PULLBACK"

    if signal is None:
        continue

    # --------------------------------------------------------
    # ENTRY = NEXT CANDLE OPEN
    # --------------------------------------------------------

    entry_index = i + 1

    if entry_index >= len(df):
        continue

    entry_row = df.iloc[entry_index]

    entry_price = entry_row["open"]

    risk_points = ATR_MULTIPLIER * atr

    if signal == "BUY":

        stop_loss = entry_price - risk_points
        take_profit = entry_price + (risk_points * RR)

    else:

        stop_loss = entry_price + risk_points
        take_profit = entry_price - (risk_points * RR)

    signals.append({
        "signal_index": i,
        "entry_index": entry_index,
        "signal_time": current["date"],
        "entry_time": entry_row["date"],
        "signal": signal,
        "setup": setup,
        "rsi14": rsi,
        "entry_price": entry_price,
        "atr14": atr,
        "risk_points": risk_points,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    })


signals_df = pd.DataFrame(signals)

print(f"V2-B signals generated: {len(signals_df):,}")


# ------------------------------------------------------------
# EARLY FOLLOW-THROUGH CALCULATION
# ------------------------------------------------------------

results = []

print("\nCalculating early follow-through...")

for _, trade in signals_df.iterrows():

    entry_index = int(trade["entry_index"])

    entry_price = trade["entry_price"]
    risk = trade["risk_points"]

    signal = trade["signal"]

    record = trade.to_dict()

    # --------------------------------------------------------
    # Evaluate each horizon
    # --------------------------------------------------------

    for horizon_name, candles_forward in HORIZONS.items():

        start = entry_index

        end = min(
            entry_index + candles_forward,
            len(df) - 1
        )

        window = df.iloc[start:end + 1]

        if signal == "BUY":

            favorable_points = (
                window["high"].max() - entry_price
            )

            adverse_points = (
                entry_price - window["low"].min()
            )

        else:

            favorable_points = (
                entry_price - window["low"].min()
            )

            adverse_points = (
                window["high"].max() - entry_price
            )

        mfe_r = favorable_points / risk
        mae_r = adverse_points / risk

        record_h = record.copy()

        record_h["horizon"] = horizon_name
        record_h["candles"] = candles_forward

        record_h["mfe_r"] = mfe_r
        record_h["mae_r"] = mae_r

        record_h["reached_025R"] = mfe_r >= 0.25
        record_h["reached_050R"] = mfe_r >= 0.50
        record_h["reached_075R"] = mfe_r >= 0.75
        record_h["reached_100R"] = mfe_r >= 1.00
        record_h["reached_150R"] = mfe_r >= 1.50
        record_h["reached_200R"] = mfe_r >= 2.00

        results.append(record_h)


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# PERIOD CLASSIFICATION
# ------------------------------------------------------------

def classify_period(dt):

    year = dt.year
    month = dt.month

    if year in [2024, 2025]:

        return "2024-2025"

    if year == 2026 and month == 3:

        return "March 2026"

    if year == 2026:

        return "2026 excl March"

    return "OTHER"


results_df["period"] = results_df["entry_time"].apply(
    classify_period
)

results_df = results_df[
    results_df["period"] != "OTHER"
].copy()


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("XAUUSD V2-B EARLY FOLLOW-THROUGH ANALYSIS")
print("=" * 80)


summary_rows = []

for period in [
    "2024-2025",
    "2026 excl March",
    "March 2026"
]:

    period_df = results_df[
        results_df["period"] == period
    ]

    print("\n")
    print("-" * 80)
    print(period)
    print("-" * 80)

    for horizon in HORIZONS:

        h = period_df[
            period_df["horizon"] == horizon
        ]

        if len(h) == 0:
            continue

        row = {
            "period": period,
            "horizon": horizon,
            "trades": len(h),

            "avg_mfe_r": h["mfe_r"].mean(),
            "median_mfe_r": h["mfe_r"].median(),

            "avg_mae_r": h["mae_r"].mean(),
            "median_mae_r": h["mae_r"].median(),

            "pct_reached_025R":
                h["reached_025R"].mean() * 100,

            "pct_reached_050R":
                h["reached_050R"].mean() * 100,

            "pct_reached_075R":
                h["reached_075R"].mean() * 100,

            "pct_reached_100R":
                h["reached_100R"].mean() * 100,

            "pct_reached_150R":
                h["reached_150R"].mean() * 100,

            "pct_reached_200R":
                h["reached_200R"].mean() * 100,
        }

        summary_rows.append(row)

        print(
            f"{horizon:>4} | "
            f"Trades={len(h):4d} | "
            f"MFE={row['avg_mfe_r']:.3f}R | "
            f"MAE={row['avg_mae_r']:.3f}R | "
            f"+0.25R={row['pct_reached_025R']:.1f}% | "
            f"+0.50R={row['pct_reached_050R']:.1f}% | "
            f"+0.75R={row['pct_reached_075R']:.1f}% | "
            f"+1R={row['pct_reached_100R']:.1f}% | "
            f"+2R={row['pct_reached_200R']:.1f}%"
        )


summary_df = pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# MARCH VS NORMAL DIFFERENCE
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("MARCH 2026 VS NORMAL DIFFERENCE")
print("=" * 80)

normal = summary_df[
    summary_df["period"] == "2026 excl March"
].set_index("horizon")

march = summary_df[
    summary_df["period"] == "March 2026"
].set_index("horizon")


for horizon in HORIZONS:

    if horizon not in normal.index or horizon not in march.index:
        continue

    mfe_difference = (
        march.loc[horizon, "avg_mfe_r"]
        -
        normal.loc[horizon, "avg_mfe_r"]
    )

    mae_difference = (
        march.loc[horizon, "avg_mae_r"]
        -
        normal.loc[horizon, "avg_mae_r"]
    )

    followthrough_difference = (
        march.loc[horizon, "pct_reached_050R"]
        -
        normal.loc[horizon, "pct_reached_050R"]
    )

    print(
        f"{horizon:>4} | "
        f"MFE difference={mfe_difference:+.3f}R | "
        f"MAE difference={mae_difference:+.3f}R | "
        f"+0.50R difference={followthrough_difference:+.1f} percentage points"
    )


# ------------------------------------------------------------
# PULLBACK VS BREAKOUT — MARCH
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("MARCH 2026 — PULLBACK VS BREAKOUT")
print("=" * 80)

march_df = results_df[
    results_df["period"] == "March 2026"
]

for setup in ["PULLBACK", "BREAKOUT"]:

    setup_df = march_df[
        march_df["setup"] == setup
    ]

    print(f"\n{setup}")

    for horizon in HORIZONS:

        h = setup_df[
            setup_df["horizon"] == horizon
        ]

        if len(h) == 0:
            continue

        print(
            f"{horizon:>4} | "
            f"Trades={len(h):2d} | "
            f"MFE={h['mfe_r'].mean():.3f}R | "
            f"MAE={h['mae_r'].mean():.3f}R | "
            f"+0.5R={h['reached_050R'].mean()*100:.1f}%"
        )


# ------------------------------------------------------------
# RSI BAND — MARCH
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("MARCH 2026 — RSI BAND EARLY FOLLOW-THROUGH")
print("=" * 80)

march_df["rsi_band"] = pd.cut(
    march_df["rsi14"],
    bins=[50, 55, 60, 65, 70],
    labels=["50-55", "55-60", "60-65", "65-70"],
    include_lowest=True
)

for band in ["50-55", "55-60", "60-65", "65-70"]:

    band_df = march_df[
        march_df["rsi_band"] == band
    ]

    if len(band_df) == 0:
        continue

    h = band_df[
        band_df["horizon"] == "1H"
    ]

    if len(h) == 0:
        continue

    print(
        f"RSI {band:>5} | "
        f"Trades={len(h):2d} | "
        f"MFE={h['mfe_r'].mean():.3f}R | "
        f"MAE={h['mae_r'].mean():.3f}R | "
        f"+0.5R={h['reached_050R'].mean()*100:.1f}%"
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 80)
print("FILES")
print("=" * 80)

print(
    f"Summary saved to:\n{OUTPUT_FILE}"
)

print("\nAnalysis complete.")