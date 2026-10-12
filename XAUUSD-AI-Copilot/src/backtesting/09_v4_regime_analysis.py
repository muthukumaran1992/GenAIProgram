import sys
import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\raw\XAUUSD_m15_20240101_20260930.csv"
)

OUTPUT_FILE = Path(
    r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\data\processed\v4_regime_analysis.csv"
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

print("=" * 70)
print("XAUUSD V4 MARKET REGIME ANALYSIS")
print("=" * 70)

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

    # V2-A breakout
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
# BACKTEST
# ============================================================

trades = []

i = START_INDEX

while i < len(df) - 1:

    signal = df.iloc[i]["signal"]

    if signal not in ["BUY", "SELL"]:
        i += 1
        continue

    entry_index = i + 1

    entry_candle = df.iloc[entry_index]

    entry = entry_candle["open"]

    atr = df.iloc[i]["atr14"]
    rsi = df.iloc[i]["rsi14"]

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

            # Conservative SL first
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

    holding_hours = (
        df.iloc[exit_index]["date"]
        - entry_candle["date"]
    ).total_seconds() / 3600

    trades.append({

        "signal_time": df.iloc[i]["date"],
        "entry_time": entry_candle["date"],
        "exit_time": df.iloc[exit_index]["date"],

        "year": df.iloc[i]["date"].year,

        "direction": signal,
        "setup_type": df.iloc[i]["setup_type"],
        "trend": df.iloc[i]["trend"],

        "rsi": rsi,
        "atr": atr,

        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "exit_price": exit_price,

        "result": result,
        "r_multiple": r_multiple,

        "holding_hours": holding_hours
    })

    i = exit_index + 1


trades = pd.DataFrame(trades)

resolved = trades[
    trades["result"].isin(["WIN", "LOSS"])
].copy()


# ============================================================
# ADD REGIME CLASSIFICATIONS
# ============================================================

# ATR quartiles based on the complete trade population
atr_q25 = resolved["atr"].quantile(0.25)
atr_q50 = resolved["atr"].quantile(0.50)
atr_q75 = resolved["atr"].quantile(0.75)

def atr_regime(value):

    if value <= atr_q25:
        return "LOW"

    elif value <= atr_q50:
        return "MEDIUM-LOW"

    elif value <= atr_q75:
        return "MEDIUM-HIGH"

    return "HIGH"


resolved["atr_regime"] = resolved["atr"].apply(
    atr_regime
)


# RSI regimes
def rsi_regime(value):

    if value < 40:
        return "<40"

    elif value < 50:
        return "40-50"

    elif value < 60:
        return "50-60"

    elif value <= 70:
        return "60-70"

    return ">70"


resolved["rsi_regime"] = resolved["rsi"].apply(
    rsi_regime
)


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def print_group_analysis(data, column, title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"{column:<18}"
        f"{'Trades':>10}"
        f"{'Win%':>10}"
        f"{'Total R':>12}"
        f"{'Avg R':>12}"
        f"{'PF':>10}"
    )

    print("-" * 70)

    for value in data[column].dropna().unique():

        subset = data[
            data[column] == value
        ]

        wins = subset[
            subset["result"] == "WIN"
        ]

        losses = subset[
            subset["result"] == "LOSS"
        ]

        gross_profit = wins["r_multiple"].sum()
        gross_loss = abs(
            losses["r_multiple"].sum()
        )

        pf = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        )

        win_rate = (
            len(wins) / len(subset) * 100
        )

        print(
            f"{str(value):<18}"
            f"{len(subset):>10}"
            f"{win_rate:>9.2f}%"
            f"{subset['r_multiple'].sum():>11.2f}R"
            f"{subset['r_multiple'].mean():>11.4f}"
            f"{pf:>10.3f}"
        )


# ============================================================
# OVERALL
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL")
print("=" * 70)

wins = resolved[
    resolved["result"] == "WIN"
]

losses = resolved[
    resolved["result"] == "LOSS"
]

gross_profit = wins["r_multiple"].sum()
gross_loss = abs(losses["r_multiple"].sum())

pf = gross_profit / gross_loss

print(f"Total trades : {len(resolved):,}")
print(
    f"Win rate    : "
    f"{len(wins) / len(resolved) * 100:.2f}%"
)
print(
    f"Total R     : "
    f"{resolved['r_multiple'].sum():.2f}R"
)
print(
    f"Average R   : "
    f"{resolved['r_multiple'].mean():.4f}R"
)
print(f"Profit factor: {pf:.3f}")


# ============================================================
# YEAR
# ============================================================

print_group_analysis(
    resolved,
    "year",
    "YEAR ANALYSIS"
)


# ============================================================
# DIRECTION
# ============================================================

print_group_analysis(
    resolved,
    "direction",
    "DIRECTION ANALYSIS"
)


# ============================================================
# SETUP
# ============================================================

print_group_analysis(
    resolved,
    "setup_type",
    "SETUP ANALYSIS"
)


# ============================================================
# TREND
# ============================================================

print_group_analysis(
    resolved,
    "trend",
    "TREND ANALYSIS"
)


# ============================================================
# ATR REGIME
# ============================================================

print_group_analysis(
    resolved,
    "atr_regime",
    "ATR / VOLATILITY REGIME"
)


# ============================================================
# RSI REGIME
# ============================================================

print_group_analysis(
    resolved,
    "rsi_regime",
    "RSI REGIME"
)


# ============================================================
# YEAR + DIRECTION
# ============================================================

print("\n")
print("=" * 70)
print("YEAR + DIRECTION")
print("=" * 70)

for year in sorted(resolved["year"].unique()):

    for direction in ["BUY", "SELL"]:

        subset = resolved[
            (resolved["year"] == year)
            & (resolved["direction"] == direction)
        ]

        if subset.empty:
            continue

        wins = subset[
            subset["result"] == "WIN"
        ]

        losses = subset[
            subset["result"] == "LOSS"
        ]

        gp = wins["r_multiple"].sum()
        gl = abs(losses["r_multiple"].sum())

        pf = gp / gl if gl > 0 else np.inf

        print(
            f"{year} {direction:<5} "
            f"Trades={len(subset):4d} "
            f"Win={len(wins)/len(subset)*100:6.2f}% "
            f"R={subset['r_multiple'].sum():7.2f} "
            f"AvgR={subset['r_multiple'].mean():7.4f} "
            f"PF={pf:.3f}"
        )


# ============================================================
# YEAR + SETUP
# ============================================================

print("\n")
print("=" * 70)
print("YEAR + SETUP")
print("=" * 70)

for year in sorted(resolved["year"].unique()):

    for setup in ["BREAKOUT", "PULLBACK"]:

        subset = resolved[
            (resolved["year"] == year)
            & (resolved["setup_type"] == setup)
        ]

        if subset.empty:
            continue

        wins = subset[
            subset["result"] == "WIN"
        ]

        losses = subset[
            subset["result"] == "LOSS"
        ]

        gp = wins["r_multiple"].sum()
        gl = abs(losses["r_multiple"].sum())

        pf = gp / gl if gl > 0 else np.inf

        print(
            f"{year} {setup:<9} "
            f"Trades={len(subset):4d} "
            f"Win={len(wins)/len(subset)*100:6.2f}% "
            f"R={subset['r_multiple'].sum():7.2f} "
            f"AvgR={subset['r_multiple'].mean():7.4f} "
            f"PF={pf:.3f}"
        )


# ============================================================
# HOLDING TIME
# ============================================================

print("\n")
print("=" * 70)
print("HOLDING TIME")
print("=" * 70)

print(
    f"Average holding : "
    f"{resolved['holding_hours'].mean():.2f} hours"
)

print(
    f"Median holding  : "
    f"{resolved['holding_hours'].median():.2f} hours"
)

print(
    f"Maximum holding : "
    f"{resolved['holding_hours'].max():.2f} hours"
)


# ============================================================
# 2024-25 VS 2026
# ============================================================

resolved["period"] = np.where(
    resolved["year"] <= 2025,
    "2024-2025",
    "2026"
)

print_group_analysis(
    resolved,
    "period",
    "2024-2025 VS 2026"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

resolved.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("Trade/regime data saved to:")
print(OUTPUT_FILE)