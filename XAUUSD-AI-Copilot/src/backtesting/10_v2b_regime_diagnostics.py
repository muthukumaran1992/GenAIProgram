r"""
XAUUSD V2-B Regime Diagnostics
--------------------------------
Diagnostic only. DOES NOT change V2-B trading rules.

Tests:
1. Trend Efficiency Ratio
2. ADX / Directional Strength
3. 50/55-day moving-average regime
4. Short-term vs long-term volatility
5. Existing EMA20 distance and trend strength for comparison

Base strategy:
- V2-B signal logic
- RSI 50-70
- Initial SL = 1.5 ATR
- Fixed TP = 2R
- Entry = next candle open
- If SL and TP are both touched in the same candle, SL is assumed first.

Run from:
E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\src\backtesting

python 10_v2b_regime_diagnostics.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot")
DATA = ROOT / "data" / "raw" / "XAUUSD_m15_20240101_20260930.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(df, n=14):
    d = df["close"].diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al
    return 100 - 100 / (1 + rs)


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)


def trend_efficiency(df, n):
    net = (df["close"] - df["close"].shift(n)).abs()
    path = df["close"].diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def adx(df, n=14):
    high = df["high"]
    low = df["low"]
    pc = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - pc).abs(),
        (low - pc).abs()
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )

    atr_w = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_w = plus_dm.ewm(alpha=1 / n, adjust=False).mean()
    minus_w = minus_dm.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * plus_w / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_w / atr_w.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    return dx.ewm(alpha=1 / n, adjust=False).mean()


def parse_datetime(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)

    # Robust parsing: first try normal datetime strings.
    parsed = pd.to_datetime(s, errors="coerce")

    # If that failed substantially, try numeric Unix timestamps.
    if parsed.notna().mean() < 0.90:
        numeric = pd.to_numeric(s, errors="coerce")
        med = numeric.dropna().abs().median()

        if med > 1e11:
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif med > 1e9:
            parsed = pd.to_datetime(numeric, unit="s", errors="coerce")

    return parsed


# ---------------------------------------------------------------------
# V2-B signal logic
# ---------------------------------------------------------------------
def build_signals(df):
    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df, 14)
    df["atr14"] = atr(df, 14)

    # Existing diagnostics
    df["ema_distance_atr"] = (
        (df["close"] - df["ema20"]).abs() / df["atr14"]
    )

    df["trend_strength_atr"] = (
        (df["ema20"] - df["ema50"]).abs() / df["atr14"]
    )

    # Candidate 1: Trend Efficiency
    for n in [8, 16, 32, 48]:
        df[f"eff_{n}"] = trend_efficiency(df, n)

    # Candidate 2: ADX
    df["adx14"] = adx(df, 14)

    # Candidate 3: 50/55-day moving average
    # M15 has approximately 96 candles/day.
    # Use rolling bars rather than calendar interpolation.
    for days in [50, 55]:
        n = days * 96
        df[f"ma_{days}d"] = df["close"].rolling(n).mean()
        df[f"ma_{days}d_slope"] = (
            df[f"ma_{days}d"] - df[f"ma_{days}d"].shift(96)
        )
        df[f"ma_{days}d_dist_atr"] = (
            (df["close"] - df[f"ma_{days}d"]) / df["atr14"]
        )

    # Candidate 4: volatility expansion
    df["tr"] = true_range(df)
    df["vol_short"] = df["tr"].rolling(16).mean()
    df["vol_long"] = df["tr"].rolling(96).mean()
    df["vol_ratio"] = df["vol_short"] / df["vol_long"].replace(0, np.nan)

    return df


def classify_base_signals(df):
    resistance = df["high"].shift(1).rolling(20).max()
    support = df["low"].shift(1).rolling(20).min()

    prev_close = df["close"].shift(1)
    bullish_candle = df["close"] > df["open"]
    bearish_candle = df["close"] < df["open"]

    trend = np.where(
        (df["close"] > df["ema20"]) & (df["ema20"] > df["ema50"]),
        "BULLISH",
        np.where(
            (df["close"] < df["ema20"]) & (df["ema20"] < df["ema50"]),
            "BEARISH",
            "SIDEWAYS"
        )
    )
    df["trend"] = trend

    bullish_breakout = (
        (df["trend"] == "BULLISH")
        & (df["close"] > resistance)
        & (prev_close <= resistance)
        & bullish_candle
    )

    bearish_breakout = (
        (df["trend"] == "BEARISH")
        & (df["close"] < support)
        & (prev_close >= support)
        & bearish_candle
    )

    bullish_pullback = (
        (df["trend"] == "BULLISH")
        & (df["low"] <= df["ema20"])
        & (df["close"] > df["ema20"])
        & bullish_candle
    )

    bearish_pullback = (
        (df["trend"] == "BEARISH")
        & (df["high"] >= df["ema20"])
        & (df["close"] < df["ema20"])
        & bearish_candle
    )

    # V2-B RSI filter: frozen.
    rsi_ok = df["rsi14"].between(50, 70, inclusive="both")

    df["signal"] = "WAIT"
    df.loc[bullish_breakout & rsi_ok, "signal"] = "BUY"
    df.loc[bearish_breakout & rsi_ok, "signal"] = "SELL"
    df.loc[bullish_pullback & rsi_ok, "signal"] = "BUY"
    df.loc[bearish_pullback & rsi_ok, "signal"] = "SELL"

    df["setup_type"] = np.where(
        bullish_breakout | bearish_breakout,
        "BREAKOUT",
        np.where(
            bullish_pullback | bearish_pullback,
            "PULLBACK",
            ""
        )
    )

    return df


def run_backtest(df):
    trades = []

    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]

        if row["signal"] not in ("BUY", "SELL"):
            i += 1
            continue

        direction = row["signal"]
        entry_idx = i + 1
        entry = df.iloc[entry_idx]["open"]
        atr_value = row["atr14"]

        if not np.isfinite(atr_value) or atr_value <= 0:
            i += 1
            continue

        risk = 1.5 * atr_value

        if direction == "BUY":
            sl = entry - risk
            tp = entry + 2 * risk
        else:
            sl = entry + risk
            tp = entry - 2 * risk

        exit_idx = None
        result_r = None
        exit_reason = None
        max_favorable = 0.0
        max_adverse = 0.0

        for j in range(entry_idx, len(df)):
            c = df.iloc[j]

            if direction == "BUY":
                mfe = (c["high"] - entry) / risk
                mae = (entry - c["low"]) / risk
                max_favorable = max(max_favorable, mfe)
                max_adverse = max(max_adverse, mae)

                hit_sl = c["low"] <= sl
                hit_tp = c["high"] >= tp

                if hit_sl and hit_tp:
                    result_r = -1.0
                    exit_reason = "STOP_LOSS"
                    exit_idx = j
                    break
                elif hit_sl:
                    result_r = -1.0
                    exit_reason = "STOP_LOSS"
                    exit_idx = j
                    break
                elif hit_tp:
                    result_r = 2.0
                    exit_reason = "TAKE_PROFIT"
                    exit_idx = j
                    break

            else:
                mfe = (entry - c["low"]) / risk
                mae = (c["high"] - entry) / risk
                max_favorable = max(max_favorable, mfe)
                max_adverse = max(max_adverse, mae)

                hit_sl = c["high"] >= sl
                hit_tp = c["low"] <= tp

                if hit_sl and hit_tp:
                    result_r = -1.0
                    exit_reason = "STOP_LOSS"
                    exit_idx = j
                    break
                elif hit_sl:
                    result_r = -1.0
                    exit_reason = "STOP_LOSS"
                    exit_idx = j
                    break
                elif hit_tp:
                    result_r = 2.0
                    exit_reason = "TAKE_PROFIT"
                    exit_idx = j
                    break

        if exit_idx is None:
            exit_idx = len(df) - 1
            final_close = df.iloc[exit_idx]["close"]
            if direction == "BUY":
                result_r = (final_close - entry) / risk
            else:
                result_r = (entry - final_close) / risk
            exit_reason = "OPEN_AT_END"

        trades.append({
            "entry_time": df.iloc[entry_idx]["date"],
            "signal_time": row["date"],
            "direction": direction,
            "setup_type": row["setup_type"],
            "rsi": row["rsi14"],
            "atr": row["atr14"],
            "result_r": result_r,
            "exit_reason": exit_reason,
            "mfe_r": max_favorable,
            "mae_r": max_adverse,

            "eff_8": row["eff_8"],
            "eff_16": row["eff_16"],
            "eff_32": row["eff_32"],
            "eff_48": row["eff_48"],
            "adx14": row["adx14"],

            "ma50_dist_atr": row["ma_50d_dist_atr"],
            "ma55_dist_atr": row["ma_55d_dist_atr"],
            "ma50_slope": row["ma_50d_slope"],
            "ma55_slope": row["ma_55d_slope"],

            "vol_ratio": row["vol_ratio"],
            "ema_distance_atr": row["ema_distance_atr"],
            "trend_strength_atr": row["trend_strength_atr"],
        })

        # One position at a time.
        i = exit_idx + 1

    return pd.DataFrame(trades)


def pf(x):
    wins = x.loc[x["result_r"] > 0, "result_r"].sum()
    losses = -x.loc[x["result_r"] < 0, "result_r"].sum()
    return wins / losses if losses > 0 else np.nan


def summarize(g):
    g = g.dropna(subset=["result_r"])
    return pd.Series({
        "trades": len(g),
        "win_rate": (g["result_r"] > 0).mean() * 100,
        "total_r": g["result_r"].sum(),
        "avg_r": g["result_r"].mean(),
        "profit_factor": pf(g),
        "avg_mfe_r": g["mfe_r"].mean(),
        "avg_mae_r": g["mae_r"].mean(),
    })


def regime_report(trades, col, bins, labels):
    x = trades.copy()
    x["regime"] = pd.cut(
        x[col], bins=bins, labels=labels, include_lowest=True
    )
    return x.groupby("regime", observed=False).apply(
        summarize, include_groups=False
    ).reset_index()


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("XAUUSD V2-B REGIME DIAGNOSTICS")
print("=" * 78)

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip().str.lower()
df["date"] = parse_datetime(df["date"])
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Start: {df['date'].min()}")
print(f"End  : {df['date'].max()}")

df = build_signals(df)
df = classify_base_signals(df)

trades = run_backtest(df)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["period"] = np.where(
    trades["entry_time"] < "2026-01-01", "2024-2025", "2026"
)
trades["month"] = trades["entry_time"].dt.to_period("M").astype(str)

march = trades[trades["month"] == "2026-03"].copy()
pre2026 = trades[trades["period"] == "2024-2025"].copy()
post_march = trades[
    (trades["period"] == "2026") & (trades["month"] != "2026-03")
].copy()
all_2026 = trades[trades["period"] == "2026"].copy()


def print_summary(name, g):
    s = summarize(g)
    print(
        f"{name:18s} trades={int(s['trades']):4d} "
        f"WR={s['win_rate']:6.2f}% "
        f"R={s['total_r']:8.2f} "
        f"AvgR={s['avg_r']:7.4f} "
        f"PF={s['profit_factor']:6.3f} "
        f"MFE={s['avg_mfe_r']:6.3f} "
        f"MAE={s['avg_mae_r']:6.3f}"
    )


print("\nBASE PERFORMANCE")
print_summary("2024-2025", pre2026)
print_summary("2026", all_2026)
print_summary("2026 excl March", post_march)
print_summary("March 2026", march)


# ---------------------------------------------------------------------
# Candidate diagnostics
# ---------------------------------------------------------------------
tests = {
    "Trend Efficiency 8": (
        "eff_8",
        [0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01],
        ["<0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70",
         "0.70-0.85", ">0.85"]
    ),
    "Trend Efficiency 16": (
        "eff_16",
        [0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01],
        ["<0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70",
         "0.70-0.85", ">0.85"]
    ),
    "Trend Efficiency 32": (
        "eff_32",
        [0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01],
        ["<0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70",
         "0.70-0.85", ">0.85"]
    ),
    "Trend Efficiency 48": (
        "eff_48",
        [0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01],
        ["<0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70",
         "0.70-0.85", ">0.85"]
    ),
    "ADX14": (
        "adx14",
        [0, 15, 20, 25, 30, 40, 1000],
        ["<15", "15-20", "20-25", "25-30", "30-40", ">40"]
    ),
    "Volatility Ratio": (
        "vol_ratio",
        [0, 0.75, 1.0, 1.25, 1.5, 2.0, 1000],
        ["<0.75", "0.75-1", "1-1.25", "1.25-1.5",
         "1.5-2", ">2"]
    ),
    "EMA20 Distance": (
        "ema_distance_atr",
        [0, 0.25, 0.5, 1, 2, 1000],
        ["<0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]
    ),
    "Trend Strength": (
        "trend_strength_atr",
        [0, 0.25, 0.5, 1, 2, 1000],
        ["<0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]
    ),
}

all_results = []

for name, (col, bins, labels) in tests.items():
    print("\n" + "-" * 78)
    print(name)
    print("-" * 78)

    for period_name, g in [
        ("2024-2025", pre2026),
        ("2026 excl March", post_march),
        ("March 2026", march),
    ]:
        report = regime_report(g, col, bins, labels)
        report.insert(0, "period", period_name)
        report.insert(0, "parameter", name)
        all_results.append(report)

        print(f"\n{period_name}")
        print(
            report[
                ["regime", "trades", "win_rate", "total_r",
                 "avg_r", "profit_factor", "avg_mfe_r", "avg_mae_r"]
            ].to_string(index=False)
        )


# ---------------------------------------------------------------------
# Special March comparison
# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("MARCH 2026 VS OTHER PERIODS — MEDIAN DIAGNOSTICS")
print("=" * 78)

compare_cols = [
    "eff_8", "eff_16", "eff_32", "eff_48",
    "adx14", "ma50_dist_atr", "ma55_dist_atr",
    "ma50_slope", "ma55_slope",
    "vol_ratio", "ema_distance_atr",
    "trend_strength_atr", "mfe_r", "mae_r"
]

groups = {
    "2024-2025": pre2026,
    "2026 excl March": post_march,
    "March 2026": march,
}

rows = []
for label, g in groups.items():
    row = {"period": label, "trades": len(g)}
    for c in compare_cols:
        row[c] = g[c].median()
    rows.append(row)

median_df = pd.DataFrame(rows)
print(median_df.to_string(index=False))

# Save
regime_df = pd.concat(all_results, ignore_index=True)
regime_file = OUT / "v2b_regime_diagnostics.csv"
median_file = OUT / "v2b_regime_median_comparison.csv"
trades_file = OUT / "v2b_regime_diagnostic_trades.csv"

regime_df.to_csv(regime_file, index=False)
median_df.to_csv(median_file, index=False)
trades.to_csv(trades_file, index=False)

print("\nFILES SAVED")
print(regime_file)
print(median_file)
print(trades_file)

print("\nINTERPRETATION RULE")
print("Do NOT select a filter just because it improves March.")
print("A useful regime parameter should:")
print("1) separate March from healthy periods,")
print("2) show a similar relationship outside March, and")
print("3) retain enough trades to be statistically meaningful.")
print("=" * 78)
