# XAUUSD V6 Regime Score Experiment
# Run from: E:\Social Eagle\GenAIProgram\XAUUSD-AI-Copilot\src\backtesting
# Command: python 12_regime_score.py

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

RSI_LOW, RSI_HIGH = 50, 70
LOOKBACK = 20
ATR_STOP_MULT, RR = 1.5, 2.0
TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
MARCH_START = pd.Timestamp("2026-03-01")
MARCH_END = pd.Timestamp("2026-03-31 23:59:59")

FEATURES = [
    "efficiency_16",
    "adx14",
    "volatility_ratio",
    "ema20_displacement",
    "trend_strength",
]

def add_regime_features(df):
    df = df.copy()

    move = (df["close"] - df["close"].shift(16)).abs()
    path = df["close"].diff().abs().rolling(16).sum()
    df["efficiency_16"] = move / path

    p = 14
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    plus = plus_dm.ewm(alpha=1/p, adjust=False).mean()
    minus = minus_dm.ewm(alpha=1/p, adjust=False).mean()

    plus_di = 100 * plus / atr
    minus_di = 100 * minus / atr
    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / den
    df["adx14"] = dx.ewm(alpha=1/p, adjust=False).mean()

    atr100 = tr.ewm(alpha=1/100, adjust=False).mean()
    df["volatility_ratio"] = df["atr14"] / atr100
    df["ema20_displacement"] = (df["close"] - df["ema20"]).abs() / df["atr14"]
    df["trend_strength"] = (df["ema20"] - df["ema50"]).abs() / df["atr14"]

    return df

def get_signal(df, i):
    c, prev = df.iloc[i], df.iloc[i-1]
    if any(pd.isna(c[x]) for x in ["ema20","ema50","rsi14","atr14","resistance","support"]):
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

    bull = c["close"] > c["open"]
    bear = c["close"] < c["open"]

    if (trend == "BULLISH" and price > c["resistance"]
        and prev["close"] <= c["resistance"] and bull):
        return "BUY", "BREAKOUT"

    if (trend == "BEARISH" and price < c["support"]
        and prev["close"] >= c["support"] and bear):
        return "SELL", "BREAKOUT"

    if (trend == "BULLISH" and c["low"] <= c["ema20"]
        and price > c["ema20"] and bull):
        return "BUY", "PULLBACK"

    if (trend == "BEARISH" and c["high"] >= c["ema20"]
        and price < c["ema20"] and bear):
        return "SELL", "PULLBACK"

    return None

def simulate_trade(df, signal_i, signal, setup):
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None

    s, e = df.iloc[signal_i], df.iloc[entry_i]
    atr = float(s["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    entry = float(e["open"])
    risk = ATR_STOP_MULT * atr
    stop = entry - risk if signal == "BUY" else entry + risk
    target = entry + RR*risk if signal == "BUY" else entry - RR*risk

    mfe = mae = 0.0

    for j in range(entry_i, len(df)):
        r = df.iloc[j]
        hi, lo = float(r["high"]), float(r["low"])

        if signal == "BUY":
            mfe = max(mfe, (hi-entry)/risk)
            mae = max(mae, (entry-lo)/risk)
            hit_sl, hit_tp = lo <= stop, hi >= target
        else:
            mfe = max(mfe, (entry-lo)/risk)
            mae = max(mae, (hi-entry)/risk)
            hit_sl, hit_tp = hi >= stop, lo <= target

        # Conservative same-candle rule: SL first.
        if hit_sl:
            exit_price, result, reason = stop, -1.0, "SL"
            exit_i = j
            break
        if hit_tp:
            exit_price, result, reason = target, RR, "TP"
            exit_i = j
            break
    else:
        exit_i = len(df)-1
        exit_price = float(df.iloc[exit_i]["close"])
        result = ((exit_price-entry)/risk if signal=="BUY"
                  else (entry-exit_price)/risk)
        reason = "OPEN_AT_END"

    return {
        "signal_time": s["date"], "entry_time": e["date"],
        "exit_time": df.iloc[exit_i]["date"],
        "signal": signal, "setup": setup,
        "entry_price": entry, "atr14": atr,
        "risk_points": risk, "stop_loss": stop, "take_profit": target,
        "rsi14": float(s["rsi14"]), "ema20": float(s["ema20"]),
        "ema50": float(s["ema50"]),
        **{f: float(s[f]) for f in FEATURES},
        "result_r": result, "max_favorable_r": mfe,
        "max_adverse_r": mae, "exit_reason": reason,
    }

def build_trades(df):
    trades = []
    i = LOOKBACK + 20

    while i < len(df)-1:
        sig = get_signal(df, i)
        if sig is None:
            i += 1
            continue

        signal, setup = sig
        trade = simulate_trade(df, i, signal, setup)
        if trade is None:
            i += 1
            continue

        trades.append(trade)
        exit_time = trade["exit_time"]

        i += 1
        while i < len(df) and df.iloc[i]["date"] <= exit_time:
            i += 1

    return pd.DataFrame(trades)

def stats(x):
    if len(x) == 0:
        return (0, np.nan, 0, np.nan, np.nan)
    wins = x.loc[x.result_r > 0, "result_r"].sum()
    losses = abs(x.loc[x.result_r < 0, "result_r"].sum())
    pf = wins/losses if losses else np.inf
    return len(x), (x.result_r > 0).mean()*100, x.result_r.sum(), x.result_r.mean(), pf

def show(label, x):
    n, wr, total, avg, pf = stats(x)
    print(f"{label:<34} Trades={n:4d} WR={wr:6.2f}% R={total:8.2f} AvgR={avg:7.4f} PF={pf:6.3f}")

def learn_maps(train):
    maps = {}
    for f in FEATURES:
        v = train[[f, "result_r"]].dropna()
        _, bins = pd.qcut(v[f], 5, retbins=True, duplicates="drop")
        bins = np.unique(bins)
        bins[0], bins[-1] = -np.inf, np.inf
        b = pd.cut(v[f], bins=bins, labels=False, include_lowest=True)
        t = v.copy()
        t["bin"] = b
        overall = t.result_r.mean()
        s = t.groupby("bin", observed=True).agg(
            trades=("result_r","size"),
            avg_r=("result_r","mean"),
            total_r=("result_r","sum")
        ).reset_index()
        s["good"] = s.avg_r >= overall
        maps[f] = {"bins": bins, "stats": s}
    return maps

def apply_score(trades, maps):
    out = trades.copy()
    out["regime_score"] = 0
    for f in FEATURES:
        b = pd.cut(out[f], bins=maps[f]["bins"], labels=False, include_lowest=True)
        good = maps[f]["stats"].set_index("bin")["good"].to_dict()
        flag = b.map(good).fillna(False).astype(bool)
        out[f+"_good"] = flag
        out["regime_score"] += flag.astype(int)
    return out

def score_table(x):
    return x.groupby("regime_score").agg(
        trades=("result_r","size"),
        win_rate=("result_r", lambda z: (z>0).mean()*100),
        total_r=("result_r","sum"),
        avg_r=("result_r","mean")
    ).reset_index()

def main():
    print("="*90)
    print("XAUUSD V6 REGIME SCORE EXPERIMENT")
    print("="*90)

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df = add_indicators(df)
    df["resistance"] = df.high.rolling(LOOKBACK).max().shift(1)
    df["support"] = df.low.rolling(LOOKBACK).min().shift(1)
    df = add_regime_features(df)

    trades = build_trades(df)
    if trades.empty:
        raise RuntimeError("No trades generated.")

    trades["signal_time"] = pd.to_datetime(trades.signal_time)
    trades["year"] = trades.signal_time.dt.year
    trades["month"] = trades.signal_time.dt.month

    train = trades[(trades.signal_time >= TRAIN_START) & (trades.signal_time <= TRAIN_END)]
    y26 = trades[trades.year == 2026]
    ex_mar = y26[~((y26.signal_time >= MARCH_START) & (y26.signal_time <= MARCH_END))]
    mar = y26[(y26.signal_time >= MARCH_START) & (y26.signal_time <= MARCH_END)]

    print(f"Rows: {len(df):,}")
    print(f"Generated V2-B trades: {len(trades)}")
    show("2024-2025", train)
    show("2026", y26)
    show("2026 excluding March", ex_mar)
    show("March 2026", mar)

    print("\nLearning regime maps from 2024-2025 ONLY...")
    maps = learn_maps(train)

    for f in FEATURES:
        print(f"\n{f}")
        print(maps[f]["stats"].to_string(index=False))

    trades = apply_score(trades, maps)

    print("\n"+"="*90)
    print("PULLBACK REGIME SCORE")
    print("="*90)

    for label, x in [
        ("2024-2025 PULLBACK", train[train.setup=="PULLBACK"]),
        ("2026 PULLBACK", y26[y26.setup=="PULLBACK"]),
        ("2026 ex-March PULLBACK", ex_mar[ex_mar.setup=="PULLBACK"]),
        ("March PULLBACK", mar[mar.setup=="PULLBACK"]),
    ]:
        # Re-select from scored frame to ensure score is present.
        if label.startswith("2024"):
            x = trades[(trades.year.isin([2024,2025])) & (trades.setup=="PULLBACK")]
        elif label == "2026 PULLBACK":
            x = trades[(trades.year==2026) & (trades.setup=="PULLBACK")]
        elif label.startswith("2026 ex"):
            x = trades[(trades.year==2026) & (trades.setup=="PULLBACK") &
                       ~((trades.signal_time>=MARCH_START)&(trades.signal_time<=MARCH_END))]
        else:
            x = trades[(trades.signal_time>=MARCH_START)&(trades.signal_time<=MARCH_END)&(trades.setup=="PULLBACK")]
        show(label, x)
        print(score_table(x).to_string(index=False))

    print("\n"+"="*90)
    print("2026 BLIND TEST - PULLBACK SCORE THRESHOLDS")
    print("="*90)

    p26 = trades[(trades.year==2026) & (trades.setup=="PULLBACK")]
    pm = p26[(p26.signal_time>=MARCH_START)&(p26.signal_time<=MARCH_END)]
    pe = p26[~((p26.signal_time>=MARCH_START)&(p26.signal_time<=MARCH_END))]

    for threshold in range(6):
        show(f"2026 PULLBACK score >= {threshold}", p26[p26.regime_score>=threshold])
        show(f"  ex-March score >= {threshold}", pe[pe.regime_score>=threshold])
        show(f"  March score >= {threshold}", pm[pm.regime_score>=threshold])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "v6_regime_score_trades.csv"
    trades.to_csv(out, index=False)

    print("\nSaved:", out)
    print("\nRULES UNCHANGED: V2-B entry | SL=1.5 ATR | TP=2R")
    print("The regime score is diagnostic first. Do NOT adopt a threshold yet.")

if __name__ == "__main__":
    main()
