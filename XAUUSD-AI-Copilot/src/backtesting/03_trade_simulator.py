import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))

from indicators.technical_indicators import add_indicators


# ==========================================
# CONFIGURATION
# ==========================================

ATR_STOP_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 2.0


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "XAUUSD_m15_20240101_20260930.csv"
)

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()

time_column = "timestamp" if "timestamp" in df.columns else "date"
required_columns = {time_column, "open", "high", "low", "close"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in {file_path.name}: "
        f"{sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError(f"No market data found in {file_path.name}.")

if time_column == "timestamp":
    df["datetime"] = pd.to_datetime(df[time_column], unit="ms")
else:
    df["datetime"] = pd.to_datetime(df[time_column])

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ==========================================
# 2. INDICATORS
# ==========================================

df = add_indicators(df)


# ==========================================
# 3. SIGNAL GENERATION
# ==========================================

df["signal"] = "WAIT"

for i in range(50, len(df) - 1):

    current = df.iloc[i]

    price = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi14"]

    # --------------------------------------
    # TREND
    # --------------------------------------

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"


    # --------------------------------------
    # PREVIOUS 20-CANDLE RANGE
    # --------------------------------------

    previous_candles = df.iloc[i - 20:i]

    resistance = previous_candles["high"].max()
    support = previous_candles["low"].min()


    # --------------------------------------
    # CANDLE
    # --------------------------------------

    bullish_candle = (
        current["close"] > current["open"]
    )

    bearish_candle = (
        current["close"] < current["open"]
    )


    # --------------------------------------
    # BREAKOUT
    # --------------------------------------

    bullish_breakout = (
        trend == "BULLISH"
        and price > resistance
    )

    bearish_breakout = (
        trend == "BEARISH"
        and price < support
    )


    # --------------------------------------
    # PULLBACK
    # --------------------------------------

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


    # --------------------------------------
    # FINAL SIGNAL
    # --------------------------------------

    if bullish_breakout or bullish_pullback:
        df.loc[i, "signal"] = "BUY"

    elif bearish_breakout or bearish_pullback:
        df.loc[i, "signal"] = "SELL"


# ==========================================
# 4. TRADE SIMULATION
# ==========================================

trades = []

position_open = False

i = 50


while i < len(df) - 1:

    # --------------------------------------
    # WAIT FOR SIGNAL
    # --------------------------------------

    if not position_open:

        signal = df.loc[i, "signal"]

        if signal == "WAIT":

            i += 1
            continue


        # ==================================
        # ENTRY ON NEXT CANDLE OPEN
        # ==================================

        entry_index = i + 1

        entry_candle = df.iloc[entry_index]

        entry_time = entry_candle["datetime"]

        entry_price = float(
            entry_candle["open"]
        )

        atr = float(
            df.loc[i, "atr14"]
        )

        risk = (
            ATR_STOP_MULTIPLIER * atr
        )


        # ==================================
        # SL / TP
        # ==================================

        if signal == "BUY":

            stop_loss = (
                entry_price - risk
            )

            take_profit = (
                entry_price
                + risk * REWARD_MULTIPLIER
            )

        else:

            stop_loss = (
                entry_price + risk
            )

            take_profit = (
                entry_price
                - risk * REWARD_MULTIPLIER
            )


        # ==================================
        # OPEN POSITION
        # ==================================

        position_open = True

        exit_index = None
        exit_price = None
        result = None
        r_multiple = None


        # ==================================
        # WALK FORWARD
        # ==================================

        for j in range(
            entry_index,
            len(df)
        ):

            future = df.iloc[j]

            future_high = float(
                future["high"]
            )

            future_low = float(
                future["low"]
            )


            # ==================================
            # BUY
            # ==================================

            if signal == "BUY":

                stop_hit = (
                    future_low <= stop_loss
                )

                target_hit = (
                    future_high >= take_profit
                )

                # Conservative assumption:
                # If both are hit in same candle,
                # assume SL was hit first.

                if stop_hit:

                    exit_index = j
                    exit_price = stop_loss
                    result = "LOSS"
                    r_multiple = -1.0

                    break

                elif target_hit:

                    exit_index = j
                    exit_price = take_profit
                    result = "WIN"
                    r_multiple = REWARD_MULTIPLIER

                    break


            # ==================================
            # SELL
            # ==================================

            else:

                stop_hit = (
                    future_high >= stop_loss
                )

                target_hit = (
                    future_low <= take_profit
                )

                # Conservative assumption:
                # If both are hit in same candle,
                # assume SL was hit first.

                if stop_hit:

                    exit_index = j
                    exit_price = stop_loss
                    result = "LOSS"
                    r_multiple = -1.0

                    break

                elif target_hit:

                    exit_index = j
                    exit_price = take_profit
                    result = "WIN"
                    r_multiple = REWARD_MULTIPLIER

                    break


        # ==================================
        # HANDLE UNRESOLVED TRADE
        # ==================================

        if exit_index is None:

            exit_index = len(df) - 1

            exit_price = float(
                df.iloc[-1]["close"]
            )

            if signal == "BUY":

                pnl = (
                    exit_price
                    - entry_price
                )

            else:

                pnl = (
                    entry_price
                    - exit_price
                )

            r_multiple = (
                pnl / risk
            )

            result = "OPEN_AT_END"


        # ==================================
        # RECORD TRADE
        # ==================================

        trade = {

            "signal_time":
                df.loc[i, "datetime"],

            "entry_time":
                entry_time,

            "exit_time":
                df.loc[exit_index, "datetime"],

            "direction":
                signal,

            "entry_price":
                entry_price,

            "stop_loss":
                stop_loss,

            "take_profit":
                take_profit,

            "exit_price":
                exit_price,

            "atr":
                atr,

            "risk":
                risk,

            "result":
                result,

            "r_multiple":
                r_multiple
        }

        trades.append(trade)

        # Position is now closed
        position_open = False

        # Continue after exit
        i = exit_index + 1

    else:

        i += 1


# ==========================================
# 5. TRADE DATAFRAME
# ==========================================

trades_df = pd.DataFrame(trades)


# ==========================================
# 6. SUMMARY
# ==========================================

print("\n==============================================")
print("       XAUUSD TRADE SIMULATOR V1")
print("==============================================")

print(
    f"\nCompleted trades : "
    f"{len(trades_df):,}"
)


if not trades_df.empty:

    print("\nResults:")

    print(
        trades_df["result"]
        .value_counts()
    )


    print("\nFirst 10 completed trades:")

    print(
        trades_df[
            [
                "signal_time",
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "stop_loss",
                "take_profit",
                "exit_price",
                "result",
                "r_multiple"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )