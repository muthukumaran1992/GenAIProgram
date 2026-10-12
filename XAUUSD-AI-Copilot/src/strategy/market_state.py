import pandas as pd


def determine_market_state(df):
    """
    Determine the current XAUUSD market state
    using EMA, RSI and ATR.
    """

    latest = df.iloc[-1]

    close = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    rsi = latest["rsi14"]
    atr = latest["atr14"]

    # -----------------------------
    # 1. Determine trend
    # -----------------------------

    if close > ema20 and ema20 > ema50:
        trend = "BULLISH"

    elif close < ema20 and ema20 < ema50:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    # -----------------------------
    # 2. Determine momentum
    # -----------------------------

    if rsi >= 70:
        momentum = "OVERBOUGHT"

    elif rsi >= 55:
        momentum = "STRONG_BULLISH"

    elif rsi >= 45:
        momentum = "NEUTRAL"

    elif rsi >= 30:
        momentum = "STRONG_BEARISH"

    else:
        momentum = "OVERSOLD"

    # -----------------------------
    # 3. Determine price position
    # -----------------------------

    if close > ema20:
        price_position = "ABOVE_EMA20"

    elif close < ema20:
        price_position = "BELOW_EMA20"

    else:
        price_position = "AT_EMA20"

    # -----------------------------
    # 4. Determine market state
    # -----------------------------

    if trend == "BULLISH" and momentum == "OVERBOUGHT":
        market_state = "BULLISH_EXTENDED"

    elif trend == "BULLISH" and momentum == "STRONG_BULLISH":
        market_state = "BULLISH_MOMENTUM"

    elif trend == "BEARISH" and momentum == "OVERSOLD":
        market_state = "BEARISH_EXTENDED"

    elif trend == "BEARISH" and momentum == "STRONG_BEARISH":
        market_state = "BEARISH_MOMENTUM"

    elif trend == "SIDEWAYS":
        market_state = "RANGE"

    else:
        market_state = "MIXED"

    return {
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi,
        "atr14": atr,
        "trend": trend,
        "momentum": momentum,
        "price_position": price_position,
        "market_state": market_state
    }