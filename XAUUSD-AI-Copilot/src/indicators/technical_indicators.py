import pandas as pd


def calculate_ema(df, period):
    return df["close"].ewm(
        span=period,
        adjust=False
    ).mean()


def calculate_rsi(df, period=14):

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = average_gain / average_loss

    return 100 - (
        100 / (1 + rs)
    )


def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # Wilder's smoothing
    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr


def add_indicators(df):

    df = df.copy()

    df["ema20"] = calculate_ema(
        df,
        20
    )

    df["ema50"] = calculate_ema(
        df,
        50
    )

    df["rsi14"] = calculate_rsi(
        df,
        14
    )

    df["atr14"] = calculate_atr(
        df,
        14
    )

    return df