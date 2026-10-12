import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

file_path = r"data\raw\xauusd-m15-bid-2025-03-19-2025-03-20.csv"

df = pd.read_csv(file_path)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ==========================================
# SWING PARAMETERS
# ==========================================

left_bars = 2
right_bars = 2


# ==========================================
# FIND SWING HIGHS / LOWS
# ==========================================

df["swing_high"] = False
df["swing_low"] = False


for i in range(
    left_bars,
    len(df) - right_bars
):

    current_high = df.loc[i, "high"]

    left_highs = df.loc[
        i-left_bars:i-1,
        "high"
    ]

    right_highs = df.loc[
        i+1:i+right_bars,
        "high"
    ]

    if (
        current_high > left_highs.max()
        and current_high > right_highs.max()
    ):

        df.loc[i, "swing_high"] = True


    current_low = df.loc[i, "low"]

    left_lows = df.loc[
        i-left_bars:i-1,
        "low"
    ]

    right_lows = df.loc[
        i+1:i+right_bars,
        "low"
    ]

    if (
        current_low < left_lows.min()
        and current_low < right_lows.min()
    ):

        df.loc[i, "swing_low"] = True


# ==========================================
# EXTRACT SWINGS
# ==========================================

swing_highs = df[
    df["swing_high"]
]

swing_lows = df[
    df["swing_low"]
]


# ==========================================
# DISPLAY RECENT SWINGS
# ==========================================

print("\n========================================")
print("       XAUUSD MARKET STRUCTURE")
print("========================================")

print("\n--- RECENT SWING HIGHS ---")

print(
    swing_highs[
        ["datetime", "high"]
    ].tail(5).to_string(index=False)
)


print("\n--- RECENT SWING LOWS ---")

print(
    swing_lows[
        ["datetime", "low"]
    ].tail(5).to_string(index=False)
)


# ==========================================
# CLASSIFY STRUCTURE
# ==========================================

if len(swing_highs) >= 2:

    last_high = swing_highs.iloc[-1]["high"]
    previous_high = swing_highs.iloc[-2]["high"]

    if last_high > previous_high:

        high_structure = "HIGHER HIGH"

    else:

        high_structure = "LOWER HIGH"

else:

    high_structure = "INSUFFICIENT DATA"


if len(swing_lows) >= 2:

    last_low = swing_lows.iloc[-1]["low"]
    previous_low = swing_lows.iloc[-2]["low"]

    if last_low > previous_low:

        low_structure = "HIGHER LOW"

    else:

        low_structure = "LOWER LOW"

else:

    low_structure = "INSUFFICIENT DATA"


# ==========================================
# OVERALL STRUCTURE
# ==========================================

if (
    high_structure == "HIGHER HIGH"
    and low_structure == "HIGHER LOW"
):

    structure = "BULLISH STRUCTURE"

elif (
    high_structure == "LOWER HIGH"
    and low_structure == "LOWER LOW"
):

    structure = "BEARISH STRUCTURE"

else:

    structure = "MIXED STRUCTURE"


# ==========================================
# OUTPUT
# ==========================================

print("\n--- STRUCTURE ANALYSIS ---")

print(
    f"High Structure : "
    f"{high_structure}"
)

print(
    f"Low Structure  : "
    f"{low_structure}"
)

print(
    f"Market Structure: "
    f"{structure}"
)

print("\n========================================")