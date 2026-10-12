"""
V10 — XAUUSD AI Trading Copilot
Monte Carlo + Position Sizing Validation

Uses the V9-B generated trade file:
    data/processed/v9b_mex_pro_cost_trades.csv

Purpose:
    1. Monte Carlo randomize the historical net-R sequence.
    2. Estimate distribution of returns and drawdowns.
    3. Test risk per trade at 0.25%, 0.50%, and 1.00%.
    4. Include a configurable conservative swap adjustment.
    5. Produce percentile-based risk statistics.

IMPORTANT
---------
This is a validation exercise, not a guarantee of future profitability.
Monte Carlo preserves the historical trade-result distribution but does
not prove that the underlying market edge will persist.

The V9-B CSV currently has swap_r = 0 because the exact live MEX XAUUSD
swap value was not available. Therefore DEFAULT_ESTIMATED_SWAP_R_PER_OVERNIGHT
is configurable. We use a small estimated R deduction as a sensitivity
assumption, not as a claimed broker value.

Strategy remains frozen:
    BUY breakout only
    RSI 50-70
    SL 1.5 ATR
    TP 2R
    no pullback
    no trailing
    no breakeven
    SELL disabled
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "v9b_mex_pro_cost_trades.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the conservative MEX Pro scenario from V9-B.
SCENARIO = "PRO_CONSERVATIVE"

# The exact swap rate is not known.
# This is a sensitivity assumption only.
# Set to 0.0 if you want pure V9-B results.
DEFAULT_ESTIMATED_SWAP_R_PER_OVERNIGHT = 0.008

RISK_LEVELS = [0.0025, 0.005, 0.01]  # 0.25%, 0.50%, 1.00%

MONTE_CARLO_RUNS = 10000
RANDOM_SEED = 42


def load_trade_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 14_v9b_mex_pro_validation.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df = df[df["scenario"] == SCENARIO].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for scenario {SCENARIO}."
        )

    required = [
        "net_r",
        "overnight_nights",
        "exit_reason",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Apply the configurable estimated overnight swap adjustment.
    df["estimated_swap_r"] = (
        df["overnight_nights"]
        * DEFAULT_ESTIMATED_SWAP_R_PER_OVERNIGHT
    )

    df["mc_net_r"] = (
        df["net_r"] - df["estimated_swap_r"]
    )

    return df.reset_index(drop=True)


def calculate_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return drawdown.min()


def calculate_max_losing_streak(returns):
    max_streak = 0
    current = 0

    for value in returns:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak


def run_monte_carlo(returns, risk_pct, rng):
    n = len(returns)

    final_returns_pct = np.empty(MONTE_CARLO_RUNS)
    max_drawdowns_pct = np.empty(MONTE_CARLO_RUNS)
    max_losing_streaks = np.empty(MONTE_CARLO_RUNS)
    min_equity_pct = np.empty(MONTE_CARLO_RUNS)

    for i in range(MONTE_CARLO_RUNS):
        sampled = rng.choice(
            returns,
            size=n,
            replace=True,
        )

        # R -> account return assuming fixed fractional risk.
        # This is a simple fixed-risk approximation:
        # each trade risks risk_pct of starting capital.
        trade_returns = sampled * risk_pct

        equity = np.cumprod(1.0 + trade_returns)

        final_returns_pct[i] = (
            (equity[-1] - 1.0) * 100.0
        )

        max_drawdowns_pct[i] = (
            calculate_drawdown(equity)
            * 100.0
        )

        max_losing_streaks[i] = (
            calculate_max_losing_streak(sampled)
        )

        min_equity_pct[i] = (
            (equity.min() - 1.0) * 100.0
        )

    return {
        "final_return_pct": final_returns_pct,
        "max_drawdown_pct": max_drawdowns_pct,
        "max_losing_streak": max_losing_streaks,
        "min_equity_pct": min_equity_pct,
    }


def percentile(series, p):
    return float(np.percentile(series, p))


def summarize_mc(mc, risk_pct):
    final_returns = mc["final_return_pct"]
    drawdowns = mc["max_drawdown_pct"]
    streaks = mc["max_losing_streak"]

    return {
        "risk_per_trade_pct": risk_pct * 100,
        "simulations": MONTE_CARLO_RUNS,

        "return_median_pct": percentile(final_returns, 50),
        "return_p05_pct": percentile(final_returns, 5),
        "return_p25_pct": percentile(final_returns, 25),
        "return_p75_pct": percentile(final_returns, 75),
        "return_p95_pct": percentile(final_returns, 95),

        # Drawdowns are negative numbers.
        "drawdown_median_pct": percentile(drawdowns, 50),
        "drawdown_p05_pct": percentile(drawdowns, 5),
        "drawdown_p95_pct": percentile(drawdowns, 95),

        "max_losing_streak_median": percentile(streaks, 50),
        "max_losing_streak_p95": percentile(streaks, 95),

        "probability_loss_pct": (
            np.mean(final_returns < 0) * 100
        ),

        "probability_drawdown_20pct": (
            np.mean(drawdowns <= -20) * 100
        ),

        "probability_drawdown_30pct": (
            np.mean(drawdowns <= -30) * 100
        ),

        "probability_drawdown_40pct": (
            np.mean(drawdowns <= -40) * 100
        ),
    }


def main():
    print("=" * 78)
    print("V10 — MONTE CARLO + POSITION SIZING")
    print("=" * 78)

    trades = load_trade_results()

    returns = trades["mc_net_r"].to_numpy(dtype=float)

    print(f"Input: {INPUT_FILE}")
    print(f"Scenario: {SCENARIO}")
    print(f"Trades: {len(returns)}")
    print(
        f"Estimated swap assumption: "
        f"{DEFAULT_ESTIMATED_SWAP_R_PER_OVERNIGHT:.4f}R "
        "per overnight night"
    )

    print("\nHistorical sample after configurable swap assumption:")
    print(f"Total R: {returns.sum():.2f}")
    print(f"Average R/trade: {returns.mean():.4f}")

    positive = returns[returns > 0].sum()
    negative = abs(returns[returns < 0].sum())

    pf = positive / negative if negative > 0 else np.inf

    print(f"Profit Factor: {pf:.3f}")
    print(
        f"Win rate: "
        f"{(returns > 0).mean() * 100:.2f}%"
    )

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for risk_pct in RISK_LEVELS:

        mc = run_monte_carlo(
            returns,
            risk_pct,
            rng,
        )

        summary = summarize_mc(
            mc,
            risk_pct,
        )

        results.append(summary)

        print("\n" + "-" * 78)
        print(
            f"RISK PER TRADE: "
            f"{risk_pct * 100:.2f}%"
        )
        print("-" * 78)

        print(
            f"Median final return : "
            f"{summary['return_median_pct']:.2f}%"
        )

        print(
            f"5th percentile     : "
            f"{summary['return_p05_pct']:.2f}%"
        )

        print(
            f"95th percentile    : "
            f"{summary['return_p95_pct']:.2f}%"
        )

        print(
            f"Median max DD      : "
            f"{summary['drawdown_median_pct']:.2f}%"
        )

        print(
            f"Worst 5% max DD    : "
            f"{summary['drawdown_p05_pct']:.2f}%"
        )

        print(
            f"Best 5% max DD     : "
            f"{summary['drawdown_p95_pct']:.2f}%"
        )

        print(
            f"95% losing streak  : "
            f"{summary['max_losing_streak_p95']:.0f} trades"
        )

        print(
            f"Probability loss   : "
            f"{summary['probability_loss_pct']:.2f}%"
        )

        print(
            f"P(DD >= 20%)       : "
            f"{summary['probability_drawdown_20pct']:.2f}%"
        )

        print(
            f"P(DD >= 30%)       : "
            f"{summary['probability_drawdown_30pct']:.2f}%"
        )

        print(
            f"P(DD >= 40%)       : "
            f"{summary['probability_drawdown_40pct']:.2f}%"
        )

    results_df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR /
        "v10_monte_carlo_position_sizing.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 78)
    print("SAVED")
    print("=" * 78)
    print(output_file)

    print("\nInterpretation:")
    print(
        "0.25% risk is intentionally conservative and should be "
        "the first risk level considered if the strategy survives."
    )
    print(
        "0.50% is a moderate research case."
    )
    print(
        "1.00% is included as a stress case, not a recommendation."
    )
    print(
        "Monte Carlo does not prove future profitability; it measures "
        "the range of outcomes implied by the historical trade distribution."
    )


if __name__ == "__main__":
    main()
