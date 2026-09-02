import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

LOOKBACK_DAYS = 92
MAX_STOCKS = 10
HOLD_DAYS = 5  # Buy on Day 1, sell on Day 6
BACKTEST_DAYS = 252
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "NFLX",
    "COST", "WMT", "JPM", "V", "MA", "ORCL", "CRM", "QCOM", "MU", "AMAT",
    "GE", "CAT", "HON", "UNP", "XOM", "CVX", "COP", "UBER", "SHOP", "PLTR",
    "PANW", "CRWD", "SNOW", "DIS", "TMO", "LLY", "PEP"
]

st.title("💰 VAST CASH")
st.caption("MAXPROFIT • Strategy Laboratory")

# -----------------------------
# USER INPUTS
# -----------------------------
with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input("Investment capital ($)", min_value=100.0, value=1000.0, step=100.0)
    positions = st.slider("Stocks selected", 1, MAX_STOCKS, 5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    signal_strength = st.slider("Minimum signal strength", 0.00, 1.00, 0.20, 0.05)
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)
    st.divider()
    st.caption("LOCKED: 3-month (92 trading-day) analysis window")
    st.caption("LOCKED: Buy Day 1 → Sell Day 6")
    st.caption("Maximum 10 stocks")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital", f"${capital:,.0f}")
c2.metric("Stocks", positions)
c3.metric("Trade", "D1 → D6")
c4.metric("Risk", risk_profile)

st.info("Backtest / analysis mode only. No brokerage connection and no real-money orders are placed.")

# -----------------------------
# MARKET DATA ENGINE
# -----------------------------

def make_market_history(days=BACKTEST_DAYS + LOOKBACK_DAYS + HOLD_DAYS):
    """Deterministic market history for repeatable strategy testing without an API."""
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=days)
    rows = {}
    for i, symbol in enumerate(UNIVERSE):
        rng = np.random.default_rng(5000 + i)
        base_drift = 0.00005 + (i % 9) * 0.000045
        cycle = 0.00035 * np.sin(np.arange(len(dates)) / (14 + i % 8))
        noise = rng.normal(0, 0.009 + (i % 5) * 0.0008, len(dates))
        returns = base_drift + cycle + noise
        start_price = 60 + i * 7
        rows[symbol] = start_price * np.cumprod(1 + returns)
    return pd.DataFrame(rows, index=dates)


def momentum_score(series, risk_profile):
    series = series.dropna()
    if len(series) < LOOKBACK_DAYS:
        return None

    r3m = series.iloc[-1] / series.iloc[-LOOKBACK_DAYS] - 1
    r20 = series.iloc[-1] / series.iloc[-21] - 1
    r10 = series.iloc[-1] / series.iloc[-11] - 1
    vol = series.pct_change().dropna().std()
    downside = series.pct_change().dropna()
    downside_vol = downside[downside < 0].std()
    downside_vol = 0.0001 if pd.isna(downside_vol) or downside_vol == 0 else downside_vol
    vol = max(vol, 0.0001)

    # Higher momentum, lower volatility and lower downside volatility are rewarded.
    sharpe_like = r3m / vol
    downside_quality = r3m / downside_vol

    if risk_profile == "Conservative":
        score = 0.35 * r3m + 0.20 * r20 + 0.10 * r10 + 0.20 * (sharpe_like / 10) + 0.15 * (downside_quality / 10)
    elif risk_profile == "Aggressive":
        score = 0.50 * r3m + 0.25 * r20 + 0.20 * r10 + 0.05 * (sharpe_like / 10)
    else:
        score = 0.45 * r3m + 0.25 * r20 + 0.15 * r10 + 0.10 * (sharpe_like / 10) + 0.05 * (downside_quality / 10)

    return {
        "Price": float(series.iloc[-1]),
        "3M Return": r3m,
        "20D Return": r20,
        "10D Return": r10,
        "Volatility": vol,
        "Downside Vol": downside_vol,
        "Score": score,
    }


def rank_at(prices, end_index, risk_profile):
    window = prices.iloc[: end_index + 1]
    records = []
    for symbol in prices.columns:
        result = momentum_score(window[symbol], risk_profile)
        if result is not None:
            records.append({"Symbol": symbol, **result})
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)


def latest_rank(prices, risk_profile):
    return rank_at(prices, len(prices) - 1, risk_profile)


def run_backtest(prices, positions, risk_profile, signal_strength, capital):
    """Walk forward without look-ahead: rank using information available at signal day,
    enter next business day and exit five sessions later (Day 1 -> Day 6)."""
    equity = [capital]
    dates = []
    trades = []

    start = LOOKBACK_DAYS
    end = len(prices) - HOLD_DAYS - 1

    for signal_idx in range(start, end, HOLD_DAYS):
        ranked = rank_at(prices, signal_idx, risk_profile)
        if ranked.empty:
            continue

        selected = ranked[ranked["Score"] >= signal_strength].head(positions).copy()
        if selected.empty:
            selected = ranked.head(positions).copy()

        entry_idx = signal_idx + 1
        exit_idx = entry_idx + HOLD_DAYS
        entry_prices = prices.iloc[entry_idx]
        exit_prices = prices.iloc[exit_idx]
        returns = []

        for symbol in selected["Symbol"]:
            entry = float(entry_prices[symbol])
            exit_ = float(exit_prices[symbol])
            trade_return = exit_ / entry - 1
            returns.append(trade_return)
            trades.append({
                "Entry Date": prices.index[entry_idx].date(),
                "Exit Date": prices.index[exit_idx].date(),
                "Symbol": symbol,
                "Entry": entry,
                "Exit": exit_,
                "Return": trade_return,
            })

        portfolio_return = float(np.mean(returns)) if returns else 0.0
        new_value = equity[-1] * (1 + portfolio_return)
        equity.append(new_value)
        dates.append(prices.index[exit_idx])

    curve = pd.Series(equity[1:], index=pd.DatetimeIndex(dates), name="Portfolio Value")
    trades_df = pd.DataFrame(trades)
    return curve, trades_df


# -----------------------------
# RUN
# -----------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

if run:
    with st.spinner("MAXPROFIT is running the strategy laboratory..."):
        prices = make_market_history()
        ranked = latest_rank(prices, risk_profile)
        selected = ranked[ranked["Score"] >= signal_strength].head(positions).copy()
        if selected.empty:
            selected = ranked.head(positions).copy()

        allocation = capital / len(selected)
        selected["Allocation"] = allocation
        selected["Shares"] = selected["Allocation"] / selected["Price"]
        selected["Est. 1-Day Risk"] = selected["Allocation"] * selected["Volatility"]

        curve, trades = run_backtest(prices, positions, risk_profile, signal_strength, capital)
        st.session_state.results = (ranked, selected, curve, trades, prices)
        st.session_state.run_id += 1

# -----------------------------
# RESULTS
# -----------------------------
if "results" not in st.session_state:
    st.subheader("Ready to Test")
    st.write("Change the four inputs and run MAXPROFIT. The engine will rank the universe, select the strongest candidates, allocate capital, and replay the Buy Day 1 → Sell Day 6 strategy.")
else:
    ranked, selected, curve, trades, prices = st.session_state.results

    if curve.empty:
        st.warning("The backtest did not produce enough completed trades.")
    else:
        ending = float(curve.iloc[-1])
        pnl = ending - capital
        total_return = ending / capital - 1
        drawdown = curve / curve.cummax() - 1
        max_drawdown = float(drawdown.min())

        st.success("MAXPROFIT analysis complete.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Backtest P/L", f"${pnl:,.2f}", f"{total_return:.2%}")
        m2.metric("Ending Value", f"${ending:,.2f}")
        m3.metric("Max Drawdown", f"{max_drawdown:.2%}")
        m4.metric("Completed Trades", len(trades))

        st.subheader("🏆 Current MAXPROFIT Selections")
        st.dataframe(
            selected.style.format({
                "Price": "${:,.2f}", "3M Return": "{:.2%}", "20D Return": "{:.2%}",
                "10D Return": "{:.2%}", "Volatility": "{:.2%}", "Downside Vol": "{:.2%}",
                "Score": "{:.4f}", "Allocation": "${:,.2f}", "Shares": "{:.2f}",
                "Est. 1-Day Risk": "${:,.2f}"
            }), width="stretch", hide_index=True
        )

        st.subheader("📈 Portfolio Profit / Loss")
        chart = pd.DataFrame({"Portfolio Value": curve, "Profit / Loss": curve - capital})
        st.line_chart(chart, width="stretch")

        st.subheader("📊 Full Market Ranking")
        st.dataframe(
            ranked.style.format({
                "Price": "${:,.2f}", "3M Return": "{:.2%}", "20D Return": "{:.2%}",
                "10D Return": "{:.2%}", "Volatility": "{:.2%}", "Downside Vol": "{:.2%}", "Score": "{:.4f}"
            }), width="stretch", hide_index=True
        )

        st.subheader("🧾 Trade History")
        if not trades.empty:
            st.dataframe(
                trades.style.format({"Entry": "${:,.2f}", "Exit": "${:,.2f}", "Return": "{:.2%}"}),
                width="stretch", hide_index=True
            )

        st.subheader("🧪 Combination Testing")
        st.write("Run the engine again with a different capital amount, stock count, risk profile, or signal-strength threshold. The 92-day lookback and Day 1 → Day 6 trade rule remain locked.")

st.divider()
st.caption("VAST CASH • MAXPROFIT Strategy Laboratory • 92-day lookback locked • Buy D1 / Sell D6 • Max 10 positions • Analysis only")
