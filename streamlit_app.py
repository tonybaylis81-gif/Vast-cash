import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

LOOKBACK_DAYS = 92
MAX_STOCKS = 10
UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","AMD","NFLX","COST","WMT","JPM","V","MA","ORCL","CRM","QCOM","MU","AMAT","GE","CAT","HON","UNP","XOM","CVX","COP","UBER","SHOP","PLTR","PANW","CRWD","SNOW","DIS","TMO","LLY","PEP"]

st.title("💰 VAST CASH")
st.caption("MAXPROFIT • Interactive strategy laboratory")

with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input("Investment capital ($)", min_value=100.0, value=1000.0, step=100.0)
    positions = st.slider("Number of stocks", 1, MAX_STOCKS, 5)
    holding_days = st.slider("Holding period (trading days)", 1, 20, 5)
    risk = st.selectbox("Risk profile", ["Balanced", "Conservative", "Aggressive"])
    st.divider()
    st.caption("Lookback is permanently locked at 3 months (92 days).")
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Capital", f"${capital:,.0f}")
col2.metric("Positions", positions)
col3.metric("Hold", f"{holding_days} days")
col4.metric("Risk", risk)

st.info("Analysis/backtest mode. No brokerage connection and no real-money orders are placed.")


def make_market_history():
    """Create deterministic sample market histories so the strategy lab works without APIs."""
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=LOOKBACK_DAYS)
    rows = {}
    for i, symbol in enumerate(UNIVERSE):
        rng = np.random.default_rng(1000 + i)
        drift = 0.00025 + (i % 7) * 0.00010
        noise = rng.normal(0, 0.012, len(dates))
        returns = drift + noise
        price = 70 + i * 8
        prices = price * np.cumprod(1 + returns)
        rows[symbol] = pd.Series(prices, index=dates)
    return pd.DataFrame(rows)


def score_market(prices):
    records = []
    for symbol in prices.columns:
        s = prices[symbol].dropna()
        ret_3m = s.iloc[-1] / s.iloc[0] - 1
        ret_20 = s.iloc[-1] / s.iloc[-21] - 1
        ret_10 = s.iloc[-1] / s.iloc[-11] - 1
        volatility = s.pct_change().dropna().std()
        momentum_quality = ret_3m / max(volatility, 0.0001)
        score = 0.45 * ret_3m + 0.25 * ret_20 + 0.15 * ret_10 + 0.15 * (momentum_quality / 10)
        records.append({"Symbol": symbol, "Price": s.iloc[-1], "3M Return": ret_3m, "20D Return": ret_20, "10D Return": ret_10, "Volatility": volatility, "Score": score})
    return pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)


def backtest(prices, picks, hold_days, capital):
    available = prices.index
    # Replay every historical day using the same ranking rule, then hold for the selected period.
    equity = pd.Series(index=available, dtype=float)
    equity.iloc[0] = capital
    for d in range(1, len(available)):
        if d < 20:
            equity.iloc[d] = equity.iloc[d-1]
            continue
        window = prices.iloc[:d+1]
        ranked = score_market(window).head(len(picks))
        symbols = ranked.Symbol.tolist()
        start_idx = max(0, d - hold_days)
        period_returns = []
        for sym in symbols:
            p0 = prices[sym].iloc[start_idx]
            p1 = prices[sym].iloc[d]
            period_returns.append(p1 / p0 - 1)
        portfolio_return = float(np.mean(period_returns)) if period_returns else 0.0
        equity.iloc[d] = equity.iloc[d-1] * (1 + portfolio_return / max(hold_days, 1))
    return equity.ffill()


if "results" not in st.session_state:
    st.session_state.results = None

if run:
    prices = make_market_history()
    results = score_market(prices)
    picks = results.head(positions).copy()
    allocation = capital / positions
    picks["Allocation"] = allocation
    picks["Shares"] = allocation / picks["Price"]
    picks["Estimated $ Risk"] = allocation * picks["Volatility"]
    equity = backtest(prices, picks.Symbol.tolist(), holding_days, capital)
    st.session_state.results = (prices, results, picks, equity)

if st.session_state.results:
    prices, results, picks, equity = st.session_state.results
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    pnl = equity.iloc[-1] - equity.iloc[0]

    st.success(f"MAXPROFIT completed a {LOOKBACK_DAYS}-day analysis using the selected settings.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Backtest P/L", f"${pnl:,.2f}", f"{total_return:.2%}")
    m2.metric("Ending Value", f"${equity.iloc[-1]:,.2f}")
    m3.metric("Selected Stocks", len(picks))

    st.subheader("🏆 MAXPROFIT Top Selections")
    st.dataframe(picks.style.format({"Price":"${:,.2f}","3M Return":"{:.2%}","20D Return":"{:.2%}","10D Return":"{:.2%}","Volatility":"{:.2%}","Score":"{:.4f}","Allocation":"${:,.2f}","Shares":"{:.2f}","Estimated $ Risk":"${:,.2f}"}), use_container_width=True, hide_index=True)

    st.subheader("📈 Portfolio Profit / Loss")
    chart = pd.DataFrame({"Portfolio Value": equity, "Profit / Loss": equity - capital})
    st.line_chart(chart, use_container_width=True)

    st.subheader("📊 Market Ranking")
    st.dataframe(results.style.format({"Price":"${:,.2f}","3M Return":"{:.2%}","20D Return":"{:.2%}","10D Return":"{:.2%}","Volatility":"{:.2%}","Score":"{:.4f}"}), use_container_width=True, hide_index=True)

    st.subheader("🧪 Combination Testing")
    st.write("Change any of the four inputs in the sidebar and press **RUN MAXPROFIT** again. Each run replaces the previous result so you can compare combinations quickly.")
else:
    st.subheader("Ready to Test")
    st.write("Choose your capital, number of positions, holding period, and risk profile. The 3-month lookback remains locked.")
    st.write("Press **RUN MAXPROFIT** to generate selections, allocations, a backtest, and a P/L curve.")

st.divider()
st.caption("VAST CASH • MAXPROFIT Interactive Strategy Laboratory • 3-month lookback locked • Maximum 10 positions • Analysis only")
