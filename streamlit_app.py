import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from ib_insync import IB, Stock, MarketOrder
except ImportError:
    IB = None

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

LOOKBACK_BARS = 92
HOLD_SESSIONS = 5
MAX_POSITIONS = 10
PAPER_TRADING_ONLY = True

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "NFLX",
    "COST", "WMT", "JPM", "V", "MA", "ORCL", "CRM", "QCOM", "MU", "AMAT",
    "GE", "CAT", "HON", "UNP", "XOM", "CVX", "COP", "UBER", "SHOP", "PLTR",
    "PANW", "CRWD", "SNOW", "DIS", "TMO", "LLY", "PEP"
]


def env(name, default):
    value = os.getenv(name, "")
    try:
        value = st.secrets.get(name, value)
    except Exception:
        pass
    return str(value).strip() or default


def ibkr_settings():
    # TWS paper trading normally uses port 7497; IB Gateway paper commonly uses 4002.
    host = env("IBKR_HOST", "127.0.0.1")
    port = int(env("IBKR_PORT", "7497"))
    client_id = int(env("IBKR_CLIENT_ID", "27"))
    return host, port, client_id


def connect_ibkr():
    if IB is None:
        raise RuntimeError("ib_insync is not installed. Add ib_insync to requirements.txt and redeploy.")
    host, port, client_id = ibkr_settings()
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=5)
    except Exception as exc:
        raise RuntimeError(
            f"IBKR paper gateway is not reachable at {host}:{port}. "
            "Run TWS/IB Gateway in paper mode and enable API connections. "
            f"Original error: {exc}"
        ) from exc
    return ib


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbols_tuple):
    symbols = list(symbols_tuple)
    end = datetime.utcnow()
    start = end - timedelta(days=450)
    data = yf.download(
        symbols,
        start=start.date().isoformat(),
        end=(end + timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no market data.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Market data did not contain closing prices.")
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": symbols[0]})
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    if len(prices) < LOOKBACK_BARS:
        raise RuntimeError(f"Only {len(prices)} sessions returned; MAXPROFIT requires {LOOKBACK_BARS}.")
    return prices


def score_symbol(series, risk_profile):
    s = series.dropna()
    if len(s) < LOOKBACK_BARS:
        return None
    r92 = s.iloc[-1] / s.iloc[-LOOKBACK_BARS] - 1
    r20 = s.iloc[-1] / s.iloc[-21] - 1
    r10 = s.iloc[-1] / s.iloc[-11] - 1
    daily = s.pct_change().dropna()
    volatility = max(float(daily.std()), 0.0001)
    negative = daily[daily < 0]
    downside_vol = max(float(negative.std()) if len(negative) > 1 else volatility, 0.0001)
    risk_adjusted = r92 / volatility
    downside_adjusted = r92 / downside_vol
    trend = (r92 + r20 + r10) / 3
    if risk_profile == "Conservative":
        score = 0.40*r92 + 0.15*r20 + 0.10*r10 + 0.20*(risk_adjusted/10) + 0.15*(downside_adjusted/10)
    elif risk_profile == "Aggressive":
        score = 0.55*r92 + 0.25*r20 + 0.15*r10 + 0.05*trend
    else:
        score = 0.45*r92 + 0.25*r20 + 0.15*r10 + 0.10*(risk_adjusted/10) + 0.05*(downside_adjusted/10)
    return {"Price": float(s.iloc[-1]), "3M Return": float(r92), "20D Return": float(r20),
            "10D Return": float(r10), "Volatility": float(volatility),
            "Downside Vol": float(downside_vol), "Score": float(score)}


def rank_market(prices, risk_profile):
    rows = []
    for symbol in prices.columns:
        result = score_symbol(prices[symbol], risk_profile)
        if result:
            rows.append({"Symbol": symbol, **result})
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def backtest(prices, positions, risk_profile, threshold, starting_capital):
    equity = float(starting_capital)
    curve, trades = [], []
    for signal_idx in range(LOOKBACK_BARS, len(prices) - HOLD_SESSIONS - 1, HOLD_SESSIONS):
        ranked = rank_market(prices.iloc[:signal_idx+1], risk_profile)
        chosen = ranked[ranked["Score"] >= threshold].head(positions)
        if chosen.empty:
            chosen = ranked.head(positions)
        entry_idx, exit_idx = signal_idx + 1, signal_idx + 1 + HOLD_SESSIONS
        returns = []
        for symbol in chosen["Symbol"]:
            entry = float(prices.iloc[entry_idx][symbol])
            exit_price = float(prices.iloc[exit_idx][symbol])
            ret = exit_price / entry - 1
            returns.append(ret)
            trades.append({"Entry Date": prices.index[entry_idx].date(), "Exit Date": prices.index[exit_idx].date(),
                           "Symbol": symbol, "Entry": entry, "Exit": exit_price, "Return": ret})
        if returns:
            equity *= 1 + float(np.mean(returns))
            curve.append((prices.index[exit_idx], equity))
    curve_df = pd.DataFrame(curve, columns=["Date", "Portfolio Value"]).set_index("Date") if curve else pd.DataFrame()
    return curve_df, pd.DataFrame(trades)


def ibkr_account_snapshot(ib):
    vals = {x.tag: x.value for x in ib.accountSummary()}
    return {
        "NetLiquidation": float(vals.get("NetLiquidation", 0) or 0),
        "BuyingPower": float(vals.get("BuyingPower", 0) or 0),
        "AvailableFunds": float(vals.get("AvailableFunds", 0) or 0),
    }


def ibkr_positions(ib):
    rows = []
    for p in ib.positions():
        rows.append({"Symbol": p.contract.symbol, "Qty": float(p.position), "Avg Cost": float(p.avgCost)})
    return pd.DataFrame(rows)


def submit_ibkr_paper_buys(ib, selected, deployment_capital):
    if deployment_capital <= 0:
        raise RuntimeError("Deployment capital must be greater than zero.")
    account = ibkr_account_snapshot(ib)
    buying_power = min(account["BuyingPower"], account["AvailableFunds"] or account["BuyingPower"])
    deploy = min(float(deployment_capital), max(0.0, buying_power))
    if deploy <= 0:
        raise RuntimeError("IBKR reports no available paper buying power.")
    existing = {p.contract.symbol.upper() for p in ib.positions() if float(p.position) != 0}
    candidates = [s for s in selected["Symbol"].tolist() if s.upper() not in existing]
    if not candidates:
        raise RuntimeError("All selected symbols are already held in the IBKR paper account.")
    allocation = deploy / len(candidates)
    results = []
    for symbol in candidates:
        contract = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            results.append({"Symbol": symbol, "Status": "Contract not qualified"})
            continue
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(1)
        price = ticker.marketPrice()
        if not price or np.isnan(price) or price <= 0:
            results.append({"Symbol": symbol, "Status": "No usable market price"})
            continue
        qty = max(1, int(allocation / float(price)))
        order = MarketOrder("BUY", qty)
        order.orderRef = f"VASTCASH-D1-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{symbol}"
        trade = ib.placeOrder(contract, order)
        ib.sleep(0.2)
        results.append({"Symbol": symbol, "Qty": qty, "Est. Notional": qty*float(price),
                        "Order ID": trade.order.orderId, "Status": trade.orderStatus.status})
    return pd.DataFrame(results)


# ---------------- UI ----------------
st.title("💰 VAST CASH")
st.caption("MAXPROFIT • Canadian-compatible IBKR paper-trading build")

with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input("Paper deployment capital ($)", min_value=1.0, value=1000.0, step=100.0)
    positions_count = st.slider("Stocks selected", 1, MAX_POSITIONS, 5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    signal_threshold = st.slider("Minimum signal strength", -0.20, 1.00, 0.20, 0.05)
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)
    st.divider()
    st.caption("LOCKED: 92 trading-day lookback")
    st.caption("LOCKED: Buy Day 1 → Sell Day 6")
    st.caption("Maximum 10 positions")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deployment", f"${capital:,.0f}")
c2.metric("Stocks", positions_count)
c3.metric("Cycle", "D1 → D6")
c4.metric("Mode", "PAPER ONLY")
st.info("🔒 PAPER-ONLY SAFETY LOCK. This application contains no live-trading path. The execution adapter is IBKR TWS/API and must be connected to an IBKR PAPER login.")
st.warning("IBKR API access requires TWS or IB Gateway running in PAPER mode and API connections enabled. A Streamlit Cloud app cannot reach 127.0.0.1 on your computer, so the execution button will remain unavailable until the gateway is reachable from the app host.")

if run:
    try:
        with st.spinner("Loading market data and running MAXPROFIT..."):
            prices = load_market_data(tuple(UNIVERSE))
            ranked = rank_market(prices, risk_profile)
            selected = ranked[ranked["Score"] >= signal_threshold].head(positions_count)
            if selected.empty:
                selected = ranked.head(positions_count)
            curve, trades = backtest(prices, positions_count, risk_profile, signal_threshold, capital)
        st.subheader("MAXPROFIT ranking")
        st.dataframe(ranked.head(20), use_container_width=True, hide_index=True)
        st.subheader("Current selections")
        st.dataframe(selected, use_container_width=True, hide_index=True)
        if not curve.empty:
            st.subheader("Backtest equity")
            st.line_chart(curve["Portfolio Value"])
        if not trades.empty:
            st.subheader("Backtest trades")
            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("IBKR Paper Execution")
        if st.button("🧪 SEND SELECTED TRADES TO IBKR PAPER", type="secondary"):
            ib = connect_ibkr()
            try:
                result = submit_ibkr_paper_buys(ib, selected, capital)
                st.dataframe(result, use_container_width=True, hide_index=True)
            finally:
                ib.disconnect()
    except Exception as exc:
        st.error(str(exc))
else:
    st.write("Press **RUN MAXPROFIT** to calculate the current ranking and backtest before any paper-order action.")
