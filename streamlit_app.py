import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

LOOKBACK_BARS = 92
HOLD_SESSIONS = 5  # Buy Day 1 -> Sell Day 6
MAX_STOCKS = 10
UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","AMD","NFLX","COST","WMT","JPM","V","MA","ORCL","CRM","QCOM","MU","AMAT","GE","CAT","HON","UNP","XOM","CVX","COP","UBER","SHOP","PLTR","PANW","CRWD","SNOW","DIS","TMO","LLY","PEP"]


def secrets_available():
    return bool(st.secrets.get("ALPACA_API_KEY", "")) and bool(st.secrets.get("ALPACA_SECRET_KEY", ""))


def alpaca_clients():
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    key = st.secrets["ALPACA_API_KEY"]
    secret = st.secrets["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def get_account():
    trading, _ = alpaca_clients()
    return trading.get_account()


def get_positions():
    trading, _ = alpaca_clients()
    return trading.get_all_positions()


def get_orders():
    trading, _ = alpaca_clients()
    return trading.get_orders()


def load_market_data(symbols, calendar_days=150):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    _, data_client = alpaca_clients()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=calendar_days)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    raw = data_client.get_stock_bars(request).df
    if raw.empty:
        raise RuntimeError("Alpaca returned no market data.")
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
        prices = raw.pivot(index="timestamp", columns="symbol", values="close")
    else:
        prices = raw[["close"]].rename(columns={"close": symbols[0]})
    prices.index = pd.to_datetime(prices.index, utc=True)
    return prices.sort_index().ffill()


def score_symbol(series, risk_profile):
    s = series.dropna()
    if len(s) < LOOKBACK_BARS:
        return None
    r92 = s.iloc[-1] / s.iloc[-LOOKBACK_BARS] - 1
    r20 = s.iloc[-1] / s.iloc[-21] - 1
    r10 = s.iloc[-1] / s.iloc[-11] - 1
    daily = s.pct_change().dropna()
    vol = max(float(daily.std()), 0.0001)
    down = daily[daily < 0]
    down_vol = max(float(down.std()) if len(down) > 1 else vol, 0.0001)
    trend = (r92 + r20 + r10) / 3
    risk_adjusted = r92 / vol
    downside_adjusted = r92 / down_vol
    if risk_profile == "Conservative":
        score = 0.40 * r92 + 0.15 * r20 + 0.10 * r10 + 0.20 * (risk_adjusted / 10) + 0.15 * (downside_adjusted / 10)
    elif risk_profile == "Aggressive":
        score = 0.55 * r92 + 0.25 * r20 + 0.15 * r10 + 0.05 * trend
    else:
        score = 0.45 * r92 + 0.25 * r20 + 0.15 * r10 + 0.10 * (risk_adjusted / 10) + 0.05 * (downside_adjusted / 10)
    return {"Price": float(s.iloc[-1]), "3M Return": r92, "20D Return": r20, "10D Return": r10, "Volatility": vol, "Downside Vol": down_vol, "Score": float(score)}


def rank_market(prices, risk_profile):
    rows = []
    for symbol in prices.columns:
        result = score_symbol(prices[symbol], risk_profile)
        if result:
            rows.append({"Symbol": symbol, **result})
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def backtest(prices, positions, risk_profile, signal_threshold, capital):
    equity = capital
    curve = []
    trades = []
    for signal_idx in range(LOOKBACK_BARS, len(prices) - HOLD_SESSIONS - 1, HOLD_SESSIONS):
        ranked = rank_market(prices.iloc[: signal_idx + 1], risk_profile)
        chosen = ranked[ranked.Score >= signal_threshold].head(positions)
        if chosen.empty:
            chosen = ranked.head(positions)
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + HOLD_SESSIONS
        returns = []
        for symbol in chosen.Symbol:
            entry = float(prices.iloc[entry_idx][symbol])
            exit_price = float(prices.iloc[exit_idx][symbol])
            ret = exit_price / entry - 1
            returns.append(ret)
            trades.append({"Entry Date": prices.index[entry_idx].date(), "Exit Date": prices.index[exit_idx].date(), "Symbol": symbol, "Entry": entry, "Exit": exit_price, "Return": ret})
        portfolio_return = float(np.mean(returns)) if returns else 0
        equity *= 1 + portfolio_return
        curve.append((prices.index[exit_idx], equity))
    curve_df = pd.DataFrame(curve, columns=["Date", "Portfolio Value"]).set_index("Date") if curve else pd.DataFrame(columns=["Portfolio Value"])
    return curve_df, pd.DataFrame(trades)


def submit_paper_orders(selected, deployment_capital):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    trading, _ = alpaca_clients()
    account = trading.get_account()
    buying_power = float(account.buying_power)
    deploy = min(float(deployment_capital), buying_power)
    if deploy <= 0:
        raise RuntimeError("Paper account has no available buying power.")
    allocation = deploy / len(selected)
    submitted = []
    for symbol in selected.Symbol.tolist():
        order = MarketOrderRequest(symbol=symbol, notional=round(allocation, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        response = trading.submit_order(order_data=order)
        submitted.append({"Symbol": symbol, "Notional": allocation, "Order ID": str(response.id), "Status": str(response.status)})
    return pd.DataFrame(submitted), deploy, buying_power


def close_paper_positions():
    trading, _ = alpaca_clients()
    return trading.close_all_positions(cancel_orders=True)


st.title("💰 VAST CASH")
st.caption("MAXPROFIT • Live market analysis + paper execution")

with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input("Paper deployment capital ($)", min_value=100.0, value=1000.0, step=100.0)
    positions_count = st.slider("Stocks selected", 1, MAX_STOCKS, 5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    signal_threshold = st.slider("Minimum signal strength", -0.20, 1.00, 0.20, 0.05)
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)
    st.divider()
    st.caption("LOCKED: 92 trading-day lookback")
    st.caption("LOCKED: Buy Day 1 → Sell Day 6")
    st.caption("Maximum 10 stocks")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deployment", f"${capital:,.0f}")
c2.metric("Stocks", positions_count)
c3.metric("Trade cycle", "D1 → D6")
c4.metric("Risk", risk_profile)

if not secrets_available():
    st.warning("Paper trading is not connected yet. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to Streamlit Secrets using your PAPER account credentials. The code is hard-locked to paper=True.")
else:
    try:
        account = get_account()
        a1, a2, a3 = st.columns(3)
        a1.metric("Paper Equity", f"${float(account.equity):,.2f}")
        a2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
        a3.metric("Account", "PAPER")
    except Exception as exc:
        st.error(f"Paper account connection failed: {exc}")

st.info("Paper mode only. This application cannot submit live orders because the trading client is explicitly configured with paper=True.")

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    if not secrets_available():
        st.error("Connect the Alpaca PAPER credentials in Streamlit Secrets first.")
    else:
        try:
            with st.spinner("Pulling market data and running MAXPROFIT..."):
                prices = load_market_data(UNIVERSE)
                ranked = rank_market(prices, risk_profile)
                selected = ranked[ranked.Score >= signal_threshold].head(positions_count).copy()
                if selected.empty:
                    selected = ranked.head(positions_count).copy()
                curve, trades = backtest(prices, positions_count, risk_profile, signal_threshold, capital)
                st.session_state.results = (prices, ranked, selected, curve, trades)
        except Exception as exc:
            st.error(f"MAXPROFIT could not complete the run: {exc}")

if st.session_state.results:
    prices, ranked, selected, curve, trades = st.session_state.results
    st.success(f"MAXPROFIT ranked {len(ranked)} symbols using the latest {LOOKBACK_BARS} trading sessions of real market data.")
    st.subheader("🏆 Current MAXPROFIT Selections")
    allocation = capital / len(selected)
    selected = selected.copy()
    selected["Allocation"] = allocation
    selected["Est. 1-Day Risk"] = allocation * selected["Volatility"]
    st.dataframe(selected.style.format({"Price":"${:,.2f}","3M Return":"{:.2%}","20D Return":"{:.2%}","10D Return":"{:.2%}","Volatility":"{:.2%}","Downside Vol":"{:.2%}","Score":"{:.4f}","Allocation":"${:,.2f}","Est. 1-Day Risk":"${:,.2f}"}), width="stretch", hide_index=True)

    if not curve.empty:
        ending = float(curve.iloc[-1]["Portfolio Value"])
        pnl = ending - capital
        total_return = ending / capital - 1
        dd = curve["Portfolio Value"] / curve["Portfolio Value"].cummax() - 1
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Backtest P/L", f"${pnl:,.2f}", f"{total_return:.2%}")
        m2.metric("Ending Value", f"${ending:,.2f}")
        m3.metric("Max Drawdown", f"{float(dd.min()):.2%}")
        m4.metric("Completed Trades", len(trades))
        st.subheader("📈 Backtest P/L")
        chart = curve.copy()
        chart["Profit / Loss"] = chart["Portfolio Value"] - capital
        st.line_chart(chart, width="stretch")

    st.subheader("📊 Full Market Ranking")
    st.dataframe(ranked.style.format({"Price":"${:,.2f}","3M Return":"{:.2%}","20D Return":"{:.2%}","10D Return":"{:.2%}","Volatility":"{:.2%}","Downside Vol":"{:.2%}","Score":"{:.4f}"}), width="stretch", hide_index=True)

    if not trades.empty:
        st.subheader("🧾 Backtest Trade History")
        st.dataframe(trades.style.format({"Entry":"${:,.2f}","Exit":"${:,.2f}","Return":"{:.2%}"}), width="stretch", hide_index=True)

    st.divider()
    st.header("🧪 PAPER TRADING")
    st.warning("The buttons below submit orders to the Alpaca PAPER account only. They do not use real money.")
    confirm = st.checkbox("I understand this will submit PAPER orders to my paper account.")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🟢 BUY CURRENT MAXPROFIT SELECTIONS", disabled=not confirm, use_container_width=True):
            try:
                orders, deployed, buying_power = submit_paper_orders(selected, capital)
                st.success(f"Submitted {len(orders)} paper buy orders. Requested deployment: ${deployed:,.2f}.")
                st.dataframe(orders, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Paper order submission failed: {exc}")
    with b2:
        if st.button("🔴 CLOSE ALL PAPER POSITIONS", disabled=not confirm, use_container_width=True):
            try:
                responses = close_paper_positions()
                st.success(f"Close-all request submitted for {len(responses)} paper positions/orders.")
            except Exception as exc:
                st.error(f"Paper close failed: {exc}")

    st.subheader("💼 Paper Account")
    try:
        positions = get_positions()
        if positions:
            pos_rows = [{"Symbol": p.symbol, "Qty": float(p.qty), "Market Value": float(p.market_value), "Unrealized P/L": float(p.unrealized_pl), "Avg Entry": float(p.avg_entry_price), "Current": float(p.current_price)} for p in positions]
            st.dataframe(pd.DataFrame(pos_rows).style.format({"Qty":"{:.4f}","Market Value":"${:,.2f}","Unrealized P/L":"${:,.2f}","Avg Entry":"${:,.2f}","Current":"${:,.2f}"}), width="stretch", hide_index=True)
        else:
            st.caption("No open paper positions.")
    except Exception as exc:
        st.error(f"Could not read paper positions: {exc}")

    try:
        orders = get_orders()
        if orders:
            order_rows = [{"ID": str(o.id), "Symbol": o.symbol, "Side": str(o.side), "Status": str(o.status), "Qty": str(o.qty), "Notional": str(o.notional), "Submitted": str(o.submitted_at)} for o in orders[:50]]
            st.subheader("📋 Recent Paper Orders")
            st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not read paper orders: {exc}")

st.divider()
st.caption("VAST CASH • MAXPROFIT • 92 trading-day lookback • Buy D1 / Sell D6 • Max 10 positions • PAPER ONLY")
