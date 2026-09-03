import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

LOOKBACK_DAYS = 92
MAX_POSITIONS = 10
DEFAULT_HOLDING = 5

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "NFLX",
    "COST", "WMT", "JPM", "V", "MA", "ORCL", "CRM", "QCOM", "MU", "AMAT",
    "GE", "CAT", "HON", "UNP", "XOM", "CVX", "COP", "UBER", "SHOP", "PLTR",
    "PANW", "CRWD", "SNOW", "DIS", "TMO", "LLY", "PEP"
]


def download_prices(symbols):
    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=450)
    data = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("No market data was returned. Try again in a moment.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Downloaded data does not contain closing prices.")
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = [symbols[0]]

    prices = prices.sort_index().ffill()
    prices = prices.dropna(axis=1, how="all")
    if len(prices) < LOOKBACK_DAYS + 25:
        raise RuntimeError(f"Only {len(prices)} trading sessions were returned. More history is required.")
    return prices


def symbol_score(series, risk_profile):
    s = series.dropna()
    if len(s) < LOOKBACK_DAYS + 1:
        return None

    last = float(s.iloc[-1])
    r92 = last / float(s.iloc[-(LOOKBACK_DAYS + 1)]) - 1
    r20 = last / float(s.iloc[-21]) - 1
    r10 = last / float(s.iloc[-11]) - 1

    daily = s.pct_change().dropna()
    vol = max(float(daily.std()), 0.0001)
    downside = daily[daily < 0]
    downside_vol = max(float(downside.std()) if len(downside) > 1 else vol, 0.0001)

    momentum = 0.55 * r92 + 0.30 * r20 + 0.15 * r10
    risk_adjusted = r92 / vol
    downside_adjusted = r92 / downside_vol

    if risk_profile == "Conservative":
        score = 0.45 * r92 + 0.15 * r20 + 0.10 * r10 + 0.20 * (risk_adjusted / 10) + 0.10 * (downside_adjusted / 10)
    elif risk_profile == "Aggressive":
        score = 0.65 * r92 + 0.25 * r20 + 0.10 * r10
    else:
        score = 0.50 * r92 + 0.25 * r20 + 0.15 * r10 + 0.05 * (risk_adjusted / 10) + 0.05 * (downside_adjusted / 10)

    return {
        "Price": last,
        "3M Return": r92,
        "20D Return": r20,
        "10D Return": r10,
        "Volatility": vol,
        "Downside Vol": downside_vol,
        "Momentum": momentum,
        "Score": float(score),
    }


def rank_market(prices, risk_profile):
    rows = []
    for symbol in prices.columns:
        result = symbol_score(prices[symbol], risk_profile)
        if result is not None:
            rows.append({"Symbol": symbol, **result})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def select_candidates(ranked, positions, threshold):
    selected = ranked[ranked["Score"] >= threshold].head(positions).copy()
    if selected.empty:
        selected = ranked.head(positions).copy()
    return selected


def build_allocation(selected, capital):
    if selected.empty:
        return selected
    result = selected.copy()
    n = len(result)
    result["Weight"] = 1.0 / n
    result["Allocation"] = capital / n
    result["Shares"] = np.floor(result["Allocation"] / result["Price"]).astype(int)
    result["Invested"] = result["Shares"] * result["Price"]
    result["Cash Left"] = result["Allocation"] - result["Invested"]
    return result


def backtest(prices, positions, risk_profile, threshold, capital, holding_period):
    equity = float(capital)
    rows = []
    trades = []

    first_signal = LOOKBACK_DAYS
    last_signal = len(prices) - holding_period - 1

    for signal_idx in range(first_signal, last_signal + 1, holding_period):
        signal_prices = prices.iloc[: signal_idx + 1]
        ranked = rank_market(signal_prices, risk_profile)
        if ranked.empty:
            continue

        chosen = select_candidates(ranked, positions, threshold)
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + holding_period
        if exit_idx >= len(prices):
            break

        returns = []
        for symbol in chosen["Symbol"]:
            entry = prices.iloc[entry_idx].get(symbol, np.nan)
            exit_price = prices.iloc[exit_idx].get(symbol, np.nan)
            if pd.isna(entry) or pd.isna(exit_price) or entry <= 0:
                continue
            ret = float(exit_price / entry - 1)
            returns.append(ret)
            trades.append({
                "Entry Date": prices.index[entry_idx].date(),
                "Exit Date": prices.index[exit_idx].date(),
                "Symbol": symbol,
                "Entry": float(entry),
                "Exit": float(exit_price),
                "Return": ret,
            })

        if returns:
            portfolio_return = float(np.mean(returns))
            equity *= 1 + portfolio_return
            rows.append({
                "Date": prices.index[exit_idx],
                "Portfolio Value": equity,
                "Period Return": portfolio_return,
            })

    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve = curve.set_index("Date")
    trade_df = pd.DataFrame(trades)
    return curve, trade_df


def performance_stats(curve, trades, starting_capital):
    if curve.empty:
        return {"Final": starting_capital, "Return": 0.0, "Win Rate": 0.0, "Max DD": 0.0, "Trades": 0}

    final_value = float(curve["Portfolio Value"].iloc[-1])
    total_return = final_value / starting_capital - 1
    peak = curve["Portfolio Value"].cummax()
    drawdown = curve["Portfolio Value"] / peak - 1
    win_rate = float((trades["Return"] > 0).mean()) if not trades.empty else 0.0

    return {
        "Final": final_value,
        "Return": total_return,
        "Win Rate": win_rate,
        "Max DD": float(drawdown.min()),
        "Trades": int(len(trades)),
    }


# ---------------- APP ----------------
st.title("💰 VAST CASH")
st.subheader("MAXPROFIT Engine")
st.caption("Market ranking + portfolio allocation + historical paper backtest")

with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input(
        "Investment capital ($)",
        min_value=10.0,
        value=1000.0,
        step=100.0,
        help="Starting paper capital for the simulation. No money is moved.",
    )
    positions = st.slider("Maximum positions", 1, MAX_POSITIONS, 5)
    holding_period = st.slider("Holding period (trading days)", 1, 20, DEFAULT_HOLDING)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    threshold = st.slider("Minimum signal strength", -0.20, 1.00, 0.20, 0.05)
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)

    st.divider()
    st.caption("LOOKBACK: 92 trading days")
    st.caption(f"CYCLE: Buy D1 → Sell D{holding_period + 1}")
    st.caption("UNIVERSE: 38 liquid large-cap stocks")
    st.caption("MODE: PAPER / BACKTEST ONLY")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital", f"${capital:,.0f}")
c2.metric("Positions", positions)
c3.metric("Hold", f"{holding_period} days")
c4.metric("Risk", risk_profile)

st.info(
    "🔒 SAFETY LOCK: VAST CASH is a research and paper-backtest engine. "
    "It does not place live orders, connect to a brokerage, or move real money."
)

if run:
    try:
        with st.spinner("Downloading market history and running MAXPROFIT..."):
            prices = download_prices(UNIVERSE)
            ranked = rank_market(prices, risk_profile)
            selected = select_candidates(ranked, positions, threshold)
            allocation = build_allocation(selected, capital)
            curve, trades = backtest(prices, positions, risk_profile, threshold, capital, holding_period)
            stats = performance_stats(curve, trades, capital)

        st.success("MAXPROFIT run complete.")

        st.subheader("🏆 Current MAXPROFIT Ranking")
        display_rank = ranked.copy()
        for col in ["3M Return", "20D Return", "10D Return", "Volatility", "Downside Vol", "Momentum", "Score"]:
            display_rank[col] = display_rank[col].map(lambda x: f"{x:.2%}" if col != "Score" else f"{x:.4f}")
        display_rank["Price"] = display_rank["Price"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display_rank.head(20), use_container_width=True, hide_index=True)

        st.subheader("💰 Proposed Paper Allocation")
        alloc_display = allocation[["Symbol", "Price", "Score", "Weight", "Allocation", "Shares", "Invested", "Cash Left"]].copy()
        alloc_display["Price"] = alloc_display["Price"].map(lambda x: f"${x:,.2f}")
        alloc_display["Score"] = alloc_display["Score"].map(lambda x: f"{x:.4f}")
        alloc_display["Weight"] = alloc_display["Weight"].map(lambda x: f"{x:.1%}")
        for col in ["Allocation", "Invested", "Cash Left"]:
            alloc_display[col] = alloc_display[col].map(lambda x: f"${x:,.2f}")
        st.dataframe(alloc_display, use_container_width=True, hide_index=True)

        st.subheader("📈 Historical Paper Backtest")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ending Value", f"${stats['Final']:,.2f}")
        m2.metric("Total Return", f"{stats['Return']:.2%}")
        m3.metric("Trade Win Rate", f"{stats['Win Rate']:.1%}")
        m4.metric("Max Drawdown", f"{stats['Max DD']:.2%}")

        if not curve.empty:
            chart = curve[["Portfolio Value"]].copy()
            st.line_chart(chart)
        else:
            st.warning("The selected settings did not produce enough complete backtest cycles.")

        if not trades.empty:
            st.subheader("📋 Backtest Trades")
            trade_display = trades.copy()
            trade_display["Entry"] = trade_display["Entry"].map(lambda x: f"${x:,.2f}")
            trade_display["Exit"] = trade_display["Exit"].map(lambda x: f"${x:,.2f}")
            trade_display["Return"] = trade_display["Return"].map(lambda x: f"{x:.2%}")
            st.dataframe(trade_display.tail(100), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Export ranking CSV",
            ranked.to_csv(index=False).encode("utf-8"),
            "vast_cash_ranking.csv",
            "text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"MAXPROFIT could not complete this run: {exc}")
        st.caption("The app itself is running. This error is from the market-data/backtest step, not the Streamlit deployment.")
else:
    st.write("### Ready")
    st.write("Set the four MAXPROFIT inputs in the sidebar, then press **RUN MAXPROFIT**.")
    st.write("The engine will rank the universe, build a paper allocation, and run the historical holding-cycle backtest.")

st.divider()
st.caption("VAST CASH • MAXPROFIT • Foundation upgraded to full paper/backtest engine")
