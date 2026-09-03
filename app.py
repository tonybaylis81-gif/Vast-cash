import math
import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True
ALPACA_TRADE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


def _find_secret(mapping, names):
    if not isinstance(mapping, Mapping):
        return None
    wanted = {n.upper() for n in names}
    for key in mapping:
        value = mapping[key]
        if str(key).strip().upper() in wanted and value is not None and str(value).strip():
            return str(value).strip()
        if isinstance(value, Mapping):
            found = _find_secret(value, names)
            if found:
                return found
    return None


def get_secret(*names):
    try:
        found = _find_secret(st.secrets, names)
        if found:
            return found
    except Exception:
        pass
    for name in names:
        value = os.getenv(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def alpaca_credentials():
    return (
        get_secret("ALPACA_API_KEY", "PAPER_API_KEY", "ALPACA_API_KEY_ID", "API_KEY"),
        get_secret("ALPACA_SECRET_KEY", "PAPER_API_SECRET", "ALPACA_API_SECRET", "API_SECRET", "SECRET_KEY"),
    )


def alpaca_headers():
    key, secret = alpaca_credentials()
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret} if key and secret else None


def period_start(period):
    days = {"6mo": 190, "1y": 370, "2y": 740, "3y": 1100, "5y": 1850}
    return (datetime.now(timezone.utc) - timedelta(days=days[period])).date().isoformat()


def load_history(symbol, period):
    headers = alpaca_headers()
    if not headers:
        return None, "Alpaca paper credentials are unavailable."
    params = {"timeframe": "1Day", "start": period_start(period), "end": datetime.now(timezone.utc).date().isoformat(), "limit": 10000, "adjustment": "all", "feed": "iex", "sort": "asc"}
    bars = []
    token = None
    try:
        for _ in range(20):
            if token:
                params["page_token"] = token
            r = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                try:
                    detail = r.json().get("message", r.text[:300])
                except Exception:
                    detail = r.text[:300]
                return None, f"Alpaca data HTTP {r.status_code}: {detail}"
            payload = r.json()
            bars.extend(payload.get("bars", []))
            token = payload.get("next_page_token")
            if not token:
                break
        if not bars:
            return None, f"No daily data returned for {symbol}."
        df = pd.DataFrame(bars)
        needed = ["t", "o", "c", "v"]
        if not all(x in df.columns for x in needed):
            return None, "Unexpected Alpaca bar format."
        df = df[needed].copy()
        df.columns = ["date", "open", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        return df[["open", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna(), None
    except Exception as exc:
        return None, f"Alpaca market-data error: {exc}"


def score(ind):
    s = 50
    if ind["price"] > ind["sma20"]: s += 15
    else: s -= 15
    if ind["sma20"] > ind["sma50"]: s += 15
    else: s -= 15
    if ind["momentum"] > 3: s += 15
    elif ind["momentum"] < -3: s -= 15
    if ind["volume_ratio"] >= 1.1: s += 5
    if ind["volatility"] > 55: s -= 15
    elif ind["volatility"] < 18: s += 5
    return max(0, min(100, s))


def indicators(d):
    c = d["close"]
    if len(c) < 51:
        return None
    sma20 = c.rolling(20).mean().iloc[-1]
    sma50 = c.rolling(50).mean().iloc[-1]
    momentum = (c.iloc[-1] / c.iloc[-21] - 1) * 100
    volatility = c.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252) * 100
    avg_volume = d["volume"].rolling(20).mean().iloc[-1]
    return {"price": float(c.iloc[-1]), "sma20": float(sma20), "sma50": float(sma50), "momentum": float(momentum), "volatility": float(volatility), "volume_ratio": float(d["volume"].iloc[-1] / avg_volume if avg_volume else 0)}


def signal(ind):
    s = score(ind)
    return "BUY" if s >= 70 else "SELL" if s <= 35 else "HOLD"


def simulate_symbol(df, starting_capital, allocation_pct, hold_days):
    cash = starting_capital
    trades = []
    i = 51
    while i < len(df) - 1:
        hist = df.iloc[:i]
        ind = indicators(hist)
        if not ind:
            i += 1
            continue
        sig = signal(ind)
        if sig != "BUY":
            i += 1
            continue
        entry_i = i + 1
        entry = float(df["open"].iloc[entry_i])
        dollars = cash * allocation_pct / 100
        qty = int(dollars // entry)
        if qty < 1:
            i += 1
            continue
        last_i = min(entry_i + hold_days, len(df) - 1)
        exit_i = entry_i
        for j in range(entry_i + 1, last_i + 1):
            h = df.iloc[:j]
            ind2 = indicators(h)
            if ind2 and signal(ind2) == "SELL":
                exit_i = j + 1 if j + 1 < len(df) else j
                break
            exit_i = j
        exit_price = float(df["open"].iloc[exit_i])
        pnl = (exit_price - entry) * qty
        cash += pnl
        trades.append({"Buy Date": df.index[entry_i].date(), "Sell Date": df.index[exit_i].date(), "Shares": qty, "Buy": round(entry, 2), "Sell": round(exit_price, 2), "Hold Days": exit_i - entry_i, "P/L": round(pnl, 2), "Return %": round((exit_price / entry - 1) * 100, 2)})
        i = max(exit_i + 1, i + 1)
    return cash, trades


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT SIMULATOR")
st.write("Test the strategy against real historical market data. No live trades are sent by the simulator.")

col1, col2, col3 = st.columns(3)
with col1:
    capital = st.number_input("Starting money", min_value=100.0, value=1000.0, step=100.0)
with col2:
    period = st.selectbox("Test period", ["6mo", "1y", "2y", "3y", "5y"], index=1)
with col3:
    hold_days = st.number_input("Maximum hold (trading days)", min_value=1, max_value=252, value=6, step=1)

stock_text = st.text_area("Stocks to test (up to 10, one per line)", "AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nAMD\nAVGO\nJPM", height=150)
stocks = list(dict.fromkeys(s.strip().upper() for s in stock_text.replace(",", "\n").splitlines() if s.strip()))[:10]
allocation = 50.0

if st.button("▶️ RUN SIMULATION", type="primary", width="stretch"):
    if not alpaca_headers():
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()
    all_trades = []
    ending_total = 0.0
    progress = st.progress(0)
    for n, ticker in enumerate(stocks):
        hist, err = load_history(ticker, period)
        if hist is None or len(hist) < 60:
            st.warning(f"{ticker}: {err or 'Not enough history'}")
            progress.progress((n + 1) / len(stocks))
            continue
        ending, trades = simulate_symbol(hist, capital, allocation, int(hold_days))
        ending_total += ending
        for t in trades:
            t["Ticker"] = ticker
            all_trades.append(t)
        progress.progress((n + 1) / len(stocks))
    if stocks:
        ending_capital = ending_total / len(stocks) if ending_total else capital
    else:
        ending_capital = capital
    pnl = ending_capital - capital
    ret = pnl / capital * 100 if capital else 0
    winners = [t for t in all_trades if t["P/L"] > 0]
    losers = [t for t in all_trades if t["P/L"] < 0]
    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Starting Capital", f"${capital:,.2f}")
    b.metric("Ending Capital", f"${ending_capital:,.2f}")
    c.metric("Profit / Loss", f"${pnl:+,.2f}")
    d.metric("Return", f"{ret:+.2f}%")
    a, b, c = st.columns(3)
    a.metric("Trades", len(all_trades))
    b.metric("Win Rate", f"{len(winners) / len(all_trades) * 100:.1f}%" if all_trades else "0.0%")
    c.metric("Best Trade", f"${max((t['P/L'] for t in all_trades), default=0):+.2f}")
    if all_trades:
        st.subheader("Trade Results")
        st.dataframe(pd.DataFrame(all_trades).sort_values("Buy Date"), width="stretch", hide_index=True)
    else:
        st.info("No BUY signals occurred during the selected test period.")
    st.caption("This is a historical simulation, not a guarantee of future performance. It uses the MAXPROFIT signal rules and next-day-open entries with an automatic maximum hold.")
