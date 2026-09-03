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

# A liquid starting universe. MAXPROFIT chooses the top 10 from this universe each quarter.
MARKET_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "AMD",
    "NFLX", "ORCL", "CRM", "ADBE", "QCOM", "INTC", "MU", "AMAT", "LRCX", "TXN",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "C", "JNJ", "UNH",
    "XOM", "CVX", "COST", "WMT", "HD", "LOW", "CAT", "GE", "BA", "DIS"
]


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


def load_history(symbol, start_date, end_date):
    headers = alpaca_headers()
    if not headers:
        return None, "Alpaca paper credentials are unavailable."
    params = {
        "timeframe": "1Day", "start": start_date, "end": end_date,
        "limit": 10000, "adjustment": "all", "feed": "iex", "sort": "asc"
    }
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


def quarter_windows(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    cursor = start
    while cursor < end:
        q_end = min(cursor + pd.DateOffset(months=3), end)
        windows.append((cursor, q_end))
        cursor = q_end
    return windows


def select_top_10(history_by_symbol, quarter_start):
    # Only use the 63 trading days immediately BEFORE the quarter begins.
    scores = []
    qstart = pd.Timestamp(quarter_start)
    for ticker, df in history_by_symbol.items():
        prior = df[df.index < qstart].tail(63)
        if len(prior) < 50:
            continue
        ret = float(prior["close"].iloc[-1] / prior["close"].iloc[0] - 1) * 100
        scores.append((ticker, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]


def simulate_quarter(df, quarter_start, quarter_end, starting_capital, buy_drop_pct=15.0, profit_target_pct=8.0):
    # Buy when price is <= 15% below the rolling 60-trading-day high.
    # Sell at the first later close >= 8% above actual entry, using next-day open execution.
    data = df[(df.index >= pd.Timestamp(quarter_start)) & (df.index <= pd.Timestamp(quarter_end))].copy()
    if len(data) < 2:
        return starting_capital, []

    cash = starting_capital
    position = None
    trades = []
    for i in range(1, len(data) - 1):
        row = data.iloc[i]
        if position is None:
            prior = df[df.index <= data.index[i]].tail(60)
            if len(prior) < 20:
                continue
            rolling_high = float(prior["close"].max())
            buy_trigger = rolling_high * (1 - buy_drop_pct / 100)
            if float(row["close"]) <= buy_trigger:
                entry_i = i + 1
                entry_price = float(data["open"].iloc[entry_i])
                dollars = cash * 0.50
                qty = int(dollars // entry_price)
                if qty >= 1:
                    position = {
                        "entry_i": entry_i,
                        "entry_price": entry_price,
                        "qty": qty,
                        "entry_date": data.index[entry_i].date(),
                    }
                    cash -= qty * entry_price
        else:
            target = position["entry_price"] * (1 + profit_target_pct / 100)
            if float(row["close"]) >= target:
                exit_i = i + 1
                exit_price = float(data["open"].iloc[exit_i])
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                cash += position["qty"] * exit_price
                trades.append({
                    "Buy Date": position["entry_date"], "Sell Date": data.index[exit_i].date(),
                    "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
                    "Sell": round(exit_price, 2), "P/L": round(pnl, 2),
                    "Return %": round((exit_price / position["entry_price"] - 1) * 100, 2),
                    "Reason": "+8% target"
                })
                position = None

    # Mark an unfinished quarter position to market. It is not counted as a completed win.
    if position is not None:
        last_close = float(data["close"].iloc[-1])
        cash += position["qty"] * last_close
        trades.append({
            "Buy Date": position["entry_date"], "Sell Date": data.index[-1].date(),
            "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
            "Sell": round(last_close, 2),
            "P/L": round((last_close - position["entry_price"]) * position["qty"], 2),
            "Return %": round((last_close / position["entry_price"] - 1) * 100, 2),
            "Reason": "Quarter end mark-to-market"
        })
    return cash, trades


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT QUARTERLY ENGINE")
st.write("Find strong stocks from the previous quarter, buy reasonable pullbacks, and sell automatically when the profit target is reached. Historical simulation only. No live trades are sent.")

col1, col2, col3 = st.columns(3)
with col1:
    capital = st.number_input("Starting money", min_value=100.0, value=1000.0, step=100.0)
with col2:
    test_days = st.number_input("Test length (calendar days)", min_value=180, max_value=3650, value=365, step=30)
with col3:
    buy_drop = st.number_input("Buy up to % below recent high", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

profit_target = st.number_input("Sell at % above actual purchase price", min_value=3.0, max_value=30.0, value=8.0, step=1.0)

st.info("MAXPROFIT automatically selects the 10 strongest stocks from the preceding 3 months. It then looks for pullbacks of the selected stocks during the following quarter. No fixed hold-days rule is used.")

if st.button("▶️ RUN MAXPROFIT SIMULATION", type="primary", width="stretch"):
    headers = alpaca_headers()
    if not headers:
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()

    end = datetime.now(timezone.utc).date()
    requested_start = (datetime.now(timezone.utc) - timedelta(days=int(test_days))).date()
    data_start = requested_start - timedelta(days=120)
    end_iso = end.isoformat()
    start_iso = data_start.isoformat()

    histories = {}
    progress = st.progress(0)
    status = st.empty()
    for n, ticker in enumerate(MARKET_UNIVERSE):
        status.write(f"Loading market history: {ticker} ({n + 1}/{len(MARKET_UNIVERSE)})")
        hist, err = load_history(ticker, start_iso, end_iso)
        if hist is not None and len(hist) >= 50:
            histories[ticker] = hist
        progress.progress((n + 1) / len(MARKET_UNIVERSE))

    if not histories:
        st.error("No usable market history was returned.")
        st.stop()

    windows = quarter_windows(requested_start, end)
    quarter_rows = []
    all_trades = []
    total_start = float(capital)
    total_end = float(capital)

    for qnum, (qstart, qend) in enumerate(windows, start=1):
        selected = select_top_10(histories, qstart)
        if not selected:
            continue
        selected_names = [x[0] for x in selected]
        # Equal starting capital per quarter, with gains/losses carried forward.
        quarter_start_capital = total_end
        quarter_end_capital = quarter_start_capital
        qtr_trades = []
        for ticker in selected_names:
            ending, trades = simulate_quarter(histories[ticker], qstart, qend, quarter_start_capital / 10, buy_drop, profit_target)
            quarter_end_capital += ending - (quarter_start_capital / 10)
            for trade in trades:
                trade["Ticker"] = ticker
                trade["Quarter"] = f"{qstart.date()} to {qend.date()}"
                qtr_trades.append(trade)
        total_end = quarter_end_capital
        all_trades.extend(qtr_trades)
        quarter_rows.append({
            "Quarter": f"{qstart.date()} to {qend.date()}",
            "Top 10": ", ".join(selected_names),
            "Start Capital": round(quarter_start_capital, 2),
            "End Capital": round(quarter_end_capital, 2),
            "Quarter P/L": round(quarter_end_capital - quarter_start_capital, 2),
            "Trades": len(qtr_trades),
        })

    pnl = total_end - total_start
    ret = pnl / total_start * 100 if total_start else 0
    completed = [t for t in all_trades if t["Reason"] == "+8% target"]
    winners = [t for t in completed if t["P/L"] > 0]

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Starting Capital", f"${total_start:,.2f}")
    b.metric("Ending Capital", f"${total_end:,.2f}")
    c.metric("Profit / Loss", f"${pnl:+,.2f}")
    d.metric("Return", f"{ret:+.2f}%")

    a, b, c = st.columns(3)
    a.metric("Completed +8% Sales", len(completed))
    b.metric("Target Win Rate", f"{len(winners) / len(completed) * 100:.1f}%" if completed else "0.0%")
    c.metric("All Entries", len(all_trades))

    if quarter_rows:
        st.subheader("Quarter-by-Quarter Results")
        st.dataframe(pd.DataFrame(quarter_rows), width="stretch", hide_index=True)

    if all_trades:
        st.subheader("Trade Results")
        st.dataframe(pd.DataFrame(all_trades).sort_values("Buy Date"), width="stretch", hide_index=True)
    else:
        st.info("No qualifying pullback trades occurred during the selected test period.")

    st.caption("MAXPROFIT uses only information available before each quarter to select the top 10. The simulator is historical evidence, not a guarantee of future performance. It does not send live orders.")
