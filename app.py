import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True
ALPACA_DATA_URL = "https://data.alpaca.markets"

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
    bars, token = [], None
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
        needed = ["t", "o", "h", "c", "v"]
        if not all(x in df.columns for x in needed):
            return None, "Unexpected Alpaca bar format."
        df = df[needed].copy()
        df.columns = ["date", "open", "high", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        return df.apply(pd.to_numeric, errors="coerce").dropna(), None
    except Exception as exc:
        return None, f"Alpaca market-data error: {exc}"


def quarter_windows(start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    windows, cursor = [], start
    while cursor < end:
        q_end = min(cursor + pd.DateOffset(months=3), end)
        windows.append((cursor, q_end))
        cursor = q_end
    return windows


def market_state(df, as_of):
    """Create a point-in-time feature vector using ONLY data before as_of."""
    prior = df[df.index < pd.Timestamp(as_of)].tail(63)
    if len(prior) < 50:
        return None
    close = prior["close"]
    ret_3m = float(close.iloc[-1] / close.iloc[0] - 1)
    daily = close.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(252)) if len(daily) else 0.0
    high = float(close.max())
    drawdown = float(close.iloc[-1] / high - 1)
    slope = float(np.polyfit(np.arange(len(close)), close.values, 1)[0] / max(close.iloc[-1], 1e-9))
    return np.array([ret_3m, vol, drawdown, slope], dtype=float)


def historical_prediction(df, quarter_start):
    """Predict next-quarter return from prior historical states, without look-ahead."""
    current = market_state(df, quarter_start)
    if current is None:
        return None

    # Walk through older historical points. Each training example uses 63 days of
    # information ending before the point and then observes the following 63 days.
    dates = df.index
    examples = []
    min_i = 70
    max_i = len(dates) - 63
    for i in range(min_i, max_i):
        state = market_state(df, dates[i])
        if state is None:
            continue
        future = df.iloc[i:i + 63]
        if len(future) < 63:
            continue
        future_ret = float(future["close"].iloc[-1] / future["close"].iloc[0] - 1)
        examples.append((state, future_ret, dates[i]))

    if len(examples) < 5:
        return None

    # Nearest historical market states. No future information from the current quarter is used.
    states = np.array([x[0] for x in examples])
    scale = np.std(states, axis=0)
    scale[scale == 0] = 1.0
    distances = np.linalg.norm((states - current) / scale, axis=1)
    order = np.argsort(distances)[:min(8, len(examples))]
    nearest = [examples[i] for i in order]
    weights = np.array([1.0 / (distances[i] + 0.05) for i in order])
    returns = np.array([x[1] for x in nearest])
    prediction = float(np.average(returns, weights=weights))
    confidence = float(np.mean(np.abs(returns - prediction)) if len(returns) else 1.0)
    return prediction, confidence, len(nearest), current


def select_predicted_top_10(history_by_symbol, quarter_start):
    results = []
    for ticker, df in history_by_symbol.items():
        pred = historical_prediction(df, quarter_start)
        if pred is None:
            continue
        prediction, uncertainty, samples, state = pred
        results.append({
            "Ticker": ticker,
            "Predicted Next Quarter %": prediction * 100,
            "Historical Uncertainty %": uncertainty * 100,
            "History Matches": samples,
            "3M Momentum %": state[0] * 100,
            "Volatility %": state[1] * 100,
            "Drawdown %": state[2] * 100,
        })
    results.sort(key=lambda x: x["Predicted Next Quarter %"], reverse=True)
    return results[:10], results


def simulate_quarter(df, quarter_start, quarter_end, starting_capital, buy_drop_pct, profit_target_pct):
    """Trade only the selected stock during the quarter using the USER variables."""
    data = df[(df.index >= pd.Timestamp(quarter_start)) & (df.index <= pd.Timestamp(quarter_end))].copy()
    if len(data) < 2:
        return starting_capital, []

    cash = float(starting_capital)
    position = None
    trades = []

    for i in range(1, len(data)):
        row = data.iloc[i]
        if position is None:
            prior = df[df.index < data.index[i]].tail(60)
            if len(prior) < 20:
                continue
            rolling_high = float(prior["close"].max())
            buy_trigger = rolling_high * (1 - buy_drop_pct / 100)
            if float(row["close"]) <= buy_trigger:
                entry_price = float(row["close"])
                qty = int((cash * 0.50) // entry_price)
                if qty >= 1:
                    position = {"entry_price": entry_price, "qty": qty, "entry_date": data.index[i].date()}
                    cash -= qty * entry_price
        else:
            target = position["entry_price"] * (1 + profit_target_pct / 100)
            # Daily high reaching the target means the target was hit intraday.
            if float(row["high"]) >= target:
                exit_price = target
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                cash += position["qty"] * exit_price
                trades.append({
                    "Buy Date": position["entry_date"], "Sell Date": data.index[i].date(),
                    "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
                    "Sell": round(exit_price, 2), "P/L": round(pnl, 2),
                    "Return %": round((exit_price / position["entry_price"] - 1) * 100, 2),
                    "Reason": f"+{profit_target_pct:.1f}% target"
                })
                position = None

    if position is not None:
        last_close = float(data["close"].iloc[-1])
        cash += position["qty"] * last_close
        trades.append({
            "Buy Date": position["entry_date"], "Sell Date": data.index[-1].date(),
            "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
            "Sell": round(last_close, 2),
            "P/L": round((last_close - position["entry_price"]) * position["qty"], 2),
            "Return %": round((last_close / position["entry_price"] - 1) * 100, 2),
            "Reason": "Quarter-end mark-to-market"
        })
    return cash, trades


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT HISTORICAL PREDICTION ENGINE")
st.write("MAXPROFIT learns from previous market patterns, predicts which stocks have the strongest historical next-quarter expectation, then tests the buy-low/sell-high rules on the following quarter. Historical simulation only. No live orders.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    capital = st.number_input("Starting money", min_value=100.0, value=1000.0, step=100.0)
with col2:
    test_days = st.number_input("Test length (calendar days)", min_value=180, max_value=3650, value=730, step=30)
with col3:
    buy_drop = st.number_input("Buy up to % below recent high", min_value=5.0, max_value=30.0, value=15.0, step=1.0)
with col4:
    profit_target = st.number_input("Sell at % above purchase", min_value=1.0, max_value=30.0, value=8.0, step=1.0)

st.info("The stock picker is no longer simply choosing the stocks that already went up most. For every quarter it builds a point-in-time market state, searches older historical states that looked similar, and uses what happened AFTER those old states to estimate the next quarter. Then the simulation applies your current buy-pullback and sell-target variables.")

if st.button("▶️ RUN MAXPROFIT SIMULATION", type="primary", width="stretch"):
    if not alpaca_headers():
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()

    end = datetime.now(timezone.utc).date()
    requested_start = (datetime.now(timezone.utc) - timedelta(days=int(test_days))).date()
    # Extra history is essential because the prediction model must learn from older quarters.
    data_start = (datetime.now(timezone.utc) - timedelta(days=int(test_days) + 1100)).date()

    histories = {}
    progress = st.progress(0)
    status = st.empty()
    for n, ticker in enumerate(MARKET_UNIVERSE):
        status.write(f"Loading market history: {ticker} ({n + 1}/{len(MARKET_UNIVERSE)})")
        hist, err = load_history(ticker, data_start.isoformat(), end.isoformat())
        if hist is not None and len(hist) >= 150:
            histories[ticker] = hist
        progress.progress((n + 1) / len(MARKET_UNIVERSE))

    if not histories:
        st.error("No usable market history was returned.")
        st.stop()

    windows = quarter_windows(requested_start, end)
    total_start = float(capital)
    total_end = float(capital)
    quarter_rows, all_trades, all_predictions = [], [], []

    for qstart, qend in windows:
        selected, full_rank = select_predicted_top_10(histories, qstart)
        if not selected:
            continue
        selected_names = [x["Ticker"] for x in selected]
        quarter_start_capital = total_end
        quarter_end_capital = quarter_start_capital
        qtr_trades = []

        for item in selected:
            ticker = item["Ticker"]
            ending, trades = simulate_quarter(histories[ticker], qstart, qend, quarter_start_capital / 10, buy_drop, profit_target)
            quarter_end_capital += ending - quarter_start_capital / 10
            for trade in trades:
                trade["Ticker"] = ticker
                trade["Quarter"] = f"{qstart.date()} to {qend.date()}"
                qtr_trades.append(trade)

        total_end = quarter_end_capital
        all_trades.extend(qtr_trades)
        for rank, item in enumerate(selected, start=1):
            item = dict(item)
            item["Quarter"] = f"{qstart.date()} to {qend.date()}"
            item["Rank"] = rank
            all_predictions.append(item)

        quarter_rows.append({
            "Quarter": f"{qstart.date()} to {qend.date()}",
            "Predicted Top 10": ", ".join(selected_names),
            "Start Capital": round(quarter_start_capital, 2),
            "End Capital": round(quarter_end_capital, 2),
            "Quarter P/L": round(quarter_end_capital - quarter_start_capital, 2),
            "Trades": len(qtr_trades),
        })

    pnl = total_end - total_start
    ret = pnl / total_start * 100 if total_start else 0
    completed = [t for t in all_trades if "target" in t["Reason"]]
    winners = [t for t in completed if t["P/L"] > 0]

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Starting Capital", f"${total_start:,.2f}")
    b.metric("Ending Capital", f"${total_end:,.2f}")
    c.metric("Profit / Loss", f"${pnl:+,.2f}")
    d.metric("Return", f"{ret:+.2f}%")

    a, b, c = st.columns(3)
    a.metric("Target Sales", len(completed))
    b.metric("Target Win Rate", f"{len(winners) / len(completed) * 100:.1f}%" if completed else "0.0%")
    c.metric("All Trade Exits", len(all_trades))

    if quarter_rows:
        st.subheader("Quarter-by-Quarter Results")
        st.dataframe(pd.DataFrame(quarter_rows), width="stretch", hide_index=True)

    if all_predictions:
        st.subheader("What MAXPROFIT Predicted From History")
        st.dataframe(pd.DataFrame(all_predictions), width="stretch", hide_index=True)

    if all_trades:
        st.subheader("Trade Results")
        st.dataframe(pd.DataFrame(all_trades).sort_values("Buy Date"), width="stretch", hide_index=True)
    else:
        st.info("No qualifying pullback trades occurred during the selected test period.")

    st.caption("Prediction uses only historical information available before each quarter. The future quarter is then used strictly to test whether that historical prediction and trading rule worked. This is evidence for testing, not a guarantee of future returns. Paper/simulation only.")
