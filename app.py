import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="⚒️", layout="wide")

# ============================================================
# VAST CASH: STOCK TRADING FOR WELDERS
# Simple front end. PAPER ONLY. No live orders.
# ============================================================
PAPER_ONLY = True
DATA_URL = "https://data.alpaca.markets"
TRADE_URL = "https://paper-api.alpaca.markets"
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","AMD",
    "NFLX","ORCL","CRM","ADBE","QCOM","INTC","MU","AMAT","LRCX","TXN",
    "JPM","BAC","WFC","GS","MS","V","MA","C","JNJ","UNH","XOM","CVX",
    "COST","WMT","HD","LOW","CAT","GE","BA","DIS"
]


def _secret(names):
    wanted = {x.strip().upper() for x in names}
    try:
        def walk(value):
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key).strip().upper() in wanted and item is not None and str(item).strip():
                        return str(item).strip()
                    found = walk(item)
                    if found:
                        return found
            return None
        found = walk(st.secrets)
        if found:
            return found
    except Exception:
        pass
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def alpaca_headers():
    key = _secret(["PAPER_API_KEY", "ALPACA_API_KEY", "ALPACA_API_KEY_ID", "API_KEY"])
    secret = _secret(["PAPER_API_SECRET", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET", "API_SECRET", "SECRET_KEY"])
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


@st.cache_data(ttl=21600, show_spinner=False)
def load_history(symbol, days=420):
    headers = alpaca_headers()
    if not headers:
        return None
    end = datetime.now(timezone.utc).date()
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    params = {
        "timeframe": "1Day", "start": start.isoformat(), "end": end.isoformat(),
        "limit": 10000, "adjustment": "all", "feed": "iex", "sort": "asc"
    }
    try:
        bars, token = [], None
        for _ in range(6):
            if token:
                params["page_token"] = token
            r = requests.get(f"{DATA_URL}/v2/stocks/{symbol}/bars", headers=headers, params=params, timeout=12)
            if r.status_code != 200:
                return None
            payload = r.json()
            bars.extend(payload.get("bars", []))
            token = payload.get("next_page_token")
            if not token:
                break
        if not bars:
            return None
        d = pd.DataFrame(bars)[["t", "o", "h", "c", "v"]]
        d.columns = ["date", "open", "high", "close", "volume"]
        d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        return d.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce").dropna()
    except Exception:
        return None


def load_universe(symbols):
    # Six workers avoids the request storm that caused Streamlit throttling.
    result = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(load_history, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data = future.result()
                if data is not None and len(data) >= 120:
                    result[symbol] = data
            except Exception:
                pass
    return result


def fingerprint(df, end=None, lookback=63):
    d = df if end is None else df.iloc[:end]
    if len(d) < lookback + 25:
        return None
    c = d["close"]
    window = c.tail(lookback)
    daily = c.pct_change().dropna().tail(30)
    recent_high = float(c.tail(60).max())
    price = float(c.iloc[-1])
    return {
        "price": price,
        "return_63": float(price / window.iloc[0] - 1),
        "return_20": float(price / c.iloc[-21] - 1),
        "return_5": float(price / c.iloc[-6] - 1),
        "volatility": float(daily.std() * np.sqrt(252)),
        "drawdown": float(price / recent_high - 1),
        "sma20": float(c.tail(20).mean()),
        "sma50": float(c.tail(50).mean()),
    }


def historical_fingerprint_test(df, hold_days, buy_drop, sell_target):
    """Walk historical pullback setups and measure how often the target was reached."""
    if len(df) < 140:
        return None
    wins, returns, holds = [], [], []
    start = 70
    stop = len(df) - hold_days - 2
    step = max(1, (stop - start) // 90)
    for i in range(start, stop, step):
        prior = df.iloc[max(0, i - 60):i]
        if len(prior) < 20:
            continue
        entry_reference = float(prior["close"].max())
        trigger = entry_reference * (1 - buy_drop / 100)
        future = df.iloc[i:i + hold_days + 1]
        entry_rows = np.where(future["low"].to_numpy(float) <= trigger)[0]
        if len(entry_rows) == 0:
            continue
        e = int(entry_rows[0])
        entry = trigger
        after = future.iloc[e:e + hold_days + 1]
        target = entry * (1 + sell_target / 100)
        hits = np.where(after["high"].to_numpy(float) >= target)[0]
        if len(hits):
            h = int(hits[0])
            returns.append(sell_target / 100)
            wins.append(1)
            holds.append(max(1, h))
        else:
            exit_price = float(after["close"].iloc[-1])
            returns.append(exit_price / entry - 1)
            wins.append(0)
            holds.append(len(after) - 1)
    if not returns:
        return None
    r = np.asarray(returns, dtype=float)
    return {
        "trades": len(r),
        "win_rate": float(np.mean(wins)),
        "median_return": float(np.median(r)),
        "mean_return": float(np.mean(r)),
        "hold": int(round(np.average(holds, weights=np.maximum(r + 0.05, 0.01)))),
    }


def score_stock(df, hold_days, buy_drop, sell_target):
    f = fingerprint(df)
    if not f:
        return None
    test = historical_fingerprint_test(df, hold_days, buy_drop, sell_target)
    if not test or test["trades"] < 4:
        return None
    # Ranking favors repeatability and return, while penalizing large volatility.
    trend_bonus = 0.10 if f["sma20"] > f["sma50"] else -0.10
    score = test["median_return"] * 100 + test["win_rate"] * 20 + trend_bonus - f["volatility"] * 5
    recent_high = float(df["close"].tail(60).max())
    trigger = recent_high * (1 - buy_drop / 100)
    return {
        "Ticker": "",
        "Score": float(score),
        "Expected Return": test["median_return"],
        "Win Rate": test["win_rate"],
        "Historical Trades": test["trades"],
        "Typical Hold": max(1, min(hold_days, test["hold"])),
        "Price": f["price"],
        "Buy Trigger": trigger,
        "Sell Target": trigger * (1 + sell_target / 100),
        "Momentum": f["return_20"],
        "Volatility": f["volatility"],
        "Fingerprint": f,
    }


def next_trading_date(days):
    d = datetime.now().date()
    count = 0
    while count < int(days):
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d.isoformat()


def paper_buy(symbol, notional):
    if not PAPER_ONLY:
        return False, "Live trading is disabled."
    h = alpaca_headers()
    if not h:
        return False, "Alpaca PAPER credentials are unavailable."
    body = {"symbol": symbol, "notional": f"{max(1.0, notional):.2f}", "side": "buy", "type": "market", "time_in_force": "day"}
    try:
        r = requests.post(f"{TRADE_URL}/v2/orders", headers={**h, "Content-Type": "application/json"}, json=body, timeout=15)
        if r.status_code not in (200, 201):
            return False, f"PAPER BUY rejected: {r.text[:250]}"
        return True, f"PAPER BUY submitted for {symbol}."
    except Exception as exc:
        return False, f"PAPER BUY error: {exc}"


def paper_account():
    h = alpaca_headers()
    if not h:
        return None
    try:
        r = requests.get(f"{TRADE_URL}/v2/account", headers=h, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ---------------- UI ----------------
st.title("⚒️ VAST CASH")
st.subheader("STOCK TRADING FOR WELDERS")
st.caption("MAXPROFIT does the math. You make the YES / NO decision. PAPER ONLY.")

with st.sidebar:
    st.header("🔧 MAXPROFIT SETTINGS")
    hold_days = st.slider("Maximum hold (trading days)", 1, 30, 4)
    buy_drop = st.slider("Buy % below recent high", 1, 20, 15)
    sell_target = st.slider("Sell % above purchase", 1, 20, 8)
    capital = st.number_input("Paper capital ($)", 100.0, 1000000.0, 1000.0, 100.0)
    allocation = st.slider("Capital used for YES selections (%)", 5, 100, 50, 5)
    st.divider()
    st.caption("The engine searches historical setups. Results are evidence, not a guarantee of future returns.")

if "top10" not in st.session_state:
    st.session_state.top10 = None
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "last_run" not in st.session_state:
    st.session_state.last_run = None

run = st.button("⚡ RUN MAXPROFIT", type="primary", use_container_width=True)
if run:
    if not alpaca_headers():
        st.error("Alpaca PAPER credentials are not available. Check Streamlit Secrets.")
    else:
        with st.spinner("MAXPROFIT is testing the market fingerprints..."):
            histories = load_universe(UNIVERSE)
            ranked = []
            for ticker, df in histories.items():
                result = score_stock(df, hold_days, buy_drop, sell_target)
                if result:
                    result["Ticker"] = ticker
                    ranked.append(result)
            ranked.sort(key=lambda x: (x["Expected Return"], x["Win Rate"], x["Score"]), reverse=True)
            st.session_state.top10 = ranked[:10]
            st.session_state.decisions = {x["Ticker"]: None for x in st.session_state.top10}
            st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")

if st.session_state.top10:
    top10 = st.session_state.top10
    st.success(f"MAXPROFIT found the current Top {len(top10)} historical setups. Run: {st.session_state.last_run}")
    st.header("🏆 TOP 10 — YOUR DECISION")
    st.caption("Every stock starts as a BUY candidate. Mark YES or NO. No paper order is sent while you choose.")

    for rank, item in enumerate(top10, 1):
        ticker = item["Ticker"]
        with st.container(border=True):
            a,b,c,d = st.columns([0.6,1.2,1.5,1.5])
            a.metric("#", rank)
            b.metric("STOCK", ticker)
            c.metric("Historical return", f"{item['Expected Return']:+.1%}")
            d.metric("Win rate", f"{item['Win Rate']:.0%}")
            sell_days = item["Typical Hold"]
            st.write(f"**Hold:** {sell_days} trading days  •  **Suggested sell date:** {next_trading_date(sell_days)}  •  **Current:** ${item['Price']:.2f}")
            st.write(f"**Buy trigger:** ${item['Buy Trigger']:.2f}  •  **Sell target:** ${item['Sell Target']:.2f}  •  **Historical tests:** {item['Historical Trades']}")
            st.caption(f"Why it ranked: historical median return {item['Expected Return']:+.1%}, {item['Win Rate']:.0%} profitable setups, 20-day momentum {item['Momentum']:+.1%}, volatility {item['Volatility']:.1%}.")
            left,right = st.columns(2)
            if left.button("✅ YES", key=f"yes_{ticker}", use_container_width=True):
                st.session_state.decisions[ticker] = "YES"
            if right.button("❌ NO", key=f"no_{ticker}", use_container_width=True):
                st.session_state.decisions[ticker] = "NO"
            decision = st.session_state.decisions.get(ticker)
            if decision == "YES":
                st.success("YES selected")
            elif decision == "NO":
                st.info("NO selected")
            else:
                st.warning("Not decided")

    decisions_complete = all(st.session_state.decisions.get(x["Ticker"]) in ("YES", "NO") for x in top10)
    yes = [x for x in top10 if st.session_state.decisions.get(x["Ticker"]) == "YES"]
    st.divider()
    st.metric("YES selections", f"{len(yes)} / {len(top10)}")

    if st.button("🚀 COMMIT SELECTED TO PAPER", type="primary", disabled=not decisions_complete, use_container_width=True):
        if not yes:
            st.info("All ten are NO. Nothing will be sent to Alpaca.")
        else:
            account = paper_account()
            if not account:
                st.error("Could not access the Alpaca PAPER account.")
            else:
                buying_power = float(account.get("buying_power", 0))
                budget = buying_power * allocation / 100 / len(yes)
                st.subheader("📨 PAPER ORDERS")
                for item in yes:
                    ok, message = paper_buy(item["Ticker"], budget)
                    st.success(message) if ok else st.error(message)

st.divider()
st.caption("🔒 PAPER ONLY. There is no live-trading path in this build.")
