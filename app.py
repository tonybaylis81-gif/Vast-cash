import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import itertools
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


def alpaca_headers():
    key = get_secret("ALPACA_API_KEY", "PAPER_API_KEY", "ALPACA_API_KEY_ID", "API_KEY")
    secret = get_secret("ALPACA_SECRET_KEY", "PAPER_API_SECRET", "ALPACA_API_SECRET", "API_SECRET", "SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret} if key and secret else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol, start_date, end_date):
    headers = alpaca_headers()
    if not headers:
        return None, "Paper credentials unavailable."
    params = {"timeframe": "1Day", "start": start_date, "end": end_date, "limit": 10000,
              "adjustment": "all", "feed": "iex", "sort": "asc"}
    bars, token = [], None
    try:
        for _ in range(20):
            if token:
                params["page_token"] = token
            r = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            payload = r.json()
            bars.extend(payload.get("bars", []))
            token = payload.get("next_page_token")
            if not token:
                break
        if not bars:
            return None, "No data"
        df = pd.DataFrame(bars)[["t", "o", "h", "c", "v"]].copy()
        df.columns = ["date", "open", "high", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        return df.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce").dropna(), None
    except Exception as exc:
        return None, str(exc)


def load_all_histories(start_date, end_date):
    histories = {}
    progress = st.progress(0)
    status = st.empty()
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(load_history, ticker, start_date, end_date): ticker for ticker in MARKET_UNIVERSE}
        for future in as_completed(futures):
            ticker = futures[future]
            hist, _ = future.result()
            if hist is not None and len(hist) >= 180:
                histories[ticker] = hist
            completed += 1
            progress.progress(completed / len(MARKET_UNIVERSE))
            status.write(f"Loading market history: {completed}/{len(MARKET_UNIVERSE)} stocks")
    progress.empty()
    status.empty()
    return histories


def market_state(df, as_of, lookback=63):
    prior = df[df.index < pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior) < max(30, int(lookback * .75)):
        return None
    close = prior.close
    daily = close.pct_change().dropna()
    return np.array([
        float(close.iloc[-1] / close.iloc[0] - 1),
        float(daily.std() * np.sqrt(252)) if len(daily) else 0.0,
        float(close.iloc[-1] / close.max() - 1),
        float(np.polyfit(np.arange(len(close)), close.values, 1)[0] / max(close.iloc[-1], 1e-9)),
    ])


def fingerprint_details(df, as_of, lookback=63):
    prior = df[df.index <= pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior) < max(30, int(lookback * .75)):
        return None
    close, daily = prior.close, prior.close.pct_change().dropna()
    recent = close.tail(min(20, len(close)))
    avg_vol = prior.volume.mean()
    return {
        "Momentum": float(close.iloc[-1] / close.iloc[0] - 1),
        "Recent 20D": float(recent.iloc[-1] / recent.iloc[0] - 1),
        "Volatility": float(daily.std() * np.sqrt(252)) if len(daily) else 0.0,
        "Drawdown": float(close.iloc[-1] / close.max() - 1),
        "Slope": float(np.polyfit(np.arange(len(close)), close.values, 1)[0] / max(close.iloc[-1], 1e-9)),
        "Up-Day Rate": float((daily > 0).mean()) if len(daily) else .5,
        "Volume Ratio": float(prior.volume.tail(10).mean() / avg_vol) if avg_vol else 1.0,
    }


def historical_prediction(df, as_of, lookback=63, analogues=8, target_pct=8.0):
    current = market_state(df, as_of, lookback)
    if current is None:
        return None
    dates = df.index
    examples = []
    min_i, max_i = int(lookback) + 7, len(dates) - 63
    for i in range(min_i, max_i):
        state = market_state(df, dates[i], lookback)
        if state is None:
            continue
        future = df.iloc[i:i + 63]
        start_price = float(future.close.iloc[0])
        future_ret = float(future.close.iloc[-1] / start_price - 1)
        hits = np.where(future.high.values >= start_price * (1 + target_pct / 100))[0]
        days_to_target = int(hits[0] + 1) if len(hits) else None
        examples.append((state, future_ret, dates[i], days_to_target))
    if len(examples) < max(5, int(analogues)):
        return None
    states = np.array([x[0] for x in examples])
    scale = np.std(states, axis=0)
    scale[scale == 0] = 1.0
    distances = np.linalg.norm((states - current) / scale, axis=1)
    order = np.argsort(distances)[:min(int(analogues), len(examples))]
    nearest = [examples[i] for i in order]
    weights = np.array([1.0 / (distances[i] + .05) for i in order])
    returns = np.array([x[1] for x in nearest])
    prediction = float(np.average(returns, weights=weights))
    uncertainty = float(np.average(np.abs(returns - prediction), weights=weights))
    positive = float(np.average((returns > 0).astype(float), weights=weights))
    best, worst = float(returns.max()), float(returns.min())
    hit_days = [(x[3], weights[j]) for j, x in enumerate(nearest) if x[3] is not None]
    if hit_days:
        hold_days = int(round(np.average([x[0] for x in hit_days], weights=[x[1] for x in hit_days])))
    else:
        hold_days = 63
    return prediction, uncertainty, len(nearest), current, positive, best, worst, nearest, hold_days


def current_track(df, as_of, lookback, analogues, buy_drop, sell_target):
    details = fingerprint_details(df, as_of, lookback)
    pred = historical_prediction(df, as_of, lookback, analogues, sell_target)
    if details is None or pred is None:
        return None
    prediction, uncertainty, samples, state, positive, best, worst, nearest, hold_days = pred
    current_price = float(df.loc[df.index <= pd.Timestamp(as_of), "close"].iloc[-1])
    recent_high = float(df.loc[df.index <= pd.Timestamp(as_of), "close"].tail(60).max())
    buy_trigger = recent_high * (1 - buy_drop / 100)
    target_price = current_price * (1 + sell_target / 100)
    buy_now = current_price <= buy_trigger and prediction > 0 and positive >= .55
    if prediction > 0 and positive >= .60 and details["Drawdown"] > -.20:
        status = "🟢 ON TRACK"
    elif prediction > 0 and positive >= .45:
        status = "🟡 WATCH / DEVIATING"
    else:
        status = "🔴 PATTERN BROKEN"
    return {
        "price": current_price, "buy_trigger": buy_trigger, "target": target_price,
        "prediction": prediction, "uncertainty": uncertainty, "positive": positive,
        "best": best, "worst": worst, "samples": samples, "details": details,
        "status": status, "nearest": nearest, "hold_days": max(1, hold_days), "buy_now": buy_now,
    }


def next_business_date(start, days):
    date = pd.Timestamp(start)
    return (date + pd.offsets.BDay(int(days))).date()


def run_strategy(histories, windows, selections, capital, buy_drop, sell_target, top_n, allocation):
    total_start = total_end = float(capital)
    all_trades, quarters = [], []
    for qstart, qend in windows:
        selected = selections.get(str(qstart.date()), [])[:int(top_n)]
        if not selected:
            continue
        q_start, q_end_cap = total_end, total_end
        per_stock = q_start / max(1, int(top_n))
        qtr_trades = []
        for item in selected:
            ending, trades = simulate_quarter(histories[item["Ticker"]], qstart, qend, per_stock, buy_drop, sell_target, allocation)
            q_end_cap += ending - per_stock
            for trade in trades:
                trade["Ticker"] = item["Ticker"]
                qtr_trades.append(trade)
        total_end = q_end_cap
        all_trades.extend(qtr_trades)
        quarters.append({"Quarter": f"{qstart.date()} to {qend.date()}", "Predicted Stocks": ", ".join(x["Ticker"] for x in selected), "Start Capital": round(q_start, 2), "End Capital": round(q_end_cap, 2), "Quarter P/L": round(q_end_cap - q_start, 2), "Trades": len(qtr_trades)})
    pnl = total_end - total_start
    return {"Ending Capital": total_end, "Profit / Loss": pnl, "Return %": pnl / total_start * 100 if total_start else 0, "quarters": quarters, "trades": all_trades}


def simulate_quarter(df, start, end, capital, buy_drop, sell_target, allocation):
    data = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if len(data) < 2:
        return capital, []
    cash, position, trades = float(capital), None, []
    allocation = min(max(float(allocation) / 100, .01), 1.0)
    for i in range(1, len(data)):
        row = data.iloc[i]
        if position is None:
            prior = df[df.index < data.index[i]].tail(60)
            if len(prior) < 20:
                continue
            trigger = float(prior.close.max()) * (1 - buy_drop / 100)
            if float(row.close) <= trigger:
                qty = int((cash * allocation) // float(row.close))
                if qty:
                    position = {"entry": float(row.close), "qty": qty, "date": data.index[i].date()}
                    cash -= qty * position["entry"]
        else:
            target = position["entry"] * (1 + sell_target / 100)
            if float(row.high) >= target:
                cash += position["qty"] * target
                trades.append({"Buy Date": position["date"], "Sell Date": data.index[i].date(), "Shares": position["qty"], "Buy": round(position["entry"], 2), "Sell": round(target, 2), "P/L": round((target - position["entry"]) * position["qty"], 2), "Return %": sell_target, "Reason": f"+{sell_target:.1f}% target"})
                position = None
    if position is not None:
        last = float(data.close.iloc[-1])
        cash += position["qty"] * last
        trades.append({"Buy Date": position["date"], "Sell Date": data.index[-1].date(), "Shares": position["qty"], "Buy": round(position["entry"], 2), "Sell": round(last, 2), "P/L": round((last - position["entry"]) * position["qty"], 2), "Return %": round((last / position["entry"] - 1) * 100, 2), "Reason": "Quarter-end mark-to-market"})
    return cash, trades


def quarter_windows(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out, cursor = [], start
    while cursor < end:
        qend = min(cursor + pd.DateOffset(months=3), end)
        out.append((cursor, qend))
        cursor = qend
    return out


def build_predictions(histories, windows, lookback, analogues):
    selections, rows = {}, []
    for qstart, qend in windows:
        ranked = []
        for ticker, df in histories.items():
            pred = historical_prediction(df, qstart, lookback, analogues)
            if pred is None:
                continue
            p, u, n, _, pos, best, worst, _, _ = pred
            ranked.append({"Ticker": ticker, "Predicted Next Quarter %": p * 100, "Historical Uncertainty %": u * 100, "Historical Positive Rate %": pos * 100, "Best Analogue %": best * 100, "Worst Analogue %": worst * 100, "History Matches": n})
        ranked.sort(key=lambda x: (x["Predicted Next Quarter %"], x["Historical Positive Rate %"]), reverse=True)
        selections[str(qstart.date())] = ranked[:10]
        for rank, item in enumerate(ranked[:20], 1):
            row = dict(item); row["Rank"] = rank; row["Quarter"] = f"{qstart.date()} to {qend.date()}"; rows.append(row)
    return selections, pd.DataFrame(rows)


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT • TOP 10 DECISION ENGINE")
st.write("Fast first. Full discovery second. MAXPROFIT builds a fingerprint for each stock, finds recurring historical patterns, ranks the Top 10, estimates the hold period, and tracks whether the pattern stays on course. PAPER/SIMULATION ONLY.")

capital = st.number_input("Simulation starting money", min_value=100.0, value=1000.0, step=100.0)
test_days = st.number_input("Historical test length (days)", min_value=365, max_value=3650, value=730, step=30)

c1, c2 = st.columns(2)
with c1:
    quick = st.button("⚡ FIND TOP 10 NOW", type="primary", width="stretch")
with c2:
    full = st.button("⚔️ RUN FULL MAXPROFIT DISCOVERY", width="stretch")

if quick or full:
    if not alpaca_headers():
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()

    now = datetime.now(timezone.utc)
    end = now.date()
    requested_start = (now - timedelta(days=int(test_days))).date()
    data_start = (now - timedelta(days=int(test_days) + 1200)).date()

    st.subheader("📡 Market scan")
    histories = load_all_histories(data_start.isoformat(), end.isoformat())
    if not histories:
        st.error("No usable market history was returned.")
        st.stop()

    lookback, analogues = 63, 8
    buy_drop, sell_target, top_n, allocation = 15, 8, 10, 30

    if full:
        windows = quarter_windows(requested_start, end)
        if len(windows) < 3:
            st.error("Use at least one year of history.")
            st.stop()
        split = max(1, int(len(windows) * .70))
        train, validation = windows[:split], windows[split:]
        st.info("Running the deeper variable search. The Top 10 appears first in the quick scan below, then the optimized settings replace it when discovery finishes.")
        train_sel, _ = build_predictions(histories, train, 63, 8)
        stage1 = []
        for b, s in itertools.product(range(1, 21), range(1, 21)):
            r = run_strategy(histories, train, train_sel, capital, b, s, 10, 30)
            stage1.append((r["Return %"], b, s))
        _, buy_drop, sell_target = max(stage1, key=lambda x: x[0])
        stage2 = []
        for n, a in itertools.product([5, 10, 15], [10, 20, 30, 40, 50]):
            r = run_strategy(histories, train, train_sel, capital, buy_drop, sell_target, n, a)
            stage2.append((r["Return %"], n, a))
        _, top_n, allocation = max(stage2, key=lambda x: x[0])
        stage3 = []
        for lb, an in itertools.product([42, 63, 84], [4, 8, 12]):
            sel, _ = build_predictions(histories, train, lb, an)
            r = run_strategy(histories, train, sel, capital, buy_drop, sell_target, top_n, allocation)
            stage3.append((r["Return %"], lb, an, sel))
        _, lookback, analogues, _ = max(stage3, key=lambda x: x[0])
        val_sel, _ = build_predictions(histories, validation, lookback, analogues)
        val_result = run_strategy(histories, validation, val_sel, capital, buy_drop, sell_target, top_n, allocation)
        st.success(f"Discovery complete. BUY pullback: {buy_drop}%. SELL target: +{sell_target}%. Top N: {top_n}. Allocation: {allocation}%. Lookback: {lookback} days. Analogues: {analogues}. Unseen validation: {val_result['Return %']:.2f}%.")

    as_of = max(df.index.max() for df in histories.values())
    ranked = []
    for ticker, df in histories.items():
        track = current_track(df, as_of, lookback, analogues, buy_drop, sell_target)
        if track:
            ranked.append((ticker, track))
    ranked.sort(key=lambda x: (x[1]["prediction"], x[1]["positive"], -x[1]["uncertainty"]), reverse=True)
    ranked = ranked[:10]

    st.subheader("🔥 MAXPROFIT TOP 10 • WHAT TO BUY / HOW LONG TO HOLD")
    st.caption(f"Model date: {as_of.date()} • BUY rule: {buy_drop}% pullback • SELL rule: +{sell_target}% from actual purchase price • Historical model only.")

    if "paper_choices" not in st.session_state:
        st.session_state.paper_choices = {}

    for rank, (ticker, t) in enumerate(ranked, 1):
        sell_date = next_business_date(as_of, t["hold_days"])
        cols = st.columns([.45, .8, 1.1, 1.1, 1.1, 1.1, 1.3, 1.4])
        cols[0].markdown(f"### #{rank}")
        cols[1].markdown(f"### {ticker}")
        cols[2].metric("Predicted", f"{t['prediction']*100:.1f}%")
        cols[3].metric("Buy Trigger", f"${t['buy_trigger']:.2f}")
        cols[4].metric("Target", f"${t['target']:.2f}")
        cols[5].metric("Hold", f"~{t['hold_days']} trading days")
        cols[6].markdown(f"**Suggested sell:** {sell_date}\n\n{t['status']}")
        yes_key, no_key = f"yes_{ticker}", f"no_{ticker}"
        if cols[7].button("✅ BUY YES", key=yes_key, use_container_width=True):
            st.session_state.paper_choices[ticker] = "YES"
        if cols[7].button("❌ BUY NO", key=no_key, use_container_width=True):
            st.session_state.paper_choices[ticker] = "NO"
        choice = st.session_state.paper_choices.get(ticker, "UNDECIDED")
        st.write(f"**MAXPROFIT paper decision:** {choice}  |  Current ${t['price']:.2f}  |  Historical positive rate {t['positive']*100:.0f}%  |  Uncertainty ±{t['uncertainty']*100:.1f}%  |  Matches {t['samples']}")
        st.write(f"**Why it ranked:** 3M momentum {t['details']['Momentum']*100:.1f}%, recent 20D {t['details']['Recent 20D']*100:.1f}%, volatility {t['details']['Volatility']*100:.1f}%, drawdown {t['details']['Drawdown']*100:.1f}%, volume ratio {t['details']['Volume Ratio']:.2f}x. Best analogue {t['best']*100:.1f}%, worst {t['worst']*100:.1f}%.")
        st.divider()

    if ranked:
        n1_ticker, n1 = ranked[0]
        st.subheader(f"👑 #1: {n1_ticker}")
        st.write(f"MAXPROFIT's current #1 is **{n1_ticker}** because its fingerprint produced the strongest expected next-quarter result among the scanned universe, supported by {n1['samples']} historical matches and a {n1['positive']*100:.0f}% positive historical match rate. The estimated hold is about **{n1['hold_days']} trading days**, with a model target around **${n1['target']:.2f}**.")
        st.write("**Keep the trend on track:** 🟢 ON TRACK means the fingerprint remains consistent with the winning historical setup. 🟡 WATCH / DEVIATING means the pattern is moving away from it. 🔴 PATTERN BROKEN means MAXPROFIT should recalculate instead of blindly holding.")

    if full:
        st.subheader("🔬 Discovery results")
        st.write(f"Final settings: buy pullback {buy_drop}%, sell target +{sell_target}%, Top {top_n}, allocation {allocation}%, lookback {lookback} days, analogues {analogues}.")
        st.metric("Unseen validation return", f"{val_result['Return %']:.2f}%")

st.caption("PAPER/SIMULATION ONLY. The YES/NO buttons record paper decisions in this session. They do not place brokerage orders.")
