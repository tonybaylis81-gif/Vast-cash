import os
from collections.abc import Mapping
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
    params = {"timeframe": "1Day", "start": start_date, "end": end_date, "limit": 10000,
              "adjustment": "all", "feed": "iex", "sort": "asc"}
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


def market_state(df, as_of, lookback=63):
    prior = df[df.index < pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior) < max(30, int(lookback * 0.75)):
        return None
    close = prior["close"]
    ret = float(close.iloc[-1] / close.iloc[0] - 1)
    daily = close.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(252)) if len(daily) else 0.0
    high = float(close.max())
    drawdown = float(close.iloc[-1] / high - 1)
    slope = float(np.polyfit(np.arange(len(close)), close.values, 1)[0] / max(close.iloc[-1], 1e-9))
    return np.array([ret, vol, drawdown, slope], dtype=float)


def fingerprint_details(df, as_of, lookback=63):
    prior = df[df.index < pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior) < max(30, int(lookback * 0.75)):
        return None
    close = prior["close"]
    daily = close.pct_change().dropna()
    momentum = float(close.iloc[-1] / close.iloc[0] - 1)
    high = float(close.max())
    drawdown = float(close.iloc[-1] / high - 1)
    vol = float(daily.std() * np.sqrt(252)) if len(daily) else 0.0
    slope = float(np.polyfit(np.arange(len(close)), close.values, 1)[0] / max(close.iloc[-1], 1e-9))
    recent = close.tail(min(20, len(close)))
    recent_ret = float(recent.iloc[-1] / recent.iloc[0] - 1)
    up_days = float((daily > 0).mean()) if len(daily) else 0.5
    avg_volume = float(prior["volume"].mean()) if len(prior) else 0.0
    recent_volume = float(prior["volume"].tail(min(10, len(prior))).mean()) if len(prior) else 0.0
    volume_ratio = recent_volume / avg_volume if avg_volume else 1.0
    return {
        "Momentum": momentum, "Recent 20D": recent_ret, "Volatility": vol,
        "Drawdown": drawdown, "Slope": slope, "Up-Day Rate": up_days,
        "Volume Ratio": volume_ratio,
    }


def historical_prediction(df, quarter_start, lookback=63, analogues=8):
    current = market_state(df, quarter_start, lookback)
    if current is None:
        return None
    dates = df.index
    examples = []
    min_i = int(lookback) + 7
    max_i = len(dates) - 63
    for i in range(min_i, max_i):
        state = market_state(df, dates[i], lookback)
        if state is None:
            continue
        future = df.iloc[i:i + 63]
        if len(future) < 63:
            continue
        future_ret = float(future["close"].iloc[-1] / future["close"].iloc[0] - 1)
        examples.append((state, future_ret, dates[i]))
    if len(examples) < max(5, int(analogues)):
        return None
    states = np.array([x[0] for x in examples])
    scale = np.std(states, axis=0)
    scale[scale == 0] = 1.0
    distances = np.linalg.norm((states - current) / scale, axis=1)
    order = np.argsort(distances)[:min(int(analogues), len(examples))]
    nearest = [examples[i] for i in order]
    weights = np.array([1.0 / (distances[i] + 0.05) for i in order])
    returns = np.array([x[1] for x in nearest])
    prediction = float(np.average(returns, weights=weights))
    uncertainty = float(np.average(np.abs(returns - prediction), weights=weights))
    positive = float(np.average((returns > 0).astype(float), weights=weights))
    best = float(np.max(returns))
    worst = float(np.min(returns))
    return prediction, uncertainty, len(nearest), current, positive, best, worst, nearest


def build_predictions(histories, windows, lookback, analogues):
    selections, prediction_rows = {}, []
    for qstart, qend in windows:
        ranked = []
        for ticker, df in histories.items():
            pred = historical_prediction(df, qstart, lookback, analogues)
            if pred is None:
                continue
            prediction, uncertainty, samples, state, positive, best, worst, _ = pred
            ranked.append({
                "Ticker": ticker, "Predicted Next Quarter %": prediction * 100,
                "Historical Uncertainty %": uncertainty * 100, "Historical Positive Rate %": positive * 100,
                "Best Analogue %": best * 100, "Worst Analogue %": worst * 100,
                "History Matches": samples, "3M Momentum %": state[0] * 100,
                "Volatility %": state[1] * 100, "Drawdown %": state[2] * 100,
            })
        ranked.sort(key=lambda x: x["Predicted Next Quarter %"], reverse=True)
        selections[str(qstart.date())] = ranked[:10]
        for rank, item in enumerate(ranked[:20], 1):
            row = dict(item)
            row["Quarter"] = f"{qstart.date()} to {qend.date()}"
            row["Rank"] = rank
            prediction_rows.append(row)
    return selections, pd.DataFrame(prediction_rows)


def simulate_quarter(df, quarter_start, quarter_end, starting_capital, buy_drop_pct, profit_target_pct, allocation_pct):
    data = df[(df.index >= pd.Timestamp(quarter_start)) & (df.index <= pd.Timestamp(quarter_end))].copy()
    if len(data) < 2:
        return float(starting_capital), []
    cash, position, trades = float(starting_capital), None, []
    allocation = max(0.01, min(float(allocation_pct) / 100.0, 1.0))
    for i in range(1, len(data)):
        row = data.iloc[i]
        if position is None:
            prior = df[df.index < data.index[i]].tail(60)
            if len(prior) < 20:
                continue
            rolling_high = float(prior["close"].max())
            trigger = rolling_high * (1 - float(buy_drop_pct) / 100.0)
            if float(row["close"]) <= trigger:
                entry = float(row["close"])
                qty = int((cash * allocation) // entry)
                if qty >= 1:
                    position = {"entry_price": entry, "qty": qty, "entry_date": data.index[i].date()}
                    cash -= qty * entry
        else:
            target = position["entry_price"] * (1 + float(profit_target_pct) / 100.0)
            if float(row["high"]) >= target:
                exit_price = target
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                cash += position["qty"] * exit_price
                trades.append({"Buy Date": position["entry_date"], "Sell Date": data.index[i].date(),
                               "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
                               "Sell": round(exit_price, 2), "P/L": round(pnl, 2),
                               "Return %": round((exit_price / position["entry_price"] - 1) * 100, 2),
                               "Reason": f"+{float(profit_target_pct):.1f}% target"})
                position = None
    if position is not None:
        last_close = float(data["close"].iloc[-1])
        pnl = (last_close - position["entry_price"]) * position["qty"]
        cash += position["qty"] * last_close
        trades.append({"Buy Date": position["entry_date"], "Sell Date": data.index[-1].date(),
                       "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
                       "Sell": round(last_close, 2), "P/L": round(pnl, 2),
                       "Return %": round((last_close / position["entry_price"] - 1) * 100, 2),
                       "Reason": "Quarter-end mark-to-market"})
    return cash, trades


def run_strategy(histories, windows, selections, starting_capital, buy_drop, sell_target, top_n, allocation_pct):
    total_start, total_end = float(starting_capital), float(starting_capital)
    all_trades, quarter_rows = [], []
    for qstart, qend in windows:
        selected = selections.get(str(qstart.date()), [])[:int(top_n)]
        if not selected:
            continue
        q_start, q_end_cap = total_end, total_end
        per_stock = q_start / max(1, int(top_n))
        qtr_trades = []
        for item in selected:
            ending, trades = simulate_quarter(histories[item["Ticker"]], qstart, qend, per_stock, buy_drop, sell_target, allocation_pct)
            q_end_cap += ending - per_stock
            for trade in trades:
                trade["Ticker"] = item["Ticker"]
                trade["Quarter"] = f"{qstart.date()} to {qend.date()}"
                qtr_trades.append(trade)
        total_end = q_end_cap
        all_trades.extend(qtr_trades)
        quarter_rows.append({"Quarter": f"{qstart.date()} to {qend.date()}",
                             "Predicted Stocks": ", ".join(x["Ticker"] for x in selected),
                             "Start Capital": round(q_start, 2), "End Capital": round(q_end_cap, 2),
                             "Quarter P/L": round(q_end_cap - q_start, 2), "Trades": len(qtr_trades)})
    pnl = total_end - total_start
    ret = pnl / total_start * 100 if total_start else 0.0
    target_sales = [t for t in all_trades if "target" in t["Reason"]]
    wins = [t for t in target_sales if t["P/L"] > 0]
    return {"Buy Pullback %": buy_drop, "Sell Target %": sell_target, "Top N Stocks": int(top_n),
            "Position Allocation %": allocation_pct, "Ending Capital": total_end, "Profit / Loss": pnl,
            "Return %": ret, "Target Sales": len(target_sales),
            "Target Win Rate %": len(wins) / len(target_sales) * 100 if target_sales else 0.0,
            "All Trade Exits": len(all_trades), "trades": all_trades, "quarters": quarter_rows}


def evaluate_combo(histories, windows, selections, capital, combo):
    return run_strategy(histories, windows, selections, capital, combo[0], combo[1], combo[2], combo[3])


def current_track(df, as_of, lookback, analogues, buy_drop, sell_target):
    details = fingerprint_details(df, as_of, lookback)
    pred = historical_prediction(df, as_of, lookback, analogues)
    if details is None or pred is None:
        return None
    prediction, uncertainty, samples, state, positive, best, worst, nearest = pred
    current_price = float(df[df.index <= pd.Timestamp(as_of)]["close"].iloc[-1])
    recent_high = float(df[df.index < pd.Timestamp(as_of)]["close"].tail(60).max())
    buy_trigger = recent_high * (1 - buy_drop / 100.0)
    sell_target = current_price * (1 + sell_target / 100.0)
    if prediction > 0 and positive >= 0.60 and details["Drawdown"] > -0.20:
        status = "🟢 ON TRACK"
    elif prediction > 0 and positive >= 0.45:
        status = "🟡 WATCH / DEVIATING"
    else:
        status = "🔴 PATTERN BROKEN"
    return {"price": current_price, "buy_trigger": buy_trigger, "sell_target": sell_target,
            "prediction": prediction, "uncertainty": uncertainty, "positive": positive,
            "best": best, "worst": worst, "samples": samples, "details": details,
            "status": status, "nearest": nearest}


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT AUTOMATIC VARIABLE-DISCOVERY ENGINE")
st.write("MAXPROFIT searches strategy variables, validates the winner on unseen history, then builds a fingerprint for each stock. It looks for recurring historical states, explains why the leaders rank where they do, and tracks whether their current pattern remains on course. Historical simulation only. No live orders.")

col1, col2 = st.columns(2)
with col1:
    capital = st.number_input("Starting money", min_value=100.0, value=1000.0, step=100.0)
with col2:
    test_days = st.number_input("Historical test length (calendar days)", min_value=365, max_value=3650, value=730, step=30)

st.info("AUTO-DISCOVERY searches buy pullbacks 1%-20%, sell targets 1%-20%, Top-N stock counts 5/10/15, historical lookbacks 42/63/84 days, analogue counts 4/8/12, and position allocations 10%-50%. The search is staged to keep the calculation practical on a phone.")

if st.button("⚔️ RUN MAXPROFIT AUTO-DISCOVERY", type="primary", width="stretch"):
    if not alpaca_headers():
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()

    now = datetime.now(timezone.utc)
    end = now.date()
    requested_start = (now - timedelta(days=int(test_days))).date()
    data_start = (now - timedelta(days=int(test_days) + 1200)).date()

    histories, progress, status = {}, st.progress(0), st.empty()
    for n, ticker in enumerate(MARKET_UNIVERSE):
        status.write(f"Loading market history: {ticker} ({n + 1}/{len(MARKET_UNIVERSE)})")
        hist, err = load_history(ticker, data_start.isoformat(), end.isoformat())
        if hist is not None and len(hist) >= 180:
            histories[ticker] = hist
        progress.progress((n + 1) / len(MARKET_UNIVERSE))

    if not histories:
        st.error("No usable market history was returned.")
        st.stop()

    windows = quarter_windows(requested_start, end)
    if len(windows) < 3:
        st.error("Use a longer historical test period so MAXPROFIT has enough quarters for training and validation.")
        st.stop()

    split = max(1, int(len(windows) * 0.70))
    train_windows, validation_windows = windows[:split], windows[split:]

    status.write("Stage 1/4: testing buy/sell variables...")
    base_lookback, base_analogues, base_top_n, base_allocation = 63, 8, 10, 30
    train_selections, _ = build_predictions(histories, train_windows, base_lookback, base_analogues)
    stage1 = []
    for buy, sell in itertools.product(range(1, 21), range(1, 21)):
        result = evaluate_combo(histories, train_windows, train_selections, capital, (buy, sell, base_top_n, base_allocation))
        stage1.append({k: result[k] for k in ["Buy Pullback %", "Sell Target %", "Top N Stocks", "Position Allocation %", "Ending Capital", "Profit / Loss", "Return %", "Target Win Rate %", "All Trade Exits"]})
    stage1_df = pd.DataFrame(stage1).sort_values(["Return %", "Profit / Loss"], ascending=False).reset_index(drop=True)
    best1 = stage1_df.iloc[0]

    status.write("Stage 2/4: testing stock-count and position-allocation variables...")
    stage2 = []
    for top_n, allocation in itertools.product([5, 10, 15], [10, 20, 30, 40, 50]):
        result = evaluate_combo(histories, train_windows, train_selections, capital, (int(best1["Buy Pullback %"]), int(best1["Sell Target %"]), top_n, allocation))
        stage2.append({k: result[k] for k in ["Buy Pullback %", "Sell Target %", "Top N Stocks", "Position Allocation %", "Ending Capital", "Profit / Loss", "Return %", "Target Win Rate %", "All Trade Exits"]})
    stage2_df = pd.DataFrame(stage2).sort_values(["Return %", "Profit / Loss"], ascending=False).reset_index(drop=True)
    best2 = stage2_df.iloc[0]

    status.write("Stage 3/4: testing historical lookback and analogue-count variables...")
    stage3 = []
    for lookback, analogues in itertools.product([42, 63, 84], [4, 8, 12]):
        selections, _ = build_predictions(histories, train_windows, lookback, analogues)
        result = evaluate_combo(histories, train_windows, selections, capital, (int(best2["Buy Pullback %"]), int(best2["Sell Target %"]), int(best2["Top N Stocks"]), int(best2["Position Allocation %"])))
        row = {k: result[k] for k in ["Buy Pullback %", "Sell Target %", "Top N Stocks", "Position Allocation %", "Ending Capital", "Profit / Loss", "Return %", "Target Win Rate %", "All Trade Exits"]}
        row["Lookback Days"], row["Analogue Count"], row["selections"] = lookback, analogues, selections
        stage3.append(row)
    stage3_df = pd.DataFrame([{k: v for k, v in r.items() if k != "selections"} for r in stage3]).sort_values(["Return %", "Profit / Loss"], ascending=False).reset_index(drop=True)
    best3 = max(stage3, key=lambda r: (r["Return %"], r["Profit / Loss"]))

    status.write("Stage 4/4: independent validation on unseen historical quarters...")
    final_buy, final_sell = int(best3["Buy Pullback %"]), int(best3["Sell Target %"])
    final_top, final_alloc = int(best3["Top N Stocks"]), int(best3["Position Allocation %"])
    final_lookback, final_analogues = int(best3["Lookback Days"]), int(best3["Analogue Count"])

    validation_selections, validation_predictions = build_predictions(histories, validation_windows, final_lookback, final_analogues)
    validation_result = run_strategy(histories, validation_windows, validation_selections, capital, final_buy, final_sell, final_top, final_alloc)
    full_selections, full_predictions = build_predictions(histories, windows, final_lookback, final_analogues)
    full_result = run_strategy(histories, windows, full_selections, capital, final_buy, final_sell, final_top, final_alloc)

    # Current prediction is anchored to the latest available market date, not an old quarter-start.
    latest_date = max(df.index.max() for df in histories.values())
    latest_ranked = []
    for ticker, df in histories.items():
        as_of = min(latest_date, df.index.max())
        track = current_track(df, as_of, final_lookback, final_analogues, final_buy, final_sell)
        if track is None:
            continue
        latest_ranked.append({
            "Rank": 0, "Ticker": ticker, "Current Price": track["price"],
            "Predicted Next Quarter %": track["prediction"] * 100,
            "Historical Uncertainty %": track["uncertainty"] * 100,
            "Historical Positive Rate %": track["positive"] * 100,
            "Best Historical Match %": track["best"] * 100,
            "Worst Historical Match %": track["worst"] * 100,
            "History Matches": track["samples"],
            "Buy Trigger": track["buy_trigger"], "Sell Target": track["sell_target"],
            "Trend Status": track["status"], "3M Momentum %": track["details"]["Momentum"] * 100,
            "Recent 20D %": track["details"]["Recent 20D"] * 100,
            "Volatility %": track["details"]["Volatility"] * 100,
            "Drawdown %": track["details"]["Drawdown"] * 100,
            "Up-Day Rate %": track["details"]["Up-Day Rate"] * 100,
            "Volume Ratio": track["details"]["Volume Ratio"],
        })
    latest_ranked.sort(key=lambda x: (x["Predicted Next Quarter %"], x["Historical Positive Rate %"]), reverse=True)
    for rank, row in enumerate(latest_ranked[:10], 1):
        row["Rank"] = rank
    current_top10 = pd.DataFrame(latest_ranked[:10])

    status.success("MAXPROFIT discovery complete.")
    st.success("MAXPROFIT found the candidate strategy, tested it on unseen history, built stock fingerprints, and generated the current Top 10.")

    st.subheader("🔥 MAXPROFIT CURRENT TOP 10 STOCKS")
    st.write("These are the stocks whose current fingerprints most strongly point toward positive future behavior under MAXPROFIT's historical analogue model. The model also reports uncertainty and whether the pattern is currently on track. This is a research watchlist, not a guarantee or personalized financial advice.")
    if not current_top10.empty:
        st.dataframe(current_top10, width="stretch", hide_index=True)
        st.caption(f"Current fingerprint date: {latest_date.date()}. Model: {final_lookback}-day state + {final_analogues} historical analogues. Discovered simulated entry rule: BUY after a {final_buy}% pullback from the recent high. Discovered exit rule: SELL at +{final_sell}% from actual purchase price.")

        number_one = current_top10.iloc[0]
        st.subheader(f"👑 #1 MAXPROFIT CHOICE: {number_one['Ticker']}")
        st.write(f"**Why #1:** MAXPROFIT predicts **{number_one['Predicted Next Quarter %']:.2f}%** for the next quarter. Its historical analogue set was positive **{number_one['Historical Positive Rate %']:.1f}%** of the time, with a best matched outcome of **{number_one['Best Historical Match %']:.1f}%** and worst of **{number_one['Worst Historical Match %']:.1f}%**. The current fingerprint has **{int(number_one['History Matches'])}** close historical matches.")
        n1a, n1b, n1c, n1d = st.columns(4)
        n1a.metric("Current Price", f"${number_one['Current Price']:.2f}")
        n1b.metric("Buy Trigger", f"${number_one['Buy Trigger']:.2f}")
        n1c.metric("Sell Target", f"${number_one['Sell Target']:.2f}")
        n1d.metric("Trend", number_one["Trend Status"])
        st.write(f"**Fingerprint:** 3M momentum {number_one['3M Momentum %']:.2f}%, recent 20-day move {number_one['Recent 20D %']:.2f}%, volatility {number_one['Volatility %']:.2f}%, drawdown from the 60-day high {number_one['Drawdown %']:.2f}%, up-days {number_one['Up-Day Rate %']:.1f}%, volume ratio {number_one['Volume Ratio']:.2f}x.")
        st.write("**How to keep the trend on track:** MAXPROFIT should continue comparing the live fingerprint against the historical fingerprint that produced the prediction. 🟢 ON TRACK means the positive historical setup remains intact. 🟡 WATCH / DEVIATING means the state is moving away from the winning pattern. 🔴 PATTERN BROKEN means the historical setup no longer supports the original thesis and the model should recalculate rather than blindly hold.")

        st.subheader("🧬 #1 Historical Fingerprint Matches")
        n1_track = current_track(histories[number_one["Ticker"]], latest_date, final_lookback, final_analogues, final_buy, final_sell)
        if n1_track:
            match_rows = []
            for state, future_ret, match_date in n1_track["nearest"]:
                match_rows.append({"Historical Match Date": match_date.date(), "Next 63D Return %": future_ret * 100,
                                   "Momentum %": state[0] * 100, "Volatility %": state[1] * 100,
                                   "Drawdown %": state[2] * 100, "Slope": state[3]})
            st.dataframe(pd.DataFrame(match_rows), width="stretch", hide_index=True)
    else:
        st.warning("MAXPROFIT could not produce a current Top 10 from the available historical data.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Buy Pullback", f"{final_buy}%")
    m2.metric("Best Sell Target", f"{final_sell}%")
    m3.metric("Top-N Stocks", str(final_top))
    m4.metric("Position Allocation", f"{final_alloc}%")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Lookback", f"{final_lookback} days")
    m6.metric("Analogues", str(final_analogues))
    m7.metric("Training Return", f"{best3['Return %']:.2f}%")
    m8.metric("UNSEEN Validation Return", f"{validation_result['Return %']:.2f}%")

    if validation_result["Return %"] > 0:
        st.success("The candidate remained profitable on the unseen validation period. That does NOT prove future profitability, but it is a stronger test than judging on the same data used for optimization.")
    else:
        st.warning("The candidate did not remain profitable on the unseen validation period. MAXPROFIT has exposed a weak or overfit candidate rather than hiding it.")

    st.subheader("🏆 Final Candidate")
    st.dataframe(pd.DataFrame([{"Buy Pullback %": final_buy, "Sell Target %": final_sell, "Top N Stocks": final_top,
                               "Position Allocation %": final_alloc, "Lookback Days": final_lookback,
                               "Analogue Count": final_analogues, "Training Return %": best3["Return %"],
                               "Validation Return %": validation_result["Return %"], "Validation P/L": validation_result["Profit / Loss"]}]), width="stretch")
    st.subheader("🔬 Stage 1: Buy/Sell Search")
    st.dataframe(stage1_df.head(25), width="stretch")
    st.subheader("🔬 Stage 2: Top-N + Allocation Search")
    st.dataframe(stage2_df.head(25), width="stretch")
    st.subheader("🔬 Stage 3: Lookback + Historical Analogue Search")
    st.dataframe(stage3_df.head(25), width="stretch")
    st.subheader("📊 Unseen Validation Quarters")
    st.dataframe(pd.DataFrame(validation_result["quarters"]), width="stretch")
    st.subheader("📈 Full Historical Run Using the Discovered Candidate")
    st.write(f"Full-period result: **{full_result['Return %']:.2f}%** from **${capital:,.2f}** to **${full_result['Ending Capital']:,.2f}**. Descriptive historical simulation only.")
    st.dataframe(pd.DataFrame(full_result["quarters"]), width="stretch")
    st.subheader("🤖 Predicted Stock Rankings")
    st.dataframe(full_predictions, width="stretch")
    st.subheader("💵 Candidate Trade Results")
    if full_result["trades"]:
        st.dataframe(pd.DataFrame(full_result["trades"]), width="stretch")
    else:
        st.info("No trades were triggered by the discovered candidate during the selected period.")
    st.caption("PAPER/SIMULATION ONLY. No live orders are placed by this application.")
