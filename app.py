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
    return prediction, uncertainty, len(nearest), current


def build_predictions(histories, windows, lookback, analogues):
    selections = {}
    prediction_rows = []
    for qstart, qend in windows:
        ranked = []
        for ticker, df in histories.items():
            pred = historical_prediction(df, qstart, lookback, analogues)
            if pred is None:
                continue
            prediction, uncertainty, samples, state = pred
            ranked.append({
                "Ticker": ticker,
                "Predicted Next Quarter %": prediction * 100,
                "Historical Uncertainty %": uncertainty * 100,
                "History Matches": samples,
                "3M Momentum %": state[0] * 100,
                "Volatility %": state[1] * 100,
                "Drawdown %": state[2] * 100,
            })
        ranked.sort(key=lambda x: x["Predicted Next Quarter %"], reverse=True)
        selected = ranked[:10]
        selections[str(qstart.date())] = selected
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
    cash = float(starting_capital)
    position = None
    trades = []
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
                trades.append({
                    "Buy Date": position["entry_date"], "Sell Date": data.index[i].date(),
                    "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
                    "Sell": round(exit_price, 2), "P/L": round(pnl, 2),
                    "Return %": round((exit_price / position["entry_price"] - 1) * 100, 2),
                    "Reason": f"+{float(profit_target_pct):.1f}% target"
                })
                position = None
    if position is not None:
        last_close = float(data["close"].iloc[-1])
        pnl = (last_close - position["entry_price"]) * position["qty"]
        cash += position["qty"] * last_close
        trades.append({
            "Buy Date": position["entry_date"], "Sell Date": data.index[-1].date(),
            "Shares": position["qty"], "Buy": round(position["entry_price"], 2),
            "Sell": round(last_close, 2), "P/L": round(pnl, 2),
            "Return %": round((last_close / position["entry_price"] - 1) * 100, 2),
            "Reason": "Quarter-end mark-to-market"
        })
    return cash, trades


def run_strategy(histories, windows, selections, starting_capital, buy_drop, sell_target, top_n, allocation_pct):
    total_start = float(starting_capital)
    total_end = total_start
    all_trades = []
    quarter_rows = []
    for qstart, qend in windows:
        ranked = selections.get(str(qstart.date()), [])
        selected = ranked[:int(top_n)]
        if not selected:
            continue
        q_start = total_end
        q_end_cap = q_start
        per_stock = q_start / max(1, int(top_n))
        qtr_trades = []
        for item in selected:
            ending, trades = simulate_quarter(
                histories[item["Ticker"]], qstart, qend,
                per_stock, buy_drop, sell_target, allocation_pct
            )
            q_end_cap += ending - per_stock
            for trade in trades:
                trade["Ticker"] = item["Ticker"]
                trade["Quarter"] = f"{qstart.date()} to {qend.date()}"
                qtr_trades.append(trade)
        total_end = q_end_cap
        all_trades.extend(qtr_trades)
        quarter_rows.append({
            "Quarter": f"{qstart.date()} to {qend.date()}",
            "Predicted Stocks": ", ".join(x["Ticker"] for x in selected),
            "Start Capital": round(q_start, 2),
            "End Capital": round(q_end_cap, 2),
            "Quarter P/L": round(q_end_cap - q_start, 2),
            "Trades": len(qtr_trades),
        })
    pnl = total_end - total_start
    ret = pnl / total_start * 100 if total_start else 0.0
    target_sales = [t for t in all_trades if "target" in t["Reason"]]
    wins = [t for t in target_sales if t["P/L"] > 0]
    return {
        "Buy Pullback %": buy_drop,
        "Sell Target %": sell_target,
        "Top N Stocks": int(top_n),
        "Position Allocation %": allocation_pct,
        "Ending Capital": total_end,
        "Profit / Loss": pnl,
        "Return %": ret,
        "Target Sales": len(target_sales),
        "Target Win Rate %": len(wins) / len(target_sales) * 100 if target_sales else 0.0,
        "All Trade Exits": len(all_trades),
        "trades": all_trades,
        "quarters": quarter_rows,
    }


def evaluate_combo(histories, windows, selections, capital, combo):
    return run_strategy(histories, windows, selections, capital, combo[0], combo[1], combo[2], combo[3])


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT AUTOMATIC VARIABLE-DISCOVERY ENGINE")
st.write("The machine searches the strategy variables for you, ranks the results, validates the winner on unseen history, and produces a current Top 10 stock watchlist from the same historical prediction engine. Historical simulation only. No live orders.")

col1, col2 = st.columns(2)
with col1:
    capital = st.number_input("Starting money", min_value=100.0, value=1000.0, step=100.0)
with col2:
    test_days = st.number_input("Historical test length (calendar days)", min_value=365, max_value=3650, value=730, step=30)

st.info("AUTO-DISCOVERY searches buy pullbacks 1%-20%, sell targets 1%-20%, Top-N stock counts 5/10/15, historical lookbacks 42/63/84 days, analogue counts 4/8/12, and position allocations 10%-50%. It uses staged optimization so MAXPROFIT can explore the variables without forcing your phone to calculate hundreds of thousands of redundant full simulations.")

if st.button("⚔️ RUN MAXPROFIT AUTO-DISCOVERY", type="primary", width="stretch"):
    if not alpaca_headers():
        st.error("Alpaca paper credentials are not available. Check Streamlit Secrets.")
        st.stop()

    end = datetime.now(timezone.utc).date()
    requested_start = (datetime.now(timezone.utc) - timedelta(days=int(test_days))).date()
    data_start = (datetime.now(timezone.utc) - timedelta(days=int(test_days) + 1200)).date()

    histories = {}
    progress = st.progress(0)
    status = st.empty()
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
    train_windows = windows[:split]
    validation_windows = windows[split:]

    status.write("Stage 1/4: testing buy/sell variables...")
    base_lookback = 63
    base_analogues = 8
    base_top_n = 10
    base_allocation = 30
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
        row["Lookback Days"] = lookback
        row["Analogue Count"] = analogues
        row["selections"] = selections
        stage3.append(row)
    stage3_df = pd.DataFrame([{k: v for k, v in r.items() if k != "selections"} for r in stage3]).sort_values(["Return %", "Profit / Loss"], ascending=False).reset_index(drop=True)
    best3 = max(stage3, key=lambda r: (r["Return %"], r["Profit / Loss"]))

    status.write("Stage 4/4: independent validation on unseen historical quarters...")
    final_buy = int(best3["Buy Pullback %"])
    final_sell = int(best3["Sell Target %"])
    final_top = int(best3["Top N Stocks"])
    final_alloc = int(best3["Position Allocation %"])
    final_lookback = int(best3["Lookback Days"])
    final_analogues = int(best3["Analogue Count"])

    validation_selections, validation_predictions = build_predictions(histories, validation_windows, final_lookback, final_analogues)
    validation_result = run_strategy(histories, validation_windows, validation_selections, capital, final_buy, final_sell, final_top, final_alloc)

    full_selections, full_predictions = build_predictions(histories, windows, final_lookback, final_analogues)
    full_result = run_strategy(histories, windows, full_selections, capital, final_buy, final_sell, final_top, final_alloc)

    # CURRENT Top 10: use the most recent completed quarter-start prediction state.
    latest_qstart, latest_qend = windows[-1]
    latest_ranked = []
    for ticker, df in histories.items():
        pred = historical_prediction(df, latest_qstart, final_lookback, final_analogues)
        if pred is None:
            continue
        prediction, uncertainty, samples, state = pred
        latest_ranked.append({
            "Rank": 0,
            "Ticker": ticker,
            "Predicted Next Quarter %": prediction * 100,
            "Historical Uncertainty %": uncertainty * 100,
            "History Matches": samples,
            "3M Momentum %": state[0] * 100,
            "Volatility %": state[1] * 100,
            "Drawdown %": state[2] * 100,
            "Suggested Buy Pullback %": final_buy,
            "Suggested Sell Target %": final_sell,
        })
    latest_ranked.sort(key=lambda x: x["Predicted Next Quarter %"], reverse=True)
    for rank, row in enumerate(latest_ranked[:10], 1):
        row["Rank"] = rank
    current_top10 = pd.DataFrame(latest_ranked[:10])

    status.success("MAXPROFIT discovery complete.")

    st.success("MAXPROFIT found a candidate strategy, validated it on unseen historical quarters, and generated the current Top 10 stocks from the same prediction engine.")

    st.subheader("🔥 MAXPROFIT CURRENT TOP 10 STOCKS")
    st.write("These are the 10 stocks MAXPROFIT currently ranks highest for the next simulated quarter, using the discovered historical-model settings. They are a research watchlist, not guaranteed winners or personalized financial advice.")
    if not current_top10.empty:
        st.dataframe(current_top10, width="stretch", hide_index=True)
        st.caption(f"Current ranking uses {final_lookback}-day historical states and {final_analogues} nearest historical analogues. The discovered simulated buy/sell settings are BUY -{final_buy}% from the recent high and SELL +{final_sell}% from the actual purchase price.")
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
        st.success("The candidate remained profitable on the unseen validation period. That does NOT prove future profitability, but it is a much stronger test than optimizing and judging on the same data.")
    else:
        st.warning("The candidate did not remain profitable on the unseen validation period. MAXPROFIT has exposed an overfit candidate rather than hiding it.")

    st.subheader("🏆 Final Candidate")
    st.dataframe(pd.DataFrame([{
        "Buy Pullback %": final_buy,
        "Sell Target %": final_sell,
        "Top N Stocks": final_top,
        "Position Allocation %": final_alloc,
        "Lookback Days": final_lookback,
        "Analogue Count": final_analogues,
        "Training Return %": best3["Return %"],
        "Validation Return %": validation_result["Return %"],
        "Validation P/L": validation_result["Profit / Loss"],
    }]), width="stretch")

    st.subheader("🔬 Stage 1: Buy/Sell Search")
    st.dataframe(stage1_df.head(25), width="stretch")
    st.subheader("🔬 Stage 2: Top-N + Allocation Search")
    st.dataframe(stage2_df.head(25), width="stretch")
    st.subheader("🔬 Stage 3: Lookback + Historical Analogue Search")
    st.dataframe(stage3_df.head(25), width="stretch")
    st.subheader("📊 Unseen Validation Quarters")
    st.dataframe(pd.DataFrame(validation_result["quarters"]), width="stretch")
    st.subheader("📈 Full Historical Run Using the Discovered Candidate")
    st.write(f"Full-period result: **{full_result['Return %']:.2f}%** from **${capital:,.2f}** to **${full_result['Ending Capital']:,.2f}**. This is descriptive historical simulation, not a forecast guarantee.")
    st.dataframe(pd.DataFrame(full_result["quarters"]), width="stretch")
    st.subheader("🤖 Predicted Stock Rankings")
    st.dataframe(full_predictions, width="stretch")
    st.subheader("💵 Candidate Trade Results")
    if full_result["trades"]:
        st.dataframe(pd.DataFrame(full_result["trades"]), width="stretch")
    else:
        st.info("No trades were triggered by the discovered candidate during the selected period.")
    st.caption("PAPER/SIMULATION ONLY. No live orders are placed by this application.")
