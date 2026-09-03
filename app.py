import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True


def make_demo_market(seed=42, periods=500):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.018, periods)
    price = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(800_000, 3_000_000, periods)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=periods, freq="B")
    return pd.DataFrame({"open": price, "close": price, "volume": volume}, index=idx)


def load_history(symbol, period="1y"):
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = ["Open", "Close", "Volume"]
        if df.empty or not all(c in df.columns for c in needed):
            return None, "No usable historical data returned."
        df = df[needed].copy().dropna()
        df.columns = ["open", "close", "volume"]
        if len(df) < 60:
            return None, "Not enough historical bars for the 50-day trend."
        return df, None
    except Exception as exc:
        return None, f"Historical data error: {exc}"


def calculate_indicators(df, end=None):
    data = df if end is None else df.iloc[:end]
    close = data["close"]
    if len(close) < 51:
        return None
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    momentum20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    volatility = close.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252) * 100
    avg_volume = data["volume"].rolling(20).mean().iloc[-1]
    volume_ratio = data["volume"].iloc[-1] / avg_volume if avg_volume else 0
    return {
        "price": float(close.iloc[-1]), "sma20": float(sma20), "sma50": float(sma50),
        "momentum20": float(momentum20), "volatility": float(volatility),
        "volume_ratio": float(volume_ratio)
    }


def maxprofit_signal(ind):
    score = 50.0
    reasons = []
    if ind["price"] > ind["sma20"]:
        score += 15; reasons.append("Price is above the 20-day average.")
    else:
        score -= 15; reasons.append("Price is below the 20-day average.")
    if ind["sma20"] > ind["sma50"]:
        score += 15; reasons.append("20-day trend is above the 50-day trend.")
    else:
        score -= 15; reasons.append("20-day trend is below the 50-day trend.")
    if ind["momentum20"] > 3:
        score += 15; reasons.append("20-day momentum is positive.")
    elif ind["momentum20"] < -3:
        score -= 15; reasons.append("20-day momentum is negative.")
    else:
        reasons.append("20-day momentum is neutral.")
    if ind["volume_ratio"] >= 1.1:
        score += 5; reasons.append("Volume confirms the move.")
    else:
        reasons.append("Volume confirmation is weak.")
    if ind["volatility"] > 55:
        score -= 15; reasons.append("Volatility is elevated.")
    elif ind["volatility"] < 18:
        score += 5; reasons.append("Volatility is relatively contained.")
    score = max(0, min(100, score))
    signal = "BUY" if score >= 70 else "SELL" if score <= 35 else "HOLD"
    return {"signal": signal, "score": score, "reasons": reasons}


def risk_gate(ind, algo, max_risk_pct, max_volatility):
    reasons = []
    if ind["volatility"] > max_volatility:
        reasons.append(f"Volatility {ind['volatility']:.1f}% exceeds {max_volatility:.1f}% limit.")
    if algo["score"] < 55:
        reasons.append("Algorithm score is below the minimum execution threshold.")
    if not 0 < max_risk_pct <= 2:
        reasons.append("Risk per trade must be between 0 and 2%.")
    return {"blocked": bool(reasons), "reason": " ".join(reasons) if reasons else "All hard paper-trading checks passed."}


def ai_helper(ind, algo, risk):
    warnings = []
    confirmations = []
    if ind["price"] > ind["sma20"]: confirmations.append("Price is aligned with the short-term trend.")
    else: warnings.append("Price is below the short-term trend.")
    if ind["sma20"] > ind["sma50"]: confirmations.append("20-day trend is above the 50-day trend.")
    else: warnings.append("Short-term trend is below the 50-day trend.")
    if ind["volatility"] > 55: warnings.append("Volatility is elevated.")
    if ind["volume_ratio"] < .75: warnings.append("Volume confirmation is weak.")
    confidence = max(0, min(100, algo["score"] - min(20, len(warnings) * 8)))
    if risk["blocked"]: action = "BLOCK"
    elif algo["signal"] in ("BUY", "SELL") and confidence >= 65: action = algo["signal"]
    else: action = "HOLD"
    return {"action": action, "confidence": round(confidence, 1), "warnings": warnings,
            "confirmations": confirmations}


def evaluate_symbol(symbol, seed):
    ind = calculate_indicators(make_demo_market(seed + sum(map(ord, symbol))))
    algo = maxprofit_signal(ind)
    risk = risk_gate(ind, algo, max_risk_pct, max_volatility)
    ai = ai_helper(ind, algo, risk)
    return ind, algo, risk, ai


def decision_record(symbol, selected_action, ind, algo, ai, risk):
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": symbol, "price": round(ind["price"], 2),
            "algorithm": algo["signal"], "algo_score": round(algo["score"], 1),
            "AI_recommendation": ai["action"], "AI_confidence": ai["confidence"],
            "user_action": selected_action, "risk_gate": "BLOCKED" if risk["blocked"] else "PASS",
            "paper_only": True}


def simulate_strategy(histories, starting_capital, allocation_pct, max_hold_days, max_risk_pct, max_volatility):
    """Historical walk-forward simulator. Signals use only data available up to that day.
    Orders execute at the following trading day's open, preventing look-ahead bias.
    One position at a time keeps the first validation version easy to audit.
    """
    dates = sorted(set().union(*[set(df.index) for df in histories.values()]))
    cash = float(starting_capital)
    position = None
    trades = []
    equity_curve = []
    blocked_count = 0

    for i, date in enumerate(dates):
        todays = {}
        for symbol, df in histories.items():
            if date not in df.index:
                continue
            end = df.index.get_loc(date) + 1
            ind = calculate_indicators(df, end)
            if ind is None:
                continue
            algo = maxprofit_signal(ind)
            risk = risk_gate(ind, algo, max_risk_pct, max_volatility)
            ai = ai_helper(ind, algo, risk)
            todays[symbol] = (df.loc[date], ind, algo, risk, ai)
            if risk["blocked"]:
                blocked_count += 1

        # Exit is evaluated before a new entry. A SELL signal exits at next day's open.
        if position and position["symbol"] in todays:
            _, ind, _, risk, ai = todays[position["symbol"]]
            held = i - position["entry_index"]
            timed_exit = held >= max_hold_days
            signal_exit = ai["action"] == "SELL"
            if timed_exit or signal_exit:
                exit_date_idx = i + 1
                if exit_date_idx < len(dates):
                    next_date = dates[exit_date_idx]
                    df = histories[position["symbol"]]
                    if next_date in df.index:
                        exit_price = float(df.loc[next_date, "open"])
                        proceeds = position["shares"] * exit_price
                        pnl = proceeds - position["cost"]
                        cash += proceeds
                        trades.append({
                            "Entry": position["entry_date"], "Exit": next_date,
                            "Ticker": position["symbol"], "Buy": round(position["entry_price"], 2),
                            "Sell": round(exit_price, 2), "Days": held,
                            "P&L": round(pnl, 2), "Return %": round(pnl / position["cost"] * 100, 2),
                            "Exit Reason": "MAX HOLD" if timed_exit else "AI SELL"
                        })
                        position = None

        # If flat, choose the highest-ranked eligible BUY signal and enter next day.
        if position is None and i + 1 < len(dates):
            candidates = []
            for symbol, (_, ind, algo, risk, ai) in todays.items():
                if not risk["blocked"] and ai["action"] == "BUY":
                    candidates.append((algo["score"], ai["confidence"], symbol))
            if candidates:
                _, _, symbol = max(candidates)
                next_date = dates[i + 1]
                df = histories[symbol]
                if next_date in df.index:
                    entry_price = float(df.loc[next_date, "open"])
                    allocation = min(cash, cash * allocation_pct / 100)
                    shares = int(allocation / entry_price)
                    if shares > 0:
                        cost = shares * entry_price
                        cash -= cost
                        position = {
                            "symbol": symbol, "entry_date": next_date,
                            "entry_price": entry_price, "shares": shares,
                            "cost": cost, "entry_index": i + 1
                        }

        # Mark-to-market equity at today's close.
        equity = cash
        if position and position["symbol"] in histories and date in histories[position["symbol"]].index:
            equity += position["shares"] * float(histories[position["symbol"]].loc[date, "close"])
        equity_curve.append({"Date": date, "Equity": equity})

    # Close any remaining position at the final available close.
    if position:
        df = histories[position["symbol"]]
        last_date = df.index[-1]
        exit_price = float(df.loc[last_date, "close"])
        proceeds = position["shares"] * exit_price
        pnl = proceeds - position["cost"]
        cash += proceeds
        held = max(0, len(dates) - 1 - position["entry_index"])
        trades.append({
            "Entry": position["entry_date"], "Exit": last_date,
            "Ticker": position["symbol"], "Buy": round(position["entry_price"], 2),
            "Sell": round(exit_price, 2), "Days": held,
            "P&L": round(pnl, 2), "Return %": round(pnl / position["cost"] * 100, 2),
            "Exit Reason": "END OF TEST"
        })
        position = None

    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        return None
    curve["Peak"] = curve["Equity"].cummax()
    curve["Drawdown %"] = (curve["Equity"] / curve["Peak"] - 1) * 100
    trade_df = pd.DataFrame(trades)
    ending = float(cash)
    net = ending - starting_capital
    win_rate = float((trade_df["P&L"] > 0).mean() * 100) if not trade_df.empty else 0.0
    max_dd = float(curve["Drawdown %"].min())
    return {
        "starting": starting_capital, "ending": ending, "net": net,
        "return_pct": net / starting_capital * 100, "trades": trade_df,
        "curve": curve, "win_rate": win_rate, "max_dd": max_dd,
        "blocked": blocked_count
    }


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT Engine • AI Helper • Paper Trading")
st.warning("🛡️ PAPER MONEY ONLY. BUY / SELL / HOLD below records simulated decisions only. No live orders are possible.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("System", "ONLINE")
c2.metric("Broker", "IBKR • PENDING")
c3.metric("Mode", "PAPER")
c4.metric("Live Orders", "DISABLED")

with st.sidebar:
    st.header("⚙️ Portfolio Setup")
    stock_text = st.text_area("Stock universe (1–10 tickers)", "AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nAMD\nAVGO\nJPM", height=220)
    seed = st.number_input("Paper-test seed", 1, 100000, 42)
    max_risk_pct = st.slider("Max risk / trade (%)", .1, 2.0, 1.0, .1)
    max_volatility = st.slider("Max volatility (%)", 20.0, 100.0, 55.0, 1.0)
    st.divider()
    st.header("🧪 Backtest Settings")
    sim_period = st.selectbox("Historical test period", ["6mo", "1y", "2y", "3y", "5y"], index=1)
    starting_capital = st.number_input("Starting paper capital ($)", 100.0, 1_000_000.0, 1000.0, 100.0)
    allocation_pct = st.slider("Capital allocated per trade (%)", 5.0, 100.0, 50.0, 5.0)
    max_hold_days = st.number_input("Automatic maximum hold (trading days)", 1, 252, 6, 1)

stocks = [s.strip().upper() for s in stock_text.replace(",", "\n").splitlines() if s.strip()]
stocks = list(dict.fromkeys(stocks))[:10]
if not stocks:
    stocks = ["DEMO"]

# Live-data scan for the dashboard. Falls back to the safe deterministic demo model if unavailable.
rows = []
all_results = {}
for i, ticker in enumerate(stocks):
    hist, _ = load_history(ticker, "6mo")
    if hist is not None:
        ind = calculate_indicators(hist)
    else:
        ind = calculate_indicators(make_demo_market(int(seed) + i + sum(map(ord, ticker))))
    algo = maxprofit_signal(ind)
    risk = risk_gate(ind, algo, max_risk_pct, max_volatility)
    ai = ai_helper(ind, algo, risk)
    all_results[ticker] = (ind, algo, risk, ai)
    rows.append({"Rank": 0, "Ticker": ticker, "MAXPROFIT": algo["signal"],
                 "Score": round(algo["score"], 1), "AI": ai["action"],
                 "Confidence": round(ai["confidence"], 1),
                 "Risk": "BLOCK" if risk["blocked"] else "PASS"})
ranking = pd.DataFrame(rows).sort_values(["Score", "Confidence"], ascending=False).reset_index(drop=True)
ranking["Rank"] = range(1, len(ranking) + 1)

st.header("🎯 AI STOCK SELECTOR")
st.caption("VAST CASH evaluates your 1–10 stock universe and ranks the candidates for paper trading.")
st.dataframe(ranking, use_container_width=True, hide_index=True)

suggested = ranking.iloc[0]["Ticker"]
if ranking.iloc[0]["Risk"] == "BLOCK":
    st.error(f"AI Helper: top candidate {suggested} is blocked by the Risk Gate.")
else:
    st.success(f"🤖 AI Helper's current top candidate: **{suggested}** • {ranking.iloc[0]['AI']} • {ranking.iloc[0]['Confidence']:.0f}% confidence")

selected = st.selectbox("👆 Select the stock to review / paper trade", ranking["Ticker"].tolist(), index=0)
ind, algo, risk, ai = all_results[selected]

st.divider()
st.header(f"🤖 AI HELPER • {selected}")
a, b, c, d = st.columns(4)
a.metric("MAXPROFIT", algo["signal"])
b.metric("Score", f"{algo['score']:.0f}/100")
c.metric("AI Confidence", f"{ai['confidence']:.0f}/100")
d.metric("AI Recommendation", ai["action"])

left, right = st.columns(2)
with left:
    st.write("**Why the helper likes/dislikes it**")
    for x in ai["confirmations"] or ["No positive confirmations."]: st.write(f"✓ {x}")
    for x in ai["warnings"] or ["No material warnings."]: st.write(f"⚠️ {x}")
with right:
    if risk["blocked"]: st.error(f"🛡️ RISK GATE: BLOCKED\n\n{risk['reason']}")
    else: st.success(f"🛡️ RISK GATE: PASS\n\n{risk['reason']}")

st.divider()
st.header("🎮 PAPER TRADE CONTROLS")
st.write(f"**Selected stock:** `{selected}`  |  **Paper price:** `${ind['price']:.2f}`")

b1, b2, b3 = st.columns(3)
with b1:
    buy = st.button("🟢 BUY", use_container_width=True, disabled=risk["blocked"])
with b2:
    sell = st.button("🔴 SELL", use_container_width=True, disabled=risk["blocked"])
with b3:
    hold = st.button("🟡 HOLD", use_container_width=True)

if "decision_log" not in st.session_state: st.session_state.decision_log = []
if buy:
    st.session_state.decision_log.insert(0, decision_record(selected, "BUY", ind, algo, ai, risk))
    st.success(f"PAPER BUY recorded for {selected}. No real order was sent.")
elif sell:
    st.session_state.decision_log.insert(0, decision_record(selected, "SELL", ind, algo, ai, risk))
    st.success(f"PAPER SELL recorded for {selected}. No real order was sent.")
elif hold:
    st.session_state.decision_log.insert(0, decision_record(selected, "HOLD", ind, algo, ai, risk))
    st.info(f"PAPER HOLD recorded for {selected}.")

st.divider()
st.header("🚀 RUN SIMULATION")
st.write("This is the fast answer to 'would this strategy have made money?' It walks forward through historical daily data, automatically buys eligible signals, holds the position, and sells on an AI SELL signal or the maximum hold period.")
st.info(f"Simulation is PAPER ONLY. Current settings: **${starting_capital:,.0f} start • {allocation_pct:.0f}% allocation/trade • {max_hold_days} trading-day max hold • {sim_period} history**.")

if st.button("▶️ RUN SIMULATION", type="primary", use_container_width=True):
    with st.spinner("Downloading historical data and running the walk-forward simulation..."):
        histories = {}
        errors = []
        for ticker in stocks:
            hist, err = load_history(ticker, sim_period)
            if hist is not None:
                histories[ticker] = hist
            else:
                errors.append(f"{ticker}: {err}")
        if histories:
            result = simulate_strategy(histories, float(starting_capital), float(allocation_pct),
                                       int(max_hold_days), float(max_risk_pct), float(max_volatility))
            st.session_state.sim_result = result
            st.session_state.sim_period = sim_period
            st.session_state.sim_unavailable = errors
        else:
            st.session_state.sim_result = None
            st.session_state.sim_unavailable = errors

if "sim_result" in st.session_state and st.session_state.sim_result is not None:
    result = st.session_state.sim_result
    st.subheader("📈 Simulation Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Starting Capital", f"${result['starting']:,.2f}")
    r2.metric("Ending Capital", f"${result['ending']:,.2f}", f"${result['net']:,.2f}")
    r3.metric("Total Return", f"{result['return_pct']:.2f}%")
    r4.metric("Max Drawdown", f"{result['max_dd']:.2f}%")
    r5, r6, r7 = st.columns(3)
    r5.metric("Completed Trades", len(result["trades"]))
    r6.metric("Win Rate", f"{result['win_rate']:.1f}%")
    r7.metric("Risk-Gated Signals", result["blocked"])
    if result["net"] > 0:
        st.success(f"Simulation finished profitable: **+${result['net']:,.2f}** over the tested period. This is historical simulation, not a promise of future profit.")
    elif result["net"] < 0:
        st.error(f"Simulation finished negative: **${result['net']:,.2f}** over the tested period. That is useful information for tuning the strategy.")
    else:
        st.info("Simulation finished approximately flat.")

    curve = result["curve"].set_index("Date")
    st.line_chart(curve[["Equity"]])
    if not result["trades"].empty:
        st.subheader("📋 Automatic BUY → HOLD → SELL Ledger")
        st.dataframe(result["trades"], use_container_width=True, hide_index=True)
    else:
        st.warning("No completed trades occurred in this historical window with the current settings. Try a longer period, different universe, or less restrictive risk settings.")
    st.caption(f"Walk-forward test used {st.session_state.get('sim_period', 'selected')} historical data. Entries execute on the next available day's open after a signal, and open positions are closed at the end of the test.")

if "sim_unavailable" in st.session_state and st.session_state.sim_unavailable:
    with st.expander("Historical data warnings"):
        for err in st.session_state.sim_unavailable:
            st.write(f"⚠️ {err}")

st.divider()
st.header("📊 Selected Stock Evidence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${ind['price']:.2f}")
m2.metric("20D Momentum", f"{ind['momentum20']:.2f}%")
m3.metric("Volatility", f"{ind['volatility']:.1f}%")
m4.metric("Volume Ratio", f"{ind['volume_ratio']:.2f}x")

chart, chart_err = load_history(selected, "6mo")
if chart is None:
    chart = make_demo_market(int(seed) + sum(map(ord, selected)))
chart["SMA20"] = chart["close"].rolling(20).mean()
chart["SMA50"] = chart["close"].rolling(50).mean()
st.line_chart(chart[["close", "SMA20", "SMA50"]])

st.header("🧪 Decision Replay / Audit Log")
if st.session_state.decision_log:
    st.dataframe(pd.DataFrame(st.session_state.decision_log), use_container_width=True, hide_index=True)
else:
    st.info("No paper decisions recorded yet.")

st.divider()
st.header("🔌 IBKR Connection")
st.info("IBKR remains intentionally disconnected. When the IBKR paper account is ready, this order-control layer will be connected to paper execution only.")
st.caption("VAST CASH • MAXPROFIT + AI HELPER • Paper Validation Build 4.0 • Historical Simulation")
