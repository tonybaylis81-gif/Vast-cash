import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True


def make_demo_market(seed=42, periods=140):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0007, 0.018, periods)
    price = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(800_000, 3_000_000, periods)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=periods, freq="B")
    return pd.DataFrame({"close": price, "volume": volume}, index=idx)


def calculate_indicators(df):
    close = df["close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    momentum20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    volatility = close.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252) * 100
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]
    volume_ratio = df["volume"].iloc[-1] / avg_volume if avg_volume else 0
    return {"price": float(close.iloc[-1]), "sma20": float(sma20), "sma50": float(sma50),
            "momentum20": float(momentum20), "volatility": float(volatility),
            "volume_ratio": float(volume_ratio)}


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
    # Demo data keeps this build safe until IBKR paper market data is connected.
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

stocks = [s.strip().upper() for s in stock_text.replace(",", "\n").splitlines() if s.strip()]
stocks = list(dict.fromkeys(stocks))[:10]
if not stocks:
    stocks = ["DEMO"]

# Rank all 1–10 candidates, then let the user select the candidate the helper recommends.
rows = []
all_results = {}
for i, ticker in enumerate(stocks):
    ind, algo, risk, ai = evaluate_symbol(ticker, int(seed) + i)
    all_results[ticker] = (ind, algo, risk, ai)
    rows.append({"Rank": 0, "Ticker": ticker, "MAXPROFIT": algo["signal"],
                 "Score": round(algo["score"], 1), "AI": ai["action"],
                 "Confidence": round(ai["confidence"], 1),
                 "Risk": "BLOCK" if risk["blocked"] else "PASS"})
ranking = pd.DataFrame(rows).sort_values(["Score", "Confidence"], ascending=False).reset_index(drop=True)
ranking["Rank"] = range(1, len(ranking) + 1)

st.header("🎯 AI STOCK SELECTOR")
st.caption("VAST CASH evaluates your 1–10 stock universe, ranks the candidates, and identifies the strongest paper-trading candidate.")
st.dataframe(ranking, use_container_width=True, hide_index=True)

suggested = ranking.iloc[0]["Ticker"]
if ranking.iloc[0]["AI"] == "BLOCK":
    st.error(f"AI Helper: no trade recommended. Top candidate {suggested} is blocked by the Risk Gate.")
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
st.header("📊 Selected Stock Evidence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${ind['price']:.2f}")
m2.metric("20D Momentum", f"{ind['momentum20']:.2f}%")
m3.metric("Volatility", f"{ind['volatility']:.1f}%")
m4.metric("Volume Ratio", f"{ind['volume_ratio']:.2f}x")

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
st.caption("VAST CASH • MAXPROFIT + AI HELPER • Paper Validation Build 3.0")
