import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")

# ============================================================
# VAST CASH | MAXPROFIT + AI HELPER
# PAPER TRADING ONLY
# ============================================================

PAPER_ONLY = True


def make_demo_market(seed: int = 42, periods: int = 140) -> pd.DataFrame:
    """Generate deterministic market data for safe local/paper testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0007, 0.018, periods)
    price = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(800_000, 3_000_000, periods)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=periods, freq="B")
    return pd.DataFrame({"close": price, "volume": volume}, index=idx)


def calculate_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    momentum20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    volatility = close.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252) * 100
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]
    volume_ratio = df["volume"].iloc[-1] / avg_volume if avg_volume else 0
    return {
        "price": float(close.iloc[-1]),
        "sma20": float(sma20),
        "sma50": float(sma50),
        "momentum20": float(momentum20),
        "volatility": float(volatility),
        "volume_ratio": float(volume_ratio),
    }


def maxprofit_signal(ind: dict) -> dict:
    """Transparent baseline scoring engine. No orders are placed here."""
    score = 50.0
    reasons = []

    if ind["price"] > ind["sma20"]:
        score += 15
        reasons.append("Price is above the 20-day average.")
    else:
        score -= 15
        reasons.append("Price is below the 20-day average.")

    if ind["sma20"] > ind["sma50"]:
        score += 15
        reasons.append("Short-term trend is above the 50-day trend.")
    else:
        score -= 15
        reasons.append("Short-term trend is below the 50-day trend.")

    if ind["momentum20"] > 3:
        score += 15
        reasons.append("20-day momentum is positive.")
    elif ind["momentum20"] < -3:
        score -= 15
        reasons.append("20-day momentum is negative.")
    else:
        reasons.append("20-day momentum is neutral.")

    if ind["volume_ratio"] >= 1.1:
        score += 5
        reasons.append("Volume confirms the move.")
    else:
        reasons.append("Volume confirmation is weak.")

    if ind["volatility"] > 55:
        score -= 15
        reasons.append("Volatility is elevated.")
    elif ind["volatility"] < 18:
        score += 5
        reasons.append("Volatility is relatively contained.")

    score = max(0, min(100, score))
    if score >= 70:
        signal = "BUY"
    elif score <= 35:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"signal": signal, "score": score, "reasons": reasons}


def ai_helper_assessment(ind: dict, algo: dict, risk: dict) -> dict:
    """Structured AI-helper layer. It evaluates the algorithm; it does not place orders."""
    warnings = []
    confirmations = []

    if ind["price"] > ind["sma20"]:
        confirmations.append("Price/trend alignment is positive.")
    else:
        warnings.append("Price is below the short-term trend.")

    if ind["sma20"] > ind["sma50"]:
        confirmations.append("20-day trend is above the 50-day trend.")
    else:
        warnings.append("Short-term trend is below the 50-day trend.")

    if ind["volatility"] > 55:
        warnings.append("Volatility exceeds the configured safety threshold.")
    if ind["volume_ratio"] < 0.75:
        warnings.append("Current volume is unusually weak.")

    confidence = algo["score"]
    confidence -= min(20, len(warnings) * 8)
    confidence = max(0, min(100, confidence))

    if risk["blocked"]:
        recommendation = "BLOCK"
    elif algo["signal"] == "BUY" and confidence >= 65:
        recommendation = "TRADE"
    elif algo["signal"] == "SELL" and confidence >= 65:
        recommendation = "TRADE"
    else:
        recommendation = "HOLD"

    if recommendation == "TRADE":
        reason = "Algorithm and AI assessment are aligned and the hard risk gate has passed."
    elif recommendation == "BLOCK":
        reason = risk["reason"]
    else:
        reason = "The signal is not strong enough to justify a paper trade yet."

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 1),
        "warnings": warnings,
        "confirmations": confirmations,
        "reason": reason,
    }


def risk_gate(ind: dict, algo: dict, max_risk_pct: float, max_volatility: float) -> dict:
    """Hard safety layer. This layer can block but never improve a signal."""
    reasons = []
    if ind["volatility"] > max_volatility:
        reasons.append(f"Volatility {ind['volatility']:.1f}% exceeds {max_volatility:.1f}% limit.")
    if algo["score"] < 55:
        reasons.append("Algorithm score is below the minimum execution threshold.")
    if max_risk_pct <= 0 or max_risk_pct > 2:
        reasons.append("Configured risk per trade must be between 0 and 2%.")

    return {
        "blocked": bool(reasons),
        "reason": " ".join(reasons) if reasons else "All hard paper-trading risk checks passed.",
        "risk_pct": max_risk_pct,
    }


def log_decision(symbol: str, ind: dict, algo: dict, ai: dict, risk: dict) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol,
        "price": round(ind["price"], 2),
        "algorithm": algo["signal"],
        "algo_score": round(algo["score"], 1),
        "ai_action": ai["recommendation"],
        "ai_confidence": ai["confidence"],
        "risk_gate": "BLOCKED" if risk["blocked"] else "PASS",
        "paper_only": True,
    }


# ----------------------------- UI -----------------------------
st.title("💰 VAST CASH")
st.subheader("MAXPROFIT Engine • AI Helper")

st.warning("🛡️ PAPER TRADING ONLY — no live orders can be submitted by this build.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("System", "ONLINE")
col2.metric("Broker", "IBKR • PENDING")
col3.metric("Mode", "PAPER")
col4.metric("Live Orders", "DISABLED")

st.divider()

with st.sidebar:
    st.header("⚙️ Engine Controls")
    symbol = st.text_input("Test symbol", value="DEMO")
    seed = st.number_input("Market simulation seed", min_value=1, value=42, step=1)
    max_risk_pct = st.slider("Max risk / trade (%)", 0.1, 2.0, 1.0, 0.1)
    max_volatility = st.slider("Max annualized volatility (%)", 20.0, 100.0, 55.0, 1.0)
    st.caption("These controls affect the paper-test decision layer only.")

market = make_demo_market(int(seed))
ind = calculate_indicators(market)
algo = maxprofit_signal(ind)
risk = risk_gate(ind, algo, max_risk_pct, max_volatility)
ai = ai_helper_assessment(ind, algo, risk)

st.header("🤖 AI HELPER")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm Signal", algo["signal"])
c2.metric("Algorithm Score", f"{algo['score']:.0f}/100")
c3.metric("AI Confidence", f"{ai['confidence']:.0f}/100")
c4.metric("Final Action", ai["recommendation"])

left, right = st.columns(2)
with left:
    st.subheader("AI Assessment")
    st.write(f"**Recommendation:** `{ai['recommendation']}`")
    st.write(f"**Reason:** {ai['reason']}")
    st.write("**Confirmations**")
    for item in ai["confirmations"] or ["No positive confirmations."]:
        st.write(f"✓ {item}")
    st.write("**Warnings**")
    for item in ai["warnings"] or ["No material warnings detected."]:
        st.write(f"⚠️ {item}")

with right:
    st.subheader("🛡️ Hard Risk Gate")
    if risk["blocked"]:
        st.error(f"ORDER BLOCKED — {risk['reason']}")
    else:
        st.success(f"RISK GATE PASS — {risk['reason']}")
    st.write(f"Configured maximum risk: **{risk['risk_pct']:.1f}%**")
    st.caption("The AI helper cannot override this gate.")

st.divider()

st.header("📊 Market & Algorithm Evidence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Demo Price", f"${ind['price']:.2f}")
m2.metric("20D Momentum", f"{ind['momentum20']:.2f}%")
m3.metric("Volatility", f"{ind['volatility']:.1f}%")
m4.metric("Volume Ratio", f"{ind['volume_ratio']:.2f}x")

chart = market.copy()
chart["SMA20"] = chart["close"].rolling(20).mean()
chart["SMA50"] = chart["close"].rolling(50).mean()
st.line_chart(chart[["close", "SMA20", "SMA50"]])

with st.expander("MAXPROFIT reasoning"):
    for reason in algo["reasons"]:
        st.write(f"• {reason}")

st.divider()

st.header("🧪 Decision Replay / Audit Log")
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []

if st.button("Run Paper Decision"):
    st.session_state.decision_log.insert(0, log_decision(symbol, ind, algo, ai, risk))
    st.success(f"Paper decision recorded: {ai['recommendation']}")

if st.session_state.decision_log:
    st.dataframe(pd.DataFrame(st.session_state.decision_log), use_container_width=True, hide_index=True)
else:
    st.info("No paper decisions recorded yet. Run the engine to begin the audit trail.")

st.divider()
st.header("🔌 Broker Connection")
st.info(
    "IBKR integration is intentionally not enabled yet. This build keeps the order path disconnected while "
    "we validate the algorithms, AI Helper, risk gate, and paper-decision logging."
)

st.caption("VAST CASH • MAXPROFIT + AI HELPER • Paper Validation Build 2.0")
