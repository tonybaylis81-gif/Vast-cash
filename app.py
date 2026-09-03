import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True
ALPACA_TRADE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def alpaca_credentials():
    return get_secret("ALPACA_API_KEY"), get_secret("ALPACA_SECRET_KEY")


def alpaca_headers():
    key, secret = alpaca_credentials()
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def alpaca_account():
    headers = alpaca_headers()
    if not headers:
        return None, "Alpaca paper credentials are not configured in Streamlit Secrets."
    try:
        r = requests.get(f"{ALPACA_TRADE_URL}/v2/account", headers=headers, timeout=15)
        if r.status_code != 200:
            return None, f"Alpaca account error {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as exc:
        return None, f"Alpaca connection error: {exc}"


def period_start(period):
    days = {"6mo": 190, "1y": 370, "2y": 740, "3y": 1100, "5y": 1850}
    return (datetime.now(timezone.utc) - timedelta(days=days.get(period, 370))).date().isoformat()


def load_history(symbol, period="1y"):
    headers = alpaca_headers()
    if not headers:
        return None, "Alpaca paper credentials are not configured."
    try:
        params = {
            "timeframe": "1Day",
            "start": period_start(period),
            "end": datetime.now(timezone.utc).date().isoformat(),
            "limit": 10000,
            "adjustment": "all",
            "feed": "iex",
            "sort": "asc",
        }
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
        bars = []
        page_token = None
        for _ in range(20):
            if page_token:
                params["page_token"] = page_token
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                return None, f"Alpaca market-data error {r.status_code}: {r.text[:300]}"
            payload = r.json()
            bars.extend(payload.get("bars", []))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        if not bars:
            return None, f"No Alpaca historical data returned for {symbol}."
        df = pd.DataFrame(bars)
        required = ["t", "o", "c", "v"]
        if not all(c in df.columns for c in required):
            return None, "Alpaca returned an unexpected bar format."
        df = df[required].copy()
        df.columns = ["date", "open", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        df = df[["open", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 60:
            return None, f"Only {len(df)} usable daily bars returned for {symbol}."
        return df, None
    except Exception as exc:
        return None, f"Alpaca historical-data error: {exc}"


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
    return {"price": float(close.iloc[-1]), "sma20": float(sma20), "sma50": float(sma50),
            "momentum20": float(momentum20), "volatility": float(volatility), "volume_ratio": float(volume_ratio)}


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
    warnings, confirmations = [], []
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
    return {"action": action, "confidence": round(confidence, 1), "warnings": warnings, "confirmations": confirmations}


def simulate_strategy(histories, starting_capital, allocation_pct, max_hold_days, max_risk_pct, max_volatility):
    dates = sorted(set().union(*[set(df.index) for df in histories.values()]))
    cash, position, trades, equity_curve, blocked_count = float(starting_capital), None, [], [], 0
    for i, date in enumerate(dates):
        todays = {}
        for symbol, df in histories.items():
            if date not in df.index: continue
            ind = calculate_indicators(df, df.index.get_loc(date) + 1)
            if ind is None: continue
            algo = maxprofit_signal(ind); risk = risk_gate(ind, algo, max_risk_pct, max_volatility); ai = ai_helper(ind, algo, risk)
            todays[symbol] = (df.loc[date], ind, algo, risk, ai)
            blocked_count += int(risk["blocked"])
        if position and position["symbol"] in todays:
            _, _, _, _, ai = todays[position["symbol"]]
            held = i - position["entry_index"]
            if held >= max_hold_days or ai["action"] == "SELL":
                if i + 1 < len(dates):
                    next_date = dates[i + 1]; df = histories[position["symbol"]]
                    if next_date in df.index:
                        exit_price = float(df.loc[next_date, "open"]); proceeds = position["shares"] * exit_price; pnl = proceeds - position["cost"]; cash += proceeds
                        trades.append({"Entry": position["entry_date"], "Exit": next_date, "Ticker": position["symbol"], "Buy": round(position["entry_price"],2), "Sell": round(exit_price,2), "Days": held, "P&L": round(pnl,2), "Return %": round(pnl/position["cost"]*100,2), "Exit Reason": "MAX HOLD" if held >= max_hold_days else "AI SELL"})
                        position = None
        if position is None and i + 1 < len(dates):
            candidates = [(algo["score"], ai["confidence"], symbol) for symbol, (_, _, algo, risk, ai) in todays.items() if not risk["blocked"] and ai["action"] == "BUY"]
            if candidates:
                _, _, symbol = max(candidates); next_date = dates[i+1]; df = histories[symbol]
                if next_date in df.index:
                    entry_price = float(df.loc[next_date, "open"]); allocation = min(cash, cash * allocation_pct / 100); shares = int(allocation / entry_price)
                    if shares > 0:
                        cost = shares * entry_price; cash -= cost; position = {"symbol": symbol, "entry_date": next_date, "entry_price": entry_price, "shares": shares, "cost": cost, "entry_index": i+1}
        equity = cash
        if position and date in histories[position["symbol"]].index: equity += position["shares"] * float(histories[position["symbol"]].loc[date, "close"])
        equity_curve.append({"Date": date, "Equity": equity})
    if position:
        df = histories[position["symbol"]]; last_date = df.index[-1]; exit_price = float(df.loc[last_date, "close"]); proceeds = position["shares"]*exit_price; pnl = proceeds-position["cost"]; cash += proceeds
        trades.append({"Entry": position["entry_date"], "Exit": last_date, "Ticker": position["symbol"], "Buy": round(position["entry_price"],2), "Sell": round(exit_price,2), "Days": max(0,len(dates)-1-position["entry_index"]), "P&L": round(pnl,2), "Return %": round(pnl/position["cost"]*100,2), "Exit Reason": "END OF TEST"})
    curve = pd.DataFrame(equity_curve)
    if curve.empty: return None
    curve["Peak"] = curve["Equity"].cummax(); curve["Drawdown %"] = (curve["Equity"]/curve["Peak"]-1)*100
    trade_df = pd.DataFrame(trades); ending = float(cash); net = ending-starting_capital
    return {"starting": starting_capital, "ending": ending, "net": net, "return_pct": net/starting_capital*100, "trades": trade_df, "curve": curve, "win_rate": float((trade_df["P&L"]>0).mean()*100) if not trade_df.empty else 0.0, "max_dd": float(curve["Drawdown %"].min()), "blocked": blocked_count}


def submit_paper_order(symbol, side, qty):
    if not PAPER_ONLY: return False, "Live trading is disabled by configuration."
    headers = alpaca_headers()
    if not headers: return False, "Alpaca paper credentials are not configured in Streamlit Secrets."
    try:
        payload = {"symbol": symbol, "qty": str(int(qty)), "side": side.lower(), "type": "market", "time_in_force": "day"}
        r = requests.post(f"{ALPACA_TRADE_URL}/v2/orders", headers={**headers, "Content-Type":"application/json"}, json=payload, timeout=15)
        if r.status_code not in (200, 201): return False, f"Paper order rejected {r.status_code}: {r.text[:300]}"
        order = r.json(); return True, f"PAPER {side.upper()} submitted: {order.get('symbol')} {order.get('qty')} shares. Order ID {order.get('id')}"
    except Exception as exc: return False, f"Paper order error: {exc}"


st.title("💰 VAST CASH")
st.subheader("MAXPROFIT Engine • AI Helper • Alpaca Paper Trading")
account, account_error = alpaca_account()
if account:
    st.success("🟢 ALPACA PAPER CONNECTED. No live trading endpoint is used by this app.")
else:
    st.warning("🟡 ALPACA PAPER NOT CONNECTED. Add the existing paper credentials to Streamlit Secrets, then reload.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("System", "ONLINE")
c2.metric("Broker", "ALPACA • PAPER" if account else "ALPACA • PENDING")
c3.metric("Mode", "PAPER ONLY")
c4.metric("Live Orders", "DISABLED")
if account:
    a,b = st.columns(2); a.metric("Paper Cash", f"${float(account.get('cash',0)):,.2f}"); b.metric("Buying Power", f"${float(account.get('buying_power',0)):,.2f}")

with st.sidebar:
    st.header("⚙️ Portfolio Setup")
    stock_text = st.text_area("Stock universe (1–10 tickers)", "AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nAMD\nAVGO\nJPM", height=220)
    max_risk_pct = st.slider("Max risk / trade (%)", .1, 2.0, 1.0, .1)
    max_volatility = st.slider("Max volatility (%)", 20.0, 100.0, 55.0, 1.0)
    st.divider(); st.header("🧪 Backtest Settings")
    sim_period = st.selectbox("Historical test period", ["6mo", "1y", "2y", "3y", "5y"], index=1)
    starting_capital = st.number_input("Starting paper capital ($)", 100.0, 1_000_000.0, 1000.0, 100.0)
    allocation_pct = st.slider("Capital allocated per trade (%)", 5.0, 100.0, 50.0, 5.0)
    max_hold_days = st.number_input("Automatic maximum hold (trading days)", 1, 252, 6, 1)

stocks = list(dict.fromkeys([s.strip().upper() for s in stock_text.replace(",","\n").splitlines() if s.strip()]))[:10]
if not stocks: stocks=["AAPL"]

st.header("📊 Live Alpaca Market Scan")
rows=[]; all_results={}
for ticker in stocks:
    hist, err = load_history(ticker, "6mo")
    if hist is None:
        rows.append({"Ticker":ticker,"MAXPROFIT":"DATA ERROR","Score":0,"AI":"BLOCK","Confidence":0,"Risk":"BLOCK"}); continue
    ind=calculate_indicators(hist); algo=maxprofit_signal(ind); risk=risk_gate(ind,algo,max_risk_pct,max_volatility); ai=ai_helper(ind,algo,risk); all_results[ticker]=(ind,algo,risk,ai)
    rows.append({"Ticker":ticker,"MAXPROFIT":algo["signal"],"Score":round(algo["score"],1),"AI":ai["action"],"Confidence":round(ai["confidence"],1),"Risk":"BLOCK" if risk["blocked"] else "PASS"})
ranking=pd.DataFrame(rows).sort_values(["Score","Confidence"],ascending=False).reset_index(drop=True); ranking.insert(0,"Rank",range(1,len(ranking)+1)); st.dataframe(ranking,use_container_width=True,hide_index=True)

if all_results:
    selected=st.selectbox("Select stock for AI Helper / paper controls", list(all_results.keys()))
    ind,algo,risk,ai=all_results[selected]
    x1,x2,x3,x4=st.columns(4); x1.metric("Price",f"${ind['price']:.2f}"); x2.metric("MAXPROFIT",f"{algo['signal']} / {algo['score']:.0f}"); x3.metric("AI",f"{ai['action']} / {ai['confidence']:.0f}%"); x4.metric("Risk Gate","BLOCK" if risk["blocked"] else "PASS")
    with st.expander("🤖 AI Helper",expanded=True):
        for text in ai["confirmations"]: st.write("✅ "+text)
        for text in ai["warnings"]: st.write("⚠️ "+text)
        st.caption(risk["reason"])

    st.subheader("🧾 Alpaca Paper Order Controls")
    st.caption("These buttons submit orders to the Alpaca PAPER account only. Live trading is hard-disabled.")
    qty=st.number_input("Whole shares",1,100000,1,1)
    confirm=st.checkbox("I understand this sends a PAPER order to Alpaca")
    b1,b2,b3=st.columns(3)
    if b1.button("🟢 PAPER BUY",use_container_width=True,disabled=not confirm):
        ok,msg=submit_paper_order(selected,"buy",qty); st.success(msg) if ok else st.error(msg)
    if b2.button("🔴 PAPER SELL",use_container_width=True,disabled=not confirm):
        ok,msg=submit_paper_order(selected,"sell",qty); st.success(msg) if ok else st.error(msg)
    if b3.button("🟡 HOLD",use_container_width=True): st.info(f"HOLD recorded for {selected}. No order sent.")

st.header("🧪 Historical Simulation")
st.caption("Walk-forward test using Alpaca historical daily bars. Signals use only information available before each simulated entry. Entries/exits execute at the following trading day's open. Default automatic maximum hold is 6 trading days.")
if st.button("▶️ RUN SIMULATION",type="primary",use_container_width=True):
    if not alpaca_headers(): st.error("Connect Alpaca PAPER credentials in Streamlit Secrets first.")
    else:
        with st.spinner(f"Loading {sim_period} Alpaca history and running the MAXPROFIT simulation..."):
            histories={}; errors=[]
            for ticker in stocks:
                hist,err=load_history(ticker,sim_period)
                if hist is not None: histories[ticker]=hist
                else: errors.append(f"{ticker}: {err}")
            if len(histories)<1: st.error("No usable Alpaca historical data was available.")
            else:
                for e in errors: st.warning(e)
                result=simulate_strategy(histories,starting_capital,allocation_pct,max_hold_days,max_risk_pct,max_volatility)
                if result:
                    m1,m2,m3,m4=st.columns(4); m1.metric("Ending Capital",f"${result['ending']:,.2f}"); m2.metric("Net P&L",f"${result['net']:,.2f}",f"{result['return_pct']:.2f}%"); m3.metric("Win Rate",f"{result['win_rate']:.1f}%"); m4.metric("Max Drawdown",f"{result['max_dd']:.2f}%")
                    st.line_chart(result["curve"].set_index("Date")["Equity"])
                    st.subheader("Trade Ledger")
                    if result["trades"].empty: st.info("No qualifying trades were generated under the current settings.")
                    else: st.dataframe(result["trades"],use_container_width=True,hide_index=True)
                    st.caption(f"Risk-gate blocks encountered during the walk-forward test: {result['blocked']}")
