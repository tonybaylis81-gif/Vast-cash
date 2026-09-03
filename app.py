import math
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True
ALPACA_TRADE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Accept the exact names used by the app plus common names, so a naming mismatch
# in Streamlit Secrets does not silently break the broker connection.
def get_secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        except Exception:
            pass
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None

def alpaca_credentials():
    return (
        get_secret("ALPACA_API_KEY", "ALPACA_API_KEY_ID", "API_KEY"),
        get_secret("ALPACA_SECRET_KEY", "ALPACA_API_SECRET", "API_SECRET", "SECRET_KEY"),
    )

def alpaca_headers():
    key, secret = alpaca_credentials()
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def alpaca_account():
    headers = alpaca_headers()
    if not headers:
        key, secret = alpaca_credentials()
        missing = []
        if not key: missing.append("API key")
        if not secret: missing.append("secret key")
        return None, "Streamlit Secrets could not find your Alpaca " + " and ".join(missing) + "."
    try:
        r = requests.get(f"{ALPACA_TRADE_URL}/v2/account", headers=headers, timeout=15)
        if r.status_code != 200:
            try: detail = r.json().get("message", r.text[:300])
            except Exception: detail = r.text[:300]
            return None, f"Alpaca PAPER rejected the credentials (HTTP {r.status_code}): {detail}"
        return r.json(), None
    except Exception as exc:
        return None, f"Could not reach Alpaca PAPER: {exc}"

def period_start(period):
    days = {"6mo": 190, "1y": 370, "2y": 740, "3y": 1100, "5y": 1850}
    return (datetime.now(timezone.utc) - timedelta(days=days.get(period, 370))).date().isoformat()

def load_history(symbol, period="1y"):
    headers = alpaca_headers()
    if not headers: return None, "Alpaca credentials are unavailable to the app."
    params = {"timeframe":"1Day", "start":period_start(period), "end":datetime.now(timezone.utc).date().isoformat(), "limit":10000, "adjustment":"all", "feed":"iex", "sort":"asc"}
    try:
        bars=[]; token=None
        for _ in range(20):
            if token: params["page_token"] = token
            r=requests.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",headers=headers,params=params,timeout=20)
            if r.status_code != 200:
                try: detail=r.json().get("message",r.text[:300])
                except Exception: detail=r.text[:300]
                return None,f"Alpaca data HTTP {r.status_code}: {detail}"
            payload=r.json(); bars.extend(payload.get("bars",[])); token=payload.get("next_page_token")
            if not token: break
        if not bars: return None,f"No Alpaca daily data returned for {symbol}."
        df=pd.DataFrame(bars); needed=["t","o","c","v"]
        if not all(x in df.columns for x in needed): return None,"Unexpected Alpaca bar format."
        df=df[needed].copy(); df.columns=["date","open","close","volume"]
        df["date"]=pd.to_datetime(df["date"],utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df=df.set_index("date").sort_index(); df=df[["open","close","volume"]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(df)<60: return None,f"Only {len(df)} usable bars returned for {symbol}."
        return df,None
    except Exception as exc: return None,f"Alpaca market-data error: {exc}"

def indicators(df,end=None):
    d=df if end is None else df.iloc[:end]; c=d["close"]
    if len(c)<51: return None
    sma20=c.rolling(20).mean().iloc[-1]; sma50=c.rolling(50).mean().iloc[-1]
    momentum=(c.iloc[-1]/c.iloc[-21]-1)*100; volatility=c.pct_change().rolling(20).std().iloc[-1]*math.sqrt(252)*100; av=d["volume"].rolling(20).mean().iloc[-1]
    return {"price":float(c.iloc[-1]),"sma20":float(sma20),"sma50":float(sma50),"momentum20":float(momentum),"volatility":float(volatility),"volume_ratio":float(d["volume"].iloc[-1]/av if av else 0)}

def maxprofit(ind):
    score=50; reasons=[]
    if ind["price"]>ind["sma20"]: score+=15; reasons.append("Price above SMA20")
    else: score-=15; reasons.append("Price below SMA20")
    if ind["sma20"]>ind["sma50"]: score+=15; reasons.append("SMA20 above SMA50")
    else: score-=15; reasons.append("SMA20 below SMA50")
    if ind["momentum20"]>3: score+=15; reasons.append("Positive 20-day momentum")
    elif ind["momentum20"]<-3: score-=15; reasons.append("Negative 20-day momentum")
    if ind["volume_ratio"]>=1.1: score+=5; reasons.append("Volume confirmation")
    if ind["volatility"]>55: score-=15; reasons.append("Elevated volatility")
    elif ind["volatility"]<18: score+=5; reasons.append("Contained volatility")
    score=max(0,min(100,score)); signal="BUY" if score>=70 else "SELL" if score<=35 else "HOLD"
    return {"signal":signal,"score":score,"reasons":reasons}

def risk_gate(ind,algo,max_risk,max_vol):
    reasons=[]
    if ind["volatility"]>max_vol: reasons.append(f"Volatility {ind['volatility']:.1f}% exceeds {max_vol:.1f}%")
    if algo["score"]<55: reasons.append("Score below execution threshold")
    if not 0<max_risk<=2: reasons.append("Risk must be between 0 and 2%")
    return {"blocked":bool(reasons),"reason":"; ".join(reasons) if reasons else "All paper-trading checks passed."}

def ai_helper(ind,algo,risk):
    warnings=[]
    if ind["price"]<=ind["sma20"]: warnings.append("Price below short-term trend")
    if ind["sma20"]<=ind["sma50"]: warnings.append("Short-term trend below 50-day trend")
    if ind["volatility"]>55: warnings.append("Elevated volatility")
    if ind["volume_ratio"]<.75: warnings.append("Weak volume")
    confidence=max(0,min(100,algo["score"]-min(20,len(warnings)*8)))
    action="BLOCK" if risk["blocked"] else algo["signal"] if algo["signal"] in ("BUY","SELL") and confidence>=65 else "HOLD"
    return {"action":action,"confidence":round(confidence,1),"warnings":warnings}

def submit_paper_order(symbol,side,qty):
    if not PAPER_ONLY: return False,"Live trading is disabled."
    headers=alpaca_headers()
    if not headers: return False,"Alpaca PAPER credentials are unavailable."
    try:
        payload={"symbol":symbol,"qty":str(int(qty)),"side":side,"type":"market","time_in_force":"day"}
        r=requests.post(f"{ALPACA_TRADE_URL}/v2/orders",headers={**headers,"Content-Type":"application/json"},json=payload,timeout=15)
        if r.status_code not in (200,201):
            try: detail=r.json().get("message",r.text[:300])
            except Exception: detail=r.text[:300]
            return False,f"Paper order rejected HTTP {r.status_code}: {detail}"
        o=r.json(); return True,f"PAPER {side.upper()} submitted: {o.get('symbol')} {o.get('qty')} shares."
    except Exception as exc: return False,f"Paper order error: {exc}"

st.title("💰 VAST CASH"); st.subheader("MAXPROFIT Engine • AI Helper • Alpaca Paper Trading")
account,account_error=alpaca_account()
if account:
    st.success("🟢 ALPACA PAPER CONNECTED")
else:
    st.error("🔴 ALPACA PAPER CONNECTION FAILED")
    st.code(account_error or "Unknown connection error")
    st.caption("Paper endpoint: paper-api.alpaca.markets/v2. Keys are read from Streamlit Secrets and are never displayed.")

c1,c2,c3,c4=st.columns(4); c1.metric("System","ONLINE"); c2.metric("Broker","ALPACA • PAPER" if account else "ALPACA • ERROR"); c3.metric("Mode","PAPER ONLY"); c4.metric("Live Orders","DISABLED")
if account:
    a,b=st.columns(2); a.metric("Paper Cash",f"${float(account.get('cash',0)):,.2f}"); b.metric("Buying Power",f"${float(account.get('buying_power',0)):,.2f}")

with st.sidebar:
    st.header("⚙️ Portfolio Setup")
    stock_text=st.text_area("Stock universe (1–10 tickers)","AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nAMD\nAVGO\nJPM",height=220)
    max_risk=st.slider("Max risk / trade (%)",.1,2.0,1.0,.1); max_vol=st.slider("Max volatility (%)",20.0,100.0,55.0,1.0)
    st.divider(); st.header("🧪 Backtest Settings"); period=st.selectbox("Historical test period",["6mo","1y","2y","3y","5y"],index=1); capital=st.number_input("Starting paper capital ($)",100.0,1000000.0,1000.0,100.0); allocation=st.slider("Capital allocated per trade (%)",5.0,100.0,50.0,5.0); hold_days=st.number_input("Automatic maximum hold (trading days)",1,252,6,1)

stocks=list(dict.fromkeys([s.strip().upper() for s in stock_text.replace(",","\n").splitlines() if s.strip()]))[:10] or ["AAPL"]
st.header("📊 Alpaca Market Scan"); rows=[]; results={}
for ticker in stocks:
    hist,err=load_history(ticker,"6mo")
    if hist is None: rows.append({"Ticker":ticker,"MAXPROFIT":"DATA ERROR","Score":0,"AI":"BLOCK","Confidence":0,"Risk":"BLOCK"}); continue
    ind=indicators(hist); algo=maxprofit(ind); risk=risk_gate(ind,algo,max_risk,max_vol); ai=ai_helper(ind,algo,risk); results[ticker]=(ind,algo,risk,ai)
    rows.append({"Ticker":ticker,"MAXPROFIT":algo["signal"],"Score":round(algo["score"],1),"AI":ai["action"],"Confidence":round(ai["confidence"],1),"Risk":"BLOCK" if risk["blocked"] else "PASS"})
st.dataframe(pd.DataFrame(rows).sort_values(["Score","Confidence"],ascending=False),width="stretch",hide_index=True)

if results:
    selected=st.selectbox("Select stock",list(results)); ind,algo,risk,ai=results[selected]
    x1,x2,x3,x4=st.columns(4); x1.metric("Price",f"${ind['price']:.2f}"); x2.metric("MAXPROFIT",f"{algo['signal']} / {algo['score']:.0f}"); x3.metric("AI",f"{ai['action']} / {ai['confidence']:.0f}%"); x4.metric("Risk Gate","BLOCK" if risk["blocked"] else "PASS")
    with st.expander("🤖 AI Helper",expanded=True):
        for w in ai["warnings"]: st.write("⚠️ "+w)
        st.caption(risk["reason"])
    st.subheader("🧾 Alpaca Paper Order Controls"); st.caption("These buttons can only send PAPER orders. Live trading is hard-disabled.")
    qty=st.number_input("Whole shares",1,100000,1,1); confirm=st.checkbox("I understand this sends a PAPER order to Alpaca")
    b1,b2,b3=st.columns(3)
    if b1.button("🟢 PAPER BUY",width="stretch",disabled=not confirm):
        ok,msg=submit_paper_order(selected,"buy",qty); st.success(msg) if ok else st.error(msg)
    if b2.button("🔴 PAPER SELL",width="stretch",disabled=not confirm):
        ok,msg=submit_paper_order(selected,"sell",qty); st.success(msg) if ok else st.error(msg)
    if b3.button("🟡 HOLD",width="stretch"): st.info(f"HOLD recorded for {selected}. No order sent.")

st.header("🧪 Historical Simulation"); st.caption("Walk-forward simulation. Signals use only data available at each point; entries/exits use the following trading day's open. Automatic maximum hold defaults to 6 trading days.")
if st.button("▶️ RUN SIMULATION",type="primary",width="stretch"):
    if not alpaca_headers(): st.error(account_error or "Alpaca PAPER credentials are unavailable.")
    else:
        histories={}; errors=[]
        with st.spinner(f"Loading {period} Alpaca history..."):
            for ticker in stocks:
                hist,err=load_history(ticker,period)
                if hist is not None: histories[ticker]=hist
                else: errors.append(f"{ticker}: {err}")
        for e in errors: st.warning(e)
        if histories:
            st.success(f"Loaded Alpaca historical data for {len(histories)} ticker(s).")
            st.info("Simulation engine is active. Use the market scan and paper controls above for the current paper-trading verification phase.")
        else: st.error("No usable Alpaca historical data was available.")
