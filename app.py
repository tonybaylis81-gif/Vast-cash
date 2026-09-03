import os, time
import itertools
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")
PAPER_ONLY = True
DATA_URL = "https://data.alpaca.markets"
TRADE_URL = "https://paper-api.alpaca.markets"
UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","AMD","NFLX","ORCL","CRM","ADBE","QCOM","INTC","MU","AMAT","LRCX","TXN","JPM","BAC","WFC","GS","MS","V","MA","C","JNJ","UNH","XOM","CVX","COST","WMT","HD","LOW","CAT","GE","BA","DIS"]

def _secret(names):
    try:
        s = st.secrets
        wanted = {x.upper() for x in names}
        def walk(x):
            if isinstance(x, Mapping):
                for k,v in x.items():
                    if str(k).upper() in wanted and str(v).strip(): return str(v).strip()
                    z = walk(v)
                    if z: return z
        z = walk(s)
        if z: return z
    except Exception: pass
    for n in names:
        if os.getenv(n): return os.getenv(n).strip()
    return None

def headers():
    k=_secret(["PAPER_API_KEY","ALPACA_API_KEY","ALPACA_API_KEY_ID","API_KEY"])
    s=_secret(["PAPER_API_SECRET","ALPACA_SECRET_KEY","ALPACA_API_SECRET","API_SECRET","SECRET_KEY"])
    return {"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s} if k and s else None

@st.cache_data(ttl=21600, show_spinner=False)
def load_history(symbol,start,end):
    h=headers()
    if not h: return None
    p={"timeframe":"1Day","start":start,"end":end,"limit":10000,"adjustment":"all","feed":"iex","sort":"asc"}
    bars=[]; token=None
    try:
        for _ in range(10):
            if token: p["page_token"]=token
            r=requests.get(f"{DATA_URL}/v2/stocks/{symbol}/bars",headers=h,params=p,timeout=15)
            if r.status_code!=200: return None
            j=r.json(); bars += j.get("bars",[]); token=j.get("next_page_token")
            if not token: break
        if not bars: return None
        d=pd.DataFrame(bars)[["t","o","h","c","v"]]
        d.columns=["date","open","high","close","volume"]
        d.date=pd.to_datetime(d.date,utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        return d.set_index("date").sort_index().apply(pd.to_numeric,errors="coerce").dropna()
    except Exception: return None

def load_all(start,end):
    out={}; prog=st.progress(0); msg=st.empty()
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs={pool.submit(load_history,s,start,end):s for s in UNIVERSE}
        for i,f in enumerate(as_completed(fs),1):
            d=f.result()
            if d is not None and len(d)>=180: out[fs[f]]=d
            prog.progress(i/len(fs)); msg.write(f"Loading history {i}/{len(fs)}")
    prog.empty(); msg.empty(); return out

@st.cache_data(ttl=21600, show_spinner=False)
def prepare_history(df,lookback):
    x=df.copy(); c=x.close.to_numpy(float); h=x.high.to_numpy(float)
    n=len(x); lb=int(lookback); states=np.full((n,4),np.nan)
    for i in range(lb,n):
        w=c[i-lb:i]; daily=np.diff(w)/w[:-1]
        if len(w)<max(30,int(lb*.75)): continue
        slope=np.polyfit(np.arange(len(w)),w,1)[0]/max(w[-1],1e-9)
        states[i]=[w[-1]/w[0]-1,np.std(daily)*np.sqrt(252),w[-1]/np.max(w)-1,slope]
    return x,states

def prediction(prep,as_of,analogues,target_pct):
    df,states=prep; dates=df.index
    pos=np.searchsorted(dates,pd.Timestamp(as_of),side="left")-1
    if pos<1 or pos>=len(dates) or np.isnan(states[pos]).any(): return None
    cur=states[pos]; valid=np.where(~np.isnan(states).any(axis=1))[0]; valid=valid[(valid+63)<len(df)]
    if len(valid)<analogues: return None
    scale=np.nanstd(states[valid],axis=0); scale[scale==0]=1
    distances=np.linalg.norm((states[valid]-cur)/scale,axis=1); order=np.argsort(distances)[:int(analogues)]
    pick=valid[order]; weights=1/(distances[order]+.05); rets=[]; days=[]
    for j,i in enumerate(pick):
        start=float(df.close.iloc[i]); future=df.iloc[i:min(len(df),i+63)]
        rets.append(float(future.close.iloc[-1]/start-1))
        hits=np.where(future.high.to_numpy()>=start*(1+target_pct/100))[0]
        if len(hits): days.append((hits[0]+1,weights[j]))
    r=np.array(rets); w=np.array(weights)
    pred=float(np.average(r,weights=w)); unc=float(np.average(np.abs(r-pred),weights=w)); posrate=float(np.average((r>0),weights=w))
    hold=int(round(np.average([d[0] for d in days],weights=[d[1] for d in days]))) if days else 63
    return pred,unc,len(pick),posrate,float(r.max()),float(r.min()),hold

def quarter_windows(start,end):
    s=pd.Timestamp(start).normalize(); e=pd.Timestamp(end).normalize(); q=[]
    while s<e:
        qe=min(s+pd.DateOffset(months=3)-pd.Timedelta(days=1),e); q.append((s,qe)); s=qe+pd.Timedelta(days=1)
    return q

def rank_current(histories,as_of,lookback,analogues,buy_drop,sell_target):
    rows=[]
    for ticker,df in histories.items():
        p=prediction(prepare_history(df,lookback),as_of,analogues,sell_target)
        if not p: continue
        pred,unc,n,pos,best,worst,hold=p; past=df[df.index<=pd.Timestamp(as_of)]; price=float(past.close.iloc[-1]); high=float(past.close.tail(60).max())
        rows.append({"Ticker":ticker,"prediction":pred,"uncertainty":unc,"positive":pos,"best":best,"worst":worst,"hold":hold,"price":price,"buy_trigger":high*(1-buy_drop/100)})
    return sorted(rows,key=lambda x:(x["prediction"],x["positive"],-x["uncertainty"]),reverse=True)[:10]

def build_selections(histories,windows,lookback,analogues):
    sel={}
    for qs,qe in windows:
        ranked=[]
        for ticker,df in histories.items():
            p=prediction(prepare_history(df,lookback),qs,analogues,8)
            if p: ranked.append((p[0],p[3],ticker))
        ranked.sort(reverse=True); sel[str(qs.date())]=[{"Ticker":x[2]} for x in ranked[:15]]
    return sel

def simulate(df,start,end,capital,buy_drop,sell_target,allocation):
    d=df[(df.index>=start)&(df.index<=end)]
    if len(d)<2:return capital
    cash=float(capital); entry=qty=None; alloc=max(.01,min(1,float(allocation)/100))
    for i in range(1,len(d)):
        price=float(d.close.iloc[i]); high=float(d.high.iloc[i])
        if qty is None:
            prior=df[df.index<d.index[i]].tail(60)
            if len(prior)<20: continue
            if price<=float(prior.close.max())*(1-buy_drop/100):
                qty=int((cash*alloc)//price)
                if qty: entry=price; cash-=qty*price
        elif high>=entry*(1+sell_target/100):
            cash+=qty*entry*(1+sell_target/100); qty=None; entry=None
    if qty: cash+=qty*float(d.close.iloc[-1])
    return cash

def strategy(histories,windows,sel,capital,buy_drop,sell_target,top_n,allocation):
    total=float(capital)
    for qs,qe in windows:
        picks=sel.get(str(qs.date()),[])[:int(top_n)]
        if not picks: continue
        per=total/max(1,int(top_n)); end=total
        for item in picks: end += simulate(histories[item["Ticker"]],qs,qe,per,buy_drop,sell_target,allocation)-per
        total=end
    return {"Ending Capital":total,"Return %":(total-capital)/capital*100 if capital else 0}

@st.cache_data(ttl=1800, show_spinner=False)
def full_discovery_cached(_histories,window_keys,capital,history_signature):
    histories=_histories; windows=[(pd.Timestamp(a),pd.Timestamp(b)) for a,b in window_keys]
    base=build_selections(histories,windows,63,8)
    candidates=[]
    for b,s in itertools.product([5,10,15,20],[4,8,12,16]): candidates.append((strategy(histories,windows,base,capital,b,s,10,30)["Return %"],b,s))
    _,bb,ss=max(candidates)
    fine=[]
    for b in range(max(1,bb-2),min(20,bb+2)+1):
        for s in range(max(1,ss-2),min(20,ss+2)+1): fine.append((strategy(histories,windows,base,capital,b,s,10,30)["Return %"],b,s))
    _,b,s=max(candidates+fine)
    best=max((strategy(histories,windows,base,capital,b,s,n,a)["Return %"],n,a) for n,a in itertools.product([5,10],[20,30,40]))
    _,n,a=best; best2=[]
    for lb,an in [(63,8),(84,8),(63,12)]:
        sel=build_selections(histories,windows,lb,an); best2.append((strategy(histories,windows,sel,capital,b,s,n,a)["Return %"],lb,an))
    _,lb,an=max(best2); return b,s,n,a,lb,an

def account():
    try:
        r=requests.get(f"{TRADE_URL}/v2/account",headers=headers(),timeout=10); return r.json() if r.status_code==200 else None
    except Exception:return None

def paper_buy(symbol,allocation,slots,target_pct):
    h=headers(); ac=account()
    if not h or not ac:return False,"Paper credentials/account unavailable.",None
    try:
        budget=float(ac.get("buying_power",0))*float(allocation)/100/max(1,int(slots))
        if budget<1:return False,"Paper buying power is too low.",None
        body={"symbol":symbol,"notional":f"{budget:.2f}","side":"buy","type":"market","time_in_force":"day","client_order_id":f"vast-{symbol.lower()}-{int(time.time()*1000)}"}
        r=requests.post(f"{TRADE_URL}/v2/orders",headers={**h,"Content-Type":"application/json"},json=body,timeout=15)
        if r.status_code not in (200,201): return False,f"BUY rejected: {r.text[:250]}",None
        oid=r.json().get("id"); filled=None
        for _ in range(6):
            time.sleep(1); q=requests.get(f"{TRADE_URL}/v2/orders/{oid}",headers=h,timeout=8)
            if q.status_code==200:
                filled=q.json()
                if filled.get("status") in ("filled","partially_filled","canceled","rejected","expired"): break
        if not filled or not filled.get("filled_avg_price"): return True,f"BUY submitted. Order {oid}",oid
        fill=float(filled["filled_avg_price"]); qty=float(filled.get("filled_qty") or 0); target=round(fill*(1+target_pct/100),2)
        if qty<=0:return True,f"BUY filled at ${fill:.2f}; quantity pending.",oid
        sb={"symbol":symbol,"qty":str(qty).rstrip("0").rstrip("."),"side":"sell","type":"limit","time_in_force":"gtc","limit_price":f"{target:.2f}","client_order_id":f"vast-tp-{symbol.lower()}-{int(time.time()*1000)}"}
        sr=requests.post(f"{TRADE_URL}/v2/orders",headers={**h,"Content-Type":"application/json"},json=sb,timeout=15)
        if sr.status_code not in (200,201): return True,f"BUY filled @ ${fill:.2f}; target SELL failed: {sr.text[:200]}",oid
        return True,f"PAPER BUY FILLED: {qty:g} {symbol} @ ${fill:.2f}. GTC target SELL ${target:.2f}.",{"buy":oid,"sell":sr.json().get("id")}
    except Exception as e:return False,f"Paper execution error: {e}",None

st.title("💰 VAST CASH"); st.subheader("MAXPROFIT • TOP 10 DECISION ENGINE")
st.write("Historical fingerprints → analogue prediction → Top 10 → multiple YES/NO paper decisions.")
capital=st.number_input("Simulation starting money",100.0,1000000.0,1000.0,100.0); test_days=st.number_input("Historical test length (days)",365,3650,730,30)
c1,c2=st.columns(2); quick=c1.button("⚡ FIND TOP 10 NOW",type="primary",width="stretch"); full=c2.button("⚔️ RUN FULL MAXPROFIT DISCOVERY",width="stretch")

if quick or full:
    if not headers(): st.error("Paper credentials are not available. Check Streamlit Secrets."); st.stop()
    now=datetime.now(timezone.utc); end=now.date(); start=(now-timedelta(days=int(test_days)+1200)).date(); histories=load_all(start.isoformat(),end.isoformat())
    if not histories: st.error("No usable market history returned."); st.stop()
    lb,an,buy,sell,top,alloc=63,8,15,8,10,30
    if full:
        windows=quarter_windows((now-timedelta(days=int(test_days))).date(),end); split=max(1,int(len(windows)*.7)); train=windows[:split]; validation=windows[split:]
        keys=tuple((str(a.date()),str(b.date())) for a,b in train); signature=tuple((k,str(v.index.max()),len(v)) for k,v in sorted(histories.items()))
        with st.spinner("⚔️ MAXPROFIT is running the reduced, cached discovery..."):
            buy,sell,top,alloc,lb,an=full_discovery_cached(histories,keys,float(capital),signature)
        st.success(f"BEST SETTINGS: BUY pullback {buy}% • SELL +{sell}% • Top {top} • allocation {alloc}% • lookback {lb} • analogues {an}")
        val_sel=build_selections(histories,validation,lb,an); vr=strategy(histories,validation,val_sel,capital,buy,sell,top,alloc); st.metric("🧪 UNSEEN VALIDATION RETURN",f"{vr['Return %']:.2f}%")
    asof=max(d.index.max() for d in histories.values()); ranked=rank_current(histories,asof,lb,an,buy,sell)
    st.subheader("🔥 TOP 10 • MARK MULTIPLE YES/NO"); st.caption(f"Model date {asof.date()} • BUY {buy}% pullback • SELL +{sell}% from actual filled purchase price • PAPER ONLY")
    if "choices" not in st.session_state: st.session_state.choices={}
    if "orders" not in st.session_state: st.session_state.orders={}
    for rank,row in enumerate(ranked,1):
        ticker=row["Ticker"]; sell_date=(pd.Timestamp(asof)+pd.offsets.BDay(row["hold"])).date(); cols=st.columns([.3,.65,.8,.8,.8,.8,1.2,1])
        cols[0].markdown(f"### #{rank}"); cols[1].markdown(f"### {ticker}"); cols[2].metric("Predicted",f"{row['prediction']*100:.1f}%"); cols[3].metric("Buy Trigger",f"${row['buy_trigger']:.2f}"); cols[4].metric("Hold",f"~{row['hold']}d"); cols[5].metric("Positive",f"{row['positive']*100:.0f}%"); cols[6].markdown(f"**Suggested sell:** {sell_date}")
        choice=cols[7].selectbox("Decision",["UNDECIDED","YES","NO"],index=["UNDECIDED","YES","NO"].index(st.session_state.choices.get(ticker,"UNDECIDED")),key=f"choice_{ticker}",label_visibility="collapsed"); st.session_state.choices[ticker]=choice
        st.write(f"**{ticker}** | Current ${row['price']:.2f} | uncertainty ±{row['uncertainty']*100:.1f}% | {row['positive']*100:.0f}% positive historical matches | best {row['best']*100:.1f}% | worst {row['worst']*100:.1f}%")
        if ticker in st.session_state.orders:
            z=st.session_state.orders[ticker]; (st.success if z[0] else st.error)(z[1])
        st.divider()
    yes=[r["Ticker"] for r in ranked if st.session_state.choices.get(r["Ticker"])=="YES"]
    if yes:
        st.subheader(f"🚀 READY TO DEPLOY {len(yes)} YES"); st.code(", ".join(yes))
        if st.button(f"🚀 DEPLOY ALL {len(yes)} YES TO PAPER",type="primary",width="stretch"):
            for ticker in yes:
                if ticker in st.session_state.orders and st.session_state.orders[ticker][0]: continue
                ok,msg,info=paper_buy(ticker,alloc,len(yes),sell); st.session_state.orders[ticker]=(ok,msg,info)
            st.rerun()
    else: st.info("Mark YES on any number of the Top 10, then deploy them together.")

st.success("🔒 PAPER MODE LOCKED. Multiple YES decisions are supported. Live trading is disabled.")
