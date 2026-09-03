import os
import time
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
ALPACA_TRADE_URL = "https://paper-api.alpaca.markets"
MARKET_UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","AMD","NFLX","ORCL","CRM","ADBE","QCOM","INTC","MU","AMAT","LRCX","TXN","JPM","BAC","WFC","GS","MS","V","MA","C","JNJ","UNH","XOM","CVX","COST","WMT","HD","LOW","CAT","GE","BA","DIS"]

def _find_secret(mapping, names):
    if not isinstance(mapping, Mapping): return None
    wanted = {n.upper() for n in names}
    for key in mapping:
        value = mapping[key]
        if str(key).strip().upper() in wanted and value is not None and str(value).strip(): return str(value).strip()
        if isinstance(value, Mapping):
            found = _find_secret(value, names)
            if found: return found
    return None

def get_secret(*names):
    try:
        found = _find_secret(st.secrets, names)
        if found: return found
    except Exception: pass
    for name in names:
        value = os.getenv(name)
        if value and str(value).strip(): return str(value).strip()
    return None

def alpaca_headers():
    key = get_secret("PAPER_API_KEY","ALPACA_API_KEY","ALPACA_API_KEY_ID","API_KEY")
    secret = get_secret("PAPER_API_SECRET","ALPACA_SECRET_KEY","ALPACA_API_SECRET","API_SECRET","SECRET_KEY")
    return {"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":secret} if key and secret else None

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol, start_date, end_date):
    headers = alpaca_headers()
    if not headers: return None, "Paper credentials unavailable."
    params={"timeframe":"1Day","start":start_date,"end":end_date,"limit":10000,"adjustment":"all","feed":"iex","sort":"asc"}
    bars=[]; token=None
    try:
        for _ in range(20):
            if token: params["page_token"]=token
            r=requests.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",headers=headers,params=params,timeout=20)
            if r.status_code!=200: return None,f"HTTP {r.status_code}"
            payload=r.json(); bars.extend(payload.get("bars",[])); token=payload.get("next_page_token")
            if not token: break
        if not bars: return None,"No data"
        df=pd.DataFrame(bars)[["t","o","h","c","v"]].copy(); df.columns=["date","open","high","close","volume"]
        df["date"]=pd.to_datetime(df["date"],utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        return df.set_index("date").sort_index().apply(pd.to_numeric,errors="coerce").dropna(),None
    except Exception as exc: return None,str(exc)

def load_all_histories(start_date,end_date):
    histories={}; progress=st.progress(0); status=st.empty(); done=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(load_history,t,start_date,end_date):t for t in MARKET_UNIVERSE}
        for f in as_completed(futures):
            h,_=f.result(); done+=1
            if h is not None and len(h)>=180: histories[futures[f]]=h
            progress.progress(done/len(MARKET_UNIVERSE)); status.write(f"Loading market history: {done}/{len(MARKET_UNIVERSE)} stocks")
    progress.empty(); status.empty(); return histories

def market_state(df,as_of,lookback=63):
    prior=df[df.index<pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior)<max(30,int(lookback*.75)): return None
    close=prior.close; daily=close.pct_change().dropna()
    return np.array([float(close.iloc[-1]/close.iloc[0]-1),float(daily.std()*np.sqrt(252)) if len(daily) else 0.0,float(close.iloc[-1]/close.max()-1),float(np.polyfit(np.arange(len(close)),close.values,1)[0]/max(close.iloc[-1],1e-9))])

def fingerprint_details(df,as_of,lookback=63):
    prior=df[df.index<=pd.Timestamp(as_of)].tail(int(lookback))
    if len(prior)<max(30,int(lookback*.75)): return None
    close=prior.close; daily=close.pct_change().dropna(); recent=close.tail(min(20,len(close))); avg_vol=prior.volume.mean()
    return {"Momentum":float(close.iloc[-1]/close.iloc[0]-1),"Recent 20D":float(recent.iloc[-1]/recent.iloc[0]-1),"Volatility":float(daily.std()*np.sqrt(252)) if len(daily) else 0.0,"Drawdown":float(close.iloc[-1]/close.max()-1),"Slope":float(np.polyfit(np.arange(len(close)),close.values,1)[0]/max(close.iloc[-1],1e-9)),"Up-Day Rate":float((daily>0).mean()) if len(daily) else .5,"Volume Ratio":float(prior.volume.tail(10).mean()/avg_vol) if avg_vol else 1.0}

def historical_prediction(df,as_of,lookback=63,analogues=8,target_pct=8.0):
    current=market_state(df,as_of,lookback)
    if current is None:return None
    dates=df.index; examples=[]; max_i=len(dates)-63
    for i in range(int(lookback)+7,max_i):
        state=market_state(df,dates[i],lookback)
        if state is None: continue
        future=df.iloc[i:i+63]; start=float(future.close.iloc[0]); ret=float(future.close.iloc[-1]/start-1)
        hits=np.where(future.high.values>=start*(1+target_pct/100))[0]; days=int(hits[0]+1) if len(hits) else None
        examples.append((state,ret,dates[i],days))
    if len(examples)<max(5,int(analogues)): return None
    states=np.array([x[0] for x in examples]); scale=np.std(states,axis=0); scale[scale==0]=1.0
    distances=np.linalg.norm((states-current)/scale,axis=1); order=np.argsort(distances)[:min(int(analogues),len(examples))]
    nearest=[examples[i] for i in order]; weights=np.array([1/(distances[i]+.05) for i in order]); returns=np.array([x[1] for x in nearest])
    prediction=float(np.average(returns,weights=weights)); uncertainty=float(np.average(np.abs(returns-prediction),weights=weights)); positive=float(np.average((returns>0).astype(float),weights=weights))
    best,worst=float(returns.max()),float(returns.min()); hit=[(x[3],weights[j]) for j,x in enumerate(nearest) if x[3] is not None]
    hold=int(round(np.average([x[0] for x in hit],weights=[x[1] for x in hit]))) if hit else 63
    return prediction,uncertainty,len(nearest),current,positive,best,worst,nearest,max(1,hold)

def current_track(df,as_of,lookback,analogues,buy_drop,sell_target):
    details=fingerprint_details(df,as_of,lookback); pred=historical_prediction(df,as_of,lookback,analogues,sell_target)
    if details is None or pred is None:return None
    prediction,uncertainty,samples,state,positive,best,worst,nearest,hold=pred; price=float(df.loc[df.index<=pd.Timestamp(as_of),"close"].iloc[-1]); high=float(df.loc[df.index<=pd.Timestamp(as_of),"close"].tail(60).max()); trigger=high*(1-buy_drop/100)
    if prediction>0 and positive>=.60 and details["Drawdown"]>-.20: status="🟢 ON TRACK"
    elif prediction>0 and positive>=.45: status="🟡 WATCH / DEVIATING"
    else: status="🔴 PATTERN BROKEN"
    return {"price":price,"buy_trigger":trigger,"target_reference":price*(1+sell_target/100),"prediction":prediction,"uncertainty":uncertainty,"positive":positive,"best":best,"worst":worst,"samples":samples,"details":details,"status":status,"nearest":nearest,"hold_days":hold}

def next_business_date(start,days): return (pd.Timestamp(start)+pd.offsets.BDay(int(days))).date()

def paper_account():
    try:
        r=requests.get(f"{ALPACA_TRADE_URL}/v2/account",headers=alpaca_headers(),timeout=15)
        return r.json() if r.status_code==200 else None
    except Exception:return None

def paper_buy_with_target(symbol,budget_pct,total_slots,sell_target):
    if not PAPER_ONLY:return False,"Live trading is locked off.",None
    headers=alpaca_headers()
    if not headers:return False,"Paper credentials unavailable.",None
    account=paper_account()
    if not account:return False,"Could not read paper account.",None
    try:
        buying_power=float(account.get("buying_power",0)); budget=buying_power*(float(budget_pct)/100)/max(1,int(total_slots))
        if budget<1:return False,"Paper buying power is too low for a share.",None
        # Use notional market buy so the paper engine can deploy the calculated allocation cleanly.
        body={"symbol":symbol,"notional":f"{budget:.2f}","side":"buy","type":"market","time_in_force":"day","client_order_id":f"vastcash-{symbol.lower()}-{int(time.time())}"}
        r=requests.post(f"{ALPACA_TRADE_URL}/v2/orders",headers={**headers,"Content-Type":"application/json"},json=body,timeout=20)
        if r.status_code not in (200,201): return False,f"BUY rejected: {r.text[:300]}",None
        order=r.json(); oid=order.get("id")
        filled=None
        for _ in range(10):
            time.sleep(1)
            q=requests.get(f"{ALPACA_TRADE_URL}/v2/orders/{oid}",headers=headers,timeout=10)
            if q.status_code==200:
                filled=q.json()
                if filled.get("status") in ("filled","partially_filled","canceled","rejected","expired"): break
        if not filled or not filled.get("filled_avg_price"):
            return True,f"Paper BUY submitted for {symbol}; waiting for fill. Order ID {oid}.",oid
        fill=float(filled["filled_avg_price"]); qty=float(filled.get("filled_qty") or 0); target=round(fill*(1+float(sell_target)/100),2)
        if qty<=0:return True,f"Paper BUY filled but no quantity was reported yet. Order ID {oid}.",oid
        sell_body={"symbol":symbol,"qty":f"{qty:.9f}".rstrip("0").rstrip("."),"side":"sell","type":"limit","time_in_force":"gtc","limit_price":f"{target:.2f}","client_order_id":f"vastcash-tp-{symbol.lower()}-{int(time.time())}"}
        sr=requests.post(f"{ALPACA_TRADE_URL}/v2/orders",headers={**headers,"Content-Type":"application/json"},json=sell_body,timeout=20)
        if sr.status_code not in (200,201): return True,f"Paper BUY filled at ${fill:.2f}, but target SELL could not be attached: {sr.text[:250]}",oid
        sid=sr.json().get("id")
        return True,f"PAPER BUY FILLED: {symbol} {qty:g} shares @ ${fill:.2f}. Automatic +{sell_target:.1f}% target SELL placed at ${target:.2f}.",{"buy_order_id":oid,"sell_order_id":sid,"fill":fill,"target":target,"qty":qty}
    except Exception as exc:return False,f"Paper execution error: {exc}",None

def run_strategy(histories,windows,selections,capital,buy_drop,sell_target,top_n,allocation):
    total_start=total_end=float(capital); all_trades=[]; quarters=[]
    for qstart,qend in windows:
        selected=selections.get(str(qstart.date()),[])[:int(top_n)]
        if not selected:continue
        q_start=total_end; q_end_cap=total_end; per=q_start/max(1,int(top_n)); qtr=[]
        for item in selected:
            ending,trades=simulate_quarter(histories[item["Ticker"]],qstart,qend,per,buy_drop,sell_target,allocation); q_end_cap+=ending-per
            for trade in trades: trade["Ticker"]=item["Ticker"]; qtr.append(trade)
        total_end=q_end_cap; all_trades+=qtr; quarters.append({"Quarter":f"{qstart.date()} to {qend.date()}","Predicted Stocks":", ".join(x["Ticker"] for x in selected),"Start Capital":round(q_start,2),"End Capital":round(q_end_cap,2),"Quarter P/L":round(q_end_cap-q_start,2),"Trades":len(qtr)})
    pnl=total_end-total_start; return {"Ending Capital":total_end,"Profit / Loss":pnl,"Return %":pnl/total_start*100 if total_start else 0,"quarters":quarters,"trades":all_trades}

def simulate_quarter(df,start,end,capital,buy_drop,sell_target,allocation):
    data=df[(df.index>=pd.Timestamp(start))&(df.index<=pd.Timestamp(end))]
    if len(data)<2:return capital,[]
    cash=float(capital); position=None; trades=[]; alloc=min(max(float(allocation)/100,.01),1)
    for i in range(1,len(data)):
        row=data.iloc[i]
        if position is None:
            prior=df[df.index<data.index[i]].tail(60)
            if len(prior)<20:continue
            trigger=float(prior.close.max())*(1-buy_drop/100)
            if float(row.close)<=trigger:
                qty=int((cash*alloc)//float(row.close))
                if qty:position={"entry":float(row.close),"qty":qty,"date":data.index[i].date()};cash-=qty*position["entry"]
        else:
            target=position["entry"]*(1+sell_target/100)
            if float(row.high)>=target:
                cash+=position["qty"]*target; trades.append({"Buy Date":position["date"],"Sell Date":data.index[i].date(),"Shares":position["qty"],"Buy":round(position["entry"],2),"Sell":round(target,2),"P/L":round((target-position["entry"])*position["qty"],2),"Return %":sell_target,"Reason":f"+{sell_target:.1f}% target"});position=None
    if position is not None:
        last=float(data.close.iloc[-1]);cash+=position["qty"]*last;trades.append({"Buy Date":position["date"],"Sell Date":data.index[-1].date(),"Shares":position["qty"],"Buy":round(position["entry"],2),"Sell":round(last,2),"P/L":round((last-position["entry"])*position["qty"],2),"Return %":round((last/position["entry"]-1)*100,2),"Reason":"Quarter-end mark-to-market"})
    return cash,trades

def quarter_windows(start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end); out=[]; cur=start
    while cur<end:
        qend=min(cur+pd.DateOffset(months=3),end);out.append((cur,qend));cur=qend
    return out

def build_predictions(histories,windows,lookback,analogues):
    selections={}; rows=[]
    for qstart,qend in windows:
        ranked=[]
        for ticker,df in histories.items():
            p=historical_prediction(df,qstart,lookback,analogues)
            if p is None:continue
            ret,u,n,_,pos,best,worst,_,_=p; ranked.append({"Ticker":ticker,"Predicted Next Quarter %":ret*100,"Historical Uncertainty %":u*100,"Historical Positive Rate %":pos*100,"Best Analogue %":best*100,"Worst Analogue %":worst*100,"History Matches":n})
        ranked.sort(key=lambda x:(x["Predicted Next Quarter %"],x["Historical Positive Rate %"]),reverse=True);selections[str(qstart.date())]=ranked[:10]
        for rank,item in enumerate(ranked[:20],1): row=dict(item);row["Rank"]=rank;row["Quarter"]=f"{qstart.date()} to {qend.date()}";rows.append(row)
    return selections,pd.DataFrame(rows)

st.title("💰 VAST CASH"); st.subheader("MAXPROFIT • TOP 10 DECISION ENGINE")
st.write("MAXPROFIT builds stock fingerprints, finds recurring historical patterns, ranks the Top 10, estimates hold time, and can deploy YES decisions into the connected **Alpaca PAPER account only**.")
capital=st.number_input("Simulation starting money",min_value=100.0,value=1000.0,step=100.0);test_days=st.number_input("Historical test length (days)",min_value=365,max_value=3650,value=730,step=30)
c1,c2=st.columns(2)
with c1: quick=st.button("⚡ FIND TOP 10 NOW",type="primary",width="stretch")
with c2: full=st.button("⚔️ RUN FULL MAXPROFIT DISCOVERY",width="stretch")

if quick or full:
    if not alpaca_headers():st.error("Paper credentials are not available. Check Streamlit Secrets.");st.stop()
    now=datetime.now(timezone.utc);end=now.date();requested_start=(now-timedelta(days=int(test_days))).date();data_start=(now-timedelta(days=int(test_days)+1200)).date()
    st.subheader("📡 Market scan");histories=load_all_histories(data_start.isoformat(),end.isoformat())
    if not histories:st.error("No usable market history was returned.");st.stop()
    lookback,analogues,buy_drop,sell_target,top_n,allocation=63,8,15,8,10,30
    if full:
        windows=quarter_windows(requested_start,end)
        if len(windows)<3:st.error("Use at least one year of history.");st.stop()
        split=max(1,int(len(windows)*.70));train,validation=windows[:split],windows[split:];train_sel,_=build_predictions(histories,train,63,8)
        stage1=[]
        for b,s in itertools.product(range(1,21),range(1,21)):stage1.append((run_strategy(histories,train,train_sel,capital,b,s,10,30)["Return %"],b,s))
        _,buy_drop,sell_target=max(stage1,key=lambda x:x[0]);stage2=[]
        for n,a in itertools.product([5,10,15],[10,20,30,40,50]):stage2.append((run_strategy(histories,train,train_sel,capital,buy_drop,sell_target,n,a)["Return %"],n,a))
        _,top_n,allocation=max(stage2,key=lambda x:x[0]);stage3=[]
        for lb,an in itertools.product([42,63,84],[4,8,12]):
            sel,_=build_predictions(histories,train,lb,an);stage3.append((run_strategy(histories,train,sel,capital,buy_drop,sell_target,top_n,allocation)["Return %"],lb,an))
        _,lookback,analogues=max(stage3,key=lambda x:x[0]);val_sel,_=build_predictions(histories,validation,lookback,analogues);val_result=run_strategy(histories,validation,val_sel,capital,buy_drop,sell_target,top_n,allocation)
        st.success(f"Discovery complete: BUY pullback {buy_drop}%, SELL +{sell_target}%, Top {top_n}, allocation {allocation}%, lookback {lookback}, analogues {analogues}. Unseen validation {val_result['Return %']:.2f}%.")
    as_of=max(df.index.max() for df in histories.values());ranked=[]
    for ticker,df in histories.items():
        track=current_track(df,as_of,lookback,analogues,buy_drop,sell_target)
        if track:ranked.append((ticker,track))
    ranked.sort(key=lambda x:(x[1]["prediction"],x[1]["positive"],-x[1]["uncertainty"]),reverse=True);ranked=ranked[:10]
    st.subheader("🔥 MAXPROFIT TOP 10 • YES = DEPLOY TO PAPER BROKERAGE")
    st.caption(f"Model date: {as_of.date()} • BUY reference: {buy_drop}% pullback • SELL target: +{sell_target}% from actual filled purchase price • PAPER ONLY")
    if "paper_choices" not in st.session_state:st.session_state.paper_choices={}
    if "paper_orders" not in st.session_state:st.session_state.paper_orders={}
    for rank,(ticker,t) in enumerate(ranked,1):
        sell_date=next_business_date(as_of,t["hold_days"]);cols=st.columns([.4,.7,1,1,1,1,1.35,1.7]);cols[0].markdown(f"### #{rank}");cols[1].markdown(f"### {ticker}");cols[2].metric("Predicted",f"{t['prediction']*100:.1f}%");cols[3].metric("Buy Trigger",f"${t['buy_trigger']:.2f}");cols[4].metric("Target Ref.",f"${t['target_reference']:.2f}");cols[5].metric("Hold",f"~{t['hold_days']} days");cols[6].markdown(f"**Suggested sell:** {sell_date}\n\n{t['status']}")
        yes_key,no_key=f"yes_{ticker}",f"no_{ticker}"
        if cols[7].button("✅ BUY YES",key=yes_key,use_container_width=True):
            if ticker not in st.session_state.paper_orders:
                ok,msg,info=paper_buy_with_target(ticker,allocation,len(ranked),sell_target);st.session_state.paper_choices[ticker]="YES";st.session_state.paper_orders[ticker]={"ok":ok,"message":msg,"info":info}
            else:st.session_state.paper_choices[ticker]="YES"
        if cols[7].button("❌ BUY NO",key=no_key,use_container_width=True):st.session_state.paper_choices[ticker]="NO"
        choice=st.session_state.paper_choices.get(ticker,"UNDECIDED");st.write(f"**Paper decision:** {choice} | Current ${t['price']:.2f} | Historical positive rate {t['positive']*100:.0f}% | Uncertainty ±{t['uncertainty']*100:.1f}% | Matches {t['samples']}")
        if ticker in st.session_state.paper_orders:
            info=st.session_state.paper_orders[ticker];(st.success if info["ok"] else st.error)(info["message"])
        st.write(f"**Why it ranked:** 3M momentum {t['details']['Momentum']*100:.1f}%, recent 20D {t['details']['Recent 20D']*100:.1f}%, volatility {t['details']['Volatility']*100:.1f}%, drawdown {t['details']['Drawdown']*100:.1f}%, volume ratio {t['details']['Volume Ratio']:.2f}x. Best analogue {t['best']*100:.1f}%, worst {t['worst']*100:.1f}%.");st.divider()
    if ranked:
        ticker,t=ranked[0];st.subheader(f"👑 #1: {ticker}");st.write(f"{ticker} ranks #1 because its current fingerprint has the strongest modelled historical outcome in the scanned universe, supported by {t['samples']} matches and a {t['positive']*100:.0f}% positive match rate. Estimated hold: {t['hold_days']} trading days.")
    if full:
        st.subheader("🔬 Discovery results");st.write(f"Final settings: buy pullback {buy_drop}%, sell target +{sell_target}%, Top {top_n}, allocation {allocation}%, lookback {lookback}, analogues {analogues}.");st.metric("Unseen validation return",f"{val_result['Return %']:.2f}%")

st.success("🔒 PAPER MODE LOCKED: YES sends a market BUY to the Alpaca PAPER account using the configured allocation. After the fill, VAST CASH places a GTC limit SELL at the configured percentage above the actual fill price. NO sends nothing. Live trading is not enabled.")