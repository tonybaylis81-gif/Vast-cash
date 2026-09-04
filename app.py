import os,time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,requests,streamlit as st
st.set_page_config(page_title='VAST CASH',page_icon='⚒️',layout='wide')
PAPER_ONLY=True; DATA='https://data.alpaca.markets'; TRADE='https://paper-api.alpaca.markets'; TARGET=0.10
UNIVERSE=['AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','AVGO','TSLA','AMD','NFLX','ORCL','CRM','ADBE','QCOM','INTC','MU','AMAT','LRCX','TXN','JPM','BAC','WFC','GS','MS','V','MA','C','JNJ','UNH','XOM','CVX','COST','WMT','HD','LOW','CAT','GE','BA','DIS']
def secret(names):
    names={x.upper() for x in names}
    try:
        def walk(v):
            if isinstance(v,Mapping):
                for k,x in v.items():
                    if str(k).upper() in names and x is not None and str(x).strip(): return str(x).strip()
                    y=walk(x)
                    if y:return y
        x=walk(st.secrets)
        if x:return x
    except Exception:pass
    for n in names:
        x=os.getenv(n)
        if x and x.strip():return x.strip()
def hdr():
    k=secret(['PAPER_API_KEY','ALPACA_API_KEY','ALPACA_API_KEY_ID','API_KEY']); s=secret(['PAPER_API_SECRET','ALPACA_SECRET_KEY','ALPACA_API_SECRET','API_SECRET','SECRET_KEY'])
    return {'APCA-API-KEY-ID':k,'APCA-API-SECRET-KEY':s} if k and s else None
@st.cache_data(ttl=21600,show_spinner=False)
def hist(symbol):
    h=hdr()
    if not h:return None
    e=datetime.now(timezone.utc).date(); q={'timeframe':'1Day','start':(e-timedelta(days=420)).isoformat(),'end':e.isoformat(),'limit':10000,'adjustment':'all','feed':'iex','sort':'asc'}
    try:
        r=requests.get(f'{DATA}/v2/stocks/{symbol}/bars',headers=h,params=q,timeout=12)
        if r.status_code!=200:return None
        b=r.json().get('bars',[])
        if not b:return None
        d=pd.DataFrame(b)[['t','o','h','c']]; d.columns=['date','open','high','close']; d.date=pd.to_datetime(d.date,utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
        return d.set_index('date').sort_index().apply(pd.to_numeric,errors='coerce').dropna()
    except Exception:return None
def all_hist():
    out={}
    with ThreadPoolExecutor(max_workers=6) as p:
        fs={p.submit(hist,s):s for s in UNIVERSE}
        for f in as_completed(fs):
            try:
                d=f.result()
                if d is not None and len(d)>=140:out[fs[f]]=d
            except Exception:pass
    return out
def score(s,d,hold,buy_drop):
    if len(d)<140:return None
    wins=[];rets=[];hs=[];start=70;stop=len(d)-hold-2;step=max(1,(stop-start)//90)
    for i in range(start,stop,step):
        prior=d.iloc[max(0,i-60):i];trigger=float(prior.high.max())*(1-buy_drop/100);fut=d.iloc[i:i+hold+1];hits=np.where(fut.low.to_numpy()<=trigger)[0]
        if not len(hits):continue
        a=fut.iloc[int(hits[0]):int(hits[0])+hold+1];th=np.where(a.high.to_numpy()>=trigger*1.10)[0]
        if len(th):wins.append(1);rets.append(.10);hs.append(max(1,int(th[0])))
        else:wins.append(0);rets.append(float(a.close.iloc[-1]/trigger-1));hs.append(len(a)-1)
    if len(rets)<4:return None
    c=d.close;price=float(c.iloc[-1]);recent=float(c.tail(60).max());trigger=recent*(1-buy_drop/100);win=float(np.mean(wins));ret=float(np.median(rets));vol=float(c.pct_change().dropna().tail(30).std()*np.sqrt(252));momentum=float(price/c.iloc[-21]-1);typical=max(1,min(hold,int(round(np.mean(hs)))))
    return {'Ticker':s,'Expected Return':ret,'Win Rate':win,'Historical Trades':len(rets),'Typical Hold':typical,'Price':price,'Buy Trigger':trigger,'Sell Target':trigger*1.10,'Momentum':momentum,'Volatility':vol,'Score':ret*100+win*20-vol*5}
def nextday(n):
    d=datetime.now().date();c=0
    while c<n:
        d+=timedelta(days=1)
        if d.weekday()<5:c+=1
    return d.isoformat()
def account():
    h=hdr()
    if not h:return None
    try:
        r=requests.get(f'{TRADE}/v2/account',headers=h,timeout=10);return r.json() if r.status_code==200 else None
    except Exception:return None
def buy(symbol,budget,hold):
    h=hdr()
    if not h:return False,'Alpaca PAPER credentials unavailable.'
    try:
        q=requests.get(f'{DATA}/v2/stocks/{symbol}/quotes/latest',headers=h,timeout=10);ask=float(q.json()['quote']['ap']) if q.status_code==200 else 0;qty=max(1,int(budget/ask)) if ask>0 else 0
        if not qty:return False,f'No usable price for {symbol}.'
        r=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(qty),'side':'buy','type':'market','time_in_force':'day'},timeout=15)
        if r.status_code not in (200,201):return False,f'PAPER BUY rejected: {r.text[:200]}'
        oid=r.json()['id'];fill=0;fq=0
        for _ in range(12):
            time.sleep(1);z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code==200:
                o=z.json();fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
                if fill>0 and fq>0:break
        if not fill:return True,f'PAPER BUY submitted for {symbol}; fill still pending.'
        target=round(fill*(1+TARGET),2)
        x=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(fq),'side':'sell','type':'limit','limit_price':f'{target:.2f}','time_in_force':'gtc'},timeout=15)
        msg=f'PAPER BUY {symbol}: {fq} shares @ ${fill:.2f}. AUTO-SELL at +10% (${target:.2f}). Time exit: {nextday(hold)}.'
        return True,msg if x.status_code in (200,201) else msg+' WARNING: target order was not accepted.'
    except Exception as e:return False,f'PAPER trade error: {e}'
st.title('⚒️ VAST CASH');st.subheader('STOCK TRADING FOR WELDERS');st.caption('MAXPROFIT does the math. You make YES / NO. PAPER ONLY.')
with st.sidebar:
    hold=st.slider('Maximum hold (trading days)',1,30,4);buy_drop=st.slider('Buy % below recent high',1,20,15);capital=st.number_input('Paper capital ($)',100.,1000000.,1000.,100.);allocation=st.slider('Capital used for YES selections (%)',5,100,50,5)
    st.caption('EXIT: first condition wins, +10% above actual filled purchase price or the suggested hold-date exit.')
if 'top10' not in st.session_state:st.session_state.top10=None
if 'decisions' not in st.session_state:st.session_state.decisions={}
if st.button('⚡ RUN MAXPROFIT',type='primary',use_container_width=True):
    if not hdr():st.error('Alpaca PAPER credentials are not available. Check Streamlit Secrets.')
    else:
        with st.spinner('MAXPROFIT is testing historical fingerprints...'):
            ranked=[]
            for s,d in all_hist().items():
                x=score(s,d,hold,buy_drop)
                if x:ranked.append(x)
            ranked.sort(key=lambda x:(x['Expected Return'],x['Win Rate'],x['Score']),reverse=True);st.session_state.top10=ranked[:10];st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
if st.session_state.top10:
    top=st.session_state.top10;st.success(f'MAXPROFIT found Top {len(top)} historical setups.');st.header('🏆 TOP 10 — YOUR DECISION')
    for i,x in enumerate(top,1):
        t=x['Ticker']
        with st.container(border=True):
            a,b,c,d=st.columns([.6,1.2,1.4,1.4]);a.metric('#',i);b.metric('STOCK',t);c.metric('Historical return',f"{x['Expected Return']:+.1%}");d.metric('Win rate',f"{x['Win Rate']:.0%}")
            st.write(f"**Hold:** {x['Typical Hold']} days • **Suggested sell date:** {nextday(x['Typical Hold'])} • **Current:** ${x['Price']:.2f}")
            st.write(f"**Buy trigger:** ${x['Buy Trigger']:.2f} • **AUTO-SELL:** +10% above actual fill • **Tests:** {x['Historical Trades']}")
            l,r=st.columns(2)
            if l.button('✅ YES',key=f'yes_{t}',use_container_width=True):st.session_state.decisions[t]='YES'
            if r.button('❌ NO',key=f'no_{t}',use_container_width=True):st.session_state.decisions[t]='NO'
            if st.session_state.decisions.get(t)=='YES':st.success('YES selected')
            elif st.session_state.decisions.get(t)=='NO':st.info('NO selected')
            else:st.warning('Not decided')
    complete=all(st.session_state.decisions.get(x['Ticker']) in ('YES','NO') for x in top);yes=[x for x in top if st.session_state.decisions.get(x['Ticker'])=='YES'];st.metric('YES selections',f'{len(yes)} / {len(top)}')
    if st.button('🚀 COMMIT SELECTED TO PAPER',type='primary',disabled=not complete,use_container_width=True):
        if not yes:st.info('All NO. Nothing sent.')
        else:
            ac=account()
            if not ac:st.error('Could not access Alpaca PAPER account.')
            else:
                budget=float(ac.get('buying_power',0))*allocation/100/len(yes);st.subheader('📨 PAPER ORDERS')
                for x in yes:
                    ok,msg=buy(x['Ticker'],budget,x['Typical Hold']);st.success(msg) if ok else st.error(msg)
st.divider();st.caption('🔒 PAPER ONLY. +10% target is based on the actual paper fill. Live trading is disabled.')
