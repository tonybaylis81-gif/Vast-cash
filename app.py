import os,time
from collections.abc import Mapping
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,requests,streamlit as st
st.set_page_config(page_title='VAST CASH',page_icon='⚒️',layout='wide')
PAPER_ONLY=True; SIM_MODE=True; SIMULATED_BUYING_POWER=1000000.00; DATA='https://data.alpaca.markets'; TRADE='https://paper-api.alpaca.markets'; TARGET=0.10
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
    # Prefer matched PAPER credentials. This avoids mixing unrelated key/secret pairs.
    pairs=[('PAPER_API_KEY','PAPER_API_SECRET'),('ALPACA_API_KEY_ID','ALPACA_API_SECRET'),('ALPACA_API_KEY','ALPACA_SECRET_KEY')]
    for key_name,secret_name in pairs:
        k=secret([key_name])
        s=secret([secret_name])
        if k and s:
            return {'APCA-API-KEY-ID':k,'APCA-API-SECRET-KEY':s}
    return None

@st.cache_data(ttl=21600,show_spinner=False)
def all_hist():
    h=hdr()
    if not h:
        return {}, 'Alpaca PAPER credentials are unavailable.'
    # Use a completed market day as the end date. Alpaca's Basic market-data
    # access can restrict very recent historical requests, so do not ask for
    # today's still-forming bar.
    end_date=(datetime.now(timezone.utc).date()-timedelta(days=1))
    start_date=end_date-timedelta(days=240)
    base_params={
        'symbols':','.join(UNIVERSE),
        'timeframe':'1Day',
        'start':start_date.isoformat(),
        'end':end_date.isoformat(),
        'limit':10000,
        'adjustment':'all',
        'feed':'iex',
        'sort':'asc'
    }
    try:
        all_raw={}
        page_token=None
        pages=0
        while True:
            params=dict(base_params)
            if page_token:
                params['page_token']=page_token
            r=requests.get(f'{DATA}/v2/stocks/bars',headers=h,params=params,timeout=30)
            if r.status_code!=200:
                try:
                    detail=r.json()
                except Exception:
                    detail=r.text[:500]
                return {}, f'Alpaca market-data HTTP {r.status_code}: {detail}'
            payload=r.json()
            raw=payload.get('bars') or {}
            for symbol,bars in raw.items():
                if bars:
                    all_raw.setdefault(symbol,[]).extend(bars)
            pages+=1
            page_token=payload.get('next_page_token')
            if not page_token or pages>=20:
                break

        out={}
        for s,b in all_raw.items():
            if not b:
                continue
            d=pd.DataFrame(b)[['t','o','h','c','l']]
            d.columns=['date','open','high','close','low']
            d.date=pd.to_datetime(d.date,utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
            d=d.drop_duplicates(subset=['date']).set_index('date').sort_index()
            d=d.apply(pd.to_numeric,errors='coerce').dropna()
            if len(d)>=140:
                out[s]=d

        if not out:
            return {}, (
                f'Alpaca returned HTTP 200 but no symbols had at least 140 usable daily bars '
                f'for {start_date.isoformat()} through {end_date.isoformat()} using the IEX historical feed. '
                f'Raw symbols returned: {list(all_raw)[:10]}.'
            )
        return out, f'Loaded {sum(len(v) for v in out.values())} daily bars across {len(out)} symbols.'
    except requests.RequestException as e:
        return {}, f'Network error while requesting Alpaca historical data: {e}'
    except Exception as e:
        return {}, f'Unexpected historical-data error: {e}'
def score(s,d,buy_drop):
    if len(d)<140:return None
    wins=[];rets=[];hs=[];start=70;stop=len(d)-2;step=max(1,(stop-start)//90)
    for i in range(start,stop,step):
        prior=d.iloc[max(0,i-60):i];trigger=float(prior['high'].max())*(1-buy_drop/100);fut=d.iloc[i:i+31];hits=np.where(fut['low'].to_numpy()<=trigger)[0]
        if not len(hits):continue
        a=fut.iloc[int(hits[0]):int(hits[0])+31];th=np.where(a['high'].to_numpy()>=trigger*(1+TARGET))[0]
        if len(th):wins.append(1);rets.append(TARGET);hs.append(max(1,int(th[0])))
        else:wins.append(0);rets.append(float(a.close.iloc[-1]/trigger-1));hs.append(len(a)-1)
    if len(rets)<4:return None
    c=d.close;price=float(c.iloc[-1]);recent=float(c.tail(60).max());trigger=recent*(1-buy_drop/100);win=float(np.mean(wins));ret=float(np.median(rets));vol=float(c.pct_change().dropna().tail(30).std()*np.sqrt(252));momentum=float(price/c.iloc[-21]-1);typical=max(1,int(round(np.mean(hs))))
    return {'Ticker':s,'Expected Return':ret,'Win Rate':win,'Historical Trades':len(rets),'Typical Hold':typical,'Price':price,'Buy Trigger':trigger,'Sell Target':trigger*(1+TARGET),'Momentum':momentum,'Volatility':vol,'Score':ret*100+win*20-vol*5}
def backtest_strategy(hist,buy_drop,starting_cash=SIMULATED_BUYING_POWER):
    # Out-of-sample style walk-forward test: each entry uses only data available before that day.
    cash=float(starting_cash); trades=[]; equity_curve=[]
    for s,d in hist.items():
        if len(d)<140: continue
        for i in range(70,len(d)-31):
            prior=d.iloc[max(0,i-60):i]
            trigger=float(prior['high'].max())*(1-buy_drop/100)
            fut=d.iloc[i:i+31]
            hits=np.where(fut['low'].to_numpy()<=trigger)[0]
            if not len(hits): continue
            entry_i=int(hits[0]); entry=trigger
            a=fut.iloc[entry_i:entry_i+31]
            target=entry*(1+TARGET)
            th=np.where(a['high'].to_numpy()>=target)[0]
            if len(th):
                exit_price=target; hold=max(1,int(th[0])); outcome='WIN'
            else:
                exit_price=float(a['close'].iloc[-1]); hold=max(1,len(a)-1); outcome='LOSS' if exit_price<entry else 'OPEN/FLAT'
            ret=exit_price/entry-1
            trades.append({'Ticker':s,'Entry':entry,'Exit':exit_price,'Return':ret,'Hold Days':hold,'Outcome':outcome})
    if not trades: return pd.DataFrame(),{}
    td=pd.DataFrame(trades).sort_values(['Ticker','Hold Days']).reset_index(drop=True)
    # Equal-dollar test: every trade receives the same fraction of starting capital.
    allocation=min(0.10,1/max(1,len(hist)))
    equity=float(starting_cash); peak=equity; max_dd=0.0
    for ret in td['Return']:
        equity*=1+allocation*float(ret)
        peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak if peak else 0)
    wins=int((td['Return']>0).sum()); losses=int((td['Return']<=0).sum())
    stats={'Starting Capital':starting_cash,'Ending Capital':equity,'Net Profit':equity-starting_cash,'Total Trades':len(td),'Wins':wins,'Losses':losses,'Win Rate':wins/len(td),'Max Drawdown':max_dd,'Avg Trade Return':float(td['Return'].mean()),'Median Trade Return':float(td['Return'].median()),'Best Trade':float(td['Return'].max()),'Worst Trade':float(td['Return'].min())}
    return td,stats

def nextday(n):
    d=datetime.now().date();c=0
    while c<n:
        d+=timedelta(days=1)
        if d.weekday()<5:c+=1
    return d.isoformat()
def account():
    if SIM_MODE:
        if 'sim_cash' not in st.session_state:st.session_state.sim_cash=SIMULATED_BUYING_POWER
        if 'sim_positions' not in st.session_state:st.session_state.sim_positions=[]
        invested=sum(float(p.get('Cost',0) or 0) for p in st.session_state.sim_positions)
        equity=float(st.session_state.sim_cash)+invested
        return {'status':'SIMULATED','cash':str(st.session_state.sim_cash),'equity':str(equity),'buying_power':str(st.session_state.sim_cash),'portfolio_value':str(equity)}
    h=hdr()
    if not h:return None
    try:
        r=requests.get(f'{TRADE}/v2/account',headers=h,timeout=10)
        if r.status_code==200:return r.json()
    except Exception:pass
    return None

def account_diagnostics():
    h=hdr()
    if not h:return None,'Alpaca PAPER credentials unavailable.'
    try:
        r=requests.get(f'{TRADE}/v2/account',headers=h,timeout=10)
        if r.status_code==200:
            return r.json(),'LIVE ALPACA PAPER ACCOUNT'
    except Exception:pass
    return {
        'status':'SIMULATED',
        'cash':str(SIMULATED_BUYING_POWER),
        'equity':str(SIMULATED_BUYING_POWER),
        'buying_power':str(SIMULATED_BUYING_POWER),
        'portfolio_value':str(SIMULATED_BUYING_POWER)
    },'LOCAL VAST CASH SIMULATION'

def reconcile_pending():
    if SIM_MODE:return []
    h=hdr()
    if not h:return []
    done=[]
    pending=st.session_state.get('pending_orders',{})
    for oid,symbol in list(pending.items()):
        try:
            z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code!=200:continue
            o=z.json();status=o.get('status','');fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
            if status in ('filled','partially_filled') and fill>0 and fq>0:
                target=round(fill*(1+TARGET),2)
                existing=requests.get(f'{TRADE}/v2/orders',headers=h,params={'status':'open','symbols':symbol,'limit':100},timeout=10)
                has_sell=False
                if existing.status_code==200:
                    for so in existing.json():
                        if so.get('side')=='sell' and so.get('symbol')==symbol:has_sell=True;break
                if not has_sell:
                    x=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(fq),'side':'sell','type':'limit','limit_price':f'{target:.2f}','time_in_force':'gtc'},timeout=15)
                    if x.status_code not in (200,201):continue
                done.append(f'{symbol}: filled {fq} @ ${fill:.2f}; +10% GTC sell at ${target:.2f} is active.')
                del pending[oid]
            elif status in ('canceled','expired','rejected','done_for_day'):del pending[oid]
        except Exception:continue
    return done

def buy(symbol,budget):
    if SIM_MODE:
        h=hdr()
        if not h:return False,'Alpaca market-data credentials unavailable.'
        try:
            q=requests.get(f'{DATA}/v2/stocks/quotes/latest',headers=h,params={'symbols':symbol,'feed':'iex'},timeout=10)
            ask=0
            if q.status_code==200:
                quote=(q.json().get('quotes',{}) or {}).get(symbol,{})
                ask=float(quote.get('ap') or 0) or float(quote.get('bp') or 0)
            if ask<=0:
                snap=requests.get(f'{DATA}/v2/stocks/{symbol}/snapshot',headers=h,params={'feed':'iex'},timeout=10)
                if snap.status_code==200:
                    sd=snap.json();ask=float((sd.get('latestTrade') or {}).get('p') or (sd.get('dailyBar') or {}).get('c') or 0)
            if ask<=0:return False,f'No usable market price for {symbol}.'
            cash=float(st.session_state.get('sim_cash',SIMULATED_BUYING_POWER))
            qty=int(min(budget,cash)/ask)
            if qty<1:return False,f'SIM BUY skipped for {symbol}: available simulated cash is ${cash:,.2f}.'.replace('$','\\$')
            cost=round(qty*ask,2)
            if cost>cash:return False,f'SIM BUY rejected for {symbol}: cost ${cost:,.2f} exceeds simulated cash ${cash:,.2f}.'.replace('$','\\$')
            target=round(ask*(1+TARGET),2)
            st.session_state.sim_cash=round(cash-cost,2)
            positions=st.session_state.setdefault('sim_positions',[])
            positions.append({'Ticker':symbol,'Quantity':qty,'Entry Price':round(ask,2),'Cost':cost,'Target Price':target,'Target Gain':f'+{TARGET:.0%}','Status':'OPEN'})
            st.session_state.setdefault('sim_orders',[]).append({'Ticker':symbol,'Side':'BUY','Quantity':qty,'Fill Price':round(ask,2),'Cost':cost,'Target Price':target,'Status':'FILLED'})
            return True,f'SIM BUY {symbol}: {qty} shares @ ${ask:.2f}. Simulated +10% target: ${target:.2f}. NO ALPACA ORDER SENT.'.replace('$','\\$')
        except Exception as e:return False,f'SIM trade error: {e}'
    h=hdr()
    if not h:return False,'Alpaca PAPER credentials unavailable.'
    try:
        q=requests.get(f'{DATA}/v2/stocks/quotes/latest',headers=h,params={'symbols':symbol,'feed':'iex'},timeout=10);ask=0
        if q.status_code==200:
            quotes=q.json().get('quotes',{});quote=quotes.get(symbol,{})
            ask=float(quote.get('ap') or 0);bid=float(quote.get('bp') or 0);ask=ask or bid
        if ask<=0:
            snap=requests.get(f'{DATA}/v2/stocks/{symbol}/snapshot',headers=h,params={'feed':'iex'},timeout=10)
            if snap.status_code==200:
                sd=snap.json();ask=float((sd.get('latestTrade') or {}).get('p') or (sd.get('dailyBar') or {}).get('c') or 0)
        qty=max(1,int(budget/ask)) if ask>0 else 0
        if not qty:return False,f'No usable price for {symbol}.'
        r=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(qty),'side':'buy','type':'market','time_in_force':'day'},timeout=15)
        if r.status_code not in (200,201):return False,f'PAPER BUY rejected: {r.text[:200]}'
        oid=r.json()['id'];st.session_state.setdefault('pending_orders',{})[oid]=symbol;fill=0;fq=0
        for _ in range(12):
            time.sleep(1);z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code==200:
                o=z.json();fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
                if fill>0 and fq>0:break
        if not fill:return True,f'PAPER BUY submitted for {symbol}; fill still pending. VAST CASH will reconcile it automatically.'
        target=round(fill*(1+TARGET),2)
        x=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(fq),'side':'sell','type':'limit','limit_price':f'{target:.2f}','time_in_force':'gtc'},timeout=15)
        msg=f'PAPER BUY {symbol}: {fq} shares @ ${fill:.2f}. AUTO-SELL at +10% (${target:.2f}) until target is reached.'
        return True,msg if x.status_code in (200,201) else msg+' WARNING: target order was not accepted.'
    except Exception as e:return False,f'PAPER trade error: {e}'
def show_account_diagnostics():
    st.header('🧾 PAPER ACCOUNT DIAGNOSTICS')
    st.caption('LOCAL SIMULATED PAPER MODE: $1,000,000 starting buying power.')
    if 'sim_cash' not in st.session_state:st.session_state.sim_cash=SIMULATED_BUYING_POWER
    if 'sim_positions' not in st.session_state:st.session_state.sim_positions=[]
    invested=sum(float(p.get('market_value',0) or 0) for p in st.session_state.sim_positions)
    equity=st.session_state.sim_cash+invested
    cols=st.columns(4)
    cols[0].metric('💵 Cash',f"${st.session_state.sim_cash:,.2f}")
    cols[1].metric('💰 Equity',f"${equity:,.2f}")
    cols[2].metric('🏦 Buying Power',f"${st.session_state.sim_cash:,.2f}")
    cols[3].metric('📊 Portfolio Value',f"${equity:,.2f}")
    st.success('ACCOUNT MODE: LOCAL VAST CASH SIMULATION')
    if st.session_state.sim_positions:
        st.subheader('📈 Simulated Positions')
        st.dataframe(pd.DataFrame(st.session_state.sim_positions),use_container_width=True,hide_index=True)
    else:st.info('No simulated positions yet.')
st.title('⚒️ VAST CASH');
try:
    reconciled=reconcile_pending()
    for msg in reconciled:st.success('🔄 '+msg)
except Exception:
    pass
show_account_diagnostics()
st.divider()
st.subheader('STOCK TRADING FOR WELDERS')
st.caption('MAXPROFIT does the math. You make YES / NO. $1M LOCAL SIMULATION. NO ALPACA ORDERS.')
with st.sidebar:
    buy_drop=st.slider('Buy % below recent high',1,20,15);capital=st.number_input('Paper capital ($)',100.,1000000.,1000000.,100.);allocation=st.slider('Capital used for YES selections (%)',5,100,50,5)
    st.caption('SIM EXIT: record a simulated +10% target above the simulated fill price. No Alpaca PAPER order is submitted.')
if 'top10' not in st.session_state:st.session_state.top10=None
if 'decisions' not in st.session_state:st.session_state.decisions={}
if 'pending_orders' not in st.session_state:st.session_state.pending_orders={}
if st.button('⚡ RUN MAXPROFIT',type='primary',use_container_width=True):
    if not hdr():
        st.error('Alpaca PAPER credentials are not available. Check Streamlit Secrets.')
    else:
        runner=st.empty()
        status=st.empty()
        progress=st.progress(0,text='MAXPROFIT is starting...')
        try:
            runner.markdown('## 🏃‍♂️💨 **MAXPROFIT RUNNING**')
            status.info('🏃 Fetching historical market data from Alpaca PAPER...')
            hist, hist_status=all_hist()
            if not hist:
                progress.progress(0,text='MAXPROFIT stopped')
                runner.markdown('## 🛑 **MAXPROFIT STOPPED**')
                st.error(f'No historical market data was returned. {hist_status}')
                st.code(hist_status, language='text')
                st.info('MAXPROFIT did not submit or modify any order. This is a market-data retrieval failure.')
            else:
                ranked=[]
                total=len(hist)
                for i,(s,d) in enumerate(hist.items(),1):
                    runner.markdown(f'## 🏃‍♂️💨 **MAXPROFIT RUNNING**  ·  {s}  ·  {i}/{total}')
                    status.info(f'🔧 Testing {s}: historical buy triggers, +10% target hits, win rate and volatility...')
                    x=score(s,d,buy_drop)
                    if x:ranked.append(x)
                    progress.progress(i/total,text=f'Calculating MAXPROFIT: {i}/{total} stocks')
                ranked.sort(key=lambda x:(x['Expected Return'],x['Win Rate'],x['Score']),reverse=True)
                st.session_state.top10=ranked[:10]
                st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
                runner.markdown('## ✅ **MAXPROFIT COMPLETE**')
                st.caption(f'📊 {hist_status}')
                status.success(f'Finished testing {total} stocks. {len(ranked)} produced usable historical setups.')
                progress.progress(1.0,text='MAXPROFIT calculation complete')
        except Exception as e:
            runner.markdown('## 🛑 **MAXPROFIT ERROR**')
            st.error(f'MAXPROFIT encountered an error: {e}')
if st.session_state.top10:
    top=st.session_state.top10;st.success(f'MAXPROFIT found Top {len(top)} historical setups.');st.header('🏆 TOP 10 — YOUR DECISION')
    for i,x in enumerate(top,1):
        t=x['Ticker']
        with st.container(border=True):
            a,b,c,d=st.columns([.6,1.2,1.4,1.4]);a.metric('#',i);b.metric('STOCK',t);c.metric('Historical return',f"{x['Expected Return']:+.1%}");d.metric('Win rate',f"{x['Win Rate']:.0%}")
            st.write(f"**Typical historical target time:** {x['Typical Hold']} trading days • **Current:** ${x['Price']:.2f}")
            st.write(f"**Buy trigger:** ${x['Buy Trigger']:.2f} • **AUTO-SELL:** +10% above actual average fill • **Tests:** {x['Historical Trades']}")
            l,r=st.columns(2)
            if l.button('✅ YES',key=f'yes_{t}',use_container_width=True):st.session_state.decisions[t]='YES'
            if r.button('❌ NO',key=f'no_{t}',use_container_width=True):st.session_state.decisions[t]='NO'
            if st.session_state.decisions.get(t)=='YES':st.success('YES selected')
            elif st.session_state.decisions.get(t)=='NO':st.info('NO selected')
            else:st.warning('Not decided')
    complete=all(st.session_state.decisions.get(x['Ticker']) in ('YES','NO') for x in top);yes=[x for x in top if st.session_state.decisions.get(x['Ticker'])=='YES'];st.metric('YES selections',f'{len(yes)} / {len(top)}')
    if st.button('🚀 COMMIT SELECTED TO SIMULATION',type='primary',disabled=not complete,use_container_width=True):
        if not yes:st.info('All NO. Nothing sent.')
        else:
            ac=account()
            if not ac:st.error('Could not access Alpaca PAPER account.')
            else:
                buying_power=float(ac.get('buying_power',0) or 0)
                budget=buying_power*allocation/100/len(yes);st.subheader('📨 PAPER ORDERS')
                for x in yes:
                    ok,msg=buy(x['Ticker'],budget)
                    if ok:st.success(msg)
                    else:st.error(msg)
st.divider();st.caption('🔒 LOCAL SIMULATION ONLY. Starting bankroll: $1,000,000. +10% target uses the simulated fill price. No Alpaca orders are submitted.')