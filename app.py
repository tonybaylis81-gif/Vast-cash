import os,time
from collections.abc import Mapping
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,requests,streamlit as st
st.set_page_config(page_title='VAST CASH',page_icon='⚒️',layout='wide')
PAPER_ONLY=True; SIM_MODE=False; SIMULATED_BUYING_POWER=5000.00; DATA='https://data.alpaca.markets'; TRADE='https://paper-api.alpaca.markets'; TARGET=0.10
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
    # Historical walk-forward test. Each entry uses only data available before that day.
    cash=float(starting_cash); trades=[]
    for s,d in hist.items():
        if len(d)<140: continue
        for i in range(70,len(d)-31):
            prior=d.iloc[max(0,i-60):i]
            trigger=float(prior['high'].max())*(1-buy_drop/100)
            fut=d.iloc[i:i+31]
            hits=np.where(fut['low'].to_numpy()<=trigger)[0]
            if not len(hits): continue
            entry=trigger; entry_i=int(hits[0]); a=fut.iloc[entry_i:entry_i+31]
            target=entry*(1+TARGET)
            th=np.where(a['high'].to_numpy()>=target)[0]
            if len(th): exit_price=target; hold=max(1,int(th[0])); outcome='WIN'
            else: exit_price=float(a['close'].iloc[-1]); hold=max(1,len(a)-1); outcome='LOSS' if exit_price<entry else 'FLAT'
            trades.append({'Ticker':s,'Entry':entry,'Exit':exit_price,'Return':exit_price/entry-1,'Hold Days':hold,'Outcome':outcome})
    if not trades:return pd.DataFrame(),{}
    td=pd.DataFrame(trades).sort_values(['Ticker','Hold Days']).reset_index(drop=True)
    allocation=min(0.10,1/max(1,len(hist)))
    equity=float(starting_cash);peak=equity;max_dd=0.0
    for ret in td['Return']:
        equity*=1+allocation*float(ret);peak=max(peak,equity);max_dd=max(max_dd,(peak-equity)/peak if peak else 0)
    wins=int((td['Return']>0).sum());losses=int((td['Return']<=0).sum())
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
    if not h:return None,'Alpaca PAPER credentials unavailable. No real account balance is available.'
    try:
        r=requests.get(f'{TRADE}/v2/account',headers=h,timeout=10)
        if r.status_code==200:
            return r.json(),'ALPACA PAPER ACCOUNT (brokerage-reported)'
        return None,f'Alpaca PAPER account HTTP {r.status_code}: {r.text[:300]}'
    except Exception as e:
        return None,f'Alpaca PAPER account connection failed: {e}'
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

def buy(symbol,budget,buy_trigger=None):
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
        a=requests.get(f'{TRADE}/v2/account',headers=h,timeout=10)
        if a.status_code!=200:return False,f'PAPER BUY blocked: account check failed (HTTP {a.status_code}); no order submitted.'
        ac=a.json()
        cash=float(ac.get('cash',0) or 0)
        buying_power=float(ac.get('buying_power',0) or 0)
        # Never add to an existing holding or stack duplicate open buy orders.
        pos=requests.get(f'{TRADE}/v2/positions/{symbol}',headers=h,timeout=10)
        if pos.status_code==200:return False,f'PAPER BUY skipped for {symbol}: a position already exists; manage the existing position first.'
        if pos.status_code not in (200,404):return False,f'PAPER BUY blocked: could not verify existing position (HTTP {pos.status_code}).'
        open_orders=requests.get(f'{TRADE}/v2/orders',headers=h,params={'status':'open','symbols':symbol,'limit':100},timeout=10)
        if open_orders.status_code!=200:return False,f'PAPER BUY blocked: could not verify open orders (HTTP {open_orders.status_code}).'
        if any(o.get('side')=='buy' for o in open_orders.json()):return False,f'PAPER BUY skipped for {symbol}: an open buy order already exists.'
        if cash<=0 or buying_power<=0:
            return False,f'PAPER BUY blocked for {symbol}: cash ${cash:,.2f}, buying power ${buying_power:,.2f}. No order submitted.'.replace('$','\\$')
        spendable=min(float(budget),cash,buying_power)
        q=requests.get(f'{DATA}/v2/stocks/quotes/latest',headers=h,params={'symbols':symbol,'feed':'iex'},timeout=10);ask=0
        if q.status_code==200:
            quotes=q.json().get('quotes',{});quote=quotes.get(symbol,{})
            ask=float(quote.get('ap') or 0);bid=float(quote.get('bp') or 0);ask=ask or bid
        if ask<=0:
            snap=requests.get(f'{DATA}/v2/stocks/{symbol}/snapshot',headers=h,params={'feed':'iex'},timeout=10)
            if snap.status_code==200:
                sd=snap.json();ask=float((sd.get('latestTrade') or {}).get('p') or (sd.get('dailyBar') or {}).get('c') or 0)
        if ask<=0:return False,f'No usable price for {symbol}; no order submitted.'
        if buy_trigger is not None and ask>float(buy_trigger):
            return False,f'PAPER BUY skipped for {symbol}: current ask ${ask:.2f} is above buy trigger ${float(buy_trigger):.2f}.'.replace('$','\\$')
        qty=int((spendable*0.98)/ask)
        if qty<1:return False,f'PAPER BUY skipped for {symbol}: no whole share fits the safe budget of ${spendable:,.2f}.'.replace('$','\\$')
        if qty*ask>spendable:return False,'PAPER BUY blocked: estimated cost exceeds budget; no order submitted.'
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
def show_positions_and_orders():
    h=hdr()
    if not h:
        st.warning('Positions and order history unavailable until Alpaca PAPER credentials connect.')
        return
    try:
        pr=requests.get(f'{TRADE}/v2/positions',headers=h,timeout=10)
        if pr.status_code==200:
            positions=pr.json()
            st.subheader('📦 CURRENT PAPER POSITIONS')
            if positions:
                rows=[{'Symbol':p.get('symbol'),'Qty':p.get('qty'),'Side':p.get('side'),'Avg Entry':p.get('avg_entry_price'),'Current Price':p.get('current_price'),'Market Value':p.get('market_value'),'Unrealized P/L':p.get('unrealized_pl')} for p in positions]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            else:st.info('No open stock positions reported by Alpaca.')
        else:st.error(f'Could not retrieve positions (HTTP {pr.status_code}): {pr.text[:200]}')
        oo=requests.get(f'{TRADE}/v2/orders',headers=h,params={'status':'open','limit':100,'direction':'desc'},timeout=10)
        if oo.status_code==200:
            orders=oo.json()
            st.subheader('🧾 OPEN PAPER ORDERS')
            if orders:
                rows=[{'Symbol':o.get('symbol'),'Side':o.get('side'),'Qty':o.get('qty'),'Filled':o.get('filled_qty'),'Type':o.get('type'),'Limit Price':o.get('limit_price'),'Status':o.get('status'),'Submitted':o.get('submitted_at')} for o in orders]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            else:st.info('No open orders reported by Alpaca.')
        else:st.error(f'Could not retrieve open orders (HTTP {oo.status_code}): {oo.text[:200]}')
        recent=requests.get(f'{TRADE}/v2/orders',headers=h,params={'status':'all','limit':20,'direction':'desc','nested':'true'},timeout=10)
        if recent.status_code==200:
            orders=recent.json()
            st.subheader('🕘 RECENT PAPER ORDER HISTORY')
            if orders:
                rows=[{'Symbol':o.get('symbol'),'Side':o.get('side'),'Qty':o.get('qty'),'Filled':o.get('filled_qty'),'Avg Fill':o.get('filled_avg_price'),'Type':o.get('type'),'Status':o.get('status'),'Submitted':o.get('submitted_at')} for o in orders]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            else:st.info('No recent orders returned by Alpaca.')
        else:st.error(f'Could not retrieve recent order history (HTTP {recent.status_code}): {recent.text[:200]}')
    except Exception as e:st.error(f'Positions/order diagnostics failed: {e}')

def show_account_diagnostics():
    st.header('🧾 PAPER ACCOUNT DIAGNOSTICS')
    ac,msg=account_diagnostics()
    if not ac:
        st.error(msg)
        st.info('No brokerage balance is being invented or displayed. Check Streamlit Secrets and the Alpaca PAPER account connection.')
        return
    cols=st.columns(4)
    cols[0].metric('💵 Cash',f"${float(ac.get('cash',0) or 0):,.2f}")
    cols[1].metric('💰 Equity',f"${float(ac.get('equity',0) or 0):,.2f}")
    cols[2].metric('🏦 Buying Power',f"${float(ac.get('buying_power',0) or 0):,.2f}")
    cols[3].metric('📊 Portfolio Value',f"${float(ac.get('portfolio_value',0) or 0):,.2f}")
    st.success('ACCOUNT MODE: ALPACA PAPER ACCOUNT')
    st.caption('Values above are reported by the Alpaca paper-trading API.')
    if float(ac.get('cash',0) or 0)<0 or float(ac.get('buying_power',0) or 0)<=0:
        st.error('TRADING PAUSED: cash is negative or buying power is zero. New buys are blocked until the account is reconciled.')

st.title('⚒️ VAST CASH');
try:
    reconciled=reconcile_pending()
    for msg in reconciled:st.success('🔄 '+msg)
except Exception:
    pass
show_account_diagnostics()
with st.expander('🔎 Inspect positions, open orders and recent order history',expanded=True):
    show_positions_and_orders()
st.divider()
st.subheader('STOCK TRADING FOR WELDERS')
st.caption('MAXPROFIT ranks historical setups. YES / NO is your decision. Orders, if submitted, go to the ALPACA PAPER account only.')
with st.sidebar:
    buy_drop=st.slider('Buy % below recent high',1,20,15);capital=st.number_input('Backtest starting capital ($)',100.,1000000.,1000000.,100.);allocation=st.slider('Maximum available buying power allocated (%)',5,25,10,5)
    auto_trade=st.checkbox('Enable automatic paper trading after MAXPROFIT runs',value=False,help='When enabled, MAXPROFIT submits paper buys only for current BUY signals. Account cash, buying power, price trigger, existing positions and open orders are checked first.')
    st.caption('PAPER EXIT: after a paper buy fills, VAST CASH attempts a +10% limit sell. Paper orders only, no live-money orders.')
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
                # Rank by score, then expected return and win rate. Keep the full trade plan with each pick.
                ranked.sort(key=lambda x:(x['Score'],x['Expected Return'],x['Win Rate']),reverse=True)
                st.session_state.top10=ranked[:10]
                st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
                runner.markdown('## ✅ **MAXPROFIT COMPLETE**')
                st.caption(f'📊 {hist_status}')
                status.success(f'Finished testing {total} stocks. {len(ranked)} produced usable historical setups.')
                progress.progress(1.0,text='MAXPROFIT calculation complete')
                if auto_trade:
                    candidates=[x for x in st.session_state.top10 if float(x.get('Price',0))<=float(x.get('Buy Trigger',0))]
                    candidates=candidates[:3]
                    if not candidates:st.info('AUTO-PAPER: no top-ranked stocks currently meet their buy trigger. No orders submitted.')
                    else:
                        ac=account()
                        cash=float((ac or {}).get('cash',0) or 0);bp=float((ac or {}).get('buying_power',0) or 0)
                        spendable=min(max(0,cash),max(0,bp))
                        if spendable<=0:st.error(f'AUTO-PAPER STOPPED. Cash ${cash:,.2f}; buying power ${bp:,.2f}. Both must be positive. No orders submitted.')
                        else:
                            budget=spendable*allocation/100/len(candidates)
                            st.subheader('🤖 AUTOMATIC PAPER ORDERS')
                            for candidate in candidates:
                                ok,msg=buy(candidate['Ticker'],budget,candidate.get('Buy Trigger'))
                                if ok:st.success(msg)
                                else:st.warning(msg)
        except Exception as e:
            runner.markdown('## 🛑 **MAXPROFIT ERROR**')
            st.error(f'MAXPROFIT encountered an error: {e}')
if st.session_state.top10:
    st.header('🎯 TODAY’S TRADE PLAN')
    st.caption('These are the ten highest-ranked setups from MAXPROFIT. The plan gives you the entry trigger, target, and historically typical time to target.')
    plan=[]
    for rank,x in enumerate(st.session_state.top10,1):
        current=float(x['Price']); trigger=float(x['Buy Trigger']); target=float(x['Sell Target'])
        action='BUY' if current<=trigger else 'WAIT'
        plan.append({'#':rank,'STOCK':x['Ticker'],'ACTION':action,'CURRENT':current,'BUY AT':trigger,'SELL TARGET':target,'TYPICAL HOLD (TRADING DAYS)':x['Typical Hold'],'WIN RATE':x['Win Rate'],'HISTORICAL RETURN':x['Expected Return'],'TESTS':x['Historical Trades']})
    st.dataframe(pd.DataFrame(plan).style.format({'CURRENT':'${:,.2f}','BUY AT':'${:,.2f}','SELL TARGET':'${:,.2f}','WIN RATE':'{:.0%}','HISTORICAL RETURN':'{:+.1%}'}),use_container_width=True,hide_index=True)
    st.info('ACTION = BUY only when the current price is at or below the calculated buy trigger. Otherwise WAIT. SELL TARGET is +10% from the planned trigger. Actual execution price will determine the final target after a simulated or real fill.')
    st.divider()
    st.subheader('🧪 STRATEGY PROFITABILITY TEST')
    if st.button('📈 TEST WHETHER VAST CASH MAKES MONEY',use_container_width=True):
        hist,_=all_hist()
        if hist:
            with st.spinner('Running the historical strategy test...'):
                trades,stats=backtest_strategy(hist,buy_drop,capital)
            if stats:
                a,b,c,d=st.columns(4)
                a.metric('Starting',f"${stats['Starting Capital']:,.0f}")
                b.metric('Ending',f"${stats['Ending Capital']:,.0f}")
                c.metric('Net Profit',f"${stats['Net Profit']:,.0f}")
                d.metric('Max Drawdown',f"{stats['Max Drawdown']:.1%}")
                st.write(f"**Trades:** {stats['Total Trades']} • **Win rate:** {stats['Win Rate']:.1%} • **Average trade:** {stats['Avg Trade Return']:+.2%} • **Best:** {stats['Best Trade']:+.2%} • **Worst:** {stats['Worst Trade']:+.2%}")
                st.caption('This is a historical simulation, not a promise of future returns. It does not include commissions, slippage, taxes, or market-impact effects.')
                st.dataframe(trades,use_container_width=True,hide_index=True)
            else: st.warning('No qualifying historical trades were found.')
        else: st.error('Historical market data could not be loaded.')

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
    if st.button('🚀 SUBMIT SELECTED TO ALPACA PAPER',type='primary',disabled=not complete,use_container_width=True):
        if not yes:st.info('All NO. Nothing sent.')
        else:
            ac=account()
            if not ac:st.error('Could not access Alpaca PAPER account.')
            else:
                cash=float(ac.get('cash',0) or 0);buying_power=float(ac.get('buying_power',0) or 0)
                spendable=min(max(0,cash),max(0,buying_power))
                if spendable<=0:st.error(f'NO ORDERS SENT. Paper account cash is ${cash:,.2f} and buying power is ${buying_power:,.2f}. Both must be positive.')
                else:
                    budget=spendable*allocation/100/len(yes);st.subheader('📨 ALPACA PAPER ORDERS')
                    for x in yes:
                        ok,msg=buy(x['Ticker'],budget,x.get('Buy Trigger'))
                        if ok:st.success(msg)
                        else:st.error(msg)
st.divider();st.caption('🔒 ALPACA PAPER TRADING ONLY. Paper orders may be submitted to Alpaca; no live-money orders are sent. Historical backtests are separate simulations.')