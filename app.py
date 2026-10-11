import os,time
from collections.abc import Mapping
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,requests,streamlit as st
st.set_page_config(page_title='VAST CASH',page_icon='⚒️',layout='wide')
PAPER_ONLY=True; SIM_MODE=False; SIMULATED_BUYING_POWER=5000.00; DATA='https://data.alpaca.markets'; TRADE='https://paper-api.alpaca.markets'; TARGET=0.05; STOP_LOSS=0.03; MIN_WIN_RATE=0.60; MIN_HISTORICAL_TRADES=6
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
    # Use a 20-session moving average as the reference. Stops are conservatively
    # assumed to trigger first if both stop and target occur in one daily candle.
    if len(d)<140:return None
    wins=[];rets=[];hs=[];start=70;stop=len(d)-2;step=max(1,(stop-start)//90)
    for i in range(start,stop,step):
        prior=d.iloc[max(0,i-20):i]
        if len(prior)<20:continue
        reference=float(prior['close'].mean());entry=reference*(1-buy_drop/100)
        fut=d.iloc[i:i+31];hits=np.where(fut['low'].to_numpy()<=entry)[0]
        if not len(hits):continue
        a=fut.iloc[int(hits[0]):int(hits[0])+31]
        target=entry*(1+TARGET);protective_stop=entry*(1-STOP_LOSS);result=None;hold=1
        for day,row in enumerate(a.itertuples(),1):
            if float(row.low)<=protective_stop:
                result=-STOP_LOSS;hold=day;break
            if float(row.high)>=target:
                result=TARGET;hold=day;break
        if result is None:
            result=float(a.close.iloc[-1]/entry-1);hold=max(1,len(a))
        rets.append(result);wins.append(1 if result>0 else 0);hs.append(hold)
    if len(rets)<4:return None
    closes=d.close;price=float(closes.iloc[-1]);reference=float(closes.tail(20).mean());trigger=reference*(1-buy_drop/100)
    win=float(np.mean(wins));ret=float(np.mean(rets));vol=float(closes.pct_change().dropna().tail(30).std()*np.sqrt(252));momentum=float(price/closes.iloc[-21]-1);typical=max(1,int(round(np.mean(hs))))
    return {'Ticker':s,'Expected Return':ret,'Win Rate':win,'Historical Trades':len(rets),'Typical Hold':typical,'Price':price,'Reference Average':reference,'Buy Trigger':trigger,'Sell Target':trigger*(1+TARGET),'Stop Loss':trigger*(1-STOP_LOSS),'Momentum':momentum,'Volatility':vol,'Score':ret*100+win*20-vol*5}
def backtest_strategy(hist,buy_drop,starting_cash=SIMULATED_BUYING_POWER):
    # Walk-forward simulation using the same moving-average entry and bracket exits.
    trades=[]
    for s,d in hist.items():
        if len(d)<140:continue
        for i in range(70,len(d)-31):
            prior=d.iloc[max(0,i-20):i]
            if len(prior)<20:continue
            reference=float(prior['close'].mean());entry=reference*(1-buy_drop/100)
            fut=d.iloc[i:i+31];hits=np.where(fut['low'].to_numpy()<=entry)[0]
            if not len(hits):continue
            a=fut.iloc[int(hits[0]):int(hits[0])+31]
            target=entry*(1+TARGET);protective_stop=entry*(1-STOP_LOSS);exit_price=None;hold=1;outcome='TIME EXIT'
            for day,row in enumerate(a.itertuples(),1):
                if float(row.low)<=protective_stop:
                    exit_price=protective_stop;hold=day;outcome='STOP LOSS';break
                if float(row.high)>=target:
                    exit_price=target;hold=day;outcome='TAKE PROFIT';break
            if exit_price is None:
                exit_price=float(a['close'].iloc[-1]);hold=max(1,len(a))
                outcome='WIN' if exit_price>entry else ('LOSS' if exit_price<entry else 'FLAT')
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
                # New orders are bracket orders: Alpaca manages their attached take-profit and stop-loss.
                # Do not submit a second, competing sell order for the same shares.
                if o.get('order_class') == 'bracket':
                    done.append(f'{symbol}: bracket buy filled {fq} @ ${fill:.2f}; attached take-profit near ${target:.2f} and protective stop are managed by Alpaca PAPER.')
                    del pending[oid]
                    continue
                existing=requests.get(f'{TRADE}/v2/orders',headers=h,params={'status':'open','symbols':symbol,'limit':100,'nested':'true'},timeout=10)
                has_sell=False
                if existing.status_code==200:
                    for so in existing.json():
                        if so.get('side')=='sell' and so.get('symbol')==symbol:has_sell=True;break
                if not has_sell:
                    x=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json={'symbol':symbol,'qty':str(fq),'side':'sell','type':'limit','limit_price':f'{target:.2f}','time_in_force':'gtc'},timeout=15)
                    if x.status_code not in (200,201):continue
                done.append(f'{symbol}: filled {fq} @ ${fill:.2f}; +{TARGET:.0%} GTC sell at ${target:.2f} is active.')
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
            return True,f'SIM BUY {symbol}: {qty} shares @ ${ask:.2f}. Simulated +{TARGET:.0%} target: ${target:.2f}. NO ALPACA ORDER SENT.'.replace('
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
        target=round(ask*(1+TARGET),2);stop=round(ask*(1-STOP_LOSS),2)
        payload={'symbol':symbol,'qty':str(qty),'side':'buy','type':'market','time_in_force':'day','order_class':'bracket','take_profit':{'limit_price':f'{target:.2f}'},'stop_loss':{'stop_price':f'{stop:.2f}'}}
        r=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json=payload,timeout=15)
        if r.status_code not in (200,201):return False,f'PAPER BRACKET BUY rejected: {r.text[:200]}'
        oid=r.json()['id'];st.session_state.setdefault('pending_orders',{})[oid]=symbol;fill=0;fq=0
        for _ in range(12):
            time.sleep(1);z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code==200:
                o=z.json();fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
                if fill>0 and fq>0:break
        if not fill:return True,f'PAPER BUY submitted for {symbol}; fill still pending. VAST CASH will reconcile it automatically.'
        target=round(fill*(1+TARGET),2)
        stop=round(fill*(1-STOP_LOSS),2)
        msg=f'PAPER BRACKET BUY {symbol}: {fq} shares @ ${fill:.2f}. Take-profit ${target:.2f} (+{TARGET:.0%}); protective stop ${stop:.2f} (-{STOP_LOSS:.0%}) submitted as attached exits.'
        return True,msg
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
with st.expander('🔎 Account details (optional)',expanded=False):
    show_positions_and_orders()
st.divider()
st.subheader('STOCK TRADING FOR WELDERS')
st.caption('One choice at a time. Press YES to submit a paper trade if the checks pass, or NO to skip it.')
with st.sidebar:
    buy_drop=st.slider('Buy % below 20-day average',1,10,5);target_choice=st.selectbox('Profit target',options=[5,10],index=0,format_func=lambda v:f'+{v}%');TARGET=target_choice/100;capital=st.number_input('Backtest starting capital ($)',100.,1000000.,1000000.,100.);allocation=st.slider('Account balance to risk per approved trade (%)',1,5,2,1)
    st.caption(f'PAPER ONLY. Take-profit +{TARGET:.0%}; protective stop −{STOP_LOSS:.0%}. You approve each trade with YES. NO skips it.')
if 'top10' not in st.session_state:st.session_state.top10=None
if 'decisions' not in st.session_state:st.session_state.decisions={}
if 'pending_orders' not in st.session_state:st.session_state.pending_orders={}
if st.button('🔎 FIND MY BEST OPTION TODAY',type='primary',use_container_width=True):
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
                    status.info(f'🔧 Testing {s}: historical +{TARGET:.0%} target hits, win rate and volatility...')
                    x=score(s,d,buy_drop)
                    if x:ranked.append(x)
                    progress.progress(i/total,text=f'Calculating MAXPROFIT: {i}/{total} stocks')
                # Rank by score, then expected return and win rate. Keep the full trade plan with each pick.
                qualified=[x for x in ranked if x['Expected Return']>0 and x['Win Rate']>=MIN_WIN_RATE and x['Historical Trades']>=MIN_HISTORICAL_TRADES]
                qualified.sort(key=lambda x:(x['Score'],x['Expected Return'],x['Win Rate']),reverse=True)
                st.session_state.top10=qualified[:10]
                st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
                runner.markdown('## ✅ **MAXPROFIT COMPLETE**')
                st.caption(f'📊 {hist_status}')
                status.success(f'Checked {total} stocks. {len(qualified)} passed the filters: positive historical average return, at least {MIN_WIN_RATE:.0%} target-hit rate, and {MIN_HISTORICAL_TRADES}+ historical tests.')
                if not qualified:st.warning('No stocks passed the filters today. The app will not force a trade just to fill ten slots.')
                progress.progress(1.0,text='MAXPROFIT calculation complete')
                st.session_state.option_index=0
                st.session_state.last_order_message=None
        except Exception as e:
            runner.markdown('## 🛑 **MAXPROFIT ERROR**')
            st.error(f'MAXPROFIT encountered an error: {e}')
if st.session_state.top10:
    top=st.session_state.top10
    if 'option_index' not in st.session_state:st.session_state.option_index=0
    idx=st.session_state.option_index
    st.header('🎯 ONE OPTION AT A TIME')
    st.caption('Review one candidate. YES submits one paper order if the price and account checks pass. NO skips it and shows the next candidate.')
    if idx>=len(top):
        st.info('You have reviewed every qualifying option today. Run a new scan later for fresh candidates.')
    else:
        x=top[idx];t=x['Ticker'];current=float(x['Price']);trigger=float(x['Buy Trigger']);reference=float(x.get('Reference Average',trigger/(1-buy_drop/100)));target=trigger*(1+TARGET);stop=trigger*(1-STOP_LOSS)
        with st.container(border=True):
            st.caption(f'OPTION {idx+1} OF {len(top)}')
            st.subheader(f'{t} · {"BUY SETUP" if current<=trigger else "WAIT FOR BUY PRICE"}')
            a,b,c1=st.columns(3)
            a.metric('Current price',f"${current:.2f}")
            b.metric('20-day average',f"${reference:.2f}")
            c1.metric('Buy only at or below',f"${trigger:.2f}",f"{(trigger/reference-1):+.1%} vs average")
            st.write(f"**After a fill:** automatic take-profit at +{TARGET:.0%} of the actual fill price and protective stop at −{STOP_LOSS:.0%} of the actual fill price. The final prices are set by Alpaca's bracket order.")
            st.write(f"**Historical target-hit rate:** {x['Win Rate']:.0%} across {x['Historical Trades']} tests")
            st.write(f"**Average historical trade return:** {x['Expected Return']:+.1%} · **Typical hold:** {x['Typical Hold']} trading days")
            st.caption('Historical performance is not a guarantee. Stop orders can execute below the stop price. Paper trading only.')
            yes_col,no_col=st.columns(2)
            if yes_col.button('✅ YES, BUY THIS OPTION',key=f'yes_option_{idx}_{t}',type='primary',use_container_width=True):
                ac=account()
                cash=float((ac or {}).get('cash',0) or 0);bp=float((ac or {}).get('buying_power',0) or 0)
                spendable=min(max(0,cash),max(0,bp))
                if not ac:st.error('Order blocked: cannot verify Alpaca PAPER account.')
                elif spendable<=0:st.error(f'Order blocked. Paper cash ${cash:,.2f}; buying power ${bp:,.2f}. Account must be reconciled first.')
                elif current>trigger:st.warning(f'No order sent. Current price ${current:.2f} is above the buy trigger ${trigger:.2f}. Choose NO to review the next option.')
                else:
                    budget=spendable*allocation/100
                    ok,msg=buy(t,budget,trigger)
                    if ok:
                        st.success(msg)
                        st.session_state.option_index=idx+1
                    else:st.error(msg)
            if no_col.button('❌ NO, SKIP THIS ONE',key=f'no_option_{idx}_{t}',use_container_width=True):
                st.session_state.option_index=idx+1
                st.rerun()
    st.divider()
    st.subheader('🧪 TEST THE STRATEGY BEFORE TRUSTING IT')
    if st.button('📈 TEST WHETHER VAST CASH MAKES MONEY',use_container_width=True):
        hist,_=all_hist()
        if hist:
            with st.spinner('Running the historical strategy test...'):
                trades,stats=backtest_strategy(hist,buy_drop,capital)
            if stats:
                a,b,c1,d=st.columns(4)
                a.metric('Starting',f"${stats['Starting Capital']:,.0f}")
                b.metric('Ending',f"${stats['Ending Capital']:,.0f}")
                c1.metric('Net Profit',f"${stats['Net Profit']:,.0f}")
                d.metric('Max Drawdown',f"{stats['Max Drawdown']:.1%}")
                st.write(f"**Trades:** {stats['Total Trades']} • **Win rate:** {stats['Win Rate']:.1%} • **Average trade:** {stats['Avg Trade Return']:+.2%} • **Best:** {stats['Best Trade']:+.2%} • **Worst:** {stats['Worst Trade']:+.2%}")
                st.caption('Historical simulation only. It does not include all real-world trading costs and cannot guarantee future results.')
                st.dataframe(trades,use_container_width=True,hide_index=True)
            else:st.warning('No qualifying historical trades were found.')
        else:st.error('Historical market data could not be loaded.')
,'\\
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
        target=round(ask*(1+TARGET),2);stop=round(ask*(1-STOP_LOSS),2)
        payload={'symbol':symbol,'qty':str(qty),'side':'buy','type':'market','time_in_force':'day','order_class':'bracket','take_profit':{'limit_price':f'{target:.2f}'},'stop_loss':{'stop_price':f'{stop:.2f}'}}
        r=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json=payload,timeout=15)
        if r.status_code not in (200,201):return False,f'PAPER BRACKET BUY rejected: {r.text[:200]}'
        oid=r.json()['id'];st.session_state.setdefault('pending_orders',{})[oid]=symbol;fill=0;fq=0
        for _ in range(12):
            time.sleep(1);z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code==200:
                o=z.json();fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
                if fill>0 and fq>0:break
        if not fill:return True,f'PAPER BUY submitted for {symbol}; fill still pending. VAST CASH will reconcile it automatically.'
        target=round(fill*(1+TARGET),2)
        stop=round(fill*(1-STOP_LOSS),2)
        msg=f'PAPER BRACKET BUY {symbol}: {fq} shares @ ${fill:.2f}. Take-profit ${target:.2f} (+{TARGET:.0%}); protective stop ${stop:.2f} (-{STOP_LOSS:.0%}) submitted as attached exits.'
        return True,msg
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
with st.expander('🔎 Account details (optional)',expanded=False):
    show_positions_and_orders()
st.divider()
st.subheader('STOCK TRADING FOR WELDERS')
st.caption('One choice at a time. Press YES to submit a paper trade if the checks pass, or NO to skip it.')
with st.sidebar:
    buy_drop=st.slider('Buy % below 20-day average',1,10,5);target_choice=st.selectbox('Profit target',options=[5,10],index=0,format_func=lambda v:f'+{v}%');TARGET=target_choice/100;capital=st.number_input('Backtest starting capital ($)',100.,1000000.,1000000.,100.);allocation=st.slider('Account balance to risk per approved trade (%)',1,5,2,1)
    st.caption(f'PAPER ONLY. Take-profit +{TARGET:.0%}; protective stop −{STOP_LOSS:.0%}. You approve each trade with YES. NO skips it.')
if 'top10' not in st.session_state:st.session_state.top10=None
if 'decisions' not in st.session_state:st.session_state.decisions={}
if 'pending_orders' not in st.session_state:st.session_state.pending_orders={}
if st.button('🔎 FIND MY BEST OPTION TODAY',type='primary',use_container_width=True):
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
                    status.info(f'🔧 Testing {s}: historical +{TARGET:.0%} target hits, win rate and volatility...')
                    x=score(s,d,buy_drop)
                    if x:ranked.append(x)
                    progress.progress(i/total,text=f'Calculating MAXPROFIT: {i}/{total} stocks')
                # Rank by score, then expected return and win rate. Keep the full trade plan with each pick.
                qualified=[x for x in ranked if x['Expected Return']>0 and x['Win Rate']>=MIN_WIN_RATE and x['Historical Trades']>=MIN_HISTORICAL_TRADES]
                qualified.sort(key=lambda x:(x['Score'],x['Expected Return'],x['Win Rate']),reverse=True)
                st.session_state.top10=qualified[:10]
                st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
                runner.markdown('## ✅ **MAXPROFIT COMPLETE**')
                st.caption(f'📊 {hist_status}')
                status.success(f'Checked {total} stocks. {len(qualified)} passed the filters: positive historical average return, at least {MIN_WIN_RATE:.0%} target-hit rate, and {MIN_HISTORICAL_TRADES}+ historical tests.')
                if not qualified:st.warning('No stocks passed the filters today. The app will not force a trade just to fill ten slots.')
                progress.progress(1.0,text='MAXPROFIT calculation complete')
                st.session_state.option_index=0
                st.session_state.last_order_message=None
        except Exception as e:
            runner.markdown('## 🛑 **MAXPROFIT ERROR**')
            st.error(f'MAXPROFIT encountered an error: {e}')
if st.session_state.top10:
    top=st.session_state.top10
    if 'option_index' not in st.session_state:st.session_state.option_index=0
    idx=st.session_state.option_index
    st.header('🎯 ONE OPTION AT A TIME')
    st.caption('Review one candidate. YES submits one paper order if the price and account checks pass. NO skips it and shows the next candidate.')
    if idx>=len(top):
        st.info('You have reviewed every qualifying option today. Run a new scan later for fresh candidates.')
    else:
        x=top[idx];t=x['Ticker'];current=float(x['Price']);trigger=float(x['Buy Trigger']);reference=float(x.get('Reference Average',trigger/(1-buy_drop/100)));target=trigger*(1+TARGET);stop=trigger*(1-STOP_LOSS)
        with st.container(border=True):
            st.caption(f'OPTION {idx+1} OF {len(top)}')
            st.subheader(f'{t} · {"BUY SETUP" if current<=trigger else "WAIT FOR BUY PRICE"}')
            a,b,c1=st.columns(3)
            a.metric('Current price',f"${current:.2f}")
            b.metric('20-day average',f"${reference:.2f}")
            c1.metric('Buy only at or below',f"${trigger:.2f}",f"{(trigger/reference-1):+.1%} vs average")
            st.write(f"**After a fill:** automatic take-profit at +{TARGET:.0%} of the actual fill price and protective stop at −{STOP_LOSS:.0%} of the actual fill price. The final prices are set by Alpaca's bracket order.")
            st.write(f"**Historical target-hit rate:** {x['Win Rate']:.0%} across {x['Historical Trades']} tests")
            st.write(f"**Average historical trade return:** {x['Expected Return']:+.1%} · **Typical hold:** {x['Typical Hold']} trading days")
            st.caption('Historical performance is not a guarantee. Stop orders can execute below the stop price. Paper trading only.')
            yes_col,no_col=st.columns(2)
            if yes_col.button('✅ YES, BUY THIS OPTION',key=f'yes_option_{idx}_{t}',type='primary',use_container_width=True):
                ac=account()
                cash=float((ac or {}).get('cash',0) or 0);bp=float((ac or {}).get('buying_power',0) or 0)
                spendable=min(max(0,cash),max(0,bp))
                if not ac:st.error('Order blocked: cannot verify Alpaca PAPER account.')
                elif spendable<=0:st.error(f'Order blocked. Paper cash ${cash:,.2f}; buying power ${bp:,.2f}. Account must be reconciled first.')
                elif current>trigger:st.warning(f'No order sent. Current price ${current:.2f} is above the buy trigger ${trigger:.2f}. Choose NO to review the next option.')
                else:
                    budget=spendable*allocation/100
                    ok,msg=buy(t,budget,trigger)
                    if ok:
                        st.success(msg)
                        st.session_state.option_index=idx+1
                    else:st.error(msg)
            if no_col.button('❌ NO, SKIP THIS ONE',key=f'no_option_{idx}_{t}',use_container_width=True):
                st.session_state.option_index=idx+1
                st.rerun()
    st.divider()
    st.subheader('🧪 TEST THE STRATEGY BEFORE TRUSTING IT')
    if st.button('📈 TEST WHETHER VAST CASH MAKES MONEY',use_container_width=True):
        hist,_=all_hist()
        if hist:
            with st.spinner('Running the historical strategy test...'):
                trades,stats=backtest_strategy(hist,buy_drop,capital)
            if stats:
                a,b,c1,d=st.columns(4)
                a.metric('Starting',f"${stats['Starting Capital']:,.0f}")
                b.metric('Ending',f"${stats['Ending Capital']:,.0f}")
                c1.metric('Net Profit',f"${stats['Net Profit']:,.0f}")
                d.metric('Max Drawdown',f"{stats['Max Drawdown']:.1%}")
                st.write(f"**Trades:** {stats['Total Trades']} • **Win rate:** {stats['Win Rate']:.1%} • **Average trade:** {stats['Avg Trade Return']:+.2%} • **Best:** {stats['Best Trade']:+.2%} • **Worst:** {stats['Worst Trade']:+.2%}")
                st.caption('Historical simulation only. It does not include all real-world trading costs and cannot guarantee future results.')
                st.dataframe(trades,use_container_width=True,hide_index=True)
            else:st.warning('No qualifying historical trades were found.')
        else:st.error('Historical market data could not be loaded.')
)
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
        target=round(ask*(1+TARGET),2);stop=round(ask*(1-STOP_LOSS),2)
        payload={'symbol':symbol,'qty':str(qty),'side':'buy','type':'market','time_in_force':'day','order_class':'bracket','take_profit':{'limit_price':f'{target:.2f}'},'stop_loss':{'stop_price':f'{stop:.2f}'}}
        r=requests.post(f'{TRADE}/v2/orders',headers={**h,'Content-Type':'application/json'},json=payload,timeout=15)
        if r.status_code not in (200,201):return False,f'PAPER BRACKET BUY rejected: {r.text[:200]}'
        oid=r.json()['id'];st.session_state.setdefault('pending_orders',{})[oid]=symbol;fill=0;fq=0
        for _ in range(12):
            time.sleep(1);z=requests.get(f'{TRADE}/v2/orders/{oid}',headers=h,timeout=10)
            if z.status_code==200:
                o=z.json();fill=float(o.get('filled_avg_price') or 0);fq=int(float(o.get('filled_qty') or 0))
                if fill>0 and fq>0:break
        if not fill:return True,f'PAPER BUY submitted for {symbol}; fill still pending. VAST CASH will reconcile it automatically.'
        target=round(fill*(1+TARGET),2)
        stop=round(fill*(1-STOP_LOSS),2)
        msg=f'PAPER BRACKET BUY {symbol}: {fq} shares @ ${fill:.2f}. Take-profit ${target:.2f} (+{TARGET:.0%}); protective stop ${stop:.2f} (-{STOP_LOSS:.0%}) submitted as attached exits.'
        return True,msg
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
with st.expander('🔎 Account details (optional)',expanded=False):
    show_positions_and_orders()
st.divider()
st.subheader('STOCK TRADING FOR WELDERS')
st.caption('One choice at a time. Press YES to submit a paper trade if the checks pass, or NO to skip it.')
with st.sidebar:
    buy_drop=st.slider('Buy % below 20-day average',1,10,5);target_choice=st.selectbox('Profit target',options=[5,10],index=0,format_func=lambda v:f'+{v}%');TARGET=target_choice/100;capital=st.number_input('Backtest starting capital ($)',100.,1000000.,1000000.,100.);allocation=st.slider('Account balance to risk per approved trade (%)',1,5,2,1)
    st.caption(f'PAPER ONLY. Take-profit +{TARGET:.0%}; protective stop −{STOP_LOSS:.0%}. You approve each trade with YES. NO skips it.')
if 'top10' not in st.session_state:st.session_state.top10=None
if 'decisions' not in st.session_state:st.session_state.decisions={}
if 'pending_orders' not in st.session_state:st.session_state.pending_orders={}
if st.button('🔎 FIND MY BEST OPTION TODAY',type='primary',use_container_width=True):
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
                    status.info(f'🔧 Testing {s}: historical +{TARGET:.0%} target hits, win rate and volatility...')
                    x=score(s,d,buy_drop)
                    if x:ranked.append(x)
                    progress.progress(i/total,text=f'Calculating MAXPROFIT: {i}/{total} stocks')
                # Rank by score, then expected return and win rate. Keep the full trade plan with each pick.
                qualified=[x for x in ranked if x['Expected Return']>0 and x['Win Rate']>=MIN_WIN_RATE and x['Historical Trades']>=MIN_HISTORICAL_TRADES]
                qualified.sort(key=lambda x:(x['Score'],x['Expected Return'],x['Win Rate']),reverse=True)
                st.session_state.top10=qualified[:10]
                st.session_state.decisions={x['Ticker']:None for x in st.session_state.top10}
                runner.markdown('## ✅ **MAXPROFIT COMPLETE**')
                st.caption(f'📊 {hist_status}')
                status.success(f'Checked {total} stocks. {len(qualified)} passed the filters: positive historical average return, at least {MIN_WIN_RATE:.0%} target-hit rate, and {MIN_HISTORICAL_TRADES}+ historical tests.')
                if not qualified:st.warning('No stocks passed the filters today. The app will not force a trade just to fill ten slots.')
                progress.progress(1.0,text='MAXPROFIT calculation complete')
                st.session_state.option_index=0
                st.session_state.last_order_message=None
        except Exception as e:
            runner.markdown('## 🛑 **MAXPROFIT ERROR**')
            st.error(f'MAXPROFIT encountered an error: {e}')
if st.session_state.top10:
    top=st.session_state.top10
    if 'option_index' not in st.session_state:st.session_state.option_index=0
    idx=st.session_state.option_index
    st.header('🎯 ONE OPTION AT A TIME')
    st.caption('Review one candidate. YES submits one paper order if the price and account checks pass. NO skips it and shows the next candidate.')
    if idx>=len(top):
        st.info('You have reviewed every qualifying option today. Run a new scan later for fresh candidates.')
    else:
        x=top[idx];t=x['Ticker'];current=float(x['Price']);trigger=float(x['Buy Trigger']);reference=float(x.get('Reference Average',trigger/(1-buy_drop/100)));target=trigger*(1+TARGET);stop=trigger*(1-STOP_LOSS)
        with st.container(border=True):
            st.caption(f'OPTION {idx+1} OF {len(top)}')
            st.subheader(f'{t} · {"BUY SETUP" if current<=trigger else "WAIT FOR BUY PRICE"}')
            a,b,c1=st.columns(3)
            a.metric('Current price',f"${current:.2f}")
            b.metric('20-day average',f"${reference:.2f}")
            c1.metric('Buy only at or below',f"${trigger:.2f}",f"{(trigger/reference-1):+.1%} vs average")
            st.write(f"**After a fill:** automatic take-profit at +{TARGET:.0%} of the actual fill price and protective stop at −{STOP_LOSS:.0%} of the actual fill price. The final prices are set by Alpaca's bracket order.")
            st.write(f"**Historical target-hit rate:** {x['Win Rate']:.0%} across {x['Historical Trades']} tests")
            st.write(f"**Average historical trade return:** {x['Expected Return']:+.1%} · **Typical hold:** {x['Typical Hold']} trading days")
            st.caption('Historical performance is not a guarantee. Stop orders can execute below the stop price. Paper trading only.')
            yes_col,no_col=st.columns(2)
            if yes_col.button('✅ YES, BUY THIS OPTION',key=f'yes_option_{idx}_{t}',type='primary',use_container_width=True):
                ac=account()
                cash=float((ac or {}).get('cash',0) or 0);bp=float((ac or {}).get('buying_power',0) or 0)
                spendable=min(max(0,cash),max(0,bp))
                if not ac:st.error('Order blocked: cannot verify Alpaca PAPER account.')
                elif spendable<=0:st.error(f'Order blocked. Paper cash ${cash:,.2f}; buying power ${bp:,.2f}. Account must be reconciled first.')
                elif current>trigger:st.warning(f'No order sent. Current price ${current:.2f} is above the buy trigger ${trigger:.2f}. Choose NO to review the next option.')
                else:
                    budget=spendable*allocation/100
                    ok,msg=buy(t,budget,trigger)
                    if ok:
                        st.success(msg)
                        st.session_state.option_index=idx+1
                    else:st.error(msg)
            if no_col.button('❌ NO, SKIP THIS ONE',key=f'no_option_{idx}_{t}',use_container_width=True):
                st.session_state.option_index=idx+1
                st.rerun()
    st.divider()
    st.subheader('🧪 TEST THE STRATEGY BEFORE TRUSTING IT')
    if st.button('📈 TEST WHETHER VAST CASH MAKES MONEY',use_container_width=True):
        hist,_=all_hist()
        if hist:
            with st.spinner('Running the historical strategy test...'):
                trades,stats=backtest_strategy(hist,buy_drop,capital)
            if stats:
                a,b,c1,d=st.columns(4)
                a.metric('Starting',f"${stats['Starting Capital']:,.0f}")
                b.metric('Ending',f"${stats['Ending Capital']:,.0f}")
                c1.metric('Net Profit',f"${stats['Net Profit']:,.0f}")
                d.metric('Max Drawdown',f"{stats['Max Drawdown']:.1%}")
                st.write(f"**Trades:** {stats['Total Trades']} • **Win rate:** {stats['Win Rate']:.1%} • **Average trade:** {stats['Avg Trade Return']:+.2%} • **Best:** {stats['Best Trade']:+.2%} • **Worst:** {stats['Worst Trade']:+.2%}")
                st.caption('Historical simulation only. It does not include all real-world trading costs and cannot guarantee future results.')
                st.dataframe(trades,use_container_width=True,hide_index=True)
            else:st.warning('No qualifying historical trades were found.')
        else:st.error('Historical market data could not be loaded.')
