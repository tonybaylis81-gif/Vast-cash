import os
from collections.abc import Mapping
import requests
import streamlit as st

st.set_page_config(page_title='PAPER Account Diagnostics', page_icon='🧾', layout='wide')
TRADE = 'https://paper-api.alpaca.markets'


def secret(names):
    names = {x.upper() for x in names}
    try:
        def walk(v):
            if isinstance(v, Mapping):
                for k, x in v.items():
                    if str(k).upper() in names and x is not None and str(x).strip():
                        return str(x).strip()
                    y = walk(x)
                    if y:
                        return y
        x = walk(st.secrets)
        if x:
            return x
    except Exception:
        pass
    for n in names:
        x = os.getenv(n)
        if x and x.strip():
            return x.strip()
    return None


def headers():
    pairs = [
        ('PAPER_API_KEY', 'PAPER_API_SECRET'),
        ('ALPACA_API_KEY_ID', 'ALPACA_API_SECRET'),
        ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY'),
    ]
    for key_name, secret_name in pairs:
        k, s = secret([key_name]), secret([secret_name])
        if k and s:
            return {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}
    return None


def get_json(path, params=None):
    h = headers()
    if not h:
        return None, 'No matched Alpaca PAPER credentials found in Streamlit Secrets.'
    try:
        r = requests.get(f'{TRADE}{path}', headers=h, params=params, timeout=15)
        if r.status_code != 200:
            return None, f'HTTP {r.status_code}: {r.text[:500]}'
        return r.json(), None
    except requests.RequestException as e:
        return None, f'Connection error: {e}'


def money(v):
    try:
        return f'${float(v or 0):,.2f}'
    except Exception:
        return str(v)

st.title('🧾 PAPER ACCOUNT DIAGNOSTICS')
st.caption('Read-only diagnostic panel. It does not submit, cancel, or modify orders.')

if st.button('🔄 REFRESH ACCOUNT DATA', type='primary', use_container_width=True):
    st.cache_data.clear()
    st.rerun()

account, account_error = get_json('/v2/account')
orders, orders_error = get_json('/v2/orders', {'status': 'open', 'limit': 100})
positions, positions_error = get_json('/v2/positions')

if account_error:
    st.error(f'Alpaca PAPER account unavailable: {account_error}')
    st.stop()

st.success(f"Alpaca PAPER account connection: {account.get('status', 'UNKNOWN')}")

st.subheader('💰 Funds and buying power')
c = st.columns(4)
c[0].metric('Equity', money(account.get('equity')))
c[1].metric('Cash', money(account.get('cash')))
c[2].metric('Buying Power', money(account.get('buying_power')))
c[3].metric('Portfolio Value', money(account.get('portfolio_value')))

c = st.columns(4)
c[0].metric('Reg-T Buying Power', money(account.get('regt_buying_power')))
c[1].metric('Non-Marginable Buying Power', money(account.get('non_marginable_buying_power')))
c[2].metric('Initial Margin', money(account.get('initial_margin')))
c[3].metric('Maintenance Margin', money(account.get('maintenance_margin')))

c = st.columns(4)
c[0].metric('Long Market Value', money(account.get('long_market_value')))
c[1].metric('Short Market Value', money(account.get('short_market_value')))
c[2].metric('Last Equity', money(account.get('last_equity')))
c[3].metric('SMA', money(account.get('sma')))

st.subheader('🛡️ Account trading state')
c = st.columns(5)
c[0].metric('Status', account.get('status', 'UNKNOWN'))
c[1].metric('Multiplier', account.get('multiplier', 'N/A'))
c[2].metric('Trading Blocked', 'YES' if account.get('trading_blocked') else 'NO')
c[3].metric('Account Blocked', 'YES' if account.get('account_blocked') else 'NO')
c[4].metric('User Trade Suspended', 'YES' if account.get('trade_suspended_by_user') else 'NO')

if float(account.get('buying_power') or 0) <= 0:
    st.warning('⚠️ Alpaca currently reports $0.00 buying power. VAST CASH should not attempt PAPER BUY orders until buying power is available.')

if orders_error:
    st.warning(f'Open orders could not be read: {orders_error}')
else:
    st.subheader(f'📨 Open PAPER orders ({len(orders) if isinstance(orders, list) else 0})')
    if orders:
        st.dataframe([
            {
                'Symbol': o.get('symbol'), 'Side': o.get('side'), 'Qty': o.get('qty'),
                'Filled Qty': o.get('filled_qty'), 'Status': o.get('status'),
                'Type': o.get('type'), 'Limit Price': o.get('limit_price'),
                'Submitted': o.get('submitted_at'), 'Order ID': o.get('id')
            } for o in orders
        ], use_container_width=True, hide_index=True)
    else:
        st.info('No open PAPER orders.')

if positions_error:
    st.warning(f'Positions could not be read: {positions_error}')
else:
    st.subheader(f'📊 Open PAPER positions ({len(positions) if isinstance(positions, list) else 0})')
    if positions:
        st.dataframe([
            {
                'Symbol': p.get('symbol'), 'Qty': p.get('qty'),
                'Avg Entry': money(p.get('avg_entry_price')), 'Market Value': money(p.get('market_value')),
                'Cost Basis': money(p.get('cost_basis')), 'Unrealized P/L': money(p.get('unrealized_pl')),
                'Unrealized P/L %': p.get('unrealized_plpc'), 'Current Price': money(p.get('current_price'))
            } for p in positions
        ], use_container_width=True, hide_index=True)
    else:
        st.info('No open PAPER positions.')

with st.expander('🔎 Full account object'):
    safe = {k: v for k, v in account.items() if k not in ('account_number', 'id')}
    st.json(safe)

st.divider()
st.caption('VAST CASH remains PAPER ONLY. This page is diagnostic and read-only.')
