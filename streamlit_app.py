import json
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VAST CASH | MAXPROFIT", page_icon="💰", layout="wide")

# ============================================================
# MAXPROFIT LOCKED CORE RULES
# ============================================================
LOOKBACK_BARS = 92                 # 3 months of trading sessions
HOLD_SESSIONS = 5                  # Buy Day 1 -> Sell Day 6
MAX_POSITIONS = 10
PAPER_TRADING_ONLY = True          # NEVER change this to False in this app

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "NFLX",
    "COST", "WMT", "JPM", "V", "MA", "ORCL", "CRM", "QCOM", "MU", "AMAT",
    "GE", "CAT", "HON", "UNP", "XOM", "CVX", "COP", "UBER", "SHOP", "PLTR",
    "PANW", "CRWD", "SNOW", "DIS", "TMO", "LLY", "PEP"
]

PAPER_API = "https://paper-api.alpaca.markets"
DATA_API = "https://data.alpaca.markets"


def get_credentials():
    try:
        key = st.secrets.get("ALPACA_API_KEY", "")
        secret = st.secrets.get("ALPACA_SECRET_KEY", "")
        return str(key).strip(), str(secret).strip()
    except Exception:
        return "", ""


def credentials_ready():
    key, secret = get_credentials()
    return bool(key and secret)


def api_request(base_url, path, method="GET", params=None, body=None):
    """Small standard-library HTTP client. Keeps the deployment dependency list tiny."""
    key, secret = get_credentials()
    if not key or not secret:
        raise RuntimeError("Alpaca PAPER credentials are not configured in Streamlit Secrets.")

    url = base_url + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca API {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error contacting Alpaca: {exc.reason}") from exc


# ============================================================
# MARKET DATA
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(symbols_tuple, calendar_days=450):
    symbols = list(symbols_tuple)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=calendar_days)
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "feed": "iex",
        "adjustment": "all",
        "sort": "asc",
    }

    all_bars = {symbol: [] for symbol in symbols}
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        elif "page_token" in params:
            del params["page_token"]
        response = api_request(DATA_API, "/v2/stocks/bars", params=params)
        bars = response.get("bars", {})
        for symbol, rows in bars.items():
            all_bars.setdefault(symbol, []).extend(rows)
        page_token = response.get("next_page_token")
        if not page_token:
            break

    frame = {}
    for symbol, rows in all_bars.items():
        if rows:
            frame[symbol] = pd.Series(
                {pd.to_datetime(row["t"]): float(row["c"]) for row in rows}
            )

    if not frame:
        raise RuntimeError("No historical bars were returned by Alpaca.")

    prices = pd.DataFrame(frame).sort_index().ffill()
    prices = prices.dropna(axis=1, how="all")
    if len(prices) < LOOKBACK_BARS:
        raise RuntimeError(f"Only {len(prices)} trading sessions were returned. MAXPROFIT requires {LOOKBACK_BARS}.")
    return prices


def score_symbol(series, risk_profile):
    s = series.dropna()
    if len(s) < LOOKBACK_BARS:
        return None

    r92 = s.iloc[-1] / s.iloc[-LOOKBACK_BARS] - 1
    r20 = s.iloc[-1] / s.iloc[-21] - 1
    r10 = s.iloc[-1] / s.iloc[-11] - 1
    daily = s.pct_change().dropna()
    volatility = max(float(daily.std()), 0.0001)
    negative = daily[daily < 0]
    downside_vol = max(float(negative.std()) if len(negative) > 1 else volatility, 0.0001)

    risk_adjusted = r92 / volatility
    downside_adjusted = r92 / downside_vol
    trend = (r92 + r20 + r10) / 3

    if risk_profile == "Conservative":
        score = (
            0.40 * r92 + 0.15 * r20 + 0.10 * r10
            + 0.20 * (risk_adjusted / 10) + 0.15 * (downside_adjusted / 10)
        )
    elif risk_profile == "Aggressive":
        score = 0.55 * r92 + 0.25 * r20 + 0.15 * r10 + 0.05 * trend
    else:
        score = (
            0.45 * r92 + 0.25 * r20 + 0.15 * r10
            + 0.10 * (risk_adjusted / 10) + 0.05 * (downside_adjusted / 10)
        )

    return {
        "Price": float(s.iloc[-1]),
        "3M Return": float(r92),
        "20D Return": float(r20),
        "10D Return": float(r10),
        "Volatility": float(volatility),
        "Downside Vol": float(downside_vol),
        "Score": float(score),
    }


def rank_market(prices, risk_profile):
    rows = []
    for symbol in prices.columns:
        result = score_symbol(prices[symbol], risk_profile)
        if result:
            rows.append({"Symbol": symbol, **result})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def backtest(prices, positions, risk_profile, threshold, starting_capital):
    """Walk forward with no look-ahead. Signal -> next session entry -> five sessions held."""
    equity = float(starting_capital)
    curve = []
    trades = []

    for signal_idx in range(LOOKBACK_BARS, len(prices) - HOLD_SESSIONS - 1, HOLD_SESSIONS):
        ranked = rank_market(prices.iloc[: signal_idx + 1], risk_profile)
        chosen = ranked[ranked["Score"] >= threshold].head(positions)
        if chosen.empty:
            chosen = ranked.head(positions)

        entry_idx = signal_idx + 1
        exit_idx = entry_idx + HOLD_SESSIONS
        returns = []

        for symbol in chosen["Symbol"]:
            entry = float(prices.iloc[entry_idx][symbol])
            exit_price = float(prices.iloc[exit_idx][symbol])
            ret = exit_price / entry - 1
            returns.append(ret)
            trades.append({
                "Entry Date": prices.index[entry_idx].date(),
                "Exit Date": prices.index[exit_idx].date(),
                "Symbol": symbol,
                "Entry": entry,
                "Exit": exit_price,
                "Return": ret,
            })

        if returns:
            equity *= 1 + float(np.mean(returns))
            curve.append((prices.index[exit_idx], equity))

    curve_df = (
        pd.DataFrame(curve, columns=["Date", "Portfolio Value"]).set_index("Date")
        if curve else pd.DataFrame(columns=["Portfolio Value"])
    )
    return curve_df, pd.DataFrame(trades)


# ============================================================
# PAPER ACCOUNT / EXECUTION
# ============================================================
def paper_account():
    return api_request(PAPER_API, "/v2/account")


def paper_positions():
    return api_request(PAPER_API, "/v2/positions")


def paper_orders(limit=100):
    return api_request(
        PAPER_API,
        "/v2/orders",
        params={"status": "all", "limit": limit, "direction": "desc"},
    )


def paper_calendar(start_date, end_date):
    return api_request(
        PAPER_API,
        "/v2/calendar",
        params={"start": start_date, "end": end_date, "date_type": "TRADING"},
    )


def submit_paper_buy(symbol, notional, client_order_id):
    # This endpoint is the Alpaca PAPER trading domain, not the live domain.
    return api_request(
        PAPER_API,
        "/v2/orders",
        method="POST",
        body={
            "symbol": symbol,
            "notional": round(float(notional), 2),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        },
    )


def close_paper_position(symbol):
    return api_request(PAPER_API, f"/v2/positions/{urllib.parse.quote(symbol)}", method="DELETE")


def submit_selected_paper_trades(selected, deployment_capital):
    account = paper_account()
    buying_power = float(account.get("buying_power", 0))
    deploy = min(float(deployment_capital), buying_power)
    if deploy <= 0:
        raise RuntimeError("Paper account has no available buying power.")

    existing = paper_positions()
    held = {str(p.get("symbol", "")).upper() for p in existing}
    open_orders = paper_orders(limit=100)
    pending = {
        str(o.get("symbol", "")).upper()
        for o in open_orders
        if str(o.get("status", "")).lower() in {"new", "accepted", "pending_new", "partially_filled"}
    }

    candidates = [s for s in selected["Symbol"].tolist() if s.upper() not in held and s.upper() not in pending]
    if not candidates:
        raise RuntimeError("None of the current selections are eligible for a new paper entry. Existing positions/open orders were skipped.")

    allocation = deploy / len(candidates)
    results = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for symbol in candidates:
        client_id = f"vastcash-buy-{stamp}-{symbol}"[:128]
        order = submit_paper_buy(symbol, allocation, client_id)
        results.append({
            "Symbol": symbol,
            "Notional": allocation,
            "Order ID": order.get("id", ""),
            "Status": order.get("status", ""),
            "Client Order ID": order.get("client_order_id", client_id),
        })
    return pd.DataFrame(results), deploy, buying_power


def eligible_d6_exits():
    """Find VAST CASH paper positions whose tagged buy order has reached five trading sessions."""
    positions = paper_positions()
    orders = paper_orders(limit=500)
    if not positions:
        return pd.DataFrame()

    buy_dates = {}
    for order in orders:
        symbol = str(order.get("symbol", "")).upper()
        cid = str(order.get("client_order_id", ""))
        if str(order.get("side", "")).lower() != "buy" or not cid.startswith("vastcash-buy-"):
            continue
        filled = order.get("filled_at") or order.get("submitted_at")
        if not filled:
            continue
        dt = pd.to_datetime(filled, utc=True).date()
        if symbol not in buy_dates or dt > buy_dates[symbol]:
            buy_dates[symbol] = dt

    today = datetime.now(timezone.utc).date()
    calendar = paper_calendar((today - timedelta(days=30)).isoformat(), today.isoformat())
    session_dates = [pd.to_datetime(x["date"]).date() for x in calendar]

    rows = []
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        entry_date = buy_dates.get(symbol)
        sessions_held = 0
        if entry_date:
            sessions_held = len([d for d in session_dates if entry_date < d <= today])
        rows.append({
            "Symbol": symbol,
            "Qty": float(position.get("qty", 0)),
            "Market Value": float(position.get("market_value", 0)),
            "Unrealized P/L": float(position.get("unrealized_pl", 0)),
            "Entry Date": entry_date,
            "Trading Sessions Held": sessions_held,
            "D6 Eligible": bool(entry_date and sessions_held >= HOLD_SESSIONS),
        })
    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
st.title("💰 VAST CASH")
st.caption("MAXPROFIT • Live market analysis + controlled paper execution")

with st.sidebar:
    st.header("MAXPROFIT Inputs")
    capital = st.number_input("Paper deployment capital ($)", min_value=100.0, value=1000.0, step=100.0)
    positions_count = st.slider("Stocks selected", 1, MAX_POSITIONS, 5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    signal_threshold = st.slider("Minimum signal strength", -0.20, 1.00, 0.20, 0.05)
    run = st.button("🚀 RUN MAXPROFIT", type="primary", use_container_width=True)
    st.divider()
    st.caption("LOCKED: 92 trading-day lookback")
    st.caption("LOCKED: Buy Day 1 → Sell Day 6")
    st.caption("Maximum 10 positions")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deployment", f"${capital:,.0f}")
c2.metric("Stocks", positions_count)
c3.metric("Cycle", "D1 → D6")
c4.metric("Mode", "PAPER ONLY")

if PAPER_TRADING_ONLY:
    st.info("🔒 PAPER-ONLY SAFETY LOCK: MAXPROFIT can analyze live market data and submit orders only to the Alpaca paper endpoint. No live trading endpoint is used.")

if not credentials_ready():
    st.warning("Paper trading is not connected yet. In Streamlit, open **Manage app → Settings → Secrets** and add your Alpaca PAPER credentials as ALPACA_API_KEY and ALPACA_SECRET_KEY. Never paste the secret key into the Python file.")
else:
    try:
        account = paper_account()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Paper Equity", f"${float(account.get('equity', 0)):,.2f}")
        a2.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
        a3.metric("Account", "PAPER")
        a4.metric("Status", str(account.get("status", "UNKNOWN")))
    except Exception as exc:
        st.error(f"Paper account connection failed: {exc}")

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    try:
        with st.spinner("Pulling real market data and running MAXPROFIT..."):
            prices = load_market_data(tuple(UNIVERSE))
            ranked = rank_market(prices, risk_profile)
            selected = ranked[ranked["Score"] >= signal_threshold].head(positions_count).copy()
            if selected.empty:
                selected = ranked.head(positions_count).copy()
            curve, trades = backtest(prices, positions_count, risk_profile, signal_threshold, capital)
            st.session_state.results = (prices, ranked, selected, curve, trades)
            st.session_state.last_run = datetime.now(timezone.utc)
    except Exception as exc:
        st.error(f"MAXPROFIT run failed: {exc}")

if st.session_state.results:
    prices, ranked, selected, curve, trades = st.session_state.results

    st.success(f"MAXPROFIT ranked {len(ranked)} symbols using the latest {LOOKBACK_BARS} trading sessions of real market data.")
    st.caption(f"Last analysis: {st.session_state.get('last_run', datetime.now(timezone.utc)).strftime('%Y-%m-%d %H:%M UTC')}")

    st.subheader("🏆 Current MAXPROFIT Selections")
    allocation = capital / len(selected)
    display_selected = selected.copy()
    display_selected["Allocation"] = allocation
    display_selected["Est. 1-Day Risk"] = allocation * display_selected["Volatility"]
    st.dataframe(display_selected.style.format({
        "Price":"${:,.2f}", "3M Return":"{:.2%}", "20D Return":"{:.2%}", "10D Return":"{:.2%}",
        "Volatility":"{:.2%}", "Downside Vol":"{:.2%}", "Score":"{:.4f}",
        "Allocation":"${:,.2f}", "Est. 1-Day Risk":"${:,.2f}"
    }), width="stretch", hide_index=True)

    if not curve.empty:
        ending = float(curve.iloc[-1]["Portfolio Value"])
        pnl = ending - capital
        total_return = ending / capital - 1
        drawdown = curve["Portfolio Value"] / curve["Portfolio Value"].cummax() - 1
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Backtest P/L", f"${pnl:,.2f}", f"{total_return:.2%}")
        m2.metric("Ending Value", f"${ending:,.2f}")
        m3.metric("Max Drawdown", f"{float(drawdown.min()):.2%}")
        m4.metric("Completed Trades", len(trades))
        st.subheader("📈 Backtest Profit / Loss")
        chart = curve.copy()
        chart["Profit / Loss"] = chart["Portfolio Value"] - capital
        st.line_chart(chart, width="stretch")

    st.subheader("📊 Full Market Ranking")
    st.dataframe(ranked.style.format({
        "Price":"${:,.2f}", "3M Return":"{:.2%}", "20D Return":"{:.2%}", "10D Return":"{:.2%}",
        "Volatility":"{:.2%}", "Downside Vol":"{:.2%}", "Score":"{:.4f}"
    }), width="stretch", hide_index=True)

    if not trades.empty:
        st.subheader("🧾 Backtest Trade History")
        st.dataframe(trades.style.format({"Entry":"${:,.2f}", "Exit":"${:,.2f}", "Return":"{:.2%}"}), width="stretch", hide_index=True)

    # --------------------------------------------------------
    # PAPER TRADING CONTROL PANEL
    # --------------------------------------------------------
    st.divider()
    st.header("🧪 PAPER TRADING CONTROL")
    st.write("MAXPROFIT will not place a paper order until you explicitly confirm the action below.")
    confirm = st.checkbox("I understand this button submits PAPER orders to my Alpaca paper account.")

    p1, p2 = st.columns(2)
    with p1:
        if st.button("🟢 BUY CURRENT MAXPROFIT SELECTIONS", disabled=not confirm, use_container_width=True):
            try:
                orders_df, deployed, buying_power = submit_selected_paper_trades(display_selected, capital)
                st.success(f"Submitted {len(orders_df)} PAPER buy orders. Requested deployment: ${deployed:,.2f} of ${buying_power:,.2f} available buying power.")
                st.dataframe(orders_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Paper buy failed: {exc}")

    with p2:
        if st.button("🔴 RUN D6 EXIT MANAGER", disabled=not confirm, use_container_width=True):
            try:
                eligible = eligible_d6_exits()
                if eligible.empty:
                    st.info("No paper positions are currently eligible for a D6 exit.")
                else:
                    st.dataframe(eligible, width="stretch", hide_index=True)
                    exits = eligible[eligible["D6 Eligible"]]
                    if exits.empty:
                        st.info("No tagged VAST CASH positions have reached five completed trading sessions yet.")
                    else:
                        exit_results = []
                        for symbol in exits["Symbol"]:
                            response = close_paper_position(symbol)
                            exit_results.append({"Symbol": symbol, "Order ID": response.get("id", ""), "Status": response.get("status", "")})
                        st.success(f"Submitted {len(exit_results)} D6 PAPER exit orders.")
                        st.dataframe(pd.DataFrame(exit_results), width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"D6 exit manager failed: {exc}")

    st.subheader("💼 Current Paper Positions")
    try:
        positions = paper_positions()
        if positions:
            position_rows = [{
                "Symbol": p.get("symbol", ""),
                "Qty": float(p.get("qty", 0)),
                "Market Value": float(p.get("market_value", 0)),
                "Unrealized P/L": float(p.get("unrealized_pl", 0)),
                "Avg Entry": float(p.get("avg_entry_price", 0)),
                "Current": float(p.get("current_price", 0)),
            } for p in positions]
            st.dataframe(pd.DataFrame(position_rows).style.format({
                "Qty":"{:.6f}", "Market Value":"${:,.2f}", "Unrealized P/L":"${:,.2f}",
                "Avg Entry":"${:,.2f}", "Current":"${:,.2f}"
            }), width="stretch", hide_index=True)
        else:
            st.caption("No open paper positions.")
    except Exception as exc:
        st.error(f"Could not read paper positions: {exc}")

    st.subheader("📋 Recent Paper Orders")
    try:
        orders = paper_orders(limit=50)
        if orders:
            order_rows = [{
                "ID": o.get("id", ""),
                "Symbol": o.get("symbol", ""),
                "Side": o.get("side", ""),
                "Status": o.get("status", ""),
                "Qty": o.get("qty", ""),
                "Notional": o.get("notional", ""),
                "Submitted": o.get("submitted_at", ""),
                "Client ID": o.get("client_order_id", ""),
            } for o in orders]
            st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
        else:
            st.caption("No paper orders found.")
    except Exception as exc:
        st.error(f"Could not read paper orders: {exc}")

else:
    st.subheader("Ready to Test")
    st.write("Choose your four strategy inputs and press **RUN MAXPROFIT**. The engine will use real historical market data, rank the universe, run the D1 → D6 backtest, and prepare the current selections for paper execution.")

st.divider()
st.caption("VAST CASH • MAXPROFIT • 92-session lookback • Buy D1 / Sell D6 • Max 10 positions • PAPER ONLY")
