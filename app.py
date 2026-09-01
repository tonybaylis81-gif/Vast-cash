import streamlit as st

st.set_page_config(page_title="VAST CASH", page_icon="💰", layout="wide")

st.title("💰 VAST CASH")
st.subheader("MAXPROFIT Engine")
st.success("VAST CASH is online.")

col1, col2, col3 = st.columns(3)
col1.metric("Status", "ONLINE")
col2.metric("Broker", "NONE")
col3.metric("Mode", "ANALYSIS")

st.divider()
st.header("Engine Configuration")
st.write("The clean VAST CASH foundation is running.")
st.write("Next layer: market-data ingestion and the MAXPROFIT ranking engine.")

lookback = st.number_input("Lookback period (days)", min_value=1, value=92, disabled=True)
max_stocks = st.number_input("Maximum positions", min_value=1, max_value=10, value=10, disabled=True)
hold_days = st.number_input("Holding period (trading days)", min_value=1, value=5, disabled=True)

st.info("No Alpaca. No brokerage connection. No API keys. No external data calls in this foundation build.")

st.divider()
st.caption("VAST CASH • Foundation Build 1.0")
