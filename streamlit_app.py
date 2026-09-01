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

st.metric("Lookback period", "92 days")
st.metric("Maximum positions", "10")
st.metric("Holding period", "5 trading days")

st.info("No Alpaca. No brokerage connection. No API keys. No external data calls in this foundation build.")

st.divider()
st.caption("VAST CASH • Foundation Build 1.0")
