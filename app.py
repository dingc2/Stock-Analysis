"""Streamlit entry point: page config, sidebar, tab routing."""

import streamlit as st

import config
from data import get_provider
from pages import overview, technicals, financials, comparison, volume

# Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize provider
provider = get_provider()

# --- Sidebar ---
with st.sidebar:
    st.title("Stock Analysis")

    # Ticker search
    ticker_input = st.text_input(
        "Ticker Symbol",
        value=st.session_state.get("selected_ticker", config.DEFAULT_TICKER),
        placeholder="e.g., AAPL",
    ).strip().upper()

    # Search suggestions
    if ticker_input and ticker_input != st.session_state.get("selected_ticker"):
        results = provider.search_ticker(ticker_input)
        if results:
            options = [f"{r['symbol']} - {r['name']}" for r in results[:5]]
            selected = st.selectbox("Search results", options, index=0)
            if selected:
                ticker_input = selected.split(" - ")[0]

    st.session_state["selected_ticker"] = ticker_input
    ticker = ticker_input

    st.divider()

    # Period selector
    period = st.selectbox(
        "Period",
        config.PERIOD_OPTIONS,
        index=config.PERIOD_OPTIONS.index(config.DEFAULT_PERIOD),
    )
    st.session_state["period"] = period

    # Interval selector (filtered by period)
    available_intervals = config.INTERVAL_OPTIONS.get(period, ["1d"])
    default_idx = (
        available_intervals.index(config.DEFAULT_INTERVAL)
        if config.DEFAULT_INTERVAL in available_intervals
        else 0
    )
    interval = st.selectbox("Interval", available_intervals, index=default_idx)
    st.session_state["interval"] = interval

    st.divider()
    st.caption(f"Data: {config.DATA_PROVIDER}")

# --- Main Content ---
tab_overview, tab_technicals, tab_financials, tab_comparison, tab_volume = st.tabs(
    ["Overview", "Technicals", "Financials", "Comparison", "Volume"]
)

with tab_overview:
    overview.render(provider, ticker, period, interval)

with tab_technicals:
    technicals.render(provider, ticker, period, interval)

with tab_financials:
    financials.render(provider, ticker)

with tab_comparison:
    comparison.render(provider, ticker, period, interval)

with tab_volume:
    volume.render(provider, ticker, period, interval)
