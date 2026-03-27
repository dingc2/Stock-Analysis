"""Streamlit entry point: page config, sidebar, tab routing."""

import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

import config
from data import get_provider
from views import overview, technicals, financials, comparison, volume

# Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize provider
provider = get_provider()

# Initialize session state defaults
if "period" not in st.session_state:
    st.session_state["period"] = config.DEFAULT_PERIOD
if "interval" not in st.session_state:
    st.session_state["interval"] = config.DEFAULT_INTERVAL

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

    # Period selector (syncs with timeframe preset buttons)
    current_period = st.session_state["period"]
    period_idx = (
        config.PERIOD_OPTIONS.index(current_period)
        if current_period in config.PERIOD_OPTIONS
        else config.PERIOD_OPTIONS.index(config.DEFAULT_PERIOD)
    )
    period = st.selectbox("Period", config.PERIOD_OPTIONS, index=period_idx)
    st.session_state["period"] = period

    # Interval selector (filtered by period, syncs with presets)
    available_intervals = config.INTERVAL_OPTIONS.get(period, ["1d"])
    current_interval = st.session_state["interval"]
    if current_interval in available_intervals:
        interval_idx = available_intervals.index(current_interval)
    else:
        interval_idx = 0
    interval = st.selectbox("Interval", available_intervals, index=interval_idx)
    st.session_state["interval"] = interval

    st.divider()

    # Auto-refresh
    st.subheader("Live Refresh")
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)
    market_open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    market_is_open = now_et.weekday() < 5 and market_open_t <= now_et <= market_close_t
    st.caption(f"Market: **{'Open' if market_is_open else 'Closed'}**")

    auto_refresh = st.toggle(
        "Auto-refresh",
        value=st.session_state.get("auto_refresh", False),
    )
    st.session_state["auto_refresh"] = auto_refresh
    if auto_refresh:
        refresh_labels = list(config.AUTO_REFRESH_INTERVALS.keys())
        refresh_key = st.selectbox(
            "Refresh interval",
            refresh_labels,
            index=refresh_labels.index(config.DEFAULT_REFRESH_INTERVAL),
        )
        st.session_state["auto_refresh_seconds"] = config.AUTO_REFRESH_INTERVALS[refresh_key]
    else:
        st.session_state["auto_refresh_seconds"] = None

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
