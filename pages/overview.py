"""Overview page: current quote, key metrics, mini sparkline chart."""

import streamlit as st
from datetime import timedelta

from data.base import DataProvider
from data.cache import get_quote_cached, get_quote_live, get_history_cached, get_history_live
from charts.price import create_line_chart
from pages import render_timeframe_buttons


def render(provider: DataProvider, ticker: str, period: str, interval: str):
    """Render the overview tab."""
    # Timeframe preset buttons
    render_timeframe_buttons(period, interval, key_prefix="overview_")

    # Content with auto-refresh support
    refresh_secs = st.session_state.get("auto_refresh_seconds")
    run_every = timedelta(seconds=refresh_secs) if refresh_secs else None

    @st.fragment(run_every=run_every)
    def content():
        live = st.session_state.get("auto_refresh", False)
        try:
            quote = get_quote_live(provider, ticker) if live else get_quote_cached(provider, ticker)
        except Exception as e:
            st.error(f"Failed to fetch quote for {ticker}: {e}")
            return

        # Price header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.metric(
                label=f"{quote.symbol} Price",
                value=f"${quote.price:,.2f}",
                delta=f"{quote.change:+.2f} ({quote.change_percent:+.2f}%)",
            )
        with col2:
            st.metric("Day High", f"${quote.day_high:,.2f}" if quote.day_high else "N/A")
        with col3:
            st.metric("Day Low", f"${quote.day_low:,.2f}" if quote.day_low else "N/A")

        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Open", f"${quote.open:,.2f}" if quote.open else "N/A")
        with col2:
            st.metric("Prev Close", f"${quote.previous_close:,.2f}" if quote.previous_close else "N/A")
        with col3:
            st.metric("Volume", f"{quote.volume:,}" if quote.volume else "N/A")
        with col4:
            if quote.market_cap:
                if quote.market_cap >= 1e12:
                    cap_str = f"${quote.market_cap / 1e12:.2f}T"
                elif quote.market_cap >= 1e9:
                    cap_str = f"${quote.market_cap / 1e9:.2f}B"
                elif quote.market_cap >= 1e6:
                    cap_str = f"${quote.market_cap / 1e6:.2f}M"
                else:
                    cap_str = f"${quote.market_cap:,.0f}"
                st.metric("Market Cap", cap_str)
            else:
                st.metric("Market Cap", "N/A")

        # Price history chart
        st.subheader("Price History")
        try:
            if live:
                df = get_history_live(provider, ticker, period=period, interval=interval)
            else:
                df = get_history_cached(provider, ticker, period=period, interval=interval)
            if not df.empty:
                fig = create_line_chart(df, title=f"{ticker} - {period}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No historical data available.")
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")

    content()
