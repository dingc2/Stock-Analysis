"""Comparison page: multi-ticker normalized price comparison."""

import streamlit as st

import config
from data.base import DataProvider
from data.cache import get_history_cached, get_quote_cached
from charts.comparison import create_comparison_chart


def render(provider: DataProvider, ticker: str, period: str, interval: str):
    """Render the comparison tab."""
    st.subheader("Compare Tickers")

    # Ticker input
    comparison_input = st.text_input(
        f"Enter tickers to compare (comma-separated, max {config.MAX_COMPARISON_TICKERS})",
        value=st.session_state.get("comparison_tickers", ""),
        placeholder="e.g., MSFT, GOOGL, AMZN",
    )

    # Parse tickers
    tickers = [ticker]  # Always include main ticker
    if comparison_input:
        additional = [t.strip().upper() for t in comparison_input.split(",") if t.strip()]
        additional = [t for t in additional if t != ticker.upper()]
        tickers.extend(additional[: config.MAX_COMPARISON_TICKERS - 1])

    st.session_state["comparison_tickers"] = comparison_input

    if len(tickers) < 2:
        st.info("Enter at least one additional ticker to compare.")
        return

    # Fetch data for all tickers
    ticker_data = {}
    for t in tickers:
        try:
            df = get_history_cached(provider, t, period=period, interval=interval)
            if not df.empty:
                ticker_data[t] = df
            else:
                st.warning(f"No data for {t}")
        except Exception as e:
            st.warning(f"Failed to fetch {t}: {e}")

    if len(ticker_data) < 2:
        st.warning("Need data for at least 2 tickers to compare.")
        return

    # Normalized comparison chart
    fig = create_comparison_chart(ticker_data, title=f"Normalized Comparison - {period}")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics comparison table
    st.subheader("Key Metrics Comparison")
    metrics_data = []
    for t in ticker_data:
        try:
            quote = get_quote_cached(provider, t)
            df = ticker_data[t]
            pct_change = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            metrics_data.append(
                {
                    "Ticker": t,
                    "Price": f"${quote.price:,.2f}",
                    "Change %": f"{quote.change_percent:+.2f}%",
                    f"Period Return": f"{pct_change:+.2f}%",
                    "Volume": f"{quote.volume:,}",
                }
            )
        except Exception:
            pass

    if metrics_data:
        st.dataframe(metrics_data, use_container_width=True, hide_index=True)
