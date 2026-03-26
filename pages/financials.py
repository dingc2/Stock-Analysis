"""Financials page: key financial metrics table."""

import streamlit as st

from data.base import DataProvider
from data.cache import get_info_cached


def render(provider: DataProvider, ticker: str):
    """Render the financials tab."""
    try:
        info = get_info_cached(provider, ticker)
    except Exception as e:
        st.error(f"Failed to fetch info for {ticker}: {e}")
        return

    if not info:
        st.warning("No financial data available.")
        return

    st.subheader(f"{info.get('shortName', ticker)} Financials")

    # Valuation metrics
    st.markdown("#### Valuation")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pe = info.get("trailingPE")
        st.metric("P/E (TTM)", f"{pe:.2f}" if pe else "N/A")
    with col2:
        fpe = info.get("forwardPE")
        st.metric("Forward P/E", f"{fpe:.2f}" if fpe else "N/A")
    with col3:
        pb = info.get("priceToBook")
        st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
    with col4:
        ps = info.get("priceToSalesTrailing12Months")
        st.metric("P/S Ratio", f"{ps:.2f}" if ps else "N/A")

    # Earnings
    st.markdown("#### Earnings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        eps = info.get("trailingEps")
        st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
    with col2:
        feps = info.get("forwardEps")
        st.metric("Forward EPS", f"${feps:.2f}" if feps else "N/A")
    with col3:
        rev = info.get("totalRevenue")
        if rev:
            if rev >= 1e9:
                rev_str = f"${rev / 1e9:.2f}B"
            elif rev >= 1e6:
                rev_str = f"${rev / 1e6:.2f}M"
            else:
                rev_str = f"${rev:,.0f}"
            st.metric("Revenue", rev_str)
        else:
            st.metric("Revenue", "N/A")
    with col4:
        margin = info.get("profitMargins")
        st.metric("Profit Margin", f"{margin * 100:.1f}%" if margin else "N/A")

    # Dividends & Risk
    st.markdown("#### Dividends & Risk")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        div_yield = info.get("dividendYield")
        st.metric("Dividend Yield", f"{div_yield * 100:.2f}%" if div_yield else "N/A")
    with col2:
        beta = info.get("beta")
        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    with col3:
        high52 = info.get("fiftyTwoWeekHigh")
        st.metric("52W High", f"${high52:,.2f}" if high52 else "N/A")
    with col4:
        low52 = info.get("fiftyTwoWeekLow")
        st.metric("52W Low", f"${low52:,.2f}" if low52 else "N/A")

    # Company info
    description = info.get("longBusinessSummary")
    if description:
        st.markdown("#### About")
        st.write(description)
