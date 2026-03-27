"""Volume page: volume analysis, patterns, and correlations."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

import config
from data.base import DataProvider
from data.cache import get_history_cached, get_history_live
from charts.volume import create_volume_chart, create_volume_profile, create_volume_price_chart
from indicators.technical import add_obv, add_ad, add_cmf
from views import render_timeframe_buttons


def render(provider: DataProvider, ticker: str, period: str, interval: str):
    """Render the volume analysis tab."""
    # Timeframe preset buttons
    render_timeframe_buttons(period, interval, key_prefix="volume_")

    # Content with auto-refresh support
    refresh_secs = st.session_state.get("auto_refresh_seconds")
    run_every = timedelta(seconds=refresh_secs) if refresh_secs else None

    @st.fragment(run_every=run_every)
    def content():
        try:
            if st.session_state.get("auto_refresh"):
                df = get_history_live(provider, ticker, period=period, interval=interval)
            else:
                df = get_history_cached(provider, ticker, period=period, interval=interval)
        except Exception as e:
            st.error(f"Failed to fetch history for {ticker}: {e}")
            return

        if df.empty:
            st.warning("No historical data available.")
            return

        # Volume bars with MA overlay
        st.subheader("Volume Analysis")
        fig = create_volume_chart(df, title=f"{ticker} Volume")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Volume Profile")
            fig = create_volume_profile(df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Price & Volume")
            fig = create_volume_price_chart(df)
            st.plotly_chart(fig, use_container_width=True)

        # Day-of-week volume patterns (only for daily intervals)
        if interval in ("1d", "5d"):
            st.subheader("Day-of-Week Volume Patterns")
            df_copy = df.copy()
            df_copy["DayOfWeek"] = df_copy.index.dayofweek
            day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
            df_copy["DayName"] = df_copy["DayOfWeek"].map(day_names)

            avg_vol = df_copy.groupby("DayName")["Volume"].mean().reindex(
                ["Mon", "Tue", "Wed", "Thu", "Fri"]
            )

            col1, col2, col3, col4, col5 = st.columns(5)
            for col, (day, vol) in zip([col1, col2, col3, col4, col5], avg_vol.items()):
                with col:
                    st.metric(day, f"{vol:,.0f}")

        # Volume indicators (OBV, A/D, CMF)
        st.subheader("Volume Indicators")
        df_vol = df.copy()
        df_vol = add_obv(df_vol)
        df_vol = add_ad(df_vol)
        df_vol = add_cmf(df_vol)

        col1, col2 = st.columns(2)
        with col1:
            if "OBV" in df_vol.columns:
                fig_obv = go.Figure()
                fig_obv.add_trace(go.Scatter(
                    x=df_vol.index, y=df_vol["OBV"],
                    line=dict(color=config.COLORS["primary"], width=1.5),
                    fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.1)",
                    name="OBV",
                ))
                fig_obv.update_layout(
                    title="On-Balance Volume (OBV)", height=250,
                    template="plotly_white", margin=dict(l=50, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_obv, use_container_width=True)
        with col2:
            if "AD" in df_vol.columns:
                fig_ad = go.Figure()
                fig_ad.add_trace(go.Scatter(
                    x=df_vol.index, y=df_vol["AD"],
                    line=dict(color=config.COLORS["secondary"], width=1.5),
                    fill="tozeroy", fillcolor="rgba(255, 127, 14, 0.1)",
                    name="A/D",
                ))
                fig_ad.update_layout(
                    title="Accumulation/Distribution", height=250,
                    template="plotly_white", margin=dict(l=50, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_ad, use_container_width=True)

        cmf_col = [c for c in df_vol.columns if c.startswith("CMF_")]
        if cmf_col:
            fig_cmf = go.Figure()
            cmf_vals = df_vol[cmf_col[0]]
            colors = [config.COLORS["up"] if v >= 0 else config.COLORS["down"] for v in cmf_vals]
            fig_cmf.add_trace(go.Bar(
                x=df_vol.index, y=cmf_vals,
                marker_color=colors, name="CMF",
            ))
            fig_cmf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_cmf.update_layout(
                title="Chaikin Money Flow (CMF)", height=250,
                template="plotly_white", margin=dict(l=50, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cmf, use_container_width=True)

        # Volume statistics
        st.subheader("Volume Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
        with col2:
            st.metric("Max Volume", f"{df['Volume'].max():,.0f}")
        with col3:
            st.metric("Min Volume", f"{df['Volume'].min():,.0f}")
        with col4:
            latest_vol = df["Volume"].iloc[-1]
            avg_vol_val = df["Volume"].mean()
            ratio = latest_vol / avg_vol_val if avg_vol_val > 0 else 0
            st.metric("Latest vs Avg", f"{ratio:.2f}x")

    content()
