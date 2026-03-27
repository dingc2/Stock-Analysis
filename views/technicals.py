"""Technicals page: full interactive chart with indicator controls and ML overlay hook."""

import streamlit as st
import pandas as pd
from datetime import timedelta

import config
from data.base import DataProvider
from data.cache import get_history_cached, get_history_live
from indicators import technical as ind
from charts.price import create_candlestick
from ml import get_available_models
from views import render_timeframe_buttons


def render(provider: DataProvider, ticker: str, period: str, interval: str):
    """Render the technicals tab."""
    # Timeframe preset buttons
    render_timeframe_buttons(period, interval, key_prefix="tech_")

    # --- Indicator Controls (outside fragment so they don't flicker) ---
    st.subheader("Indicators")

    # Trend Overlays
    with st.expander("Trend Overlays", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_sma = st.checkbox("SMA", value=True)
            sma_period = st.slider("SMA Period", 5, 200, config.SMA_PERIOD) if show_sma else config.SMA_PERIOD
        with c2:
            show_ema = st.checkbox("EMA", value=False)
            ema_period = st.slider("EMA Period", 5, 200, config.EMA_PERIOD) if show_ema else config.EMA_PERIOD
        with c3:
            show_vwap = st.checkbox("VWAP", value=False)
            show_supertrend = st.checkbox("Supertrend", value=False)
        with c4:
            show_ichimoku = st.checkbox("Ichimoku Cloud", value=False)

    # Bands & Channels
    with st.expander("Bands & Channels"):
        c1, c2, c3 = st.columns(3)
        with c1:
            show_bollinger = st.checkbox("Bollinger Bands", value=False)
            if show_bollinger:
                bb_period = st.slider("BB Period", 5, 50, config.BOLLINGER_PERIOD)
                bb_std = st.slider("BB Std Dev", 1.0, 3.0, config.BOLLINGER_STD, 0.5)
            else:
                bb_period = config.BOLLINGER_PERIOD
                bb_std = config.BOLLINGER_STD
        with c2:
            show_keltner = st.checkbox("Keltner Channels", value=False)
        with c3:
            show_squeeze = st.checkbox("Bollinger Squeeze", value=False,
                                       help="Detects when Bollinger Bands are inside Keltner Channels")

    # Momentum Oscillators
    with st.expander("Momentum Oscillators"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_rsi = st.checkbox("RSI", value=False)
            rsi_period = st.slider("RSI Period", 5, 50, config.RSI_PERIOD) if show_rsi else config.RSI_PERIOD
        with c2:
            show_macd = st.checkbox("MACD", value=False)
        with c3:
            show_stochastic = st.checkbox("Stochastic", value=False)
        with c4:
            show_williams_r = st.checkbox("Williams %R", value=False)

    # Trend Strength & Volatility
    with st.expander("Trend Strength & Volatility"):
        c1, c2 = st.columns(2)
        with c1:
            show_adx = st.checkbox("ADX (Trend Strength)", value=False)
        with c2:
            show_atr = st.checkbox("ATR (Volatility)", value=False)

    # Patterns & Levels
    with st.expander("Patterns & Levels"):
        c1, c2, c3 = st.columns(3)
        with c1:
            show_patterns = st.checkbox("Candlestick Patterns", value=False)
        with c2:
            show_sr = st.checkbox("Support/Resistance", value=False)
        with c3:
            show_fib = st.checkbox("Fibonacci Retracement", value=False)

    # --- Chart with auto-refresh ---
    st.subheader("Chart")

    refresh_secs = st.session_state.get("auto_refresh_seconds")
    run_every = timedelta(seconds=refresh_secs) if refresh_secs else None

    @st.fragment(run_every=run_every)
    def chart_section():
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

        df = df.copy()

        # --- Compute indicators ---
        if show_sma:
            df = ind.add_sma(df, period=sma_period)
        if show_ema:
            df = ind.add_ema(df, period=ema_period)
        if show_vwap:
            df = ind.add_vwap(df)
        if show_ichimoku:
            df = ind.add_ichimoku(df)
        if show_supertrend:
            df = ind.add_supertrend(df)
        if show_bollinger or show_squeeze:
            df = ind.add_bollinger(df, period=bb_period, std=bb_std)
        if show_keltner or show_squeeze:
            df = ind.add_keltner(df)
        if show_rsi:
            df = ind.add_rsi(df, period=rsi_period)
        if show_macd:
            df = ind.add_macd(df)
        if show_stochastic:
            df = ind.add_stochastic(df)
        if show_williams_r:
            df = ind.add_williams_r(df)
        if show_adx:
            df = ind.add_adx(df)
        if show_atr:
            df = ind.add_atr(df)
        if show_patterns:
            df = ind.detect_candlestick_patterns(df)
        if show_squeeze:
            df = ind.detect_squeeze(df)

        sr_levels = ind.detect_support_resistance(df) if show_sr else None
        fib_levels = ind.compute_fibonacci_levels(df) if show_fib else None

        # --- Render chart ---
        fig = create_candlestick(
            df,
            title=f"{ticker} - {period}",
            show_sma=show_sma, show_ema=show_ema,
            show_bollinger=show_bollinger, show_rsi=show_rsi, show_macd=show_macd,
            sma_period=sma_period, ema_period=ema_period, rsi_period=rsi_period,
            show_vwap=show_vwap, show_ichimoku=show_ichimoku,
            show_supertrend=show_supertrend, show_keltner=show_keltner,
            show_stochastic=show_stochastic, show_williams_r=show_williams_r,
            show_adx=show_adx, show_atr=show_atr,
            show_patterns=show_patterns, show_squeeze=show_squeeze,
            sr_levels=sr_levels, fib_levels=fib_levels,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Detected patterns table ---
        if show_patterns and "CDL_Pattern" in df.columns:
            detected = df[df["CDL_Pattern"] != ""][["Close", "CDL_Pattern", "CDL_Signal"]].copy()
            if not detected.empty:
                detected["CDL_Signal"] = detected["CDL_Signal"].map(
                    {1: "Bullish", -1: "Bearish", 0: "Neutral"}
                )
                detected.columns = ["Close", "Pattern", "Signal"]
                st.caption(f"{len(detected)} candlestick pattern(s) detected")
                st.dataframe(detected.tail(15), use_container_width=True)

        # --- Squeeze status ---
        if show_squeeze and "Squeeze" in df.columns:
            squeeze_active = df["Squeeze"].iloc[-1]
            st.caption(f"Bollinger Squeeze: {'**ACTIVE** - volatility compression detected' if squeeze_active else 'Not active'}")

        # --- ML model overlay hook ---
        models = get_available_models()
        if models:
            st.subheader("ML Predictions")
            for model in models:
                with st.expander(f"{model.get_name()} - {model.get_description()}", expanded=True):
                    try:
                        prediction_df = model.predict(df)

                        last = prediction_df.iloc[-1]
                        pred_dir = last.get("Pred_Direction")
                        pred_prob = last.get("Pred_Probability")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if pred_dir is not None and not pd.isna(pred_dir):
                                direction = "Up" if pred_dir == 1 else "Down"
                                st.metric("Next-Day Prediction", direction)
                            else:
                                st.metric("Next-Day Prediction", "N/A")
                        with col2:
                            if pred_prob is not None and not pd.isna(pred_prob):
                                st.metric("Confidence", f"{pred_prob:.1%}")
                            else:
                                st.metric("Confidence", "N/A")
                        with col3:
                            train_acc = prediction_df.attrs.get("train_accuracy")
                            if train_acc is not None:
                                st.metric("Train Accuracy", f"{train_acc:.1%}")

                        display_cols = ["Close", "Pred_Direction", "Pred_Probability"]
                        display_cols = [c for c in display_cols if c in prediction_df.columns]
                        recent = prediction_df[display_cols].tail(10).copy()
                        if "Pred_Direction" in recent.columns:
                            recent["Pred_Direction"] = recent["Pred_Direction"].map(
                                {1.0: "Up", 0.0: "Down"}
                            )
                        if "Pred_Probability" in recent.columns:
                            recent["Pred_Probability"] = recent["Pred_Probability"].apply(
                                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                            )
                        st.dataframe(recent, use_container_width=True)
                    except Exception as e:
                        st.error(f"Model error: {e}")

    chart_section()
