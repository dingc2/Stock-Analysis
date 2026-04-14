"""Overview page: current quote, key metrics, mini sparkline chart, and composite signal score."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta

from data.base import DataProvider
from data.cache import get_quote_cached, get_quote_live, get_history_cached, get_history_live
from charts.price import create_line_chart
from indicators import technical as ind
from indicators.composite import compute_signal_score, get_signal_color
from ml import get_available_models
from views import render_timeframe_buttons


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def _make_gauge(score: float, label: str) -> go.Figure:
    """Build a Plotly gauge figure for the composite signal score."""
    color = get_signal_color(label)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"font": {"size": 36, "color": color}, "suffix": ""},
        title={"text": f"<b>Signal Score</b><br><span style='font-size:1.1em;color:{color}'>{label}</span>",
               "font": {"size": 14}},
        gauge={
            "axis": {
                "range": [-100, 100],
                "tickwidth": 1,
                "tickcolor": "#555",
                "tickvals": [-100, -50, -20, 0, 20, 50, 100],
                "ticktext": ["-100", "-50", "-20", "0", "+20", "+50", "+100"],
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-100, -50], "color": "rgba(213,0,0,0.25)"},
                {"range": [-50, -20], "color": "rgba(255,109,0,0.2)"},
                {"range": [-20,  20], "color": "rgba(255,215,64,0.15)"},
                {"range": [ 20,  50], "color": "rgba(105,240,174,0.2)"},
                {"range": [ 50, 100], "color": "rgba(0,200,83,0.25)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=60, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    return fig


# ---------------------------------------------------------------------------
# Sub-score breakdown bar
# ---------------------------------------------------------------------------

def _render_sub_score_bar(label: str, value: float | None, weight: float, display_override: str | None = None):
    """Render a single sub-score row: label | mini bar | value."""
    if value is None or np.isnan(value):
        display = "N/A"
        bar_pct = 50            # centred
        bar_color = "#555"
    else:
        display = display_override if display_override is not None else f"{value:+.2f}"
        bar_pct = int((value + 1) / 2 * 100)   # -1→0 %, +1→100 %
        bar_color = (
            "#00c853" if value >= 0.2 else
            "#d50000" if value <= -0.2 else
            "#ffd740"
        )

    # Filled bar centred on zero (50 %)
    bar_html = (
        f"<div style='background:#2a2a2a;border-radius:4px;height:10px;position:relative;'>"
        f"<div style='position:absolute;left:50%;top:0;bottom:0;width:1px;background:#555;'></div>"
        f"<div style='position:absolute;"
        f"{'left:50%;width:' + str(abs(bar_pct - 50)) + '%;' if bar_pct >= 50 else 'right:' + str(50 - bar_pct) + '%;left:' + str(bar_pct) + '%;width:' + str(50 - bar_pct) + '%;'}"
        f"top:0;bottom:0;background:{bar_color};border-radius:4px;'></div>"
        f"</div>"
    )

    col_a, col_b, col_c, col_d = st.columns([2, 4, 1.2, 1])
    with col_a:
        st.markdown(f"<div style='padding-top:2px;font-size:0.85em;color:#aaa;'>{label}</div>",
                    unsafe_allow_html=True)
    with col_b:
        st.markdown(bar_html, unsafe_allow_html=True)
    with col_c:
        color = "#00c853" if (value is not None and not np.isnan(value) and value >= 0.2) else \
                "#d50000" if (value is not None and not np.isnan(value) and value <= -0.2) else "#ffd740"
        st.markdown(
            f"<div style='text-align:right;font-size:0.85em;color:{color};font-weight:600;'>{display}</div>",
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f"<div style='text-align:right;font-size:0.75em;color:#666;'>{int(weight*100)}%</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Score section (runs inside the auto-refresh fragment)
# ---------------------------------------------------------------------------

def _render_signal_section(provider: DataProvider, ticker: str, period: str, interval: str,
                            live: bool, include_ml: bool):
    """Fetch data, compute score, render gauge and breakdown."""
    try:
        if live:
            df = get_history_live(provider, ticker, period=period, interval=interval, include_vix=True)
        else:
            df = get_history_cached(provider, ticker, period=period, interval=interval, include_vix=True)
    except Exception as e:
        st.error(f"Failed to fetch history for signal score: {e}")
        return

    if df.empty or len(df) < 20:
        st.warning("Not enough historical data to compute a signal score (need at least 20 bars).")
        return

    df = df.copy()

    # Compute all indicators needed by the scorer
    df = ind.add_all(df, include_advanced=True)   # SMA, EMA, RSI, MACD, BB, Stoch, ADX, ATR, ROC, MFI
    df = ind.add_cmf(df)
    df = ind.add_obv(df)
    df = ind.detect_candlestick_patterns(df)

    # Optionally run ML models
    ml_results: list[dict] = []
    ml_errors: list[str] = []
    if include_ml:
        models = get_available_models()
        for model in models:
            try:
                pred_df = model.predict(df)
                last_pred = pred_df.iloc[-1]
                prob_up = last_pred.get("Prob_Up") if hasattr(last_pred, "get") else \
                          (last_pred["Prob_Up"] if "Prob_Up" in pred_df.columns else None)
                if prob_up is not None and not pd.isna(prob_up):
                    ml_results.append({"prob_up": float(prob_up), "name": model.get_name()})
            except Exception as e:
                ml_errors.append(f"{model.get_name()}: {str(e)}")

    df = compute_signal_score(df, ml_results=ml_results if ml_results else None)

    last = df.iloc[-1]
    score = float(last["Signal_Score"]) if not pd.isna(last["Signal_Score"]) else 0.0
    label = str(last["Signal_Label"]) if last["Signal_Label"] else "Neutral"

    import config as cfg
    weights = cfg.SIGNAL_WEIGHTS

    # Calculate effective weights based on available scores
    available_cols = []
    if "Score_Trend" in df.columns and not pd.isna(last["Score_Trend"]): available_cols.append("trend")
    if "Score_Momentum" in df.columns and not pd.isna(last["Score_Momentum"]): available_cols.append("momentum")
    if "Score_Volume" in df.columns and not pd.isna(last["Score_Volume"]): available_cols.append("volume")
    if "Score_Pattern" in df.columns and not pd.isna(last["Score_Pattern"]): available_cols.append("pattern")
    if "Score_ML" in df.columns and not pd.isna(last["Score_ML"]): available_cols.append("ml")
    
    total_available_weight = sum(weights[k] for k in available_cols)
    effective_weights = {
        k: (weights[k] / total_available_weight) if total_available_weight > 0 else 0.0
        for k in weights
    }

    # Layout: gauge left, breakdown right
    g_col, b_col = st.columns([1, 1])

    with g_col:
        fig = _make_gauge(score, label)
        st.plotly_chart(fig, use_container_width=True)

    with b_col:
        st.markdown("**Score breakdown**")
        st.caption("Each bar shows the sub-score (-1 to +1). Weight = share of final score.")

        sub_map = [
            ("Trend",    "Score_Trend",    effective_weights["trend"]),
            ("Momentum", "Score_Momentum", effective_weights["momentum"]),
            ("Volume",   "Score_Volume",   effective_weights["volume"]),
            ("Pattern",  "Score_Pattern",  effective_weights["pattern"]),
            ("ML Models","Score_ML",       effective_weights["ml"]),
        ]
        for name, col, w in sub_map:
            val = float(last[col]) if col in df.columns and not pd.isna(last[col]) else None
            override = "No pattern" if name == "Pattern" and val == 0.0 else None
            _render_sub_score_bar(name, val, w, display_override=override)

        if ml_results:
            names = ", ".join(r["name"] for r in ml_results)
            st.caption(f"ML: {names}")
        elif include_ml:
            if ml_errors:
                error_str = " | ".join(ml_errors)
                st.caption(f"ML models unavailable ({error_str}) — ML weight redistributed.")
            else:
                st.caption("ML models unavailable — ML weight redistributed.")

    # Interpretation guide (collapsible)
    with st.expander("How to read this score", expanded=False):
        st.markdown("""
| Score range | Label | Interpretation |
|---|---|---|
| +50 to +100 | **Strong Buy** | Most indicators align bullishly; elevated ML confidence |
| +20 to +50  | **Buy** | Majority of signals lean bullish |
| −20 to +20  | **Neutral** | Mixed or conflicting signals; no clear edge |
| −50 to −20  | **Sell** | Majority of signals lean bearish |
| −100 to −50 | **Strong Sell** | Most indicators align bearishly |

**Component weights** (configurable in `config.py → SIGNAL_WEIGHTS`):

- **Trend (25%)** — Price vs SMA/EMA (ATR-normalised), MACD histogram direction, ADX directional index
- **Momentum (25%)** — RSI, Stochastic %K, MFI, Rate of Change
- **Volume (10%)** — Chaikin Money Flow, OBV trend vs 10-bar average
- **Patterns (10%)** — Most recent candlestick pattern (+1 bullish / −1 bearish)
- **ML Models (30%)** — XGBoost + LSTM next-day up-probability consensus

> **Disclaimer:** This score is a quantitative summary of technical signals, not financial advice.
> Past patterns do not guarantee future results.
        """)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(provider: DataProvider, ticker: str, period: str, interval: str):
    """Render the overview tab."""
    render_timeframe_buttons(period, interval, key_prefix="overview_")

    refresh_secs = st.session_state.get("auto_refresh_seconds")
    run_every = timedelta(seconds=refresh_secs) if refresh_secs else None

    @st.fragment(run_every=run_every)
    def content():
        live = st.session_state.get("auto_refresh", False)

        # --- Quote ---
        try:
            quote = get_quote_live(provider, ticker) if live else get_quote_cached(provider, ticker)
        except Exception as e:
            st.error(f"Failed to fetch quote for {ticker}: {e}")
            return

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

        st.divider()

        # --- Composite Signal Score ---
        st.subheader("Composite Signal Score")
        include_ml = st.toggle(
            "Include ML model predictions",
            value=True,
            help="XGBoost and LSTM models retrain on each load — uncheck to skip (faster).",
            key="overview_include_ml",
        )
        _render_signal_section(provider, ticker, period, interval, live, include_ml)

        st.divider()

        # --- Price history ---
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
