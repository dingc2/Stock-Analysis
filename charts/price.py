"""Candlestick and line price charts with indicator overlays and subplots."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def create_candlestick(
    df: pd.DataFrame,
    title: str = "",
    # Phase 1 overlays
    show_sma: bool = False,
    show_ema: bool = False,
    show_bollinger: bool = False,
    show_rsi: bool = False,
    show_macd: bool = False,
    sma_period: int = config.SMA_PERIOD,
    ema_period: int = config.EMA_PERIOD,
    rsi_period: int = config.RSI_PERIOD,
    # Phase 2 overlays
    show_vwap: bool = False,
    show_ichimoku: bool = False,
    show_supertrend: bool = False,
    show_keltner: bool = False,
    # Phase 2 subplots
    show_stochastic: bool = False,
    show_williams_r: bool = False,
    show_adx: bool = False,
    show_atr: bool = False,
    # Phase 2 extras
    show_patterns: bool = False,
    show_squeeze: bool = False,
    sr_levels: list[float] | None = None,
    fib_levels: dict | None = None,
) -> go.Figure:
    """Create a candlestick chart with optional indicator overlays and subplots."""

    # Build subplot layout dynamically
    subplots = []  # list of (name, height_ratio)
    if show_rsi:
        subplots.append("RSI")
    if show_macd:
        subplots.append("MACD")
    if show_stochastic:
        subplots.append("Stochastic")
    if show_williams_r:
        subplots.append("Williams %R")
    if show_adx:
        subplots.append("ADX")
    if show_atr:
        subplots.append("ATR")

    subplot_count = 1 + len(subplots)
    subplot_titles = [title] + subplots
    price_height = max(0.4, 0.8 - 0.12 * len(subplots))
    sub_height = (1.0 - price_height) / max(1, len(subplots))
    row_heights = [price_height] + [sub_height] * len(subplots)

    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    # --- Row 1: Candlestick + overlays ---
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=config.COLORS["up"],
            decreasing_line_color=config.COLORS["down"],
            name="Price",
        ),
        row=1, col=1,
    )

    # SMA
    if show_sma and f"SMA_{sma_period}" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"SMA_{sma_period}"],
            line=dict(color=config.COLORS["sma"], width=1.5),
            name=f"SMA({sma_period})",
        ), row=1, col=1)

    # EMA
    if show_ema and f"EMA_{ema_period}" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"EMA_{ema_period}"],
            line=dict(color=config.COLORS["ema"], width=1.5),
            name=f"EMA({ema_period})",
        ), row=1, col=1)

    # Bollinger Bands
    if show_bollinger:
        _add_bollinger_overlay(fig, df)

    # VWAP
    if show_vwap and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"],
            line=dict(color=config.COLORS["vwap"], width=1.5, dash="dot"),
            name="VWAP",
        ), row=1, col=1)

    # Ichimoku Cloud
    if show_ichimoku:
        _add_ichimoku_overlay(fig, df)

    # Supertrend
    if show_supertrend:
        _add_supertrend_overlay(fig, df)

    # Keltner Channels
    if show_keltner:
        _add_keltner_overlay(fig, df)

    # Support/Resistance lines
    if sr_levels:
        current_price = df["Close"].iloc[-1]
        for lvl in sr_levels:
            is_support = lvl < current_price
            fig.add_hline(
                y=lvl, line_dash="dash",
                line_color=config.COLORS["support"] if is_support else config.COLORS["resistance"],
                opacity=0.6, row=1, col=1,
                annotation_text=f"{'S' if is_support else 'R'} {lvl:.2f}",
                annotation_position="right",
            )

    # Fibonacci levels
    if fib_levels:
        for name, lvl in fib_levels.items():
            fig.add_hline(
                y=lvl, line_dash="dot",
                line_color=config.COLORS["fibonacci"], opacity=0.4,
                row=1, col=1,
                annotation_text=f"Fib {name} ({lvl:.2f})",
                annotation_position="left",
            )

    # Candlestick pattern annotations
    if show_patterns and "CDL_Pattern" in df.columns:
        pattern_df = df[df["CDL_Pattern"] != ""]
        if not pattern_df.empty:
            colors = [
                config.COLORS["up"] if s > 0
                else config.COLORS["down"] if s < 0
                else "gray"
                for s in pattern_df["CDL_Signal"]
            ]
            positions = [
                df.loc[idx, "Low"] * 0.995 if sig >= 0
                else df.loc[idx, "High"] * 1.005
                for idx, sig in zip(pattern_df.index, pattern_df["CDL_Signal"])
            ]
            fig.add_trace(go.Scatter(
                x=pattern_df.index, y=positions,
                mode="markers+text",
                marker=dict(symbol="diamond", size=8, color=colors),
                text=pattern_df["CDL_Pattern"],
                textposition="bottom center",
                textfont=dict(size=8),
                name="Patterns",
                showlegend=True,
            ), row=1, col=1)

    # Squeeze markers on price chart
    if show_squeeze and "Squeeze" in df.columns:
        squeeze_on = df[df["Squeeze"]]
        if not squeeze_on.empty:
            fig.add_trace(go.Scatter(
                x=squeeze_on.index,
                y=[df["Low"].min()] * len(squeeze_on),
                mode="markers",
                marker=dict(symbol="square", size=5,
                            color=config.COLORS["squeeze_on"]),
                name="Squeeze ON",
            ), row=1, col=1)

    # --- Subplots ---
    current_row = 2

    # RSI
    if show_rsi and f"RSI_{rsi_period}" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"RSI_{rsi_period}"],
            line=dict(color=config.COLORS["primary"], width=1.5),
            name=f"RSI({rsi_period})",
        ), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5,
                      row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5,
                      row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # MACD
    if show_macd:
        _add_macd_subplot(fig, df, current_row)
        current_row += 1

    # Stochastic
    if show_stochastic:
        stk = [c for c in df.columns if c.startswith("STOCHk_")]
        std = [c for c in df.columns if c.startswith("STOCHd_")]
        if stk:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[stk[0]],
                line=dict(color=config.COLORS["stoch_k"], width=1.5),
                name="%K",
            ), row=current_row, col=1)
        if std:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[std[0]],
                line=dict(color=config.COLORS["stoch_d"], width=1.5),
                name="%D",
            ), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5,
                      row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5,
                      row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # Williams %R
    if show_williams_r:
        wr_col = [c for c in df.columns if c.startswith("WILLR_")]
        if wr_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[wr_col[0]],
                line=dict(color=config.COLORS["willr"], width=1.5),
                name="Williams %R",
            ), row=current_row, col=1)
            fig.add_hline(y=-20, line_dash="dash", line_color="red", opacity=0.5,
                          row=current_row, col=1)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", opacity=0.5,
                          row=current_row, col=1)
            fig.update_yaxes(range=[-100, 0], row=current_row, col=1)
        current_row += 1

    # ADX
    if show_adx:
        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        dmp_col = [c for c in df.columns if c.startswith("DMP_")]
        dmn_col = [c for c in df.columns if c.startswith("DMN_")]
        if adx_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[adx_col[0]],
                line=dict(color=config.COLORS["adx"], width=2),
                name="ADX",
            ), row=current_row, col=1)
        if dmp_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[dmp_col[0]],
                line=dict(color=config.COLORS["di_plus"], width=1, dash="dash"),
                name="+DI",
            ), row=current_row, col=1)
        if dmn_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[dmn_col[0]],
                line=dict(color=config.COLORS["di_minus"], width=1, dash="dash"),
                name="-DI",
            ), row=current_row, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="gray", opacity=0.5,
                      row=current_row, col=1)
        current_row += 1

    # ATR
    if show_atr:
        atr_col = [c for c in df.columns if c.startswith("ATR_")]
        if atr_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[atr_col[0]],
                line=dict(color=config.COLORS["atr"], width=1.5),
                name="ATR",
                fill="tozeroy",
                fillcolor="rgba(23, 190, 207, 0.1)",
            ), row=current_row, col=1)
        current_row += 1

    # Layout
    total_height = 400 + 150 * len(subplots)
    fig.update_layout(
        height=total_height,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=20),
    )

    return fig


# ---------------------------------------------------------------------------
# Private helpers for overlay/subplot rendering
# ---------------------------------------------------------------------------

def _add_bollinger_overlay(fig: go.Figure, df: pd.DataFrame):
    bb_upper = [c for c in df.columns if c.startswith("BBU_")]
    bb_mid = [c for c in df.columns if c.startswith("BBM_")]
    bb_lower = [c for c in df.columns if c.startswith("BBL_")]

    if bb_upper:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[bb_upper[0]],
            line=dict(color=config.COLORS["bollinger_upper"], width=1, dash="dash"),
            name="BB Upper",
        ), row=1, col=1)
    if bb_mid:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[bb_mid[0]],
            line=dict(color=config.COLORS["bollinger_mid"], width=1, dash="dot"),
            name="BB Mid",
        ), row=1, col=1)
    if bb_lower:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[bb_lower[0]],
            line=dict(color=config.COLORS["bollinger_lower"], width=1, dash="dash"),
            name="BB Lower", fill="tonexty",
            fillcolor="rgba(148, 103, 189, 0.1)",
        ), row=1, col=1)


def _add_ichimoku_overlay(fig: go.Figure, df: pd.DataFrame):
    # Tenkan-sen (conversion line)
    if "ITS_9" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ITS_9"],
            line=dict(color=config.COLORS["ichimoku_tenkan"], width=1),
            name="Tenkan-sen",
        ), row=1, col=1)
    # Kijun-sen (base line)
    if "IKS_26" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["IKS_26"],
            line=dict(color=config.COLORS["ichimoku_kijun"], width=1),
            name="Kijun-sen",
        ), row=1, col=1)
    # Cloud (Senkou Span A and B)
    if "ISA_9" in df.columns and "ISB_26" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ISA_9"],
            line=dict(color="green", width=0),
            name="Senkou A", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ISB_26"],
            line=dict(color="red", width=0),
            name="Ichimoku Cloud", fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.15)",
        ), row=1, col=1)


def _add_supertrend_overlay(fig: go.Figure, df: pd.DataFrame):
    stl_col = [c for c in df.columns if c.startswith("SUPERTl_")]
    sts_col = [c for c in df.columns if c.startswith("SUPERTs_")]

    if stl_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[stl_col[0]],
            line=dict(color=config.COLORS["supertrend_up"], width=2),
            name="Supertrend Up", connectgaps=False,
        ), row=1, col=1)
    if sts_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sts_col[0]],
            line=dict(color=config.COLORS["supertrend_down"], width=2),
            name="Supertrend Down", connectgaps=False,
        ), row=1, col=1)


def _add_keltner_overlay(fig: go.Figure, df: pd.DataFrame):
    kc_upper = [c for c in df.columns if c.startswith("KCU")]
    kc_mid = [c for c in df.columns if c.startswith("KCB")]
    kc_lower = [c for c in df.columns if c.startswith("KCL")]

    if kc_upper:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[kc_upper[0]],
            line=dict(color=config.COLORS["keltner_upper"], width=1, dash="dash"),
            name="KC Upper",
        ), row=1, col=1)
    if kc_mid:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[kc_mid[0]],
            line=dict(color=config.COLORS["keltner_mid"], width=1, dash="dot"),
            name="KC Mid",
        ), row=1, col=1)
    if kc_lower:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[kc_lower[0]],
            line=dict(color=config.COLORS["keltner_lower"], width=1, dash="dash"),
            name="KC Lower", fill="tonexty",
            fillcolor="rgba(188, 189, 34, 0.08)",
        ), row=1, col=1)


def _add_macd_subplot(fig: go.Figure, df: pd.DataFrame, row: int):
    macd_col = [c for c in df.columns
                if c.startswith("MACD_") and "S" not in c and "H" not in c]
    signal_col = [c for c in df.columns if c.startswith("MACDs_")]
    hist_col = [c for c in df.columns if c.startswith("MACDh_")]

    if macd_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[macd_col[0]],
            line=dict(color=config.COLORS["macd"], width=1.5),
            name="MACD",
        ), row=row, col=1)
    if signal_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[signal_col[0]],
            line=dict(color=config.COLORS["macd_signal"], width=1.5),
            name="Signal",
        ), row=row, col=1)
    if hist_col:
        colors = [
            config.COLORS["macd_hist_pos"] if v >= 0
            else config.COLORS["macd_hist_neg"]
            for v in df[hist_col[0]]
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df[hist_col[0]],
            marker_color=colors, name="Histogram",
        ), row=row, col=1)


def create_line_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Simple line chart for sparklines / overview."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines",
            line=dict(color=config.COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)",
            name="Close",
        )
    )
    fig.update_layout(
        title=title, height=300, template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_title="", yaxis_title="Price ($)",
    )
    return fig
