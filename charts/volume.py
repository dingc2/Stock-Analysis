"""Volume charts: color-coded bars and volume profile."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def create_volume_chart(df: pd.DataFrame, title: str = "Volume") -> go.Figure:
    """Color-coded volume bars (green=up day, red=down day) with MA overlay."""
    colors = [
        config.COLORS["volume_up"] if row["Close"] >= row["Open"] else config.COLORS["volume_down"]
        for _, row in df.iterrows()
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7,
        )
    )

    # Volume moving average (20-day)
    if len(df) >= 20:
        vol_ma = df["Volume"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vol_ma,
                line=dict(color=config.COLORS["primary"], width=2),
                name="Vol MA(20)",
            )
        )

    fig.update_layout(
        title=title,
        height=350,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_volume_profile(df: pd.DataFrame, bins: int = 30) -> go.Figure:
    """Horizontal volume profile showing volume at price levels."""
    price_min = df["Low"].min()
    price_max = df["High"].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_profile = np.zeros(bins)

    for _, row in df.iterrows():
        for i in range(bins):
            if row["Low"] <= bin_edges[i + 1] and row["High"] >= bin_edges[i]:
                vol_profile[i] += row["Volume"] / max(
                    1, int((row["High"] - row["Low"]) / (bin_edges[1] - bin_edges[0])) + 1
                )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=bin_centers,
            x=vol_profile,
            orientation="h",
            marker_color=config.COLORS["primary"],
            opacity=0.7,
            name="Volume Profile",
        )
    )

    fig.update_layout(
        title="Volume Profile",
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_title="Volume",
        yaxis_title="Price ($)",
    )

    return fig


def create_volume_price_chart(df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: price line + volume bars."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            line=dict(color=config.COLORS["primary"], width=2),
            name="Close Price",
        ),
        secondary_y=False,
    )

    colors = [
        config.COLORS["volume_up"] if row["Close"] >= row["Open"] else config.COLORS["volume_down"]
        for _, row in df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.4,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Price & Volume",
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=50, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)

    return fig
