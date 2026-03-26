"""Normalized percentage-change overlay for multi-ticker comparison."""

import pandas as pd
import plotly.graph_objects as go

import config

# Colors for comparison lines
COMPARISON_COLORS = [
    config.COLORS["primary"],
    config.COLORS["secondary"],
    config.COLORS["success"],
    config.COLORS["danger"],
    config.COLORS["info"],
]


def create_comparison_chart(
    ticker_data: dict[str, pd.DataFrame],
    title: str = "Normalized Price Comparison (%)",
) -> go.Figure:
    """Overlay normalized percentage change for multiple tickers.

    Args:
        ticker_data: dict mapping ticker symbol -> OHLCV DataFrame
        title: chart title
    """
    fig = go.Figure()

    for i, (ticker, df) in enumerate(ticker_data.items()):
        if df.empty:
            continue
        normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized,
                mode="lines",
                line=dict(color=color, width=2),
                name=ticker,
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Change (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
