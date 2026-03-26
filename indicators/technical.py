"""Pure-function technical indicators using pandas-ta.

Every function takes an OHLCV DataFrame and returns it with added columns.
No Streamlit dependency, no side effects.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

import config


# ---------------------------------------------------------------------------
# Phase 1 indicators
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, period: int = config.SMA_PERIOD) -> pd.DataFrame:
    """Add Simple Moving Average column."""
    df[f"SMA_{period}"] = ta.sma(df["Close"], length=period)
    return df


def add_ema(df: pd.DataFrame, period: int = config.EMA_PERIOD) -> pd.DataFrame:
    """Add Exponential Moving Average column."""
    df[f"EMA_{period}"] = ta.ema(df["Close"], length=period)
    return df


def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
    """Add Relative Strength Index column."""
    df[f"RSI_{period}"] = ta.rsi(df["Close"], length=period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """Add MACD, MACD Signal, and MACD Histogram columns."""
    macd = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    return df


def add_bollinger(
    df: pd.DataFrame,
    period: int = config.BOLLINGER_PERIOD,
    std: float = config.BOLLINGER_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands columns (upper, mid, lower)."""
    bbands = ta.bbands(df["Close"], length=period, std=std)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    return df


# ---------------------------------------------------------------------------
# Phase 2A: Advanced indicators
# ---------------------------------------------------------------------------

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price."""
    vwap = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    if vwap is not None:
        df["VWAP"] = vwap
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ichimoku Cloud components."""
    try:
        ich = ta.ichimoku(df["High"], df["Low"], df["Close"])
        if ich is not None and len(ich) >= 1 and ich[0] is not None:
            span_df = ich[0]  # ISA, ISB, ITS, IKS, ICS
            for col in span_df.columns:
                df[col] = span_df[col]
    except Exception:
        pass
    return df


def add_supertrend(
    df: pd.DataFrame, length: int = 7, multiplier: float = 3.0,
) -> pd.DataFrame:
    """Add Supertrend indicator."""
    st = ta.supertrend(df["High"], df["Low"], df["Close"],
                       length=length, multiplier=multiplier)
    if st is not None:
        df = pd.concat([df, st], axis=1)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index with +DI/-DI."""
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=period)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
    return df


def add_stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3,
) -> pd.DataFrame:
    """Add Stochastic Oscillator (%K and %D)."""
    stoch = ta.stoch(df["High"], df["Low"], df["Close"],
                     k=k, d=d, smooth_k=smooth_k)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Williams %R oscillator."""
    wr = ta.willr(df["High"], df["Low"], df["Close"], length=period)
    if wr is not None:
        df[f"WILLR_{period}"] = wr
    return df


def add_keltner(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0,
) -> pd.DataFrame:
    """Add Keltner Channels (upper, basis, lower)."""
    kc = ta.kc(df["High"], df["Low"], df["Close"],
               length=period, scalar=multiplier)
    if kc is not None:
        df = pd.concat([df, kc], axis=1)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range."""
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=period)
    if atr is not None:
        df[f"ATR_{period}"] = atr
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume."""
    obv = ta.obv(df["Close"], df["Volume"])
    if obv is not None:
        df["OBV"] = obv
    return df


def add_ad(df: pd.DataFrame) -> pd.DataFrame:
    """Add Accumulation/Distribution Line."""
    ad = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"])
    if ad is not None:
        df["AD"] = ad
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Chaikin Money Flow."""
    cmf = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"],
                 length=period)
    if cmf is not None:
        df[f"CMF_{period}"] = cmf
    return df


# ---------------------------------------------------------------------------
# Phase 2B: Candlestick pattern detection (pure Python, no TA-Lib)
# ---------------------------------------------------------------------------

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect common candlestick patterns. Adds a 'CDL_Pattern' column
    with pattern name strings and a 'CDL_Signal' column (+1 bullish, -1 bearish, 0 none).
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = c - o
    body_abs = body.abs()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    avg_body = body_abs.rolling(20, min_periods=1).mean()

    patterns = pd.Series("", index=df.index)
    signals = pd.Series(0, index=df.index, dtype=int)

    # Doji: very small body relative to range
    range_ = h - l
    is_doji = (body_abs < range_ * 0.1) & (range_ > 0)
    patterns = patterns.where(~is_doji, "Doji")
    # Doji is neutral

    # Hammer: small body at top, long lower shadow (>= 2x body), bullish after downtrend
    is_hammer = (
        (lower_shadow >= 2 * body_abs)
        & (upper_shadow < body_abs * 0.5)
        & (body_abs > 0)
        & ~is_doji
    )
    patterns = patterns.where(~is_hammer, "Hammer")
    signals = signals.where(~is_hammer, 1)

    # Shooting Star: small body at bottom, long upper shadow, bearish
    is_shooting = (
        (upper_shadow >= 2 * body_abs)
        & (lower_shadow < body_abs * 0.5)
        & (body_abs > 0)
        & ~is_doji
    )
    patterns = patterns.where(~is_shooting, "Shooting Star")
    signals = signals.where(~is_shooting, -1)

    # Bullish Engulfing: previous red candle fully inside current green candle
    prev_body = body.shift(1)
    is_bull_engulf = (
        (body > 0)
        & (prev_body < 0)
        & (o <= c.shift(1))
        & (c >= o.shift(1))
        & (body_abs > body_abs.shift(1))
    )
    patterns = patterns.where(~is_bull_engulf, "Bull Engulfing")
    signals = signals.where(~is_bull_engulf, 1)

    # Bearish Engulfing
    is_bear_engulf = (
        (body < 0)
        & (prev_body > 0)
        & (o >= c.shift(1))
        & (c <= o.shift(1))
        & (body_abs > body_abs.shift(1))
    )
    patterns = patterns.where(~is_bear_engulf, "Bear Engulfing")
    signals = signals.where(~is_bear_engulf, -1)

    # Morning Star (3-candle bullish reversal)
    body_2ago = body.shift(2)
    body_abs_1ago = body_abs.shift(1)
    is_morning = (
        (body_2ago < 0)
        & (body_abs.shift(2) > avg_body.shift(2))
        & (body_abs_1ago < avg_body.shift(1) * 0.5)
        & (body > 0)
        & (c > (o.shift(2) + c.shift(2)) / 2)
    )
    patterns = patterns.where(~is_morning, "Morning Star")
    signals = signals.where(~is_morning, 1)

    # Evening Star (3-candle bearish reversal)
    is_evening = (
        (body_2ago > 0)
        & (body_abs.shift(2) > avg_body.shift(2))
        & (body_abs_1ago < avg_body.shift(1) * 0.5)
        & (body < 0)
        & (c < (o.shift(2) + c.shift(2)) / 2)
    )
    patterns = patterns.where(~is_evening, "Evening Star")
    signals = signals.where(~is_evening, -1)

    df["CDL_Pattern"] = patterns
    df["CDL_Signal"] = signals
    return df


# ---------------------------------------------------------------------------
# Phase 2C: Support/Resistance and Fibonacci
# ---------------------------------------------------------------------------

def detect_support_resistance(
    df: pd.DataFrame, window: int = 20, num_levels: int = 5,
) -> list[float]:
    """Detect support and resistance levels using local peaks/troughs.
    Returns a list of price levels.
    """
    from scipy.signal import argrelextrema

    highs = df["High"].values
    lows = df["Low"].values

    # Find local maxima (resistance) and minima (support)
    res_idx = argrelextrema(highs, np.greater, order=window)[0]
    sup_idx = argrelextrema(lows, np.less, order=window)[0]

    levels = list(highs[res_idx]) + list(lows[sup_idx])
    if not levels:
        return []

    # Cluster nearby levels (within 1.5% of each other)
    levels.sort()
    clustered = []
    cluster = [levels[0]]
    for lvl in levels[1:]:
        if lvl <= cluster[-1] * 1.015:
            cluster.append(lvl)
        else:
            clustered.append(np.mean(cluster))
            cluster = [lvl]
    clustered.append(np.mean(cluster))

    # Return the most significant levels (closest to current price)
    current_price = df["Close"].iloc[-1]
    clustered.sort(key=lambda x: abs(x - current_price))
    return clustered[:num_levels]


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Compute Fibonacci retracement levels from swing high/low.
    Returns dict mapping level name to price.
    """
    recent = df.tail(lookback)
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()
    diff = swing_high - swing_low

    if diff == 0:
        return {}

    return {
        "0.0%": swing_high,
        "23.6%": swing_high - 0.236 * diff,
        "38.2%": swing_high - 0.382 * diff,
        "50.0%": swing_high - 0.500 * diff,
        "61.8%": swing_high - 0.618 * diff,
        "78.6%": swing_high - 0.786 * diff,
        "100.0%": swing_low,
    }


# ---------------------------------------------------------------------------
# Phase 2D: Bollinger Squeeze
# ---------------------------------------------------------------------------

def detect_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Bollinger Squeeze: Bollinger Bands inside Keltner Channels.
    Adds 'Squeeze' column (True when squeeze is on).
    """
    bb_upper = [c for c in df.columns if c.startswith("BBU_")]
    bb_lower = [c for c in df.columns if c.startswith("BBL_")]
    kc_upper = [c for c in df.columns if c.startswith("KCU")]
    kc_lower = [c for c in df.columns if c.startswith("KCL")]

    if bb_upper and bb_lower and kc_upper and kc_lower:
        df["Squeeze"] = (
            (df[bb_lower[0]] > df[kc_lower[0]])
            & (df[bb_upper[0]] < df[kc_upper[0]])
        )
    else:
        df["Squeeze"] = False
    return df


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add all default indicators to the DataFrame."""
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    return df
