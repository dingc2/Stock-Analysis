"""Composite Signal Score.

Pure function — no Streamlit, no side effects.

Takes a DataFrame that already has technical indicators computed (via
``indicators/technical.py``) and optional ML probability results, then
returns it with the following columns added to the *last row only*
(historical rows are filled with NaN to keep the DataFrame shape stable):

    Signal_Score   float  -100 to +100
    Signal_Label   str    "Strong Buy" / "Buy" / "Neutral" / "Sell" / "Strong Sell"
    Score_Trend    float  -1 to +1  (trend sub-score)
    Score_Momentum float  -1 to +1  (momentum sub-score)
    Score_Volume   float  -1 to +1  (volume sub-score)
    Score_Pattern  float  -1 to +1  (candlestick pattern sub-score)
    Score_ML       float  -1 to +1  (ML model consensus sub-score)

``ml_results`` is an optional list of dicts, each with key ``"prob_up"``
(float 0-1) produced by each ML model that was run.  Pass ``None`` or an
empty list to exclude the ML category (its weight is redistributed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Internal helpers — each returns a single float in [-1, +1] or None
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe(series: pd.Series, col: str) -> float | None:
    """Return the last non-NaN scalar value for *col* if it exists."""
    if col not in series.index or pd.isna(series[col]):
        return None
    return float(series[col])


def _normalize_trend(df: pd.DataFrame) -> float | None:
    """Trend sub-score from price vs SMA/EMA, MACD histogram, ADX direction.

    Each component normalises to [-1, +1]:
    - Price vs SMA : clamp((Close-SMA) / (2*ATR), -1, 1)
    - Price vs EMA : clamp((Close-EMA) / (2*ATR), -1, 1)
    - MACD hist    : clamp(hist / mean(|hist|, 20 bars), -1, 1)
    - ADX dir      : (DMP - DMN) / (DMP + DMN)   [natural ±1 range]
    """
    last = df.iloc[-1]
    close = float(df["Close"].iloc[-1])
    signals: list[float] = []

    # ATR used as normaliser; fall back to 2 % of price if missing
    atr_col = next((c for c in df.columns if c.startswith("ATR_")), None)
    atr = float(df[atr_col].iloc[-1]) if (atr_col and not pd.isna(df[atr_col].iloc[-1])) else close * 0.02
    if atr == 0:
        atr = close * 0.02

    sma_col = next((c for c in df.columns if c.startswith("SMA_")), None)
    if sma_col:
        v = _safe(last, sma_col)
        if v is not None:
            signals.append(_clamp((close - v) / (2 * atr)))

    ema_col = next((c for c in df.columns if c.startswith("EMA_")), None)
    if ema_col:
        v = _safe(last, ema_col)
        if v is not None:
            signals.append(_clamp((close - v) / (2 * atr)))

    # MACD histogram — normalised by its own rolling magnitude
    hist_col = next((c for c in df.columns if c.startswith("MACDh_")), None)
    if hist_col and hist_col in df.columns:
        hist_series = df[hist_col].dropna()
        if len(hist_series) >= 5:
            mean_abs = hist_series.abs().rolling(20, min_periods=5).mean().iloc[-1]
            if mean_abs and mean_abs > 0:
                signals.append(_clamp(float(hist_series.iloc[-1]) / mean_abs))

    # ADX direction: (DMP - DMN) / (DMP + DMN)
    adx_col = next((c for c in df.columns if c.startswith("ADX_")), None)
    dmp_col = next((c for c in df.columns if c.startswith("DMP_")), None)
    dmn_col = next((c for c in df.columns if c.startswith("DMN_")), None)
    if dmp_col and dmn_col:
        dmp = _safe(last, dmp_col)
        dmn = _safe(last, dmn_col)
        if dmp is not None and dmn is not None and (dmp + dmn) > 0:
            raw = (dmp - dmn) / (dmp + dmn)
            # Scale by ADX strength: ADX > 25 = full weight, < 15 = dampened
            adx_val = _safe(last, adx_col) if adx_col else None
            if adx_val is not None:
                strength = _clamp(adx_val / 25.0, 0.2, 1.0)
                raw *= strength
            signals.append(_clamp(raw))

    return float(np.mean(signals)) if signals else None


def _normalize_momentum(df: pd.DataFrame) -> float | None:
    """Momentum sub-score from RSI, Stochastic %K, MFI, ROC.

    Oscillators (RSI/Stoch/MFI): linear map 0→-1, 50→0, 100→+1.
    ROC: clamp(ROC / 10, -1, 1)  (10% move = full score).
    """
    last = df.iloc[-1]
    signals: list[float] = []

    for col in (c for c in df.columns if c.startswith("RSI_")):
        v = _safe(last, col)
        if v is not None:
            signals.append(_clamp((v - 50.0) / 50.0))
            break  # one RSI is enough

    stoch_col = next((c for c in df.columns if c.startswith("STOCHk_")), None)
    if stoch_col:
        v = _safe(last, stoch_col)
        if v is not None:
            signals.append(_clamp((v - 50.0) / 50.0))

    mfi_col = next((c for c in df.columns if c.startswith("MFI_")), None)
    if mfi_col:
        v = _safe(last, mfi_col)
        if v is not None:
            signals.append(_clamp((v - 50.0) / 50.0))

    roc_col = next((c for c in df.columns if c.startswith("ROC_")), None)
    if roc_col:
        v = _safe(last, roc_col)
        if v is not None:
            signals.append(_clamp(v / 10.0))

    return float(np.mean(signals)) if signals else None


def _normalize_volume(df: pd.DataFrame) -> float | None:
    """Volume sub-score from CMF and OBV slope.

    CMF: typically -0.2..+0.2, amplify × 5 then clamp.
    OBV slope: +1 if OBV > its 10-bar mean, -1 otherwise.
    """
    last = df.iloc[-1]
    signals: list[float] = []

    cmf_col = next((c for c in df.columns if c.startswith("CMF_")), None)
    if cmf_col:
        v = _safe(last, cmf_col)
        if v is not None:
            signals.append(_clamp(v * 5.0))

    if "OBV" in df.columns:
        obv = df["OBV"].dropna()
        if len(obv) >= 10:
            slope = 1.0 if float(obv.iloc[-1]) > float(obv.rolling(10).mean().iloc[-1]) else -1.0
            signals.append(slope)

    return float(np.mean(signals)) if signals else None


def _normalize_pattern(df: pd.DataFrame) -> float | None:
    """Pattern sub-score from the most recent CDL_Signal on the last row."""
    if "CDL_Signal" not in df.columns:
        return None
    v = _safe(df.iloc[-1], "CDL_Signal")
    if v is None:
        return None
    # CDL_Signal is already -1 / 0 / +1
    return _clamp(float(v))


def _normalize_ml(ml_results: list[dict] | None) -> float | None:
    """ML sub-score: average (Prob_Up - 0.5) * 2 across all models.

    Each model dict must have key ``"prob_up"`` (float 0-1).
    """
    if not ml_results:
        return None
    scores = []
    for r in ml_results:
        prob = r.get("prob_up")
        if prob is not None and not np.isnan(prob):
            scores.append(_clamp((float(prob) - 0.5) * 2.0))
    return float(np.mean(scores)) if scores else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signal_score(
    df: pd.DataFrame,
    ml_results: list[dict] | None = None,
) -> pd.DataFrame:
    """Compute the composite signal score for the most recent bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + any indicator columns already attached.  At minimum the
        DataFrame should have a ``Close`` column.  The more indicators are
        present, the richer the score.
    ml_results : list[dict] | None
        Optional list of ``{"prob_up": float}`` dicts from ML models.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with six new columns appended.  All rows except the
        last receive NaN / "" for the score columns so the shape is stable.
    """
    score_cols = [
        "Score_Trend", "Score_Momentum", "Score_Volume",
        "Score_Pattern", "Score_ML", "Signal_Score", "Signal_Label",
    ]

    # Compute each sub-score (None means no data available)
    sub_scores: dict[str, float | None] = {
        "trend":    _normalize_trend(df),
        "momentum": _normalize_momentum(df),
        "volume":   _normalize_volume(df),
        "pattern":  _normalize_pattern(df),
        "ml":       _normalize_ml(ml_results),
    }

    # Build effective weights — redistribute weight from missing categories
    weights = dict(config.SIGNAL_WEIGHTS)
    available = {k: v for k, v in sub_scores.items() if v is not None}
    if not available:
        final_score = 0.0
        label = "Neutral"
    else:
        total_available_weight = sum(weights[k] for k in available)
        effective: dict[str, float] = {
            k: weights[k] / total_available_weight for k in available
        }
        final_score = sum(effective[k] * available[k] for k in available) * 100.0
        final_score = _clamp(final_score, -100.0, 100.0)
        label = _score_to_label(final_score)

    # Assign all score columns — NaN for historical rows, value for last row
    n = len(df)
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan if col != "Signal_Label" else ""

    idx = df.index[-1]
    df.loc[idx, "Score_Trend"]    = sub_scores["trend"]    if sub_scores["trend"]    is not None else np.nan
    df.loc[idx, "Score_Momentum"] = sub_scores["momentum"] if sub_scores["momentum"] is not None else np.nan
    df.loc[idx, "Score_Volume"]   = sub_scores["volume"]   if sub_scores["volume"]   is not None else np.nan
    df.loc[idx, "Score_Pattern"]  = sub_scores["pattern"]  if sub_scores["pattern"]  is not None else np.nan
    df.loc[idx, "Score_ML"]       = sub_scores["ml"]       if sub_scores["ml"]       is not None else np.nan
    df.loc[idx, "Signal_Score"]   = final_score
    df.loc[idx, "Signal_Label"]   = label

    return df


def _score_to_label(score: float) -> str:
    t = config.SIGNAL_THRESHOLDS
    if score >= t["strong_buy"]:
        return "Strong Buy"
    if score >= t["buy"]:
        return "Buy"
    if score >= t["neutral_low"]:
        return "Neutral"
    if score >= t["sell"]:
        return "Sell"
    return "Strong Sell"


def get_signal_color(label: str) -> str:
    """Return a hex colour for a given Signal_Label string."""
    return {
        "Strong Buy":  "#00c853",
        "Buy":         "#69f0ae",
        "Neutral":     "#ffd740",
        "Sell":        "#ff6d00",
        "Strong Sell": "#d50000",
    }.get(label, "#9e9e9e")
