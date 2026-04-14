import numpy as np
import pandas as pd

def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features used exclusively by ML models.
    
    Assumes `VIX_Close` and `RSI_14` may be present in the DataFrame.
    Calculates Fear/Greed proxies, price returns, volatility, and
    normalised versions of price-scale indicators.
    Replaces infinities with np.nan to maintain ML compatibility.
    """
    df = df.copy()

    # Layer 2: Fear & Greed Proxy
    # Combining RSI (momentum) with VIX level normalized to 0-100
    if "RSI_14" in df.columns and "VIX_Close" in df.columns:
        # Normalized VIX (where 50 is extreme fear=0, 10 is greed=100)
        vix_norm = 100 - np.clip((df["VIX_Close"] - 10) / (40 - 10) * 100, 0, 100)
        df["Fear_Greed_Proxy"] = (df["RSI_14"] + vix_norm) / 2
    else:
        df["Fear_Greed_Proxy"] = 50.0  # Neutral

    # ----------------------------------------------------------------
    # Price-derived features
    # ----------------------------------------------------------------
    if "Close" in df.columns:
        df["Return_1d"] = df["Close"].pct_change()
        df["Return_2d"] = df["Close"].pct_change(2)
        df["Return_3d"] = df["Close"].pct_change(3)
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_10d"] = df["Close"].pct_change(10)
        df["Return_20d"] = df["Close"].pct_change(20)
        df["Volatility_10d"] = df["Return_1d"].rolling(10).std()

    if "Close" in df.columns and "SMA_20" in df.columns:
        df["Price_vs_SMA"] = (df["Close"] / df["SMA_20"] - 1)
    else:
        df["Price_vs_SMA"] = np.nan

    # ----------------------------------------------------------------
    # Normalised price-scale indicators (ratios instead of raw values)
    # ----------------------------------------------------------------
    if "Close" in df.columns and "EMA_12" in df.columns:
        df["Price_vs_EMA"] = (df["Close"] / df["EMA_12"] - 1)
    else:
        df["Price_vs_EMA"] = np.nan

    # MACD components normalised by Close price so they are
    # comparable across tickers / price levels.
    for raw_col, norm_col in [
        ("MACD_12_26_9", "MACD_Norm"),
        ("MACDh_12_26_9", "MACDh_Norm"),
        ("MACDs_12_26_9", "MACDs_Norm"),
    ]:
        if raw_col in df.columns and "Close" in df.columns:
            df[norm_col] = df[raw_col] / df["Close"]
        else:
            df[norm_col] = np.nan

    # ATR normalised by Close (percentage-based volatility measure)
    if "ATR_14" in df.columns and "Close" in df.columns:
        df["ATR_Norm"] = df["ATR_14"] / df["Close"]
    else:
        df["ATR_Norm"] = np.nan

    # ----------------------------------------------------------------
    # Volume and gap features
    # ----------------------------------------------------------------
    if "Volume" in df.columns:
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    if "Open" in df.columns and "Close" in df.columns:
        df["Gap_Return"] = df["Open"] / df["Close"].shift(1) - 1
        
    if "High" in df.columns and "Low" in df.columns and "Close" in df.columns:
        df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # ----------------------------------------------------------------
    # Replace infinities before returning
    # ----------------------------------------------------------------
    feature_cols = [
        "Return_1d",
        "Return_2d",
        "Return_3d",
        "Return_5d",
        "Return_10d",
        "Return_20d",
        "Volatility_10d",
        "Price_vs_SMA",
        "Price_vs_EMA",
        "MACD_Norm",
        "MACDh_Norm",
        "MACDs_Norm",
        "ATR_Norm",
        "Volume_Ratio",
        "Gap_Return",
        "Daily_Range",
    ]
    
    existing_cols = [col for col in feature_cols if col in df.columns]
    df[existing_cols] = df[existing_cols].replace([np.inf, -np.inf], np.nan)

    return df
