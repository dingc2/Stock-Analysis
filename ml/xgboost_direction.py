"""XGBoost 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Features are computed from indicators/technical.py to ensure consistency with charts.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from ml.base import ModelProvider
from indicators.technical import add_all


class XGBoostDirection(ModelProvider):
    """Predicts next-day price direction (up/down) using XGBoost."""

    FEATURE_COLS = [
        "SMA_20", "EMA_12", "RSI_14",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0",
        "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
    ]
    MIN_TRAIN_ROWS = 60

    def get_name(self) -> str:
        return "XGBoost Direction"

    def get_description(self) -> str:
        return "Predicts next-day price direction (up/down) using technical indicators"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure all indicators are present
        df = add_all(df)

        # Build feature matrix using only columns that exist
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError("No indicator columns found. Need at least SMA/EMA/RSI.")

        # Add price-derived features
        df["Return_1d"] = df["Close"].pct_change()
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
        df["Price_vs_SMA"] = (df["Close"] / df["SMA_20"] - 1) if "SMA_20" in df.columns else 0
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        feature_cols = available + [
            "Return_1d", "Return_5d", "Volatility_10d",
            "Price_vs_SMA", "Volume_Ratio",
        ]

        # Target: 1 if next day close > today close, else 0
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # Drop rows with NaN in features or target
        model_df = df.dropna(subset=feature_cols + ["Target"])

        if len(model_df) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_ROWS} rows after indicator "
                f"warm-up, got {len(model_df)}. Try a longer period."
            )

        X = model_df[feature_cols].values
        y = model_df["Target"].values

        # Train on all but last 5 rows, predict on everything
        split = max(self.MIN_TRAIN_ROWS, len(X) - 5)
        X_train, y_train = X[:split], y[:split]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Predict probabilities for all rows
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        # Map predictions back onto original df
        df["Pred_Direction"] = np.nan
        df["Pred_Probability"] = np.nan
        df.loc[model_df.index, "Pred_Direction"] = preds
        df.loc[model_df.index, "Pred_Probability"] = proba

        # Also predict the very last row (next-day forecast)
        last_row = df.iloc[[-1]]
        last_features = last_row[feature_cols].dropna(axis=1)
        if len(last_features.columns) == len(feature_cols):
            last_X = last_features.values
            df.loc[df.index[-1], "Pred_Direction"] = model.predict(last_X)[0]
            df.loc[df.index[-1], "Pred_Probability"] = model.predict_proba(last_X)[0, 1]

        # Compute train accuracy for display
        train_preds = model.predict(X_train)
        train_acc = (train_preds == y_train).mean()
        df.attrs["train_accuracy"] = train_acc
        df.attrs["train_size"] = len(X_train)
        df.attrs["test_size"] = len(X) - split

        return df
