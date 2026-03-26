"""XGBoost 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Features are computed from indicators/technical.py to ensure consistency with charts.

Important output columns:
- Pred_Direction: 1 for Up, 0 for Down
- Pred_Probability: confidence of the predicted direction
- Prob_Up: raw probability that the next day is Up
"""

import numpy as np
import pandas as pd
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
    HOLDOUT_ROWS = 5

    def get_name(self) -> str:
        return "XGBoost Direction"

    def get_description(self) -> str:
        return "Predicts next-day price direction (up/down) using technical indicators"

    def _build_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

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
        df["Price_vs_SMA"] = (df["Close"] / df["SMA_20"] - 1) if "SMA_20" in df.columns else np.nan
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        feature_cols = available + [
            "Return_1d",
            "Return_5d",
            "Volatility_10d",
            "Price_vs_SMA",
            "Volume_Ratio",
        ]

        # Replace inf values before dropping missing rows
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Target: 1 if next day close > today close, else 0
        # IMPORTANT: last row has no known next-day close, so target should be NaN, not 0
        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(
            next_close.notna(),
            (next_close > df["Close"]).astype(int),
            np.nan,
        )

        # Drop rows with NaN in features or target
        model_df = df.dropna(subset=feature_cols + ["Target"]).copy()

        if len(model_df) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_ROWS} rows after indicator "
                f"warm-up, got {len(model_df)}. Try a longer period."
            )

        X = model_df[feature_cols].to_numpy()
        y = model_df["Target"].astype(int).to_numpy()

        # Chronological split for simple validation
        split = max(self.MIN_TRAIN_ROWS, len(X) - self.HOLDOUT_ROWS)
        X_train, y_train = X[:split], y[:split]

        eval_model = self._build_model()
        eval_model.fit(X_train, y_train)

        # Predict on labeled rows using the evaluation model
        prob_up = eval_model.predict_proba(X)[:, 1]
        preds = (prob_up >= 0.5).astype(int)

        # Confidence should match the predicted class, not always P(Up)
        confidence = np.where(preds == 1, prob_up, 1.0 - prob_up)

        # Map predictions back onto original df
        df["Pred_Direction"] = np.nan        # 1 = Up, 0 = Down
        df["Pred_Probability"] = np.nan      # confidence of predicted direction
        df["Prob_Up"] = np.nan               # raw probability of Up

        df.loc[model_df.index, "Pred_Direction"] = preds
        df.loc[model_df.index, "Pred_Probability"] = confidence
        df.loc[model_df.index, "Prob_Up"] = prob_up

        # Train a final model on all labeled data for the live next-day forecast
        final_model = self._build_model()
        final_model.fit(X, y)

        # Predict the very last row (true next-day forecast)
        last_features = df.iloc[[-1]][feature_cols]
        if not last_features.isna().any(axis=None):
            last_X = last_features.to_numpy()
            last_prob_up = final_model.predict_proba(last_X)[0, 1]
            last_pred = int(last_prob_up >= 0.5)
            last_confidence = last_prob_up if last_pred == 1 else 1.0 - last_prob_up

            df.loc[df.index[-1], "Pred_Direction"] = last_pred
            df.loc[df.index[-1], "Pred_Probability"] = last_confidence
            df.loc[df.index[-1], "Prob_Up"] = last_prob_up

        # Metrics for display
        train_preds = eval_model.predict(X_train)
        train_acc = float((train_preds == y_train).mean())

        val_acc = np.nan
        if split < len(X):
            X_val, y_val = X[split:], y[split:]
            val_preds = eval_model.predict(X_val)
            val_acc = float((val_preds == y_val).mean())

        df.attrs["train_accuracy"] = train_acc
        df.attrs["validation_accuracy"] = val_acc
        df.attrs["train_size"] = len(X_train)
        df.attrs["test_size"] = max(0, len(X) - split)
        df.attrs["feature_cols"] = feature_cols

        return df