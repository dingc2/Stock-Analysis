"""Shared base class for direction prediction models.

Extracts all common feature-engineering, train/val split, prediction-assignment,
attrs-population, and session-scoped caching logic so that XGBoost and LSTM
only provide their model-specific training routines.
"""

from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
import pandas as pd

from indicators.technical import add_all
from indicators.ml_features import add_ml_features
from ml.base import ModelProvider


def _dataframe_hash(df: pd.DataFrame) -> str:
    """Fast hash of OHLCV columns for cache invalidation."""
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    buf = ohlcv.to_numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


class BaseDirectionModel(ModelProvider):
    """Shared logic for next-day direction classifiers.

    Subclasses must implement the abstract methods and class attributes:
    - MODEL_NAME: human-readable name
    - MODEL_DESCRIPTION: short description
    - MIN_TRAIN_ROWS: minimum rows required after indicator warm-up
    - FEATURE_COLS: list of indicator column names to use as base features
    - DERIVED_FEATURES: list of ml_features column names to append

    Session-scoped caching prevents redundant retraining when the same
    (ticker, period, interval, data-hash) is seen across tab renders.
    """

    MODEL_NAME: str = "BaseDirection"
    MODEL_DESCRIPTION: str = ""

    # Cache TTL in seconds (matches history cache TTL)
    _CACHE_TTL: int = 3600

    # Base indicator columns shared by all direction models
    FEATURE_COLS: list[str] = [
        "RSI_14",
        "STOCHk_14_3_3", "STOCHd_14_3_3",
        "ADX_14", "DMP_14", "DMN_14",
        "ROC_10",
        "MFI_14",
        "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
        "VIX_Close", "VIX_10d_MA", "VIX_20d_MA", "VIX_30d_MA",
        "Fear_Greed_Proxy",
    ]

    # Engineered features from ml_features.py
    DERIVED_FEATURES: list[str] = [
        "Price_vs_SMA",
        "Price_vs_EMA",
        "MACD_Norm",
        "MACDh_Norm",
        "MACDs_Norm",
        "ATR_Norm",
        "Return_1d",
        "Return_2d",
        "Return_3d",
        "Return_5d",
        "Return_10d",
        "Return_20d",
        "Volatility_10d",
        "Volume_Ratio",
        "Gap_Return",
        "Daily_Range",
    ]

    MIN_TRAIN_ROWS: int = 60
    HOLDOUT_ROWS: int = 30

    def __init__(self) -> None:
        # Per-instance cache: key = dataframe-hash → (result_df, timestamp)
        self._pred_cache: dict[str, tuple[pd.DataFrame, float]] = {}

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self.MODEL_NAME

    def get_description(self) -> str:
        return self.MODEL_DESCRIPTION

    def _train_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
    ) -> tuple[object, int]:
        """Train the model and return (trained_model, best_iteration).

        Args:
            X_train: training features (chronological first portion)
            y_train: training labels
            X_val: holdout features (never seen during training — for out-of-sample metrics)
            y_val: holdout labels
            X_es: optional early-stopping split features (subset of X_train)
            y_es: optional early-stopping split labels

        Returns:
            (trained_model, best_iteration) where best_iteration is used to
            size the final model retrained on all labeled data.
        """
        raise NotImplementedError

    def _final_train_predict(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        n_estimators: int,
    ) -> tuple[object, np.ndarray]:
        """Retrain on all labeled data and predict Prob_Up for the last row.

        Args:
            X_all: all labeled features
            y_all: all labeled labels
            n_estimators: number of boosting rounds / epochs for final model

        Returns:
            (final_model, last_prob_up) where last_prob_up is a scalar probability of Up.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared predict() — data prep + train/val split + caching + attrs
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run direction prediction on the given OHLCV + indicators DataFrame.

        Returns the DataFrame with Pred_Direction, Pred_Probability, Prob_Up
        columns added. Predictions are assigned only to holdout rows (no
        leakage). The final row gets a live forecast from a model retrained
        on all labeled data.

        Results are cached per-instance using a hash of the OHLCV columns.
        A subsequent call with identical price data returns the cached result.
        """
        # Session-scoped cache: skip retraining if we already saw this data
        now = time.monotonic()
        cache_key = _dataframe_hash(df)
        if cache_key in self._pred_cache:
            cached_df, cached_at = self._pred_cache[cache_key]
            if now - cached_at < self._CACHE_TTL:
                return cached_df.copy()
            # expired — remove and proceed to retrain
            del self._pred_cache[cache_key]

        df = df.copy()

        # 1. Feature engineering (shared by all models)
        df = add_all(df, include_advanced=True)
        df = add_ml_features(df)

        # 2. Build feature list
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError("No indicator columns found. Need at least RSI/Stochastic.")
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            warnings.warn(
                f"{self.MODEL_NAME}: {len(missing)}/{len(self.FEATURE_COLS)} indicator "
                f"columns missing ({', '.join(missing)}). Training on reduced feature set.",
                stacklevel=2,
            )

        feature_cols = available + [f for f in self.DERIVED_FEATURES if f in df.columns]

        # 3. Target: 1 if next day close > today close, else 0
        #    Last row has no known next-day close → NaN
        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(
            next_close.notna(),
            (next_close > df["Close"]).astype(int),
            np.nan,
        )

        # 4. Drop rows with NaN in features or target
        model_df = df.dropna(subset=feature_cols + ["Target"]).copy()

        if len(model_df) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_ROWS} rows after indicator warm-up, "
                f"got {len(model_df)}. Try a longer period."
            )

        X = model_df[feature_cols].to_numpy()
        y = model_df["Target"].astype(int).to_numpy()

        # 5. Chronological split for validation
        split = max(self.MIN_TRAIN_ROWS, len(X) - self.HOLDOUT_ROWS)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # 6. Early stopping split (subset of X_train)
        #    Subclasses that don't use early stopping can ignore these
        es_split = int(len(X_train) * 0.85)
        X_fit, y_fit = X_train[:es_split], y_train[:es_split]
        X_es, y_es = X_train[es_split:], y_train[es_split:]

        # 7. Train eval model + get best iteration
        eval_model, best_iter = self._train_and_eval(
            X_fit, y_fit, X_val, y_val, X_es, y_es,
        )

        # 8. Out-of-sample predictions on holdout rows only
        df["Pred_Direction"] = np.nan
        df["Pred_Probability"] = np.nan
        df["Prob_Up"] = np.nan

        if len(X_val) > 0:
            val_prob_up = self._prob_up_from_model(eval_model, X_val)
            val_preds = (val_prob_up >= 0.5).astype(int)
            val_confidence = np.where(val_preds == 1, val_prob_up, 1.0 - val_prob_up)
            val_indices = model_df.index[split:]
            df.loc[val_indices, "Pred_Direction"] = val_preds
            df.loc[val_indices, "Pred_Probability"] = val_confidence
            df.loc[val_indices, "Prob_Up"] = val_prob_up

        # 9. Retrain final model on all labeled data for last-row live forecast
        final_model, last_prob_up = self._final_train_predict(X, y, best_iter)

        # 10. Last-row live forecast
        last_features = df.iloc[[-1]][feature_cols]
        if not last_features.isna().any(axis=None):
            last_X = last_features.to_numpy()
            last_prob = self._predict_last_row(final_model, last_X)
            last_pred = int(last_prob >= 0.5)
            last_conf = last_prob if last_pred == 1 else 1.0 - last_prob
            df.loc[df.index[-1], "Pred_Direction"] = last_pred
            df.loc[df.index[-1], "Pred_Probability"] = last_conf
            df.loc[df.index[-1], "Prob_Up"] = last_prob

        # 11. Metrics
        train_prob = self._prob_up_from_model(eval_model, X_train)
        train_acc = float(((train_prob >= 0.5).astype(int) == y_train).mean())

        val_acc = np.nan
        if len(X_val) > 0:
            val_acc = float(((val_prob_up >= 0.5).astype(int) == y_val).mean())

        df.attrs["train_accuracy"] = train_acc
        df.attrs["validation_accuracy"] = val_acc
        df.attrs["train_size"] = len(X_train)
        df.attrs["test_size"] = len(X_val)
        df.attrs["feature_cols"] = feature_cols

        # Store in session cache
        self._pred_cache[cache_key] = (df.copy(), now)

        return df

    def _prob_up_from_model(self, model: object, X: np.ndarray) -> np.ndarray:
        """Return Prob_Up array from a trained model (subclass-specific)."""
        raise NotImplementedError

    def _predict_last_row(self, final_model: object, last_X: np.ndarray) -> float:
        """Return scalar Prob_Up for a single row from the final model."""
        raise NotImplementedError
