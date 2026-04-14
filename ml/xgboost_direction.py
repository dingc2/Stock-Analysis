"""XGBoost 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Features are computed from indicators/technical.py to ensure consistency with charts.

Important output columns:
- Pred_Direction: 1 for Up, 0 for Down
- Pred_Probability: confidence of the predicted direction
- Prob_Up: raw probability that the next day is Up
"""

import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
from ml.base import ModelProvider
from indicators.technical import add_all
from indicators.ml_features import add_ml_features


class XGBoostDirection(ModelProvider):
    """Predicts next-day price direction (up/down) using XGBoost."""

    FEATURE_COLS = [
        # Momentum oscillators (already 0-100 or percentage-based)
        "RSI_14",
        "STOCHk_14_3_3", "STOCHd_14_3_3",
        "ADX_14", "DMP_14", "DMN_14",
        "ROC_10",
        "MFI_14",
        # Bollinger relative metrics (unitless)
        "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
        # Fear indicators (index-level, comparable across tickers)
        "VIX_Close", "VIX_10d_MA", "VIX_20d_MA", "VIX_30d_MA",
        "Fear_Greed_Proxy",
    ]
    MIN_TRAIN_ROWS = 60
    HOLDOUT_ROWS = 30
    EARLY_STOPPING_ROUNDS = 10
    EARLY_STOP_FRAC = 0.15

    def get_name(self) -> str:
        return "XGBoost Direction"

    def get_description(self) -> str:
        return "Predicts next-day price direction (up/down) using technical indicators"

    def _build_model(self, early_stopping: bool = False, n_estimators: int | None = None) -> XGBClassifier:
        params = dict(
            n_estimators=n_estimators or config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
            reg_alpha=config.XGB_REG_ALPHA,
            reg_lambda=config.XGB_REG_LAMBDA,
            eval_metric=config.XGB_EVAL_METRIC,
            random_state=config.XGB_RANDOM_STATE,
            verbosity=0,
        )
        if early_stopping:
            params["early_stopping_rounds"] = self.EARLY_STOPPING_ROUNDS
        return XGBClassifier(**params)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure all indicators are present (including advanced for ML)
        df = add_all(df, include_advanced=True)
        df = add_ml_features(df)

        # Build feature matrix using only columns that exist
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError("No indicator columns found. Need at least RSI/Stochastic.")
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            warnings.warn(
                f"XGBoostDirection: {len(missing)}/{len(self.FEATURE_COLS)} indicator columns "
                f"missing ({', '.join(missing)}). Training on reduced feature set.",
                stacklevel=2,
            )

        # Normalised price-scale features + price-derived features
        feature_cols = available + [
            # Normalised indicator ratios (from ml_features)
            "Price_vs_SMA",
            "Price_vs_EMA",
            "MACD_Norm",
            "MACDh_Norm",
            "MACDs_Norm",
            "ATR_Norm",
            # Price-derived features
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

        # Chronological split for validation
        split = max(self.MIN_TRAIN_ROWS, len(X) - self.HOLDOUT_ROWS)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Split training data further for early stopping
        es_split = int(len(X_train) * (1 - self.EARLY_STOP_FRAC))
        X_fit, y_fit = X_train[:es_split], y_train[:es_split]
        X_es, y_es = X_train[es_split:], y_train[es_split:]

        eval_model = self._build_model(early_stopping=True)
        eval_model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

        # =====================================================================
        # Predictions — OUT-OF-SAMPLE ONLY to prevent data leakage
        # =====================================================================
        df["Pred_Direction"] = np.nan        # 1 = Up, 0 = Down
        df["Pred_Probability"] = np.nan      # confidence of predicted direction
        df["Prob_Up"] = np.nan               # raw probability of Up

        # Predict on holdout rows only (eval_model never saw these)
        if len(X_val) > 0:
            val_prob_up = eval_model.predict_proba(X_val)[:, 1]
            val_preds = (val_prob_up >= 0.5).astype(int)
            val_confidence = np.where(val_preds == 1, val_prob_up, 1.0 - val_prob_up)

            val_indices = model_df.index[split:]
            df.loc[val_indices, "Pred_Direction"] = val_preds
            df.loc[val_indices, "Pred_Probability"] = val_confidence
            df.loc[val_indices, "Prob_Up"] = val_prob_up

        # =====================================================================
        # Final model for live next-day forecast
        # Retrained on ALL labeled data (standard practice — we need every
        # sample for the actual live prediction, but we don't report this
        # model's accuracy as "validation" accuracy).
        # =====================================================================
        best_n_estimators = (
            eval_model.best_iteration + 1
            if hasattr(eval_model, "best_iteration") and eval_model.best_iteration is not None
            else config.XGB_N_ESTIMATORS
        )
        final_model = self._build_model(early_stopping=False, n_estimators=best_n_estimators)
        final_model.fit(X, y, verbose=False)

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

        # =====================================================================
        # Metrics — clearly separated by model
        # =====================================================================
        # Train accuracy: eval_model on its own training data (for monitoring only)
        train_preds = eval_model.predict(X_train)
        train_acc = float((train_preds == y_train).mean())

        # Validation accuracy: eval_model on truly unseen holdout data
        val_acc = np.nan
        if len(X_val) > 0:
            val_preds_check = eval_model.predict(X_val)
            val_acc = float((val_preds_check == y_val).mean())

        df.attrs["train_accuracy"] = train_acc
        df.attrs["validation_accuracy"] = val_acc
        df.attrs["train_size"] = len(X_train)
        df.attrs["test_size"] = len(X_val)
        df.attrs["feature_cols"] = feature_cols

        return df