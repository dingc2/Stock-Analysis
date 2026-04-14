"""PyTorch LSTM 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Returns the same prediction columns as other models for UI compatibility.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

import config
from indicators.technical import add_all
from indicators.ml_features import add_ml_features
from ml.base import ModelProvider

try:
    import torch
    from torch import nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError as exc:  # pragma: no cover - handled by registry fallback
    raise ImportError("PyTorch is required for LSTMDirection") from exc


class _LSTMBinaryClassifier(nn.Module):
    """Minimal LSTM binary classifier for sequence data."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.drop(last_step)
        logits = self.head(last_step).squeeze(1)
        return logits


class LSTMDirection(ModelProvider):
    """Predicts next-day price direction (up/down) using an LSTM."""

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

    MIN_TRAIN_ROWS = 120
    HOLDOUT_ROWS = 30
    SEQ_LEN = 20

    def get_name(self) -> str:
        return "LSTM Direction"

    def get_description(self) -> str:
        return "Predicts next-day price direction (up/down) using an LSTM over indicator sequences"

    def _build_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_x, seq_y, seq_idx = [], [], []
        for i in range(self.SEQ_LEN - 1, len(features)):
            start = i - self.SEQ_LEN + 1
            seq_x.append(features[start : i + 1])
            seq_y.append(labels[i])
            seq_idx.append(indices[i])
        return np.array(seq_x), np.array(seq_y), np.array(seq_idx)

    def _fit_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int | None = None,
    ) -> tuple[_LSTMBinaryClassifier, int]:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)

        model = _LSTMBinaryClassifier(
            input_size=x_train.shape[-1],
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT,
        )
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LSTM_LEARNING_RATE, weight_decay=config.LSTM_WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler — reduce LR on val loss plateau
        has_val = x_val is not None and y_val is not None and len(x_val) > 0
        scheduler = None
        if has_val:
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
            )

        x_t = torch.tensor(x_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        if has_val:
            x_v = torch.tensor(x_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        wait = 0
        best_epoch = 0

        n = x_t.shape[0]
        target_epochs = epochs if epochs is not None else config.LSTM_EPOCHS
        for epoch in range(target_epochs):
            model.train()
            order = torch.randperm(n, generator=g)
            for start in range(0, n, config.LSTM_BATCH_SIZE):
                batch_idx = order[start : start + config.LSTM_BATCH_SIZE]
                xb = x_t[batch_idx]
                yb = y_t[batch_idx]

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.LSTM_GRAD_CLIP)
                optimizer.step()

            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(x_v), y_v).item()

                # Step the LR scheduler
                if scheduler is not None:
                    scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                    if wait >= config.LSTM_PATIENCE:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        elif not has_val:
            best_epoch = target_epochs - 1

        model.eval()
        return model, best_epoch

    def _predict_prob_up(self, model: _LSTMBinaryClassifier, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = add_all(df, include_advanced=True)
        df = add_ml_features(df)

        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError("No indicator columns found. Need at least RSI/Stochastic.")
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            warnings.warn(
                f"LSTMDirection: {len(missing)}/{len(self.FEATURE_COLS)} indicator columns "
                f"missing ({', '.join(missing)}). Training on reduced feature set.",
                stacklevel=2,
            )

        # Feature list matches XGBoost for parity — all normalised/unitless
        feature_cols = available + [
            # Normalised indicator ratios (from ml_features)
            "Price_vs_SMA",
            "Price_vs_EMA",
            "MACD_Norm",
            "MACDh_Norm",
            "MACDs_Norm",
            "ATR_Norm",
            # Price-derived features (all percentage-based)
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

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(next_close.notna(), (next_close > df["Close"]).astype(int), np.nan)

        model_df = df.dropna(subset=feature_cols + ["Target"]).copy()
        if len(model_df) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_ROWS} rows after indicator warm-up, got {len(model_df)}. "
                "Try a longer period."
            )

        x_raw = model_df[feature_cols].to_numpy(dtype=np.float32)
        y_raw = model_df["Target"].to_numpy(dtype=np.float32)
        row_indices = model_df.index.to_numpy()

        if len(x_raw) < self.SEQ_LEN:
            raise ValueError(f"Need at least {self.SEQ_LEN} labeled rows for LSTM sequences.")

        split_row = max(self.MIN_TRAIN_ROWS, len(x_raw) - self.HOLDOUT_ROWS)
        split_row = min(split_row, len(x_raw) - 1)

        train_mean = x_raw[:split_row].mean(axis=0)
        train_std = x_raw[:split_row].std(axis=0)
        train_std = np.where(train_std < 1e-8, 1.0, train_std)
        x_scaled = (x_raw - train_mean) / train_std

        seq_x, seq_y, seq_idx = self._build_sequences(x_scaled, y_raw, row_indices)
        if len(seq_x) < 2:
            raise ValueError("Not enough data after sequence construction.")

        train_seq_mask = seq_idx < row_indices[split_row]
        x_train, y_train = seq_x[train_seq_mask], seq_y[train_seq_mask]
        x_val, y_val = seq_x[~train_seq_mask], seq_y[~train_seq_mask]

        if len(x_train) < 20:
            raise ValueError("Not enough training sequences for LSTM. Try a longer period.")

        eval_model, best_epoch = self._fit_model(x_train, y_train, seed=42, x_val=x_val, y_val=y_val)

        # =====================================================================
        # Predictions — OUT-OF-SAMPLE ONLY to prevent data leakage
        # =====================================================================
        df["Pred_Direction"] = np.nan
        df["Pred_Probability"] = np.nan
        df["Prob_Up"] = np.nan

        # Only predict on validation sequences (eval_model never saw these)
        if len(x_val) > 0:
            val_prob_up = self._predict_prob_up(eval_model, x_val)
            val_preds = (val_prob_up >= 0.5).astype(int)
            val_confidence = np.where(val_preds == 1, val_prob_up, 1.0 - val_prob_up)

            val_seq_idx = seq_idx[~train_seq_mask]
            df.loc[val_seq_idx, "Pred_Direction"] = val_preds
            df.loc[val_seq_idx, "Pred_Probability"] = val_confidence
            df.loc[val_seq_idx, "Prob_Up"] = val_prob_up

        # =====================================================================
        # Final model for live next-day forecast
        # Retrained on ALL sequences with the same epoch count (using same
        # seed for reproducibility).
        # =====================================================================
        final_epochs = best_epoch + 1
        final_model, _ = self._fit_model(seq_x, seq_y, seed=42, epochs=final_epochs)

        # Last-row prediction: check the entire sequence for NaN, not just the last row
        all_features = df[feature_cols].to_numpy(dtype=np.float32)
        all_scaled = (all_features - train_mean) / train_std
        if len(all_scaled) >= self.SEQ_LEN:
            last_seq = all_scaled[-self.SEQ_LEN :]
            if not np.isnan(last_seq).any():
                last_seq = last_seq[None, :, :]
                last_prob_up = float(self._predict_prob_up(final_model, last_seq)[0])
                last_pred = int(last_prob_up >= 0.5)
                last_conf = last_prob_up if last_pred == 1 else 1.0 - last_prob_up
                df.loc[df.index[-1], "Pred_Direction"] = last_pred
                df.loc[df.index[-1], "Pred_Probability"] = last_conf
                df.loc[df.index[-1], "Prob_Up"] = last_prob_up

        # =====================================================================
        # Metrics — clearly separated
        # =====================================================================
        train_preds = (self._predict_prob_up(eval_model, x_train) >= 0.5).astype(int)
        train_acc = float((train_preds == y_train.astype(int)).mean())

        val_acc = np.nan
        if len(x_val) > 0:
            val_preds_check = (self._predict_prob_up(eval_model, x_val) >= 0.5).astype(int)
            val_acc = float((val_preds_check == y_val.astype(int)).mean())

        df.attrs["train_accuracy"] = train_acc
        df.attrs["validation_accuracy"] = val_acc
        df.attrs["train_size"] = int(len(x_train))
        df.attrs["test_size"] = int(len(x_val))
        df.attrs["feature_cols"] = feature_cols

        return df
