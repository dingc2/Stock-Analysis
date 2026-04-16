"""PyTorch LSTM 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Returns the same prediction columns as other models for UI compatibility.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

import config
from indicators.technical import add_all
from indicators.ml_features import add_ml_features
from ml.direction_base import BaseDirectionModel, _dataframe_hash

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


class LSTMDirection(BaseDirectionModel):
    """Predicts next-day price direction (up/down) using an LSTM."""

    MODEL_NAME = "LSTM Direction"
    MODEL_DESCRIPTION = "Predicts next-day price direction (up/down) using an LSTM over indicator sequences"
    MIN_TRAIN_ROWS = 120
    SEQ_LEN = 20

    # LSTM uses the same base + derived features as XGBoost
    FEATURE_COLS = BaseDirectionModel.FEATURE_COLS
    DERIVED_FEATURES = BaseDirectionModel.DERIVED_FEATURES

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

    def _prob_up_from_model(self, model: _LSTMBinaryClassifier, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _predict_last_row(self, final_model: _LSTMBinaryClassifier, last_X: np.ndarray) -> float:
        with torch.no_grad():
            # last_X shape: (1, seq_len, n_features) already sequenced by caller
            logits = final_model(torch.tensor(last_X, dtype=torch.float32))
            return float(torch.sigmoid(logits).item())

    # ------------------------------------------------------------------
    # Override predict() — LSTM needs sequence construction + z-score norm
    # before the chronological split.  Caching and attrs still come from
    # the base class via super().
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """LSTM predict with sequence construction + z-score normalization.

        Caching and shared attrs are inherited from BaseDirectionModel.
        """
        # Session-scoped cache
        now = time.monotonic()
        cache_key = _dataframe_hash(df)
        if cache_key in self._pred_cache:
            cached_df, cached_at = self._pred_cache[cache_key]
            if now - cached_at < self._CACHE_TTL:
                return cached_df.copy()
            del self._pred_cache[cache_key]

        df = df.copy()

        # 1. Feature engineering (shared)
        df = add_all(df, include_advanced=True)
        df = add_ml_features(df)

        # 2. Build feature list
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError("No indicator columns found. Need at least RSI/Stochastic.")

        feature_cols = available + [f for f in self.DERIVED_FEATURES if f in df.columns]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # 3. Target
        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(
            next_close.notna(),
            (next_close > df["Close"]).astype(int),
            np.nan,
        )

        # 4. Drop rows with NaN
        model_df = df.dropna(subset=feature_cols + ["Target"]).copy()
        if len(model_df) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_ROWS} rows after indicator warm-up, "
                f"got {len(model_df)}. Try a longer period."
            )

        x_raw = model_df[feature_cols].to_numpy(dtype=np.float32)
        y_raw = model_df["Target"].to_numpy(dtype=np.float32)
        row_indices = model_df.index.to_numpy()

        # 5. Chronological train/val split (same logic as base, but before sequencing)
        split_row = max(self.MIN_TRAIN_ROWS, len(x_raw) - self.HOLDOUT_ROWS)
        split_row = min(split_row, len(x_raw) - 1)

        # 6. Z-score normalization — fit on train portion only
        train_mean = x_raw[:split_row].mean(axis=0)
        train_std = x_raw[:split_row].std(axis=0)
        train_std = np.where(train_std < 1e-8, 1.0, train_std)
        x_scaled = (x_raw - train_mean) / train_std

        # 7. Build sequences
        seq_x, seq_y, seq_idx = self._build_sequences(x_scaled, y_raw, row_indices)
        if len(seq_x) < 2:
            raise ValueError("Not enough data after sequence construction.")

        # 8. Split sequences into train/val
        train_seq_mask = seq_idx < row_indices[split_row]
        x_train_seq = seq_x[train_seq_mask]
        y_train_seq = seq_y[train_seq_mask]
        x_val_seq = seq_x[~train_seq_mask]
        y_val_seq = seq_y[~train_seq_mask]

        if len(x_train_seq) < 20:
            raise ValueError("Not enough training sequences for LSTM. Try a longer period.")

        # 9. Train eval model with early stopping
        eval_model, best_epoch = self._fit_model(
            x_train_seq, y_train_seq, seed=42, x_val=x_val_seq, y_val=y_val_seq,
        )

        # 10. Out-of-sample predictions on holdout sequences only
        df["Pred_Direction"] = np.nan
        df["Pred_Probability"] = np.nan
        df["Prob_Up"] = np.nan

        if len(x_val_seq) > 0:
            val_prob_up = self._prob_up_from_model(eval_model, x_val_seq)
            val_preds = (val_prob_up >= 0.5).astype(int)
            val_confidence = np.where(val_preds == 1, val_prob_up, 1.0 - val_prob_up)
            val_seq_idx = seq_idx[~train_seq_mask]
            df.loc[val_seq_idx, "Pred_Direction"] = val_preds
            df.loc[val_seq_idx, "Pred_Probability"] = val_confidence
            df.loc[val_seq_idx, "Prob_Up"] = val_prob_up

        # 11. Final model retrained on all sequences for last-row live forecast
        final_epochs = best_epoch + 1
        final_model, _ = self._fit_model(seq_x, seq_y, seed=42, epochs=final_epochs)

        # 12. Last-row live forecast
        all_features = model_df[feature_cols].to_numpy(dtype=np.float32)
        all_scaled = (all_features - train_mean) / train_std
        if len(all_scaled) >= self.SEQ_LEN:
            last_seq = all_scaled[-self.SEQ_LEN:]
            if not np.isnan(last_seq).any():
                last_seq = last_seq[None, :, :]  # (1, seq_len, n_features)
                last_prob_up = self._predict_last_row(final_model, last_seq)
                last_pred = int(last_prob_up >= 0.5)
                last_conf = last_prob_up if last_pred == 1 else 1.0 - last_prob_up
                df.loc[df.index[-1], "Pred_Direction"] = last_pred
                df.loc[df.index[-1], "Pred_Probability"] = last_conf
                df.loc[df.index[-1], "Prob_Up"] = last_prob_up

        # 13. Metrics
        train_prob = self._prob_up_from_model(eval_model, x_train_seq)
        train_acc = float(((train_prob >= 0.5).astype(int) == y_train_seq.astype(int)).mean())

        val_acc = np.nan
        if len(x_val_seq) > 0:
            val_acc = float(((val_prob_up >= 0.5).astype(int) == y_val_seq.astype(int)).mean())

        df.attrs["train_accuracy"] = train_acc
        df.attrs["validation_accuracy"] = val_acc
        df.attrs["train_size"] = int(len(x_train_seq))
        df.attrs["test_size"] = int(len(x_val_seq))
        df.attrs["feature_cols"] = feature_cols

        # Store in session cache
        self._pred_cache[cache_key] = (df.copy(), now)

        return df
