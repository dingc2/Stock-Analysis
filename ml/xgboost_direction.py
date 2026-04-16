"""XGBoost 1-day price direction classifier.

Predicts whether tomorrow's close will be higher (1) or lower (0) than today's close.
Features are computed from indicators/technical.py to ensure consistency with charts.

Important output columns:
- Pred_Direction: 1 for Up, 0 for Down
- Pred_Probability: confidence of the predicted direction
- Prob_Up: raw probability that the next day is Up
"""

import numpy as np
from xgboost import XGBClassifier

import config
from ml.direction_base import BaseDirectionModel


class XGBoostDirection(BaseDirectionModel):
    """Predicts next-day price direction (up/down) using XGBoost."""

    MODEL_NAME = "XGBoost Direction"
    MODEL_DESCRIPTION = "Predicts next-day price direction (up/down) using technical indicators"
    MIN_TRAIN_ROWS = 60

    def _train_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
    ):
        model = XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
            reg_alpha=config.XGB_REG_ALPHA,
            reg_lambda=config.XGB_REG_LAMBDA,
            eval_metric=config.XGB_EVAL_METRIC,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
            random_state=config.XGB_RANDOM_STATE,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=False)
        best_iter = (
            model.best_iteration
            if hasattr(model, "best_iteration") and model.best_iteration is not None
            else config.XGB_N_ESTIMATORS
        )
        return model, best_iter

    def _final_train_predict(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        n_estimators: int,
    ):
        final_model = XGBClassifier(
            n_estimators=n_estimators,
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
        final_model.fit(X_all, y_all, verbose=False)
        last_prob_up = final_model.predict_proba(X_all[[-1]])[:, 1][0]
        return final_model, last_prob_up

    def _prob_up_from_model(self, model: XGBClassifier, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def _predict_last_row(self, final_model: XGBClassifier, last_X: np.ndarray) -> float:
        return float(final_model.predict_proba(last_X)[0, 1])
