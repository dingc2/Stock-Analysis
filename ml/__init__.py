"""ML model registry. Add new models here for auto-discovery by the UI."""

from ml.base import ModelProvider


def get_available_models() -> list[ModelProvider]:
    """Return all registered ML models.

    To add a model:
    1. Create ml/your_model.py implementing ModelProvider
    2. Import it here and append to the list
    """
    models: list[ModelProvider] = []
    try:
        from ml.xgboost_direction import XGBoostDirection
        models.append(XGBoostDirection())
    except Exception:
        pass  # xgboost unavailable or native library not loadable
    try:
        from ml.lstm_direction import LSTMDirection
        models.append(LSTMDirection())
    except Exception:
        pass  # torch unavailable or model import failed
    return models
