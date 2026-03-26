"""Abstract base class for ML model providers."""

from abc import ABC, abstractmethod

import pandas as pd


class ModelProvider(ABC):
    """Contract for all ML models in the stock analysis platform."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on the given OHLCV + indicators DataFrame.
        Returns the DataFrame with prediction columns added.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def get_description(self) -> str:
        """Short description of what this model does."""
