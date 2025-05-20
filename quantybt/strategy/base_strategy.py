from abc import ABC,abstractmethod
from typing import Dict, Any
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params) -> Dict[str, pd.Series]:
        """Generate signals on preprocessed data"""
        pass

    @property
    @abstractmethod
    def param_space(self) -> Dict[str, Any]:
        """Define hyperopt search space"""
        pass