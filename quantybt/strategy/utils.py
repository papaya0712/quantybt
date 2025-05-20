
import pandas as pd
from typing import Tuple

class Utils:
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
     df = df.copy()

     if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')
            df.index.name = None
            df['timestamp'] = df.index
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            df['timestamp'] = df.index
        else:
            raise ValueError("DataFrame has no DatetimeIndex and no 'timestamp' column")
     else:
        df.index.name = None
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index

     df = df[~df.index.duplicated(keep='first')]

     if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        if not df.index.is_monotonic_increasing:
            raise ValueError("Data cannot be sorted chronologically")

     return df

    def time_based_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.validate_data(df)

        delta = df.index.to_series().diff().dropna().min()
        if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
            raise ValueError("Cannot determine a positive time delta from index")

        split_idx = int(len(df) * (1 - test_size))
        split_date = df.index[split_idx]
        train = df.loc[: split_date - delta]
        test  = df.loc[ split_date :]

        if not train.empty and not test.empty:
            if train.index.max() >= test.index.min():
                raise RuntimeError("Overlap detected between train and test sets")

        return train, test
    
#