import os
import pandas as pd
import numpy as np
from typing import Dict

####
class BaseModel:
    def __init__(self):
        pass

    def preprocess_trade_sources(self, trade_sources: Dict[str, Dict[str, str]], tz='UTC') -> Dict[str, pd.DataFrame]:
        
        mapped = {}
        for name, paths in trade_sources.items():
            trades_path = paths['trades']
            df_path = paths['df']

            if not os.path.isfile(trades_path):
                raise FileNotFoundError(f"Trade file not found: {trades_path}")
            if not os.path.isfile(df_path):
                raise FileNotFoundError(f"Market data file not found: {df_path}")

            trades_df = pd.read_feather(trades_path)
            market_df = pd.read_feather(df_path)

            trades_df = self._map_bars_to_time(trades_df, market_df)
            return_df = self._aggregate_returns(trades_df, tz=tz)
            mapped[name] = return_df
        return mapped

    def _map_bars_to_time(self, trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
     trades_df = trades_df.copy()

     trades_df['Entry Timestamp'] = pd.to_datetime(trades_df['Entry Timestamp'])
     trades_df['Exit Timestamp'] = pd.to_datetime(trades_df['Exit Timestamp'])

     if 'timestamp' in market_df.columns:
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
        market_df = market_df.set_index('timestamp')

     trades_df['entry_dt'] = trades_df['Entry Timestamp']
     trades_df['exit_dt'] = trades_df['Exit Timestamp']

     return trades_df

    def _aggregate_returns(self, trades_df: pd.DataFrame, tz='UTC') -> pd.DataFrame:
        df = trades_df.copy()
        df['exit_dt'] = pd.to_datetime(df['exit_dt'], utc=True)
        df = df.set_index('exit_dt')
        df.index = df.index.tz_convert(tz)

        daily_returns = (1 + df['Return']).resample('1D').prod() - 1
        daily_returns = daily_returns.fillna(0)
        equity_curve = (1 + daily_returns).cumprod()

        return pd.DataFrame({
            'DailyReturn': daily_returns,
            'Equity': equity_curve
        })

####