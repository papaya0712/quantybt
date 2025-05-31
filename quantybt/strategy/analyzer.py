import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio import Portfolio
from vectorbt.portfolio.enums import SizeType
from typing import Dict, Any, Optional, Union, Callable
from .plots import _PlotBacktest
from .utils import Utils
from .stats import Stats
from .base_strategy import Strategy

import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Analyzer:
    def __init__(
        self,
        strategy: Strategy,
        params: Dict[str, Any],
        full_data: pd.DataFrame,
        timeframe: str,
        price_col: str = "close",
        test_size: float = 0,
        init_cash: float = 1000.0,
        fees: float = 0.0002,
        slippage: float = 0.000,
        trade_side: Optional[str] = 'longonly',
        tp_stop: Union[None, float, pd.Series, Callable] = None,
        sl_stop: Union[None, float, pd.Series, Callable] = None,
        size: Union[None, float, pd.Series, Callable] = None,):

        self.s = Stats(price_col=price_col)
        self.util = Utils()
        self.strategy = strategy
        self.params = params
        self.timeframe = timeframe
        self.test_size = test_size
        self.init_cash = float(init_cash)
        self.fees = fees
        self.slippage = slippage
        self.tp_stop = tp_stop
        self.sl_stop = sl_stop
        self.size = size

        #self.size_type = SizeType.Percent
        self.size_type = SizeType.Value
        self.trade_side = trade_side

        self.full_data = self.util.validate_data(full_data)
        self.full_data['timestamp'] = pd.to_datetime(self.full_data['timestamp'])
        self.full_data = (
            self.full_data
            .set_index('timestamp', drop=True)
            .sort_index()
        )
        if not isinstance(self.full_data.index, pd.DatetimeIndex):
            warnings.warn(
                "Data index is not datetime-based. "
                "Time-based features may not work as expected.",
                stacklevel=2
            )

        if test_size > 0:
            self.train_df, self.test_df = self.util.time_based_split(self.full_data, test_size)
            self.train_df = self.strategy.preprocess_data(self.train_df.copy(), params)
        else:
            self.train_df = self.strategy.preprocess_data(self.full_data.copy(), params)
            self.test_df = None

        
        raw_signals = self.strategy.generate_signals(self.train_df, **params)
        self._validate_signals(raw_signals)

        self.signals = raw_signals

        sl_series = self._expand_param(self.sl_stop, self.train_df)
        tp_series = self._expand_param(self.tp_stop, self.train_df)
        size_series = self._expand_param(self.size, self.train_df)

        portfolio_kwargs = dict(
            close=self.train_df[self.s.price_col],
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=self.trade_side,
            size_type=self.size_type,
            )
        


        if self.trade_side == 'shortonly':
            portfolio_kwargs['short_entries'] = self.signals['short_entries']
            portfolio_kwargs['short_exits'] = self.signals['short_exits']
        else:
            # Long
            portfolio_kwargs['entries'] = self.signals['entries']
            portfolio_kwargs['exits'] = self.signals['exits']
            # Short 
            if 'short_entries' in self.signals:
                portfolio_kwargs['short_entries'] = self.signals['short_entries']
            if 'short_exits' in self.signals:
                portfolio_kwargs['short_exits'] = self.signals['short_exits']

        
        if tp_series is not None:
            portfolio_kwargs['tp_stop'] = tp_series
        if sl_series is not None:
            portfolio_kwargs['sl_stop'] = sl_series
        if size_series is not None:
            portfolio_kwargs['size'] = size_series

        self.pf = Portfolio.from_signals(**portfolio_kwargs)

    def _validate_signals(self, signals: Dict[str, pd.Series]):
        if self.trade_side == 'shortonly':
            required = ['short_entries', 'short_exits']
        elif self.trade_side == 'longonly':
            required = ['entries', 'exits']
        else:  
            required = ['entries', 'exits', 'short_entries', 'short_exits']

        for key in required:
            if key not in signals or not isinstance(signals[key], pd.Series):
                raise ValueError(f"Signal '{key}' missing or no pandas Series")
            
            if signals[key].dtype != bool:
                signals[key] = signals[key].astype(bool)

        
        if self.trade_side == 'shortonly' and not signals['short_entries'].any():
            raise ValueError("No short entry signals generated.")
        if self.trade_side != 'shortonly' and not signals['entries'].any():
            raise ValueError("No long entry signals generated.")
  
    def _expand_param(self, param, df: pd.DataFrame) -> Optional[Union[float, pd.Series, Dict[str, Any]]]:
        
        if param is None:
            return None
        if isinstance(param, (float, int)):
            return param
        if isinstance(param, pd.Series):
            return param
        if isinstance(param, str):
            if param not in df.columns:
                raise KeyError(f"'{param}' not found")
            return df[param]
        if isinstance(param, dict):
            result = {}
            for side, value in param.items():
                result[side] = self._expand_param(value, df)
            return result
        if callable(param):
            result = param(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            return result
        raise TypeError(f"Parameter-Typ {type(param)} not supported")

    def oos_test(self) -> Optional[vbt.Portfolio]:
        if self.test_df is None or self.test_df.empty:
            return None

        test_df = self.strategy.preprocess_data(self.test_df.copy(), self.params)
        raw_signals = self.strategy.generate_signals(test_df, **self.params)
        self._validate_signals(raw_signals)
        signals = self.signals

        sl_series = self._expand_param(self.sl_stop, test_df)
        tp_series = self._expand_param(self.tp_stop, test_df)
        size_series = self._expand_param(self.size, test_df)

        pk = dict(
            close=test_df[self.s.price_col],
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=self.trade_side,
            size_type=self.size_type
        )
        if self.trade_side == 'shortonly':
            pk['short_entries'] = signals['short_entries']
            pk['short_exits'] = signals['short_exits']
        else:
            pk['entries'] = signals['entries']
            pk['exits'] = signals['exits']
            if 'short_entries' in signals:
                pk['short_entries'] = signals['short_entries']
            if 'short_exits' in signals:
                pk['short_exits'] = signals['short_exits']

        if tp_series is not None:
            pk['tp_stop'] = tp_series
        if sl_series is not None:
            pk['sl_stop'] = sl_series
        if size_series is not None:
            pk['size'] = size_series

        return Portfolio.from_signals(**pk)

    def backtest_results(self) -> pd.DataFrame:
        return self.s.backtest_summary(self.pf, self.timeframe)

    def plot_backtest(self, title: str = 'Backtest Results [Plot]'): 
        return _PlotBacktest(self).plot_backtest(title=title)

    def export_trades(self, file_name: Optional[str] = 'strategy_report', save_dir: str = r"path"):
        trades = self.pf.trades.records_readable.copy()
        os.makedirs(save_dir, exist_ok=True)

        if not file_name.endswith('.feather'):
            file_name += '.feather'

        file_path = os.path.join(save_dir, file_name)
        trades.reset_index(drop=True).to_feather(file_path)
        print(f"Trades successfully exported to: {file_path}")

####

