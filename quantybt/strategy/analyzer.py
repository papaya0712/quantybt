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
        size:    Union[None, float, pd.Series, Callable] = None,
    ):
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
        self.size    = size
        # Fix size type to percent
        self.size_type = SizeType.Percent
        self.trade_side = trade_side

        # Prepare data
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

        # Split into train/test
        if test_size > 0:
            self.train_df, self.test_df = self.util.time_based_split(self.full_data, test_size)
            self.train_df = self.strategy.preprocess_data(self.train_df.copy(), params)
        else:
            self.train_df = self.strategy.preprocess_data(self.full_data.copy(), params)
            self.test_df = None

        # Generate and validate signals
        self.signals = self.strategy.generate_signals(self.train_df, **params)
        self._validate_signals()

        # Expand dynamic params to per-bar Series
        sl_series = self._expand_param(self.sl_stop, self.train_df)
        tp_series = self._expand_param(self.tp_stop, self.train_df)
        size_series = self._expand_param(self.size, self.train_df)

        # Build kwargs for Portfolio
        portfolio_kwargs = dict(
            close=self.train_df[self.s.price_col],
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=self.trade_side,
            size_type=self.size_type
        )
        # Signals
        if self.trade_side == 'shortonly':
            portfolio_kwargs['short_entries'] = self.signals['short_entries']
            portfolio_kwargs['short_exits'] = self.signals.get('short_exits')
        else:
            portfolio_kwargs['entries'] = self.signals.get('entries')
            portfolio_kwargs['exits'] = self.signals.get('exits')
            if 'short_entries' in self.signals:
                portfolio_kwargs['short_entries'] = self.signals['short_entries']
            if 'short_exits' in self.signals:
                portfolio_kwargs['short_exits'] = self.signals['short_exits']

        # Dynamic stops & size
        if tp_series is not None:
            portfolio_kwargs['tp_stop'] = tp_series
        if sl_series is not None:
            portfolio_kwargs['sl_stop'] = sl_series
        if size_series is not None:
            portfolio_kwargs['size'] = size_series

        # Create portfolio
        self.pf = Portfolio.from_signals(**portfolio_kwargs)

    def _validate_signals(self):
        if self.trade_side == 'shortonly':
            required = ['short_entries', 'short_exits']
        elif self.trade_side == 'longonly':
            required = ['entries', 'exits']
        else:
            required = ['entries', 'exits', 'short_entries', 'short_exits']

        for k in required:
            if k not in self.signals or not isinstance(self.signals[k], pd.Series):
                raise ValueError(f"Signal '{k}' fehlt oder ist kein pd.Series.")
            if self.signals[k].dtype != bool:
                self.signals[k] = self.signals[k].astype(bool)

        if self.trade_side == 'shortonly' and not self.signals['short_entries'].any():
            raise ValueError("No short entry signals generated")
        if self.trade_side != 'shortonly' and not self.signals['entries'].any():
            raise ValueError("No entry signals generated")

    def _expand_param(self, param, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Convert a scalar, pd.Series, or Callable to a pd.Series aligned with df.index.
        """
        if param is None:
            return None
        result = param(df) if callable(param) else param
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)
        return result

    def oos_test(self) -> Optional[vbt.Portfolio]:
        if self.test_df is None or self.test_df.empty:
            return None
        test_df = self.strategy.preprocess_data(self.test_df.copy(), self.params)
        test_signals = self.strategy.generate_signals(test_df, **self.params)
        self._validate_signals()

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
            pk['short_entries'] = test_signals['short_entries']
            pk['short_exits'] = test_signals.get('short_exits')
        else:
            pk['entries'] = test_signals.get('entries')
            pk['exits'] = test_signals.get('exits')
            if 'short_entries' in test_signals:
                pk['short_entries'] = test_signals['short_entries']
            if 'short_exits' in test_signals:
                pk['short_exits'] = test_signals['short_exits']

        if tp_series is not None:
            pk['tp_stop'] = tp_series
        if sl_series is not None:
            pk['sl_stop'] = sl_series
        if size_series is not None:
            pk['size'] = size_series

        return Portfolio.from_signals(**pk)

    def backtest_results(self) -> pd.DataFrame:
        return self.s.backtest_summary(self.pf, self.timeframe)

    def plot_backtest(self, title: str = 'Backtest Results'):
        return _PlotBacktest(self).plot_backtest(title=title)

    def export_trades(self, file_name: Optional[str] = 'strategy_report',save_dir: str = r"path"):
        trades = self.pf.trades.records_readable.copy()
        os.makedirs(save_dir, exist_ok=True)

        if not file_name.endswith('.feather'):
            file_name += '.feather'

        file_path = os.path.join(save_dir, file_name)
        trades.reset_index(drop=True).to_feather(file_path)
        print(f"Trades successfully exported to: {file_path}")
#
