import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Optional
from .plots import _PlotBacktest
from .utils import Utils
from .stats import Stats
from .base_strategy import Strategy

import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Analyzer:
    def __init__(self,strategy: Strategy,
        params: Dict[str, Any],
        full_data: pd.DataFrame,
        timeframe: str,
        price_col: str = "close",
        test_size: float = 0,
        init_cash: float = 1000.0,
        fees: float = 0.0002,
        slippage: float = 0.000,
        trade_side: Optional[str] = 'longonly',
        tp_stop: Optional[float] = None,
        sl_stop: Optional[float] = None):
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
        self.trade_side = trade_side
        self.full_data = self.util.validate_data(full_data)

        if not isinstance(self.full_data.index, pd.DatetimeIndex):
            warnings.warn("Data index is not datetime-based. Time-based features may not work as expected.", stacklevel=2)

        if test_size > 0:
            self.train_df, self.test_df = self.util.time_based_split(self.full_data, test_size)
            self.train_df = self.strategy.preprocess_data(self.train_df.copy(), params)
        else:
            self.train_df = self.strategy.preprocess_data(self.full_data.copy(), params)
            self.test_df = None

        self.signals = self.strategy.generate_signals(self.train_df, **params)
        self._validate_signals()

        if self.trade_side == 'shortonly':
            portfolio_kwargs = dict(
                close=self.train_df[self.s.price_col],
                short_entries=self.signals['short_entries'],
                short_exits=self.signals.get('short_exits'),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction='shortonly'
            )
        else:
            portfolio_kwargs = dict(
                close=self.train_df[self.s.price_col],
                entries=self.signals.get('entries'),
                exits=self.signals.get('exits'),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction=self.trade_side
            )
            if 'short_entries' in self.signals:
                portfolio_kwargs['short_entries'] = self.signals['short_entries']
            if 'short_exits' in self.signals:
                portfolio_kwargs['short_exits'] = self.signals['short_exits']

        if self.tp_stop is not None:
            portfolio_kwargs['tp_stop'] = self.tp_stop
        if self.sl_stop is not None:
            portfolio_kwargs['sl_stop'] = self.sl_stop

        self.pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
    
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
    
    def oos_test(self) -> Optional[vbt.Portfolio]:
        if self.test_df is None or self.test_df.empty:
            return None
        test_df = self.strategy.preprocess_data(self.test_df.copy(), self.params)
        test_signals = self.strategy.generate_signals(test_df, **self.params)

        if self.trade_side == 'shortonly':
            pk = dict(
                close=test_df[self.s.price_col],
                short_entries=test_signals['short_entries'],
                short_exits=test_signals.get('short_exits'),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction='shortonly'
            )
        else:
            pk = dict(
                close=test_df[self.s.price_col],
                entries=test_signals.get('entries'),
                exits=test_signals.get('exits'),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction='longonly'
            )
            if 'short_entries' in test_signals:
                pk['short_entries'] = test_signals['short_entries']
            if 'short_exits' in test_signals:
                pk['short_exits'] = test_signals['short_exits']

        if self.tp_stop is not None:
            pk['tp_stop'] = self.tp_stop
        if self.sl_stop is not None:
            pk['sl_stop'] = self.sl_stop

        return vbt.Portfolio.from_signals(**pk)

    def backtest_results(self) -> pd.DataFrame:
        return self.s.backtest_summary(self.pf, self.timeframe)

    def plot_backtest(self, title: str = 'Backtest Results'):
        return _PlotBacktest(self).plot_backtest(title=title)
    
    def export_trades(self, file_name: Optional[str] = 'strategy_report', save_dir: str = r"path"):
     """
     Exports trades to CSV, needed for portfolio optimization later
     """
     trades = self.pf.trades.records_readable.copy()
     os.makedirs(save_dir, exist_ok=True)
     file_path = os.path.join(save_dir, f"{file_name}")
     trades.to_feather(file_path, index=False)
     print(f"Trades successfully exported to: {file_path}")

#