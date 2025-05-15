# QuantyBT 🪐

**A lightweight backtesting framework based on [vectorbt](https://github.com/polakowo/vectorbt), focused on statistical robustness, modularity, and seamless strategy integration with custom models and crypto-native data loading.**  
---

## Features

- **Simple integration** with vectorbt as the backtesting engine.
- **Custom Model Support**: Native wrappers for custom-implemented models (e.g., Kalman Filters) and statistical frameworks.  
- **Built-in data loaders** for cryptocurrencies from Binance (no api needed).
- **Modular architecture**: define strategies by inheriting from a base `Strategy` class (`preprocess`, `generate_signals`, `param_space`).
- **Robust Validation**: Out-of-sample splits and hyperparameter tuning via [Hyperopt](https://github.com/hyperopt/hyperopt).
- **Walkforward-Optimization**: with generalization loss function for dynamic overfitting control  
- **Statistical analysis tools**: Monte Carlo simulations
- **Parameter Sensitivity**: finite differences for local sensitivity (global incoming) 
- **Performance reporting**: generate equity curves, heatmaps, and metric summaries with minimal boilerplate.

---

## Incoming Features

- **More Custom Models**  
- **Portfolio Optimization** with advanced methods (HRP, CVaR, Maximum Entropy, ...)  
- **Live integration**
---

### Install
```bash
git clone https://github.com/papaya0712/quantybt.git
```

## Define ur strategy

```python
import pandas as pd
import numpy as np
from quantybt import Strategy
from typing import Dict, Any
from hyperopt import hp

class YourStrategy(Strategy):
    def preprocess_data(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Data preprocessing and feature engineering"""
        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Calculate indicators/features
        df['feature1'] = df['close'].rolling(params['feature1_window']).mean()
        df['feature2'] = df['high'].rolling(params['feature2_window']).max()
        df['trendfilter'] = df['close'] > df['close'].rolling(params['trendfilter_len']).mean()
        
        # Generate signals (avoid lookahead bias)
        df['entries'] = np.where(
            (df['feature1'] > params['feature1_thresh']) & 
            (df['feature2'] > params['feature2_thresh']) & 
            df['trendfilter'],
            1, 0
        ).shift(1)  # Critical: shift to prevent lookahead
        
        # Optional: exit signals
        df['exits'] = np.where(
            df['close'] < df['close'].rolling(params['exit_window']).mean(), 1, 0).shift(1)
        
        return df.dropna()

    def generate_signals(self, df: pd.DataFrame, **params) -> Dict[str, pd.Series]:
        """Extract signals from preprocessed data"""
        return {
            'entries': df['entries'].astype(bool),
            'exits': df['exits'].astype(bool),
            # Optional for short trades:
            # 'short_entries': df['short_entries'].astype(bool),
            # 'short_exits': df['short_exits'].astype(bool)
        }

    @property
    def param_space(self) -> Dict[str, Any]:
        """Hyperparameter optimization space"""
        return {
            "feature1_window": hp.choice("feature1_window", [20, 50, 100]),
            "feature1_thresh": hp.uniform("feature1_thresh", 0.9, 1.1),
            "feature2_window": hp.choice("feature2_window", [5, 10, 20]),
            "feature2_thresh": hp.uniform("feature2_thresh", 0.95, 1.05),
            "trendfilter_len": hp.choice("trendfilter_len", [100, 200, 300]),
            "exit_window": hp.choice("exit_window", [5, 10, 20])
        }

df = pd.read_feather("path/to/BTC_1d.feather")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

strategy = YourStrategy()

```

---

## Simple Backtesting

```python

from quantybt import Analyzer

analyzer = Analyzer(
    strategy=strat,
    params=params,
    full_data=df,
    timeframe='15m',
    price_col="close",
    init_cash=1000,
    fees=0.0006, # = 0.06% 
    slippage=0, 
    trade_side='longonly', 
    sl_stop=params['sl_pct'])

print(analyzer.backtest_results())


anaylzer.plot_backtest()
![Backtest Plot](backtest_plt.png)
```
