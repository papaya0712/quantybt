import pandas as pd
import numpy as np

from typing import Dict
from quantybt.portfolio.functions import *
from quantybt.portfolio.base import BaseModel

class EqualWeightPortfolio(BaseModel):
    """
    Simulates a 1/n weighted portfolio and calculates portfolio metrics. 
    Ensure that the input strategy records comming from an robust and realistic backtesting run and are not overfitted or biased
    
    Setting:
    - base settings are fine, dont change the timeframe, rf = 0 is already correct for crypto
    """
    def __init__(self, trade_sources: Dict[str, Dict[str, str]], tz: str = 'UTC'):
        super().__init__()
        self.trade_sources = trade_sources
        self.tz = tz
        
        self.mapped_trades = self.preprocess_trade_sources(trade_sources, tz=tz)
        self.results: Dict[str, float] = {}
        self.portfolio: pd.DataFrame = None

    def run(self, freq: str = '1d', rf: float = 0.) -> pd.DataFrame:

     df = pd.concat([df['DailyReturn'].rename(name) for name, df in self.mapped_trades.items()],axis=1).fillna(0)
     weights   = np.repeat(1/df.shape[1], df.shape[1])
     port_ret  = df.dot(weights)
     port_eq   = (1 + port_ret).cumprod()
     periods_per_year = annual_factor(freq, root=False)

     cagr    = CAGR(port_ret, periods_per_year)
     sr      = sharpe(port_ret, freq=freq, rf=rf)
     st      = sortino(port_ret, freq=freq, target=rf)
     st_adj  = sortino_adjusted(port_ret, freq=freq, target=rf)
     cm      = calmar(port_ret, port_eq, freq=freq)
     dd      = max_drawdown(port_eq)

     global_cvar_99 = global_CVaR(returns=port_ret, alpha=0.99)
     global_cvar_95 = global_CVaR(returns=port_ret, alpha=0.95)
     global_cvar_50 = global_CVaR(returns=port_ret, alpha=0.50)

     rolling_cvar_99 = rolling_CVaR(returns=port_ret, alpha=0.99, window=365)
     rolling_cvar_95 = rolling_CVaR(returns=port_ret, alpha=0.95, window=365)
     rolling_cvar_50 = rolling_CVaR(returns=port_ret, alpha=0.50, window=365)

     self.results = pd.DataFrame({
      'total_return_pct':      [(port_eq.iloc[-1] - 1) * 100],
      'CAGR %':                [cagr*100],
      'max_drawdown_pct':      [dd * 100],
      'sharpe':                [sr],
      'sortino':               [st],
      'sortino_adj':           [st_adj],
      'calmar':                [cm],
      'CVaR_99_pct':           [global_cvar_99 * 100],
      'CVaR_95_pct':           [global_cvar_95 * 100],
      'CVaR_50_pct':           [global_cvar_50 * 100],
      })
     
     self.portfolio = pd.DataFrame({
        'return': port_ret,
        'equity':      port_eq,
        'rolling_cvar_99': rolling_cvar_99,
        'rolling_cvar_95': rolling_cvar_95,
        'rolling_cvar_50': rolling_cvar_50,
        
        })
     

     return self.results

    def plot(self):
     import matplotlib.pyplot as plt
   
     if self.portfolio is None:
        raise ValueError("No portfolio data. Use .run() first")
     
     df = self.portfolio
     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

     ax1.plot(df.index, df['equity'], label='Portfolio Equity (1/n weighted)', linewidth=2)
     ax1.set_title('Equal-Weight Portfolio Equity Curve (Daily Aggregated)')
     ax1.legend()
     ax1.grid(True)

     for name, strategy_df in self.mapped_trades.items():
        ax1.plot(strategy_df.index, strategy_df['Equity'], label=f"{name}", alpha=0.5)

     ax2.plot(df.index, df['return'] * 100, label='Portfolio Daily Return')
     ax2.set_title('Equal-Weight Portfolio Daily Returns')
     ax2.set_ylabel('Return %')
     ax2.legend()
     ax2.grid(True)
     
     ax3.plot(df.index, df['rolling_cvar_99'] * 100, label = 'Rolling CVaR_99%')
     ax3.plot(df.index, df['rolling_cvar_95'] * 100, label = 'Rolling CVaR_95%')
     ax3.plot(df.index, df['rolling_cvar_50'] * 100, label = 'Rolling CVaR_50%')
     ax3.set_title('Rolling CVaR values, window = 365d')
     ax3.set_ylabel("%")
     ax3.legend()
     ax3.grid(True)
     

     plt.tight_layout()
     plt.show() 

#
