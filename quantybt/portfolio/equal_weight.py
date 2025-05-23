import pandas as pd
import numpy as np
from typing import Dict

from quantybt.portfolio.functions import max_drawdown, sharpe, sortino_adjusted, sortino, calmar
from quantybt.portfolio.base import BaseModel

class EqualWeightPortfolio(BaseModel):
    """
    Simulates a 1/n weighted portfolio and calculates portfolio metrics 
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
     self.portfolio = pd.DataFrame({
        'DailyReturn': port_ret,
        'Equity':      port_eq})
     

     sr      = sharpe(port_ret, freq=freq, rf=rf)
     st      = sortino(port_ret, freq=freq, target=rf)
     st_adj  = sortino_adjusted(port_ret, freq=freq, target=rf)
     cm      = calmar(port_ret, port_eq, freq=freq)
     dd      = max_drawdown(port_eq)

     self.results = pd.DataFrame([{
        'total_return_portfolio':    (port_eq.iloc[-1] - 1) * 100,
        'max_drawdown_portfolio':     dd * 100,
        'sharpe_portfolio':           sr,
        'sortino_portfolio':          st,
        'sortino_portfolio_adj':      st_adj,
        'calmar_portfolio':           cm}]
        ).round(3)

     return self.results

    def plot(self):
     if self.portfolio is None:
        raise ValueError("No portfolio data. Use .run() first")

     import matplotlib.pyplot as plt

     df = self.portfolio
     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

     ax1.plot(df.index, df['Equity'], label='Portfolio Equity (1/n weighted)', linewidth=2)
     ax1.set_title('Equal-Weight Portfolio Equity Curve (Daily Aggregated)')
     ax1.legend()
     ax1.grid(True)

     for name, strategy_df in self.mapped_trades.items():
        ax1.plot(strategy_df.index, strategy_df['Equity'], label=f"{name}", alpha=0.5)

     ax2.plot(df.index, df['DailyReturn'], label='Portfolio Daily Return')
     ax2.set_title('Equal-Weight Portfolio Daily Returns')
     ax2.legend()
     ax2.grid(True)

     plt.tight_layout()
     plt.show()

#
