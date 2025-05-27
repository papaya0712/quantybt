import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple
from quantybt.portfolio.functions import (
    annual_factor, CAGR, sharpe, sortino, sortino_adjusted,
    calmar, max_drawdown, global_CVaR, rolling_CVaR
)
from quantybt.portfolio.base import BaseModel
from quantybt.strategy import Bootstrapping
from .functions import EmpiricalCVaR

####

class WeightedPortfolio(BaseModel):
    """
    Simulates a 1/n weighted portfolio and calculates portfolio metrics.

    Methods:
    - run: computes the equal-weight portfolio performance and metrics
    - plot: standard plots of equity, returns, and rolling CVaR

    Note:
    - due to special crypto condition in markets and exchanges there is no rebalancing cost simulation needed 
      assuming a cross margin mode is used on your crypto-exchange
      
    

    """
    def __init__(self, trade_sources: Dict[str, Dict[str, str]], tz: str = 'UTC'):
        super().__init__()
        self.trade_sources = trade_sources
        self.tz = tz
        self.mapped_trades = self.preprocess_trade_sources(trade_sources, tz=tz)
        self.results: pd.DataFrame = None
        self.portfolio: pd.DataFrame = None

    def run(self, freq: str = '1d', rf: float = 0.) -> pd.DataFrame:
        df = pd.concat(
            [df['DailyReturn'].rename(name) for name, df in self.mapped_trades.items()], axis=1
        ).fillna(0)
        weights = np.repeat(1 / df.shape[1], df.shape[1])
        port_ret = df.dot(weights)
        port_eq = (1 + port_ret).cumprod()
        periods_per_year = annual_factor(freq, root=False)

        cagr = CAGR(port_ret, periods_per_year)
        sr = sharpe(port_ret, freq=freq, rf=rf)
        st = sortino(port_ret, freq=freq, target=rf)
        st_adj = sortino_adjusted(port_ret, freq=freq, target=rf)
        cm = calmar(port_ret, port_eq, freq=freq)
        dd = max_drawdown(port_eq)

        gc99 = global_CVaR(returns=port_ret, alpha=0.99)
        gc95 = global_CVaR(returns=port_ret, alpha=0.95)
        gc50 = global_CVaR(returns=port_ret, alpha=0.50)

        rc99 = rolling_CVaR(returns=port_ret, alpha=0.99, window=365)
        rc95 = rolling_CVaR(returns=port_ret, alpha=0.95, window=365)
        rc50 = rolling_CVaR(returns=port_ret, alpha=0.50, window=365)
        

        # bootstrapped CVaR

        bs = EmpiricalCVaR(returns=port_ret, n_sims=10000, random_seed=42)
        cvar_list = bs.to_cvar_list(alpha=0.05)
        empirical_cvar_95 = np.nanmean(cvar_list)

        

        self.results = pd.DataFrame({
            'total_return_pct': [(port_eq.iloc[-1] - 1) * 100],
            'CAGR_pct': [cagr * 100],
            'max_drawdown_pct': [dd * 100],
            'Sharpe': [sr],
            #'Sortino': [st],
            'Sortino': [st_adj],
            'Calmar': [cm],
            'CVaR_99_pct': [gc99 * 100],
            'CVaR_95_pct': [gc95 * 100],
            'CVaR_50_pct': [gc50 * 100],
            'Empirical_CVaR_95_pct': [empirical_cvar_95 * 100]
        }).round(2)

        self.portfolio = pd.DataFrame({
            'return': port_ret,
            'equity': port_eq,
            'rolling_cvar_99': rc99,
            'rolling_cvar_95': rc95,
            'rolling_cvar_50': rc50,
        })
        return self.results

    def plot(self):
        if self.portfolio is None:
            raise ValueError("No portfolio data. Use .run() first.")
        df = self.portfolio
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax1, ax2, ax3 = axes
        ax1.plot(df.index, df['equity'], label='Portfolio Equity (1/n)', lw=2)
        ax1.set_title('Equal-Weight Portfolio Equity Curve')
        for name, s in self.mapped_trades.items():
            ax1.plot(s.index, s['Equity'], label=name, alpha=0.5)
        ax1.legend(); ax1.grid(True)
        ax2.plot(df.index, df['return']*100, label='Daily Return')
        ax2.set_title('Portfolio Daily Returns'); ax2.set_ylabel('%'); ax2.grid(True)
        ax3.plot(df.index, df['rolling_cvar_99']*100, label='CVaR 99%')
        ax3.plot(df.index, df['rolling_cvar_95']*100, label='CVaR 95%')
        ax3.plot(df.index, df['rolling_cvar_50']*100, label='CVaR 50%')
        ax3.set_title('Rolling CVaR (365-day)'); ax3.set_ylabel('%'); ax3.legend(); ax3.grid(True)
        plt.tight_layout(); plt.show()  