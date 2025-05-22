import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from quantybt.portfolio.base import BaseModel

# 
class SimpleCorrelationAnalyzer(BaseModel):
    def __init__(self, trade_sources: Dict[str, Dict[str, str]]):
        super().__init__()
        self.trade_sources = trade_sources
        self.mapped_trades = self.preprocess_trade_sources(trade_sources)
        self.results = {}
        self.combined = None

    def run(self, rolling_window: int = 180) -> Dict[str, float]:
        a_data = self.mapped_trades['strategy_A']
        b_data = self.mapped_trades['strategy_B']

        combined = pd.concat([a_data['DailyReturn'], b_data['DailyReturn']], axis=1, keys=['A', 'B']).fillna(0)
        self.combined = combined

        corr_pearson = combined['A'].corr(combined['B'])
        corr_spearman = combined['A'].corr(combined['B'], method='spearman')
        active_days = combined[(combined['A'] != 0) & (combined['B'] != 0)]
        corr_active = active_days['A'].corr(active_days['B'])

        self.results = {
            'pearson': corr_pearson,
            'spearman': corr_spearman,
            'active_days_corr': corr_active,}

        print(f"Pearson Correlation: {corr_pearson:.4f}")
        print(f"Spearman Correlation: {corr_spearman:.4f}")
        print(f"Correlation only on active days: {corr_active:.4f}")

        return self.results

    def plot(self, rolling_window: int = 180):
        if self.combined is None:
            raise ValueError("no data, use .run() first")

        a_data = self.mapped_trades['strategy_A']
        b_data = self.mapped_trades['strategy_B']
        combined = self.combined

        rolling_corr = combined['A'].rolling(window=rolling_window).corr(combined['B'])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Equity Curves
        ax1.plot(a_data.index, a_data['Equity'], label='Strategy A')
        ax1.plot(b_data.index, b_data['Equity'], label='Strategy B')
        ax1.set_title("Cummulative equity curves")
        ax1.legend()
        ax1.grid(True)

        #  Daily Returns
        ax2.plot(a_data.index, a_data['DailyReturn'], label='A Daily Return', alpha=0.6)
        ax2.plot(b_data.index, b_data['DailyReturn'], label='B Daily Return', alpha=0.6)
        ax2.set_title("Daily Returns")
        ax2.legend()
        ax2.grid(True)

        #Rolling Correlation
        ax3.plot(rolling_corr.index, rolling_corr, label=f'{rolling_window}-Day rolling Correlation', linewidth=2)
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.set_ylabel('Correlation')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

# 
