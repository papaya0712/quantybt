import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from quantybt.portfolio.base import BaseModel
from quantybt.portfolio.functions import neg_log_lik_clayton, neg_log_lik_gumbel
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from scipy.optimize import minimize
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=InterpolationWarning)

####
class CorrelationAnalyzer(BaseModel):
    """
    Correlation Analyzer aiming or robust correlation analysis

    Parameter:
    - rolling_window: defines the lookback window for the rolling correlation
    """

    def __init__(self, trade_sources: Dict[str, Dict[str, str]]):
        super().__init__()
        self.trade_sources = trade_sources
        self.mapped_trades = self.preprocess_trade_sources(trade_sources)
        self.results = {}
        self.combined = None
        self.n_strategies = len(self.mapped_trades)

        print(f"Loaded {self.n_strategies} strategy{'ies' if self.n_strategies != 1 else ''}: {list(self.mapped_trades.keys())}")

    def _is_stationary(self, returns):
     eps = 1e-6
     series = returns.dropna()
     adf_stat, adf_p, _, _, _, _ = adfuller(series, regression='c', autolag='AIC')
     kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
     return {
        "stationary": (adf_p < 0.05) and (kpss_p > 0.05),
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p}   

    def run(self, rolling_window: int = 180, test_stationary: bool = True) -> Dict[str, float]:
        if self.n_strategies == 2:
            return self._run_bivariate_analysis(rolling_window, test_stationary)
        else:
            return self._run_multivariate_analysis(rolling_window, test_stationary)
    
    def _run_bivariate_analysis(self, rolling_window, test_stationary):
        a_data = self.mapped_trades['strategy_A']
        b_data = self.mapped_trades['strategy_B']
        combined = pd.concat([a_data['DailyReturn'], b_data['DailyReturn']], axis=1, keys=['A', 'B']).fillna(0)
        self.combined = combined

        if test_stationary:
            warnings.filterwarnings("ignore", message=".*InterpolationWarning.*")
            a_stat = self._is_stationary(combined['A'])
            b_stat = self._is_stationary(combined['B'])

            if not a_stat['stationary']:
                print(f"Strategy A not stationary | ADF: {a_stat['adf_p']:.4f}, KPSS: {a_stat['kpss_p']:.4f}")
            if not b_stat['stationary']:
                print(f"Strategy B not stationary | ADF: {b_stat['adf_p']:.4f}, KPSS: {b_stat['kpss_p']:.4f}")

        corr_pearson = combined['A'].corr(combined['B'])
        corr_spearman = combined['A'].corr(combined['B'], method='spearman')
        corr_kendall = combined['A'].corr(combined['B'], method='kendall')

        active_days = combined[(combined['A'] != 0) & (combined['B'] != 0)]
        corr_active = active_days['A'].corr(active_days['B'])

        eps = 1e-6
        u = np.clip(combined['A'].rank(pct=True).values, eps, 1 - eps)
        v = np.clip(combined['B'].rank(pct=True).values, eps, 1 - eps)

        res_clayton = minimize(lambda t: neg_log_lik_clayton(t, u, v), x0=np.array([1.0]), bounds=[(1e-6, None)])
        theta_clayton = res_clayton.x[0]
        lambda_lower = 2 ** (-1.0 / theta_clayton)

        res_gumbel = minimize(lambda t: neg_log_lik_gumbel(t, u, v), x0=np.array([1.5]), bounds=[(1.0, None)])
        theta_gumbel = res_gumbel.x[0]
        lambda_upper = 2 - 2 ** (1.0 / theta_gumbel)

        self.results = {
            'pearson': corr_pearson,
            'spearman': corr_spearman,
            'kendall_tau': corr_kendall,
            'active_days_corr': corr_active,
            'theta_clayton': theta_clayton,
            'lambda_lower': lambda_lower,
            'theta_gumbel': theta_gumbel,
            'lambda_upper': lambda_upper,
        }

        return self.results
    
    def _run_multivariate_analysis(self, rolling_window, test_stationary):
        """dummy"""
        return 

    def plot(self, rolling_window: int = 180):
        if self.combined is None:
            raise ValueError("No data to plot. Run .run() first.")

        if self.n_strategies == 2:
            a_data = self.mapped_trades['strategy_A']
            b_data = self.mapped_trades['strategy_B']
            combined = self.combined

            rolling_corr = combined['A'].rolling(window=rolling_window).corr(combined['B'])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

            ax1.plot(a_data.index, a_data['Equity'], label='Strategy A')
            ax1.plot(b_data.index, b_data['Equity'], label='Strategy B')
            ax1.set_title("Cumulative Equity Curves")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(rolling_corr.index, rolling_corr, label=f'{rolling_window}-Day Rolling Correlation', linewidth=2)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.set_ylabel('Correlation')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        else:
            print("dummy")

####

