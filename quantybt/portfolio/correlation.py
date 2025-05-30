import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
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
    Correlation Analyzer for robust correlation analysis

    Computes lambda_lower and lambda_upper for Clayton and Gumbel copulas
    over both full sample and active days only.
    """
    def __init__(self,trade_sources: Dict[str, Dict[str, str]]):
        super().__init__()
        self.trade_sources = trade_sources
        self.mapped_trades = self.preprocess_trade_sources(trade_sources)
        self.n_strategies = len(self.mapped_trades)
        self.combined: pd.DataFrame = pd.DataFrame()
        self.results: Dict[str, float] = {}

        print(f"Loaded {self.n_strategies} strategy{'ies' if self.n_strategies != 1 else ''}: {list(self.mapped_trades.keys())}")

    def _is_stationary(self, series: pd.Series) -> Dict[str, float]:
        s = series.dropna()
        adf_stat, adf_p, *_ = adfuller(s, regression='c', autolag='AIC')
        kpss_stat, kpss_p, *_ = kpss(s, regression='c', nlags='auto')
        return {
            'stationary': (adf_p < 0.05 and kpss_p > 0.05),
            'adf_p': adf_p,
            'kpss_p': kpss_p
        }

    def run(self,rolling_window: int = 180,test_stationary: bool = True) -> Dict[str, float]:
        if self.n_strategies == 2:
            return self._run_bivariate_analysis(rolling_window, test_stationary)
        else:
            return self._run_multivariate_analysis(test_stationary)

    def _run_bivariate_analysis(self, rolling_window: int,test_stationary: bool) -> Dict[str, float]:

        names = list(self.mapped_trades.keys())
        A, B = names[0], names[1]

        a_df = self.mapped_trades[A]['DailyReturn'].rename(A)
        b_df = self.mapped_trades[B]['DailyReturn'].rename(B)

        combined = pd.concat([a_df, b_df], axis=1, join='inner').dropna(how='any')
        self.combined = combined

        if test_stationary:
            warnings.filterwarnings("ignore", message=".*InterpolationWarning.*")
            stat_a = self._is_stationary(combined[A])
            stat_b = self._is_stationary(combined[B])
            if not stat_a['stationary']:
                print(f"{A} not stationary | ADF p={stat_a['adf_p']:.4f}, KPSS p={stat_a['kpss_p']:.4f}")
            if not stat_b['stationary']:
                print(f"{B} not stationary | ADF p={stat_b['adf_p']:.4f}, KPSS p={stat_b['kpss_p']:.4f}")

        mask_A = combined[A] != 0
        mask_B = combined[B] != 0
        mask_B_forward = mask_B.shift(1, fill_value=False)
        mask_B_backward = mask_B.shift(-1, fill_value=False)
        active_mask = mask_A & (mask_B | mask_B_forward | mask_B_backward)
        active = combined.loc[active_mask]

        eps = 1e-6
        u_full = np.clip(combined[A].rank(pct=True).values, eps, 1-eps)
        v_full = np.clip(combined[B].rank(pct=True).values, eps, 1-eps)
        res_c_full = minimize(lambda t: neg_log_lik_clayton(t, u_full, v_full), x0=[1.0], bounds=[(1e-6, None)])
        theta_c_full = res_c_full.x[0]
        lambda_lower_full = 2 ** (-1.0 / theta_c_full)
        res_g_full = minimize(lambda t: neg_log_lik_gumbel(t, u_full, v_full), x0=[1.5], bounds=[(1.0, None)])
        theta_g_full = res_g_full.x[0]
        lambda_upper_full = 2 - 2 ** (1.0 / theta_g_full)

        if not active.empty:
            u_act = np.clip(active[A].rank(pct=True).values, eps, 1-eps)
            v_act = np.clip(active[B].rank(pct=True).values, eps, 1-eps)
            res_c_act = minimize(lambda t: neg_log_lik_clayton(t, u_act, v_act), x0=[1.0], bounds=[(1e-6, None)])
            theta_c_act = res_c_act.x[0]
            lambda_lower_act = 2 ** (-1.0 / theta_c_act)
            res_g_act = minimize(lambda t: neg_log_lik_gumbel(t, u_act, v_act), x0=[1.5], bounds=[(1.0, None)])
            theta_g_act = res_g_act.x[0]
            lambda_upper_act = 2 - 2 ** (1.0 / theta_g_act)
        else:
            lambda_lower_act = np.nan
            lambda_upper_act = np.nan


        self.results = {
            'pearson_full': round(combined[A].corr(combined[B]), 2),
            'pearson_active': round((pearson_act := (active[A].corr(active[B]) if not active.empty else np.nan)), 2),
            'spearman_full': round(combined[A].corr(combined[B], method='spearman'), 2),
            'spearman_active': round((spearman_act := (active[A].corr(active[B], method='spearman') if not active.empty else np.nan)), 2),
            'kendall_full': round(combined[A].corr(combined[B], method='kendall'), 2),
            'kendall_active': round((kendall_act := (active[A].corr(active[B], method='kendall') if not active.empty else np.nan)), 2),
            'clayton_lambda_full': round(lambda_lower_full, 2),
            'clayton_lambda_active': round(lambda_lower_act, 2) if not active.empty else np.nan,
            'gumbel_lambda_full': round(lambda_upper_full, 2),
            'gumbel_lambda_active': round(lambda_upper_act, 2) if not active.empty else np.nan
        }
        return self.results
    
    def _run_multivariate_analysis(self, test_stationary: bool):
         names = list(self.mapped_trades.keys())
         combined = pd.concat([self.mapped_trades[name]['DailyReturn'].rename(name) for name in names],axis=1, join='inner').dropna(how='any')
         self.combined = combined

         if test_stationary:
          for name in names:
            stat = self._is_stationary(combined[name])
            if not stat['stationary']:
                print(f"{name} not stationary | ADF p={stat['adf_p']:.4f}, KPSS p={stat['kpss_p']:.4f}")
         
         active_masks = []
         for name in names:
          mask = combined[name] != 0
          mask_fwd = mask.shift(1, fill_value=False)
          mask_bwd = mask.shift(-1, fill_value=False)
          active_masks.append(mask | mask_fwd | mask_bwd)

         active_mask = np.logical_or.reduce(active_masks)
         active = combined.loc[active_mask]

         pearson_corr = combined.corr(method='pearson')
         pearson_corr_active = active.corr(method='pearson')

         
         self.results = {
         "pearson_corr_full": pearson_corr.round(2),
         "pearson_corr_active": pearson_corr_active.round(2),
         }

         return self.results
    
    def plot(self, rolling_window: int = 180) -> None:
     if self.combined is None or self.combined.empty:
        raise ValueError("No data to plot. Run .run() first.")

     names = list(self.mapped_trades.keys())
     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

     for name, df in self.mapped_trades.items():
        ax1.plot(df['Equity'], label=name)
     ax1.set_title("Equity Curves")
     ax1.legend()
     ax1.grid(True)

     if self.n_strategies == 2 and 'pearson_full' in self.results:
        A, B = names[0], names[1]
        roll_full = self.combined[A].rolling(rolling_window).corr(self.combined[B])
        ax2.plot(roll_full, label=f"{rolling_window}-Day Rolling Corr (Full)")
        ax2.axhline(0, linestyle='--', color='gray')
        ax2.set_title("Rolling Correlation")
        ax2.set_ylabel("Correlation")
        ax2.legend()
        ax2.grid(True)

     elif 'pearson_corr_active' in self.results:
        import seaborn as sb
        corr_matrix = self.results['pearson_corr_active']
        mask = np.eye(corr_matrix.shape[0], dtype=bool) 
        sb.heatmap(corr_matrix, mask=mask, annot=True, cmap='crest',center=0, ax=ax2)
        ax2.set_title("Pearson Corr. Matrix (Active ±1)")


      
     else:
      ax2.text(0.5, 0.5, "No analysis results to plot.",
                 ha='center', va='center', fontsize=12)
      ax2.axis('off')

     plt.tight_layout()
     plt.show()

####