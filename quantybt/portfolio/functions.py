import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.optimize import minimize


# ---------------------------------------------------------------- # tail dependence correlation

def _clayton_logpdf(theta, u, v):
    term = (u**(-theta) + v**(-theta) - 1.0)
    return np.log(theta + 1) - (theta + 1)*(np.log(u) + np.log(v)) - (2 + 1/theta)*np.log(term)

def _gumbel_logpdf(theta, u, v):
    tu = (-np.log(u))**theta
    tv = (-np.log(v))**theta
    inner = (tu + tv)**(1.0/theta)
    return (
        np.log(theta)
        + (theta - 1)*(np.log(-np.log(u)) + np.log(-np.log(v)))
        - (2*theta - 1)*np.log(inner)
        - inner
    )

def neg_log_lik_clayton(theta, u, v):
    if theta <= 0: 
        return np.inf
    return -np.sum(_clayton_logpdf(theta, u, v))

def neg_log_lik_gumbel(theta, u, v):
    if theta < 1:
        return np.inf
    return -np.sum(_gumbel_logpdf(theta, u, v))

# ---------------------------------------------------------------- # standard metrics
PERIODS_PER_YEAR = {
    '1m': 525_600, '5m': 105_120, '15m': 35_040, '30m': 17_520,
    '1h':   8_760, '2h':   4_380, '4h':    2_190,
    '1d':     365, '1w':      52
    }

def _periods(freq: str) -> int:
    return PERIODS_PER_YEAR.get(freq, 365)

def annual_factor(freq: str, root: bool = False) -> float:
    periods = {
        '1m': 525_600, '5m': 105_120, '15m': 35_040, '30m': 17_520,
        '1h':   8_760, '2h':   4_380, '4h':    2_190,
        '1d':     365, '1w':      52
    }
    factor = periods.get(freq, 365)
    return np.sqrt(factor) if root else factor

def max_drawdown(equity: pd.Series) -> float:
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max
    return abs(drawdown.min())

def CAGR(dr: pd.Series, periods_per_year: float) -> float:
     """Standard formula for geometric annualized mean"""
     dr = dr.dropna()
     if dr.empty:
         return np.nan
     end_val = (1.0 + dr).prod()
     years   = len(dr) / periods_per_year
     return (end_val ** (1.0 / years) - 1.0) if years > 0 and end_val > 0 else np.nan

def sharpe(return_series: list, freq: str = "1d", rf: float = 0.) -> float:
    arr = np.asarray(return_series) - rf
    mean_ret = np.mean(arr)
    std_ret  = np.std(arr, ddof=1)
    ann      = annual_factor(freq, root=True)
    return (mean_ret / std_ret) * ann if std_ret else np.nan

def sortino(returns, freq: str = "1d", target: float = 0.0) -> float:
    """similar to sharpe ratio but only penalize downside volatility """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    excess  = returns - target
    periods = annual_factor(freq, root=False)
    downside = np.minimum(excess, 0)
    if not downside.any():
        return np.nan
    rms_down    = np.sqrt(np.mean(downside**2))
    downside_ann = rms_down * np.sqrt(periods)
    mean_exc    = excess.mean() * periods
    return mean_exc / downside_ann if downside_ann else np.nan

def calmar(returns, equity_curve, freq: str = "1d") -> float:
    periods   = annual_factor(freq, root=False)
    cum_ret   = equity_curve.iloc[-1] - 1
    annual_ret = (1 + cum_ret)**(periods/len(returns)) - 1
    return annual_ret / max_drawdown(equity_curve)

def sortino_adjusted(returns, freq: str = "1d", target: float = 0.0) -> float:
    """more realistic sortino when working with aggregated data and many 0% return days"""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    p      = annual_factor(freq, root=False)
    excess = returns - target
    neg    = excess[excess < 0]
    if neg.size == 0:
        return np.nan
    
    rms_down = np.sqrt((neg**2).mean())
    ann_down = rms_down * np.sqrt(p)
    ann_exc  = excess.mean() * p
    return ann_exc / ann_down if ann_down else np.nan

def global_CVaR(returns, alpha: float) -> float:
    losses = -returns
    var_lvl = np.percentile(losses, 100 * alpha)
    tail_loss = losses[losses >= var_lvl]
    return tail_loss.mean() if tail_loss.size >  0 else np.nan

def rolling_CVaR(returns, alpha: float, window: int = 365) -> float:
    def _cvar(x):
        losses = -x
        var_level = np.percentile(losses, 100 * alpha)
        tail_losses = losses[losses >= var_level]
        return tail_losses.mean() if tail_losses.size > 0 else np.nan
    
    return returns.rolling(window).apply(_cvar, raw=False)

class EmpiricalCVaR:
    def __init__(self, returns: pd.Series, n_sims: int = 10000, random_seed: int = 42):
        self.returns = returns.dropna().values
        self.n_sims = n_sims
        self.random_seed = random_seed
        self.simulated_returns = []

    def run(self):
        np.random.seed(self.random_seed)
        n = len(self.returns)

        self.simulated_returns = [
            np.random.choice(self.returns, size=n, replace=True)
            for _ in range(self.n_sims)
        ]
        return self.simulated_returns

    def to_cvar_list(self, alpha=0.05, cvar_func=None):
        if not self.simulated_returns:
            self.run()
        if cvar_func is None:
            cvar_func = global_CVaR  

        return [cvar_func(pd.Series(sim), alpha=alpha) for sim in self.simulated_returns]

# ---------------------------------------------------------------- #


