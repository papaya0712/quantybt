import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.optimize import minimize

"""
References:

- Copulas: https://en.wikipedia.org/wiki/Copula_(statistics)
- 
-
"""


# ---------------------------------------------------------------- #

def clayton_logpdf(theta, u, v):
    term = (u**(-theta) + v**(-theta) - 1.0)
    return np.log(theta + 1) - (theta + 1)*(np.log(u) + np.log(v)) - (2 + 1/theta)*np.log(term)

def gumbel_logpdf(theta, u, v):
    tu = (-np.log(u))**theta
    tv = (-np.log(v))**theta
    c_val = np.exp(- (tu + tv)**(1.0/theta))
    inner = (tu + tv)**(1.0/theta)
    return (
        np.log(theta)
        + (theta - 1)*(np.log(-np.log(u)) + np.log(-np.log(v)))
        - (2*theta - 1)*np.log(inner)
        - inner
    )

def neg_log_lik_clayton(theta, u, v):
    if theta <= 0: return np.inf
    return -np.sum(clayton_logpdf(theta, u, v))

def neg_log_lik_gumbel(theta, u, v):
    if theta < 1:
        return np.inf
    return -np.sum(gumbel_logpdf(theta, u, v))

# ---------------------------------------------------------------- # 

def _annual_factor(self, timeframe: str, root: bool = True) -> float:
     periods = {
        '1m': 525600, '5m': 105120, '15m': 35040, '30m': 17520,
        '1h': 8760, '2h': 4380, '4h': 2190, '1d': 365, '1w': 52}
     factor = periods.get(timeframe, 365)
     return np.sqrt(factor) if root else factor

def max_drawdown():
    return

def sharpe(return_series: List = "", freq: str = "1d", rf: Optional[float] = 0):
    returns = return_series.values
    periods = _annual_factor(freq, root=False) 
    mean_ret = np.mean(returns - rf)
    std_ret = np.std(returns, ddof=1)
    return (mean_ret / std_ret) * np.sqrt(periods) if std_ret else np.nan

# value at risk
def VaR():
    return

# conditional VaR
def CVaR():
    return

# entropic VaR
def EVaR():
    return

# ---------------------------------------------------------------- #

