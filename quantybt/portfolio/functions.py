import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.optimize import minimize


# ---------------------------------------------------------------- # tail dependence correlation

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
    if theta <= 0: 
        return np.inf
    return -np.sum(clayton_logpdf(theta, u, v))

def neg_log_lik_gumbel(theta, u, v):
    if theta < 1:
        return np.inf
    return -np.sum(gumbel_logpdf(theta, u, v))

# ---------------------------------------------------------------- # metrics
PERIODS_PER_YEAR = {
    '1m': 525_600, '5m': 105_120, '15m': 35_040, '30m': 17_520,
    '1h':   8_760, '2h':   4_380, '4h':    2_190,
    '1d':     365, '1w':      52
    }

def _periods(freq: str) -> int:
    return PERIODS_PER_YEAR.get(freq, 365)

def _annual_factor(freq: str, root: bool = False) -> float:
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

def sharpe(return_series: list, freq: str = "1d", rf: float = 0.) -> float:
    arr = np.asarray(return_series) - rf
    mean_ret = np.mean(arr)
    std_ret  = np.std(arr, ddof=1)
    ann      = _annual_factor(freq, root=True)
    return (mean_ret / std_ret) * ann if std_ret else np.nan

def sortino(returns, freq: str = "1d", target: float = 0.0) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    excess  = returns - target
    periods = _annual_factor(freq, root=False)
    downside = np.minimum(excess, 0)
    if not downside.any():
        return np.nan
    rms_down    = np.sqrt(np.mean(downside**2))
    downside_ann = rms_down * np.sqrt(periods)
    mean_exc    = excess.mean() * periods
    return mean_exc / downside_ann if downside_ann else np.nan

def calmar(returns, equity_curve, freq: str = "1d") -> float:
    periods   = _annual_factor(freq, root=False)
    cum_ret   = equity_curve.iloc[-1] - 1
    annual_ret = (1 + cum_ret)**(periods/len(returns)) - 1
    return annual_ret / max_drawdown(equity_curve)

def sortino_adjusted(returns, freq: str = "1d", target: float = 0.0) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    p      = _annual_factor(freq, root=False)
    excess = returns - target
    neg    = excess[excess < 0]
    if neg.size == 0:
        return np.nan
    
    rms_down = np.sqrt((neg**2).mean())
    ann_down = rms_down * np.sqrt(p)
    ann_exc  = excess.mean() * p
    return ann_exc / ann_down if ann_down else np.nan


# ---------------------------------------------------------------- #

