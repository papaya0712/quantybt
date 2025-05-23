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


# ---------------------------------------------------------------- #

