# some interesting background theory

This document outlines the theoretical foundations used the framework, including metric definitions/interpretations statistical methods and algorithms

---

# Basics
### Sharpe Ratio
$$
\text{Sharpe} = \frac{(\mathbb{E}[r] - r_f)}{\sigma} \cdot \sqrt{T}
$$

Where:

- $\mathbb{E}[r]$: expected return  
- $\sigma$: standard deviation  
- $T$: annualization factor (using 365 days as base for crypto)
- $r_f$: risk-free return rate which usually derived from treasury bills. for crypto its typically 0 

### Sortino Ratio

$$
\text{Sortino} = \frac{(\mathbb{E}[r] - r_f)}{\sigma_{\text{down}}} \cdot \sqrt{T}
$$

Where:

- $\sigma_{\text{down}}$: downside standard deviation (only measures negative return deviations)

Unlike the Sharpe Ratio, the Sortino Ratio **only penalizes downside volatility**, which is a more realistic risk measure, especially for strategies with asymmetric return distributions.

### Calmar Ratio

$$
\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}, \quad \text{where} \quad 
\text{CAGR} = \left(1 + \text{Cumulative Return} \right)^{T / N} - 1
$$

Where:
- $N$: total number of periods in the return series
- Cumulative Return = $\prod (1 + r_t) - 1$

---

# Montecarlo Simulation
### Why?
Using Monte Carlo methods gives you a better understanding of real risks in your trading system. In general we rely here on the **Weak Law of Large Numbers (WLLN)** and the **Central Limit Theorem (CLT)** from probability theory:

### Weak Law of Large Numbers (WLLN)
$$
\lim_{N \to \infty} \Pr\left(|\bar X_N - \mathbb{E}[X]| > \varepsilon\right) = 0,\quad \text{where} \quad \bar X_N = \frac{1}{N} \sum_{i=1}^N X_i,\; \varepsilon > 0
$$

### Central Limit Theorem (CLT)
$$
\frac{\bar X_N - \mathbb{E}[X]}{\sigma / \sqrt{N}} \xrightarrow{d} \mathcal{N}(0,1),\quad \text{with} \quad \sigma^2 = \mathrm{Var}(X)
$$

In our framework, we apply simple bootstrap resampling with replacement to generate thousands of synthetic equity curves. While this breaks temporal dependencies such as autocorrelation and volatility clustering, it provides a first-order approximation of the sampling distribution of key performance metrics (Sharpe, Sortino, Calmar, ...).

The statistical rationale rests on two things:
1. Weak Law of Large Numbers: With enough resamples, the bootstrapped estimates stabilise around their expected values
2. Asymptotic normality (heuristically linked to the CLT): The empirical distributions tend to become approximately normal allowing us to derive confidence intervals and p-values

Although the strict i.i.d. assumptions are violated, empirical evidence often shows sufficiently normal-shaped distributions. If strong serial dependence is suspected, a block or stationary bootstrap is preferable.

---

# Walkfoward Optimization and Generalization loss
### Why ?

Relying on a single train-test split where you optimize your hyperparameters on in-sample data (train fold) and evaluate on out-of-sample data (test fold) is better than nothing, but still prone to overfitting.

A more robust approach is **Walkforward Optimization**, where you use a rolling (or anchored) train/test window. This generates multiple smaller train-test splits across the entire dataset, providing a more reliable estimate of generalization performance and robustness.

One step further is using a **Genralization loss function** which acts as penalty for massive underperfromance on unseend data and prevents even more overfitting. There are many different ways to define your GL-function. One for example:

$$
\text{Loss} = -\overline{\text{ValMetric}} + \beta \cdot \frac{\max(\text{GL})}{\text{scale}}
$$

> **Note:** This formula penalizes even if \( \text{ValMetric}_{\text{OOS}} > \text{Benchmark} \), as long as \( \text{ValMetric}_{\text{OOS}} < \text{TrainMetric}_{\text{IS}} \).


This penalizes sharp degradation between in-sample and out-of-sample performance, especially when it’s unstable over recent evaluations.