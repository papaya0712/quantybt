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
#### Why?
Using Monte Carlo methods gives you a better understanding of real risks in your trading system. In general we rely here on the **Weak Law of Large Numbers (WLLN)** and the **Central Limit Theorem (CLT)** from probability theory:

| **WLLN** | 
|:-------- | 
| $\displaystyle \lim_{N \to \infty} \Pr\left(|\bar X_N - \mathbb{E}[X]| > \varepsilon\right) = 0,\quad \text{where} \quad \bar X_N = \frac{1}{N} \sum_{i=1}^N X_i,\; \varepsilon > 0$ |

| **CLT** |
|:-------- |
| $\displaystyle \frac{\bar X_N - \mathbb{E}[X]}{\sigma / \sqrt{N}} \xrightarrow{d} \mathcal{N}(0,1),\quad \text{with} \quad \sigma^2 = \mathrm{Var}(X)$ |
