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

### P-Value

To quantify how clearly our strategy outperforms the benchmark, we apply a **randomisation/bootstrap test** that makes **no distributional assumptions** about the performance metric (e.g., Sharpe, Sortino, Calmar).

**Procedure**

Under the null hypothesis  
\( H_0 \) (*no outperformance*), we generate \( N \) synthetic performance values \( T_i \) via **block bootstrap** of the excess return series (alternatively: permutation or sign-flip can be used).

The **two-sided p-value** is computed as:

\[
p = \frac{2 \cdot \min \left( \#\{T_i \leq T_{\text{orig}}\},\; \#\{T_i \geq T_{\text{orig}}\} \right) + 1}{N + 1}
\]

For a one-sided test (pure outperformance or underperformance), the factor 2 is omitted.

**Interpretation**

The p-value \( p \) expresses the probability, under \( H_0 \), of observing a test statistic as extreme as \( T_{\text{orig}} \) purely by chance.

This method is robust to **non-normality**, **skewness**, and **heteroscedasticity**. However, if strong **serial correlation** is present in the data, a **block bootstrap** should be used to avoid variance distortion.

**Hypotheses**

- \( H_0 \): The strategy is statistically indistinguishable from random performance.  
- \( H_1 \): The observed performance deviates significantly from randomness (better or worse, depending on test direction).
