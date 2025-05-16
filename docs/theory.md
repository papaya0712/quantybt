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
\text{Calmar} = \frac{\text{CAGR}}{\text{Max Drawdown}}
$$
