# 📘 Quantitative Strategy Theory

This document outlines the theoretical foundations used in the QuantyBT framework, including metric definitions and statistical methods.

---

## 📐 Sharpe Ratio

We define:

$$
\text{Sharpe} = \frac{\mathbb{E}[r]}{\sigma[r]} \cdot \sqrt{T}
$$

Where:

- \( \mathbb{E}[r] \): expected return  
- \( \sigma[r] \): standard deviation  
- \( T \): annualization factor (e.g. 365 for daily crypto)

---

## 📉 Maximum Drawdown

Defined as the worst peak-to-trough decline in equity:

$$
\text{MaxDrawdown} = \min_t \left( \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s} \right)
$$

Where \( V_t \) is the equity curve value at time \( t \).

---

## 📊 Monte Carlo P-Value Estimation

P-values for simulation significance testing are computed as:

$$
p = \frac{\text{rank(original)} + 1}{\text{num simulations} + 1}
$$

A lower `p` indicates that the original result is unlikely under random sampling.
