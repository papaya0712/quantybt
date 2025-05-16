# some nice and interesting background theory about stuff in this framework

This document outlines the theoretical foundations used in the QuantyBT framework, including metric definitions statistical methods and algorithms

---

# Basics
### Sharpe Ratio

We define:

$$
\text{Sharpe} = \frac{\mathbb{E}[r]}{\sigma[r]} \cdot \sqrt{T}
$$

Where:

- $$\( \mathbb{E}[r] \)§§: expected return  
- §§\( \sigma[r] \)§§: standard deviation  
- §§\( T \)§§: annualization factor (im using 365days as base for crypot currencies)

---















### Monte Carlo Simulation and P-Value Estimation

P-values for simulation significance testing are computed as:

$$
p = \frac{\text{rank(original)} + 1}{\text{num simulations} + 1}
$$

A lower `p` indicates that the original result is unlikely under random sampling.
