
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Tuple

class Stats:
    def __init__(self, price_col: str = "close"):
        self.price_col = price_col
    
    def _annual_factor(self, timeframe: str, root: bool = True) -> float:
     periods = {
        '1m': 525600, '5m': 105120, '15m': 35040, '30m': 17520,
        '1h': 8760, '2h': 4380, '4h': 2190, '1d': 365, '1w': 52}
     factor = periods.get(timeframe, 365)
     return np.sqrt(factor) if root else factor
    
    def _returns(self, pf: vbt.Portfolio) -> Tuple[float, float]:
        performance = pf.total_return() * 100
        benchmark = pf.total_benchmark_return() * 100
        
        return performance, benchmark
    
    @staticmethod
    def _cagr_from_returns(dr: pd.Series, periods_per_year: float) -> float:
        dr = dr.dropna()
        if dr.empty:
            return np.nan
        end_val = (1.0 + dr).prod()
        years   = len(dr) / periods_per_year
        return (end_val ** (1.0 / years) - 1.0) if years > 0 and end_val > 0 else np.nan

    def _risk_metrics(self, timeframe: str, pf: vbt.Portfolio) -> Tuple[float, float, float, float]:
        equity = pf.value().values
        benchmark_equity = pf.benchmark_value().values

        # Max Drawdown
        rolling_max_strat = np.maximum.accumulate(equity)
        dd_strat = (equity - rolling_max_strat) / rolling_max_strat
        max_dd_strat = dd_strat.min() * 100

        rolling_max_bench = np.maximum.accumulate(benchmark_equity)
        dd_bench = (benchmark_equity - rolling_max_bench) / rolling_max_bench
        max_dd_bench = dd_bench.min() * 100

        # Volatility - annualized
        strat_returns = pf.returns().values
        bench_returns = pf.benchmark_returns().values

        af = self._annual_factor(timeframe, root=True)
        vola_strat = np.std(strat_returns) * af * 100
        vola_bench = np.std(bench_returns) * af * 100

        return max_dd_strat, max_dd_bench, vola_strat, vola_bench
    
    def _risk_adjusted_metrics(self, timeframe: str, pf: vbt.Portfolio) -> Tuple[float, float, float]:
     returns = pf.returns().values  
     periods = self._annual_factor(timeframe, root=False)  
     rf = 0.0

     # Sharpe Ratio
     mean_ret = np.mean(returns - rf)
     std_ret = np.std(returns, ddof=1)
     sharpe = (mean_ret / std_ret) * np.sqrt(periods) if std_ret else np.nan        

     # Sortino Ratio
     rets = returns
     target = 0.0
     excess = rets - target
     mean_exc = excess.mean()
     downside = np.minimum(excess, 0)
     rms_down = np.sqrt(np.mean(np.square(downside)))
     downside_ann = rms_down * np.sqrt(periods)
     sortino = (mean_exc * periods) / downside_ann if downside_ann else np.nan

     # Calmar Ratio
     cum_ret = np.prod(1 + returns) - 1
     annual_ret = (1 + cum_ret) ** (periods / len(returns)) - 1
     max_dd = abs(pf.max_drawdown())
     calmar = annual_ret / max_dd if max_dd else np.nan

     return sharpe, sortino, calmar
    
    def _correlation_to_benchmark(self, pf: vbt.Portfolio) -> float:
     strat_returns = pf.returns().values
     bench_returns = pf.benchmark_returns().values
     if len(strat_returns) != len(bench_returns):
        return np.nan  

     return np.corrcoef(strat_returns, bench_returns)[0, 1]

    def backtest_summary(self, pf: vbt.Portfolio, timeframe: str) -> pd.DataFrame:
     stats = pf.stats()   
     perf_strat, perf_bench = self._returns(pf)              # [%]
     periods_per_year       = self._annual_factor(timeframe, root=False)

     cagr_strat = self._cagr_from_returns(pf.returns(), periods_per_year) * 100
     try:
        cagr_bench = self._cagr_from_returns(pf.benchmark_returns(), periods_per_year) * 100
     except AttributeError:
        cagr_bench = np.nan

     dd_strat, dd_bench, vola_strat, vola_bench = self._risk_metrics(timeframe, pf)
     sharpe, sortino, calmar                    = self._risk_adjusted_metrics(timeframe, pf)
     corr                                        = self._correlation_to_benchmark(pf)

     g = stats.get
     summary = {
        # performance
        "CAGR [%]":                            round(cagr_strat, 2),
        "Benchmark CAGR [%]":                  round(cagr_bench, 2),
        "Strategy Performance [%]":            round(perf_strat, 2),
        "Benchmark Performance [%]":           round(perf_bench, 2),
        # risk
        "Strategy Max Drawdown [%]":           round(dd_strat, 2),
        "Benchmark Max Drawdown [%]":          round(dd_bench, 2),
        "Annualized Strategy Volatility [%]":  round(vola_strat, 2),
        "Annualized Benchmark Volatility [%]": round(vola_bench, 2),
        "Sharpe Ratio":                        round(sharpe, 2),
        "Sortino Ratio":                       round(sortino, 2),
        "Calmar Ratio":                        round(calmar, 2),
        # other
        "Profit Factor":                       round(g("Profit Factor", np.nan), 2),
        "Correlation to Benchmark":            round(corr, 2),
        "Total Trades":                        int(g("Total Trades", 0)),
        "Win Rate [%]":                        round(g("Win Rate [%]", np.nan), 2),
        "Best Trade [%]":                      round(g("Best Trade [%]", np.nan), 2),
        "Worst Trade [%]":                     round(g("Worst Trade [%]", np.nan), 2),
        "Avg Winning Trade [%]":               round(g("Avg Winning Trade [%]", np.nan), 2),
        "Avg Losing Trade [%]":                round(g("Avg Losing Trade [%]", np.nan), 2),
        "Avg Winning Trade Duration":          g("Avg Winning Trade Duration", np.nan),
        "Avg Losing Trade Duration":           g("Avg Losing Trade Duration", np.nan),
     }

     return pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])

#