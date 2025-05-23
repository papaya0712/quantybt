
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
    
    def _risk_adjusted_metrics(self, timeframe: str, pf: vbt.Portfolio) -> Tuple[float, float, float, float, float, float]:
     strat_returns = pf.returns().values  
     bench_returns = pf.benchmark_returns().values  
     periods = self._annual_factor(timeframe, root=False)  
     rf = 0.0

     def sharpe(returns):
        mean_ret = np.mean(returns - rf)
        std_ret = np.std(returns, ddof=1)
        return (mean_ret / std_ret) * np.sqrt(periods) if std_ret else np.nan

     def sortino(returns):
        target = 0.0
        excess = returns - target
        downside = np.minimum(excess, 0)
        rms_down = np.sqrt(np.mean(np.square(downside)))
        downside_ann = rms_down * np.sqrt(periods)
        mean_exc = excess.mean()
        return (mean_exc * periods) / downside_ann if downside_ann else np.nan

     def calmar(returns, max_dd):
        cum_ret = np.prod(1 + returns) - 1
        annual_ret = (1 + cum_ret) ** (periods / len(returns)) - 1
        return annual_ret / max_dd if max_dd else np.nan
     
     max_dd_strat = abs(pf.max_drawdown())
     max_dd_bench = abs(pf.benchmark_max_drawdown()) if hasattr(pf, "benchmark_max_drawdown") else np.nan
     return (
        sharpe(strat_returns),
        sortino(strat_returns),
        calmar(strat_returns, max_dd_strat),
        sharpe(bench_returns),
        sortino(bench_returns),
        calmar(bench_returns, max_dd_bench)
     )
    
    def _correlation_to_benchmark(self, pf: vbt.Portfolio) -> float:
     strat_returns = pf.returns().values
     bench_returns = pf.benchmark_returns().values
     if len(strat_returns) != len(bench_returns):
        return np.nan  

     return np.corrcoef(strat_returns, bench_returns)[0, 1]

    def _alpha_beta(self, pf: vbt.Portfolio, timeframe: str, rf: float = 0.0) -> Tuple[float, float]:
     strat_returns = pf.returns().dropna().values
     bench_returns = pf.benchmark_returns().dropna().values

     if len(strat_returns) != len(bench_returns):
        return np.nan, np.nan

     periods = self._annual_factor(timeframe, root=False)

   
     excess_strat = strat_returns - rf
     excess_bench = bench_returns - rf
     cov = np.cov(excess_strat, excess_bench, ddof=0)[0, 1]
     var = np.var(excess_bench, ddof=0)
     beta = cov / var if var != 0 else np.nan

     mean_strat_ann = (1 + np.mean(strat_returns)) ** periods - 1
     mean_bench_ann = (1 + np.mean(bench_returns)) ** periods - 1

     alpha = mean_strat_ann - (rf + beta * (mean_bench_ann - rf)) if not np.isnan(beta) else np.nan

     return alpha, beta  
    
    def _risk_of_ruin(self, win_rate: float, avg_win: float, avg_loss: float, risk_per_trade: float = 0.01, ruin_threshold: float = 1.0) -> float:
     if avg_loss == 0:
         return np.nan 
     R = avg_win / abs(avg_loss)
     edge = win_rate * R - (1 - win_rate)
     if edge <= 0:
        return 1.0  # 100%r
     base = (1 - edge) / (1 + edge)
     n_trades = ruin_threshold / risk_per_trade
     return base ** n_trades
    
    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
    
     if avg_loss == 0 or avg_win is None or avg_loss is None or win_rate is None:
        return np.nan

     R = avg_win / abs(avg_loss)
     kelly = win_rate - (1 - win_rate) / R
     return max(0.0, kelly)
    
    def backtest_summary(self, pf: vbt.Portfolio, timeframe: str) -> pd.DataFrame:
        

        perf_strat, perf_bench          = self._returns(pf)       
        periods_per_year                = self._annual_factor(timeframe, root=False)

        cagr_strat = self._cagr_from_returns(pf.returns(),              periods_per_year) * 100
        cagr_bench = self._cagr_from_returns(pf.benchmark_returns(),    periods_per_year) * 100 \
                     if hasattr(pf, "benchmark_returns") else np.nan

        dd_strat, dd_bench, vola_strat, vola_bench = self._risk_metrics(timeframe, pf)
        sharpe, sortino, calmar, sharpe_b, sortino_b, calmar_b = self._risk_adjusted_metrics(timeframe, pf)
        corr = self._correlation_to_benchmark(pf)
        
        stats = pf.stats()
        g = stats.get 

        alpha, beta = self._alpha_beta(pf, timeframe=timeframe)
        win_rate  = g("Win Rate [%]", np.nan) / 100
        avg_win   = g("Avg Winning Trade [%]", np.nan) / 100
        avg_loss  = g("Avg Losing Trade [%]", np.nan) / 100

        kelly = self.kelly_fraction(win_rate, avg_win, avg_loss)

        ror_100   = self._risk_of_ruin(win_rate, avg_win, avg_loss, risk_per_trade=kelly, ruin_threshold=1.0)
        ror_50    = self._risk_of_ruin(win_rate, avg_win, avg_loss, risk_per_trade=kelly, ruin_threshold=0.5)


        data = {
            "CAGR [%]":                             (cagr_strat,            cagr_bench),
            "Total Return [%]":                     (perf_strat,            perf_bench),
            "Max Drawdown [%]":                     (dd_strat,              dd_bench),
            "Annualized Volatility [%]":            (vola_strat,            vola_bench),
            "Sharpe Ratio":                         (sharpe,                sharpe_b),
            "Sortino Ratio":                        (sortino,               sortino_b),
            "Calmar Ratio":                         (calmar,                calmar_b),
            "Profit Factor":                        (g("Profit Factor"),         ""),
            "Correlation to Benchmark":             (corr,                       ""),
            "Alpha    ": (round(alpha, 4), ""),
            "Beta":      (round(beta, 4),  ""),
            "Kelly [%]":                            (round(kelly * 100, 2),      ""),
            "Risk of Ruin 100% , risk=kelly":       (ror_100,                    ""),
            "Risk of Ruin 50%  , risk=kelly":       (ror_50,                     ""),
            "--------------------------------":     ("",                         ""),
            "Total Trades":                         (g("Total Trades", 0),       ""),
            "Win Rate [%]":                         (g("Win Rate [%]"),          ""),
            "Best Trade [%]":                       (g("Best Trade [%]"),        ""),
            "Worst Trade [%]":                      (g("Worst Trade [%]"),       ""),
            "Avg Winning Trade [%]":                (g("Avg Winning Trade [%]"), ""),
            "Avg Losing Trade [%]":                 (g("Avg Losing Trade [%]"),  ""),
            "Avg Winning Trade Duration":           (g("Avg Winning Trade Duration"), ""),
            "Avg Losing Trade Duration":            (g("Avg Losing Trade Duration"),  ""),
            
            }


        summary_df = pd.DataFrame.from_dict(data, orient="index", columns=["Strategy", "Benchmark"])
        for col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: round(x, 2) if isinstance(x, (int, float, np.floating)) else x
            )


        return summary_df

#