import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from tqdm import tqdm
from .plots import _PlotBootstrapping
from .analyzer import Analyzer

try:
    from numba import njit
    print(">>> Successfully imported numba.")

    @njit(cache=True, fastmath=True)
    def _cumprod_numba(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            acc = np.float32(1.0)
            for j in range(a.shape[1]):
                acc *= a[i, j]
                out[i, j] = acc
        return out

except Exception as e:
    print(">>> Numba not found. Error:", e)

    def _cumprod_numba(a: np.ndarray) -> np.ndarray:
        return np.cumprod(a, axis=1)

class Bootstrapping:
    """
    Recommended: Use at least 2,000-5,000 simulations for robust statistical results. 
    For extreme risk estimation, such as 1%-VaR or severe tail events, consider 10,000+ simulations.
    
    Warning: Standard bootstrapping methods can destroy the autocorrelation structure in your return series.

    Incoming: stationary Bootstrapping
    """
    _PERIODS = {
        '1m': 525_600, 
        '5m': 105_120, 
        '15m': 35_040, 
        '30m': 17_520,
        '1h': 8_760, 
        '2h': 4_380, 
        '4h': 2_190, 
        '1d': 365, 
        '1w': 52
        
    }

    def __init__(self, analyzer=None, *, timeframe='1d', ret_series=None, n_sims=1000, random_seed=69, batch_size=500):
        if analyzer is not None:
            self.pf = analyzer.pf
            self.init_cash = analyzer.init_cash
            self.timeframe = analyzer.timeframe
            self.ret_series = analyzer.pf.returns()
        else:
            if ret_series is None:
                raise ValueError("Provide a return series if no analyzer is given")
            self.pf = None
            self.init_cash = 1.0
            self.timeframe = timeframe
            self.ret_series = ret_series.copy()

        if self.timeframe not in self._PERIODS:
            raise ValueError(f"Unsupported timeframe '{self.timeframe}'.")

        self.n_sims = n_sims
        self.random_seed = random_seed
        self.ann_factor = self._PERIODS[self.timeframe]
        self.batch_size = batch_size

    def _frequency(self, ret: pd.Series) -> pd.Series:
        rs = ret.copy()
        rs.index = pd.to_datetime(rs.index)
        if self.timeframe.endswith(('m', 'h')) or self.timeframe == '1d':
            return rs
        if self.timeframe == '1w':
            return rs.resample('W').apply(lambda x: (1 + x).prod() - 1)
        return rs.resample('M').apply(lambda x: (1 + x).prod() - 1)

    def _analyze_simulations(self, samples: np.ndarray):
        ann_factor = self.ann_factor
        init_cash = self.init_cash
        cumprod = _cumprod_numba(1.0 + samples) * init_cash
        cum_ret = (1.0 + samples).prod(axis=1) - 1.0

        mean_ret = samples.mean(axis=1)
        std = samples.std(axis=1, ddof=1)
        sharpe = np.where(std > 0, mean_ret / std * np.sqrt(ann_factor), np.nan)

        excess = samples
        mean_exc = np.mean(excess, axis=1)
        rms_down = np.sqrt(np.mean(np.square(np.minimum(excess, 0)), axis=1))
        mean_exc_ann = mean_exc * ann_factor
        downside_ann = rms_down * np.sqrt(ann_factor)
        sortino = np.where(downside_ann > 0, mean_exc_ann / downside_ann, np.nan)

        rolling_max = np.maximum.accumulate(cumprod, axis=1)
        rolling_max = np.where(rolling_max == 0, 1e-9, rolling_max)
        max_dd = ((cumprod - rolling_max) / rolling_max).min(axis=1)

        years = np.clip(samples.shape[1] / ann_factor, 1e-6, None)
        cagr = np.where(cumprod[:, -1] > 0, (cumprod[:, -1] / init_cash) ** (1 / years) - 1, np.nan)
        calmar = np.where(max_dd < 0, cagr / abs(max_dd), np.nan)

        out = []
        for i in range(samples.shape[0]):
            out.append({
                'CumulativeReturn': cum_ret[i],
                'Sharpe': sharpe[i],
                'Sortino': sortino[i],
                'Calmar': calmar[i],
                'MaxDrawdown': max_dd[i]
            })
        return out

    def _analyze_series(self, ret: pd.Series):
        if len(ret) == 0:
            return dict.fromkeys(['CumulativeReturn', 'Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown'], np.nan)
        arr = np.asarray(ret, dtype=np.float64)[np.newaxis, :]
        return self._analyze_simulations(arr)[0]

    def mc_with_replacement(self):
        np.random.seed(self.random_seed)
        returns = self._frequency(self.ret_series)
        arr = returns.values.astype(np.float32)
        n_obs = arr.size

        stats = []
        equity_list = []
        for i in range(0, self.n_sims, self.batch_size):
            end = min(i + self.batch_size, self.n_sims)
            bsize = end - i
            idx = np.random.randint(0, n_obs, size=(bsize, n_obs))
            samples = arr[idx]
            equity = _cumprod_numba(1.0 + samples) * self.init_cash

            equity_list.append(equity)
            stats.extend(self._analyze_simulations(samples))

        all_equity = np.vstack(equity_list).T
        sim_equity = pd.DataFrame(
            all_equity,
            index=returns.index,
            columns=[f"Sim_{i}" for i in range(self.n_sims)]
        )

        orig_stats = self._analyze_simulations(arr[np.newaxis, :])[0]
        return {
            'original_stats': orig_stats,
            'simulated_stats': stats,
            'simulated_equity_curves': sim_equity
        }

    def benchmark_equity(self):
        if self.pf is not None and hasattr(self.pf, 'benchmark_value'):
            bench = self.pf.benchmark_value()
        else:
            orig_ret = self._frequency(self.ret_series)
            bench = (1 + orig_ret).cumprod() * self.init_cash
        bench.index = pd.to_datetime(bench.index)
        return bench

    def run(self):
        res = self.mc_with_replacement()
        df = pd.DataFrame(res['simulated_stats'])
        df.loc['Original'] = res['original_stats']
        df_sim = df.drop(index='Original')

        summary = df_sim.describe().drop(index=['count', 'mean'])
        #print("=== Monte Carlo Simulation Summary ===")
        #print(summary)

        if self.pf is not None and hasattr(self.pf, 'benchmark_returns'):
            bench_ret = self._frequency(self.pf.benchmark_returns())
            bench_stats = self._analyze_series(bench_ret)

            print("\n=== Empirical P-Value Tests (Simulated vs Benchmark) ===")
            for metric in ['CumulativeReturn', 'Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown']:
                sim_values = df_sim[metric].dropna().values
                bench_value = bench_stats[metric]

                rank = np.sum(sim_values <= bench_value)
                p_left = (rank + 1) / (len(sim_values) + 1)
                p_right = 1 - p_left
                p_val = 2 * min(p_left, p_right)

                print(f"{metric:>18}: p-value = {p_val:.5f} | benchmark = {bench_value:.4f} | sim_mean = {sim_values.mean():.4f}")

        return df

    def plot_histograms(self, mc_results: pd.DataFrame = None):
        if mc_results is None:
            mc_data = self.mc_with_replacement()
            mc_results = pd.DataFrame(mc_data['simulated_stats'])
        return _PlotBootstrapping(self).plot_histograms(mc_results)

####

class Permutation:
    """
    Permutation test for detecting data mining bias. Recommended at least 200 runs (Comp. Cost are much higher n_sims = n_backtests).
    Outputs a p-value

    Note:
     - Permutation of orginal asset price series will destroy autocorrelation and other timeseries features
     - the higher the p-value -> the more your strategy sucks. should be < 0.10
    """
    def __init__(
        self,
        analyzer: Analyzer,
        n_sims: int = 200,
        seed: Optional[int] = None):

        self.analyzer = analyzer
        self.full_data = analyzer.full_data.copy()
        self.strategy = analyzer.strategy
        self.params = analyzer.params
        self.timeframe = analyzer.timeframe
        self.init_cash = analyzer.init_cash
        self.fees = analyzer.fees
        self.slippage = analyzer.slippage
        self.trade_side = analyzer.trade_side
        self.price_col = analyzer.s.price_col
        self.sl_stop = analyzer.sl_stop
        self.tp_stop = analyzer.tp_stop
        self.size = analyzer.size

        self.n_sims = n_sims
        self.seed = seed

        self.synthetic_paths: List[pd.DataFrame] = []
        self.synthetic_metrics: List[float] = []
        self.original_metric: Optional[float] = None
        self.p_value: Optional[float] = None

    def _get_synthetic_price_paths(self) -> None:
        ohlc = self.full_data[['open', 'high', 'low', 'close']]
        n_bars = len(ohlc)
        time_index = ohlc.index
        log_bars = np.log(ohlc[['open', 'high', 'low', 'close']]).to_numpy()

        base_seed = self.seed if self.seed is not None else 0

        for i in tqdm(range(self.n_sims), desc="Generating permutations"):
            np.random.seed(base_seed + i)
            start_bar = log_bars[0].copy()

            r_o = (log_bars[:, 0] - np.concatenate(([np.nan], log_bars[:-1, 3])))[1:]
            r_h = (log_bars[:, 1] - log_bars[:, 0])[1:]
            r_l = (log_bars[:, 2] - log_bars[:, 0])[1:]
            r_c = (log_bars[:, 3] - log_bars[:, 0])[1:]

            perm_n = n_bars - 1
            idx = np.arange(perm_n)
            perm_indices_hlc = np.random.permutation(idx)
            perm_indices_o = np.random.permutation(idx)

            perm_high = r_h[perm_indices_hlc]
            perm_low = r_l[perm_indices_hlc]
            perm_close = r_c[perm_indices_hlc]
            perm_open = r_o[perm_indices_o]

            perm_bars_log = np.zeros_like(log_bars)
            perm_bars_log[0] = start_bar

            for t in range(1, n_bars):
                k = t - 1
                perm_bars_log[t, 0] = perm_bars_log[t-1, 3] + perm_open[k]
                perm_bars_log[t, 1] = perm_bars_log[t, 0] + perm_high[k]
                perm_bars_log[t, 2] = perm_bars_log[t, 0] + perm_low[k]
                perm_bars_log[t, 3] = perm_bars_log[t, 0] + perm_close[k]

            perm_bars = np.exp(perm_bars_log)
            perm_df = pd.DataFrame(
                perm_bars,
                index=time_index,
                columns=['open', 'high', 'low', 'close']
            )
            self.synthetic_paths.append(perm_df)

    def _compute_profit_factor(self, analyzer: Analyzer) -> float:
        pf = analyzer.pf
        stats = pf.stats()
        return stats.get("Profit Factor", np.nan)

    def run(self) -> pd.DataFrame:
        
        self.original_metric = self._compute_profit_factor(self.analyzer)
        self.synthetic_paths.clear()
        self._get_synthetic_price_paths()
        perm_better_count = 1
        self.synthetic_metrics.clear()

        for perm_path in tqdm(self.synthetic_paths, desc="Backtesting Permutations"):
            perm_full = self.full_data.copy()
            perm_full[['open', 'high', 'low', 'close']] = perm_path[['open', 'high', 'low', 'close']]

            analyzer_perm = Analyzer(
                strategy=self.strategy,
                params=self.params,
                full_data=perm_full.reset_index().rename(columns={'index': 'timestamp'}),
                timeframe=self.timeframe,
                price_col=self.price_col,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                trade_side=self.trade_side,
                sl_stop=self.sl_stop,
                tp_stop=self.tp_stop,
                size=self.size
            )

            perm_pf = self._compute_profit_factor(analyzer_perm)
            if perm_pf >= self.original_metric:
                perm_better_count += 1

            self.synthetic_metrics.append(perm_pf)

        
        self.p_value = perm_better_count / self.n_sims
        print(f"MCPT P-Value: {self.p_value:.4f}")

        results = pd.DataFrame({
            'ProfitFactor': self.synthetic_metrics
        })
        results.loc['Original', 'ProfitFactor'] = self.original_metric
        results.loc['p-value', 'ProfitFactor'] = self.p_value

        return results

    def plot(self, df_results: Optional[pd.DataFrame] = None) -> None:
        
        if self.original_metric is None or self.p_value is None:
            raise ValueError("use run() first")

        if df_results is None:
            df_results = self.run()

        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))

        synth_vals = df_results.loc[df_results.index != 'Original', 'ProfitFactor'].dropna().astype(float)
        orig_val = df_results.loc['Original', 'ProfitFactor']

        plt.hist(synth_vals, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(orig_val, color='red', linestyle='--', linewidth=2, label=f'Original PF: {orig_val:.4f}')
        plt.title(f'Permutation test: Profit Factor (p-Value = {self.p_value:.4f})')
        plt.xlabel("Profit Factor")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

####