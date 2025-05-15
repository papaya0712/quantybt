import pandas as pd
import numpy as np
from .plots import _PlotBootstrapping

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
        '1m': 525_600, '5m': 105_120, '15m': 35_040, '30m': 17_520,
        '1h': 8_760, '2h': 4_380, '4h': 2_190, '1d': 365, '1w': 52
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

    def _convert_frequency(self, ret: pd.Series) -> pd.Series:
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
        returns = self._convert_frequency(self.ret_series)
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
            orig_ret = self._convert_frequency(self.ret_series)
            bench = (1 + orig_ret).cumprod() * self.init_cash
        bench.index = pd.to_datetime(bench.index)
        return bench

    def results(self):
        res = self.mc_with_replacement()
        df = pd.DataFrame(res['simulated_stats'])
        df.loc['Original'] = res['original_stats']
        df_sim = df.drop(index='Original')

        summary = df_sim.describe().drop(index=['count', 'mean'])
        print("=== Monte Carlo Simulation Summary ===")
        print(summary)

        print("\n=== Empirical P-Value Tests (Simulated vs Original) ===")
        for metric in ['CumulativeReturn', 'Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown']:
            sim_values = df_sim[metric].dropna().values
            orig_value = df.loc['Original', metric]

            rank = np.sum(sim_values <= orig_value)
            p_left = (rank + 1) / (len(sim_values) + 1)
            p_right = 1 - p_left
            p_val = 2 * min(p_left, p_right)
            print(f"{metric:>18}: p-value = {p_val:.5f} | original = {orig_value:.4f} | sim_mean = {sim_values.mean():.4f}")

        if self.pf is not None and hasattr(self.pf, 'benchmark_returns'):
            bench_ret = self._convert_frequency(self.pf.benchmark_returns())
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
    
#