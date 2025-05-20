import pandas as pd
import numpy as np
import vectorbt as vbt
import holoviews as hv
from typing import Dict, Any, Optional, List, Callable, Union, Sequence
from tqdm import tqdm

from .plots import _lsa
from .stats import Stats
from .analyzer import Analyzer

MetricKey = str
MetricFunc = Callable[["Stats", str, "vbt.Portfolio"], float]

class LocalSensitivityAnalyzer:
    """
    Local Sensitivity Analyzer using discrete neighbor evaluation.
    Tests only the immediate left and right neighbors in the defined param_space for each parameter
    """
    _METRIC_MAPS: Dict[MetricKey, MetricFunc] = {
        "sharpe_ratio":  lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[0],
        "sortino_ratio": lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[1],
        "calmar_ratio":  lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[2],
        "total_return":  lambda s, tf, pf: s._returns(pf)[0],
        "max_drawdown":  lambda s, tf, pf: s._risk_metrics(tf, pf)[0],
        "volatility":    lambda s, tf, pf: s._risk_metrics(tf, pf)[2],
        "profit_factor": lambda s, tf, pf: pf.stats().get("Profit Factor", np.nan),
    }

    def __init__(
        self,
        analyzer: "Analyzer",
        target_metrics: Union[MetricKey, Sequence[MetricKey]] = "sharpe_ratio",
        seed: Optional[int] = 123,
    ):
        if isinstance(target_metrics, str):
            self.metrics: List[MetricKey] = [target_metrics]
        else:
            self.metrics = list(target_metrics)
        unknown = [m for m in self.metrics if m not in self._METRIC_MAPS]
        if unknown:
            raise ValueError(f"unknown metric: {', '.join(unknown)}")

        self.tpl = analyzer
        self.stats = analyzer.s
        self.tf = analyzer.timeframe
        self.seed = seed

        self._metric_funcs: Dict[MetricKey, Callable] = {
            m: (lambda pf, m=m: self._METRIC_MAPS[m](self.stats, self.tf, pf))
            for m in self.metrics
        }

    def _get_neighbors(self, param_name: str, current_value: Any) -> List[Any]:
        space = self.tpl.strategy.param_space[param_name]
        if getattr(space, 'name', None) == 'choice':
            values = list(space.pos_args[1])
        elif getattr(space, 'name', None) == 'quniform':
            _, low, high, q = space.pos_args
            values = list(np.arange(low, high + 1e-9, q))
        else:
            values = list(space)
        try:
            idx = values.index(current_value)
        except ValueError:
            idx = next((i for i, v in enumerate(values) if np.isclose(v, current_value)), None)
        if idx is None:
            return []
        neighbors = []
        if idx > 0:
            neighbors.append(values[idx - 1])
        if idx < len(values) - 1:
            neighbors.append(values[idx + 1])
        return neighbors

    def finite_diff(self, base_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate metrics at immediate discrete neighbors for each numeric parameter.
        Returns DataFrame with metrics at lower and upper neighbors. Shows progress bar.
        """
        f0 = self._objective(base_params)
        if not all(np.isfinite(list(f0.values()))):
            raise RuntimeError("Baseline metric not finite")

        rows = []
        # iterate parameters with progress bar
        for name, val in tqdm(base_params.items(), desc="Params", unit="param"):
            if not isinstance(val, (int, float, np.floating)):
                continue
            nbrs = self._get_neighbors(name, val)
            if not nbrs:
                continue
            row = {'parameter': name, 'baseline': val}
            # evaluate neighbors with progress
            for nbr in tqdm(nbrs, desc=f"Neighbors of {name}", leave=False, unit="nbr"):
                params = dict(base_params, **{name: nbr})
                f_nbr = self._objective(params)
                for m in self.metrics:
                    col = f"metric_{m}_{nbr}" if len(nbrs) > 1 else f"metric_{m}"
                    row[col] = f_nbr[m]
            rows.append(row)

        if not rows:
            raise ValueError("no numeric params with neighbors found")

        return pd.DataFrame(rows).set_index('parameter')

    def _objective(self, params: Dict[str, Any]) -> Dict[MetricKey, float]:
        state = None
        if self.seed is not None:
            state = np.random.get_state()
            np.random.seed(self.seed)
        A = self.tpl.__class__
        a = A(
            strategy=self.tpl.strategy,
            params=params,
            full_data=self.tpl.full_data,
            timeframe=self.tf,
            test_size=0.0,
            init_cash=self.tpl.init_cash,
            fees=self.tpl.fees,
            slippage=self.tpl.slippage,
            tp_stop=params.get('tp_pct'),
            sl_stop=params.get('sl_pct'),
        )
        vals = {m: func(a.pf) for m, func in self._metric_funcs.items()}
        if state is not None:
            np.random.set_state(state)
        return vals

    def plot_finite_differences(self, matrix: Optional[pd.DataFrame] = None, title: Optional[str] = None,) -> hv.HeatMap:
        if matrix is None:
            matrix = self.finite_differences(self.tpl.strategy.param_space)
        title = title or "Discrete Neighbor Metrics"
        return _lsa(matrix, title).heatmap()
    
#