import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union, Sequence
from hyperopt import space_eval, STATUS_OK, tpe, fmin, Trials
from .plots import _PlotWFOSummary
from .montecarlo import Bootstrapping
from .analyzer import Analyzer
from .stats import Stats
import logging

logger = logging.getLogger(__name__)  
from pandas.tseries.frequencies import to_offset
from pandas import DateOffset

@dataclass
class WFOSplitCfg:
    """
    Attributes:
        n_folds: Number of folds (only used in anchored_time mode).
        mode: 'rolling' or 'anchored'.
        train_period: e.d. '3M', '90D', DateOffset(...).
        test_period: e.d. '1M', '7D', DateOffset(...).
    """
    n_folds: int = 3
    mode: str = "rolling"
    train_period: Union[str, DateOffset] = "24M"
    test_period: Union[str, DateOffset]  = "12M"

def _parse_period(period: Union[str, DateOffset]) -> DateOffset:
    if isinstance(period, DateOffset):
        return period
    return to_offset(period)

class AdvancedOptimizer:
    """
    Advanced optimizer using walkforward optimization with anchored or rolling_time mode.
    - max_evals = tested parameter combinations
    - beta = penalty factor between robustness and performance
    """
    def __init__(
        self,
        analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio",
        beta: float = 0.3,
        split_cfg: Union[WFOSplitCfg, Sequence[WFOSplitCfg]] = WFOSplitCfg(),
    ):
        self.analyzer = analyzer
        self.strategy = analyzer.strategy
        self.timeframe = analyzer.timeframe
        self.max_evals = max_evals
        self.target_metric = target_metric
        self.beta = beta
        self.init_cash = analyzer.init_cash
        self.fees = analyzer.fees
        self.slippage = analyzer.slippage
        self.s = analyzer.s

        
        self.split_cfgs: List[WFOSplitCfg] = (
            [split_cfg] if isinstance(split_cfg, WFOSplitCfg) else list(split_cfg)
        )
        logging.debug(f"Configured Walk-Forward Split Configs: {self.split_cfgs}")
        
        self._splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = self._prepare_splits()
        if not self._splits:
            raise ValueError(
                f"Keine Walk-Forward-Splits erzeugt. Data range: "
                f"{analyzer.train_df.index[0]}–{analyzer.train_df.index[-1]}, "
                f"train_period: {self.split_cfgs[0].train_period}, "
                f"test_period: {self.split_cfgs[0].test_period}"
            )
        logging.debug(f"Generated total of {len(self._splits)} Walk-Forward splits")

        self.best_params: Optional[dict] = None
        self.trials: Optional[Trials] = None
        self.train_pf = None
        self.test_pf = None
        self.oos_pfs: List[vbt.Portfolio] = []
        self._history_diffs: List[float] = []
        self._history_gl_max: List[float] = []
        self.trial_metrics: List[Tuple[float, float]] = []

        self.metrics_map = {
            "sharpe_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[0],
            "sortino_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[1],
            "calmar_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[2],
            "total_return": lambda pf: self.s._returns(pf)[0],
            "max_drawdown": lambda pf: self.s._risk_metrics(self.timeframe, pf)[0],
            "volatility": lambda pf: self.s._risk_metrics(self.timeframe, pf)[2],
            "profit_factor": lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

    def _generate_splits(self, df: pd.DataFrame, cfg: WFOSplitCfg) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
   
     df = df.sort_index()
     if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

     train_off = _parse_period(cfg.train_period)
     test_off  = _parse_period(cfg.test_period)
     splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
     last_ts = df.index[-1]

     if cfg.mode == "rolling":
        window_start = df.index[0]
       
        while window_start + train_off <= last_ts:
            train_start = window_start
            train_end   = train_start + train_off
            test_start  = train_end
            raw_test_end = test_start + test_off
            test_end     = raw_test_end if raw_test_end <= last_ts else last_ts

            train_df = df.loc[train_start:train_end]
            test_df  = df.loc[test_start:test_end]
            splits.append((train_df, test_df))

            if test_end == last_ts:
                break

            window_start = window_start + test_off

     elif cfg.mode == "anchored":
        train_start = df.index[0]
        current_end = train_start + train_off
        for _ in range(cfg.n_folds):
            test_start = current_end
            raw_test_end = test_start + test_off
            test_end     = raw_test_end if raw_test_end <= last_ts else last_ts

            train_df = df.loc[train_start:current_end]
            test_df  = df.loc[test_start:test_end]
            splits.append((train_df, test_df))

            if test_end == last_ts:
                break

            current_end = current_end + test_off

     else:
        raise ValueError(f"Unknown WFO mode: {cfg.mode}")

     return splits

    def print_fold_periods(self):
        print("=== Walk-Forward Fold Periods ===")
        for i, (train_df, test_df) in enumerate(self._splits, 1):
            ts, te = train_df.index[0], train_df.index[-1]
            vs, ve = test_df.index[0], test_df.index[-1]
            print(f"Fold {i}: Train {ts} → {te}, Test {vs} → {ve}")

    def _prepare_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = self.analyzer.train_df.sort_index()
        all_splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for cfg in self.split_cfgs:
            all_splits.extend(self._generate_splits(df, cfg))
        return all_splits

    def _metric(self, pf: vbt.Portfolio) -> float:
        if self.target_metric in self.metrics_map:
            return self.metrics_map[self.target_metric](pf)
        try:
            return getattr(pf, self.target_metric)()
        except Exception:
            return pf.stats().get(self.target_metric, np.nan)

    @staticmethod
    def _choose_direction(sig: Dict[str, Any]) -> str:
        has_short = sig.get("short_entries") is not None and sig.get("short_exits") is not None
        has_long = sig.get("entries") is not None and sig.get("exits") is not None
        if has_long and has_short:
            return "both"
        if has_short:
            return "shortonly"
        return "longonly"

    def _objective(self, params: dict) -> dict:
     try:
        seed = int(abs(hash(frozenset(params.items()))) % 2**32)
        np.random.seed(seed)
        losses, is_metrics, val_metrics = [], [], []
        higher_is_better = self.target_metric not in ["max_drawdown", "volatility"]

        for train_df, val_df in self._splits:
            # In-Sample
            df_train = self.strategy.preprocess_data(train_df.copy(), params)
            sig_train = self.strategy.generate_signals(df_train, **params)

            sl_series_train = self.analyzer._expand_param(params.get("sl_stop"), df_train)
            tp_series_train = self.analyzer._expand_param(params.get("tp_stop"), df_train)
            size_series_train = self.analyzer._expand_param(params.get("size"), df_train)

            pf_train = vbt.Portfolio.from_signals(
                close=df_train[self.s.price_col],
                entries=sig_train.get("entries"),
                exits=sig_train.get("exits"),
                short_entries=sig_train.get("short_entries"),
                short_exits=sig_train.get("short_exits"),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction=self._choose_direction(sig_train),
                sl_stop=sl_series_train,
                tp_stop=tp_series_train,
                size=size_series_train
            )
            m_is = self._metric(pf_train)

            # Out-of-Sample
            df_val = self.strategy.preprocess_data(val_df.copy(), params)
            sig_val = self.strategy.generate_signals(df_val, **params)

            sl_series_val = self.analyzer._expand_param(params.get("sl_stop"), df_val)
            tp_series_val = self.analyzer._expand_param(params.get("tp_stop"), df_val)
            size_series_val = self.analyzer._expand_param(params.get("size"), df_val)

            pf_val = vbt.Portfolio.from_signals(
                close=df_val[self.s.price_col],
                entries=sig_val.get("entries"),
                exits=sig_val.get("exits"),
                short_entries=sig_val.get("short_entries"),
                short_exits=sig_val.get("short_exits"),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction=self._choose_direction(sig_val),
                sl_stop=sl_series_val,
                tp_stop=tp_series_val,
                size=size_series_val
            )
            m_val = self._metric(pf_val)

            # Generalization Loss
            if higher_is_better:
                if m_is <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    gl = max(0.0, min(1.0, 1.0 - (m_val / m_is)))
            else:
                if m_val <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    gl = max(0.0, min(1.0, 1.0 - (m_is / m_val)))

            losses.append((-m_val, gl))
            is_metrics.append(m_is)
            val_metrics.append(m_val)

        if not losses:
            return {"loss": np.inf, "status": STATUS_OK, "params": params}

        m_val_avg = -np.mean([l[0] for l in losses])
        gl_max = max(l[1] for l in losses)
        scale_raw = np.std(self._history_diffs[-10:]) if len(self._history_diffs) >= 10 else 1.0
        scale = np.clip(scale_raw if scale_raw > 0 else 1.0, 0.1, 10.0)
        loss = -m_val_avg + self.beta * (gl_max / scale)

        self._history_gl_max.append(gl_max)
        self._history_diffs.append(loss)
        self.trial_metrics.append((np.mean(is_metrics), np.mean(val_metrics)))

        return {"loss": loss, "status": STATUS_OK, "params": params}

     except Exception as e:
        logging.getLogger(__name__).error(f"Objective error: {e}", exc_info=True)
        return {"loss": np.inf, "status": STATUS_OK}
    
    def optimize(self) -> Tuple[dict, Trials]:
        self.trials = Trials()
        best = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(42),
        )
        self.best_params = space_eval(self.strategy.param_space, best)
        self.print_fold_periods()
        print("=== Top 5 Parameter combinations after Generalization-Loss penalty ===")
        top: List[Tuple[float, dict]] = []

        for trial, gl in zip(self.trials.trials, self._history_gl_max):
            flat = {k: v[0] for k, v in trial['misc']['vals'].items()}
            params = space_eval(self.strategy.param_space, flat)
            top.append((gl, params))
        for rank, (gl, params) in enumerate(sorted(top, key=lambda x: x[0])[:5], start=1):
            print(f"{rank:>2}. GL={gl:.4f} → Params={params}")
            
        self.is_values  = [is_m  for (is_m,  oos_m) in self.trial_metrics]
        self.oos_values = [oos_m for (is_m,  oos_m) in self.trial_metrics]
        return self.best_params, self.trials

    def evaluate(self) -> dict:
     if self.best_params is None:
        raise ValueError("Call optimize() before evaluate().")

     # In-sample final
     df_is = self.strategy.preprocess_data(self.analyzer.train_df.copy(), self.best_params)
     sig_is = self.strategy.generate_signals(df_is, **self.best_params)

     sl_series_is = self.analyzer._expand_param(self.best_params.get("sl_stop"), df_is)
     tp_series_is = self.analyzer._expand_param(self.best_params.get("tp_stop"), df_is)
     size_series_is = self.analyzer._expand_param(self.best_params.get("size"), df_is)

     dir_is = self._choose_direction(sig_is)
     self.train_pf = vbt.Portfolio.from_signals(
        close=df_is[self.s.price_col],
        entries=sig_is.get('entries'),
        exits=sig_is.get('exits'),
        short_entries=sig_is.get('short_entries'),
        short_exits=sig_is.get('short_exits'),
        freq=self.timeframe,
        init_cash=self.init_cash,
        fees=self.fees,
        slippage=self.slippage,
        direction=dir_is,
        sl_stop=sl_series_is,
        tp_stop=tp_series_is,
        size=size_series_is
     )

     # Out-of-sample splits
     self.oos_pfs = []
     for train_df, val_df in self._splits:
        df_val = self.strategy.preprocess_data(val_df.copy(), self.best_params)
        sig_val = self.strategy.generate_signals(df_val, **self.best_params)

        sl_series_val = self.analyzer._expand_param(self.best_params.get("sl_stop"), df_val)
        tp_series_val = self.analyzer._expand_param(self.best_params.get("tp_stop"), df_val)
        size_series_val = self.analyzer._expand_param(self.best_params.get("size"), df_val)

        dir_val = self._choose_direction(sig_val)
        pf_val = vbt.Portfolio.from_signals(
            close=df_val[self.s.price_col],
            entries=sig_val.get('entries'),
            exits=sig_val.get('exits'),
            short_entries=sig_val.get('short_entries'),
            short_exits=sig_val.get('short_exits'),
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=dir_val,
            sl_stop=sl_series_val,
            tp_stop=tp_series_val,
            size=size_series_val
        )
        self.oos_pfs.append(pf_val)

     self.test_pf = self.oos_pfs[-1] if self.oos_pfs else None
     train_summary = self.s.backtest_summary(self.train_pf, self.timeframe)
     test_summary = self.s.backtest_summary(self.test_pf, self.timeframe) if self.test_pf else None

     return {
        'train_pf': self.train_pf,
        'test_pf': self.test_pf,
        'train_summary': train_summary,
        'test_summary': test_summary,
        'oos_pfs': self.oos_pfs,
        'trial_metrics': self.trial_metrics
     }

    def montecarlo_oos(self, n_sims: int = 2_000, random_seed: int = 69, batch_size: int = 500, timeframe: Optional[str] = None) -> Dict[str, pd.DataFrame]:
     if not getattr(self, 'oos_pfs', None):
        raise RuntimeError("use `evaluate()` first.")

     tf = timeframe or self.analyzer.timeframe
     results = {}

     for i, pf in enumerate(self.oos_pfs, start=1):
        ret_series = pf.returns()
        bs = Bootstrapping(
            analyzer=None,
            ret_series=ret_series,
            timeframe=tf,
            n_sims=n_sims,
            random_seed=random_seed,
            batch_size=batch_size
        )
        mc_data = bs.mc_with_replacement()
        
        equity_curves: pd.DataFrame = mc_data['simulated_equity_curves']
        stats = pd.DataFrame(mc_data['simulated_stats'])

        q05 = equity_curves.quantile(0.05, axis=1)
        q95 = equity_curves.quantile(0.95, axis=1)
        quantiles = pd.DataFrame({'q05': q05, 'q95': q95})
        
        fold_key = f'Fold_{i}'
        results[fold_key] = {
                'equity_curves': equity_curves,
                'stats':         stats,
                'quantiles':     quantiles
            }
        
     print("\n=== Bootstrap Summaries per OOS Fold ===")
     for fold, fold_res in results.items():
        summary = fold_res['stats'].describe().loc[['mean', 'std', '25%', '50%', '75%'], :]
        print(f"\n--- {fold} ---")
        print(summary)

        
       
     return results

    def plot_walkforward_summary(self,
                                 bootstrap_results: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
                                 title: str = "Walk-Forward OOS Folds"
                                 ) -> go.Figure:
        """
        Plots the Walk-Forward OOS summary. If bootstrap_results is provided,
        95% confidence bands are included; otherwise only strategy vs. benchmark.

        Args:
            bootstrap_results: Optional dict from bootstrap_oos();
                if None, omits confidence bands.
            title: Title of the plot.
        Returns:
            Plotly Figure.
        """
        
        if bootstrap_results is None and hasattr(self, 'bs_results'):
            bootstrap_results = self.bs_results

        return _PlotWFOSummary(self, bootstrap_results).plot(title=title)

#
