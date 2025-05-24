import pandas as pd
import numpy as np
import vectorbt as vbt

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union, Sequence
from hyperopt import space_eval, STATUS_OK, tpe, fmin, Trials
from .plots import _PlotWFOSummary
from .analyzer import Analyzer
from .stats import Stats
import logging


logger = logging.getLogger(__name__)  


def _to_dict(obj: Any) -> Dict:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].to_dict()
        return {col: obj[col].to_dict() for col in obj.columns}
    return dict(obj)

@dataclass
class _WFOSplitCfg:
    """
    Configuration for Walk-Forward splits.

    Attributes:
        n_folds: Number of folds to generate.
        test_size: Proportion of data to use for each test set.
        train_size: Proportion of data for the training window (only for rolling mode).
        mode: Split mode, either 'anchored' (growing train window) or 'rolling' (fixed train window).
    """
    n_folds: int = 3
    test_size: float = 0.3
    train_size: Optional[float] = None  # Only used in rolling mode
    mode: str = "anchored"  # "anchored" or "rolling"

#
class AdvancedOptimizer:
    """
    Advanced optimizer using walkforward optimization with anchored or rolling mode
    - max_evals = tested parameter combinations, note that the TPE-sampler needs less trials tahn regular grid or random search
    - beta = penalty factor for getting a good tradeoff between robustness and performance

    Note: 
    - via "from quantybt.strategy.optimizer import _WFOSplitCfg()" you can import the Split configuration
    - warm-up for features will be ignored for saving more of the timeseries for backtesting. this may result in minimal differences. 
    - ensure your used features are only calculated on historical data and not a global computed z-score for example
    - shifting your signals or features is recommended for maximum safety and realistic results    
    """
    def __init__(
        self,
        analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio",
        beta: float = 0.3, 
        split_cfg: Union[_WFOSplitCfg, Sequence[_WFOSplitCfg]] = _WFOSplitCfg(),
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

        self.split_cfgs: List[_WFOSplitCfg] = (
            [split_cfg] if isinstance(split_cfg, _WFOSplitCfg) else list(split_cfg)
        )
        logging.debug(f"Configured Walk-Forward Split Configs: {self.split_cfgs}")
        self._splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = self._prepare_splits()
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

    def _generate_splits(self, df: pd.DataFrame, cfg: _WFOSplitCfg) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        total_samples = len(df)
        test_samples = int(cfg.test_size * total_samples)
        logging.debug(
            f"Generating splits: mode={cfg.mode}, n_folds={cfg.n_folds}, "
            f"train_size={cfg.train_size}, test_samples={test_samples}"
        )

        if cfg.mode == "rolling":
            if cfg.train_size is None:
                raise ValueError("train_size must be set for rolling mode")
            train_samples = int(cfg.train_size * total_samples)
            min_required = train_samples + cfg.n_folds * test_samples
            if min_required > total_samples:
                raise ValueError(
                    f"not enough data: required  {min_required}, available {total_samples}"
                )

            splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
            start = 0
            for fold in range(cfg.n_folds):
                if start + train_samples + test_samples > total_samples:
                    logging.debug(f"Rolling: Exiting at fold {fold} to avoid overflow")
                    break
                train_df = df.iloc[start : start + train_samples]
                test_df = df.iloc[
                    start + train_samples : start + train_samples + test_samples
                ]
                splits.append((train_df, test_df))
                logging.debug(
                    f"Fold {fold + 1} (Rolling): train {train_df.index[0]}–{train_df.index[-1]}, "
                    f"test {test_df.index[0]}–{test_df.index[-1]}"
                )
                start += test_samples
            return splits

        # Anchored mode (growing training window)
        min_train_samples = int(0.2 * total_samples)
        if min_train_samples + cfg.n_folds * test_samples > total_samples:
            raise ValueError(f"not enough data for {cfg.n_folds} Folds")

        splits = []
        current_start = min_train_samples
        for fold in range(cfg.n_folds):
            if current_start + test_samples > total_samples:
                logging.debug(f"Anchored: Exiting at fold {fold} to avoid overflow.")
                break
            train_df = df.iloc[:current_start]
            test_df = df.iloc[current_start : current_start + test_samples]
            splits.append((train_df, test_df))
            logging.debug(
                f"Fold {fold + 1} (Anchored): train {train_df.index[0]}–{train_df.index[-1]}, "
                f"test {test_df.index[0]}–{test_df.index[-1]}"
            )
            current_start += test_samples
        return splits
    
    def print_fold_periods(self):
     print("=== Walk-Forward Fold Periods ===")
     df = self.analyzer.train_df

     for i, (train_df, test_df) in enumerate(self._splits, 1):
        def to_timestamp(idx):
            if isinstance(idx, pd.DatetimeIndex):
                return idx[0], idx[-1]
            elif 'timestamp' in df.columns:
                try:
                    start = df.loc[idx[0], 'timestamp']
                    end   = df.loc[idx[-1], 'timestamp']
                    return start, end
                except Exception:
                    return idx[0], idx[-1]
            else:
                return idx[0], idx[-1]

        train_start, train_end = to_timestamp(train_df.index)
        test_start, test_end   = to_timestamp(test_df.index)

        print(f"Fold {i}:")
        print(f"  Train: {train_start} → {train_end}")
        print(f"  Test : {test_start} → {test_end}")
        print("-" * 40)

    def _prepare_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = self.analyzer.train_df.sort_index()
        all_splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for cfg in self.split_cfgs:
            splits = self._generate_splits(df, cfg)
            all_splits.extend(splits)
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
        elif has_short:
            return "shortonly"
        else:
            return "longonly"

    def _objective(self, params: dict) -> dict:
     try:
        
        seed = int(abs(hash(frozenset(params.items()))) % 2**32)
        np.random.seed(seed)
        
        losses, is_metrics, val_metrics = [], [], []
        higher_is_better = self.target_metric not in ["max_drawdown", "volatility"]

        for train_df, val_df in self._splits:
            df_train = self.strategy.preprocess_data(train_df.copy(), params)
            sig_train = self.strategy.generate_signals(df_train, **params)
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
                sl_stop=params.get("sl_pct"),
                tp_stop=params.get("tp_pct"),
            )
            m_is = self._metric(pf_train)

            df_val = self.strategy.preprocess_data(val_df.copy(), params)
            sig_val = self.strategy.generate_signals(df_val, **params)
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
                sl_stop=params.get("sl_pct"),
                tp_stop=params.get("tp_pct"),
            )
            m_val = self._metric(pf_val)

            
            if higher_is_better:
                if m_is <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    raw_gl = 1.0 - (m_val / m_is)
                    gl = max(0.0, min(1.0, raw_gl))
            else:
                if m_val <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    raw_gl = 1.0 - (m_is / m_val)
                    gl = max(0.0, min(1.0, raw_gl))

            losses.append((-m_val, gl))
            is_metrics.append(m_is)
            val_metrics.append(m_val)

        m_val_avg = -np.mean([l[0] for l in losses]) 
        gl_max    = max([l[1] for l in losses])          

        scale_raw = np.std(self._history_diffs[-10:]) if len(self._history_diffs) >= 10 else 1.0
        scale     = np.clip(scale_raw if scale_raw > 0 else 1.0, 0.1, 10.0)

        loss = -m_val_avg + self.beta * (gl_max / scale)

        self._history_gl_max.append(gl_max)
        self._history_diffs.append(loss)
        self.trial_metrics.append((np.mean(is_metrics), np.mean(val_metrics)))

        return {"loss": loss, "status": STATUS_OK, "params": params}

     except Exception as e:
        logger.error(f"Objective error: {e}", exc_info=True)
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

     top: list[tuple[float, dict]] = []
     for trial, gl in zip(self.trials.trials, self._history_gl_max):
        raw_vals = trial['misc']['vals']
        flat_vals = {k: v[0] for k, v in raw_vals.items()}
        params = space_eval(self.strategy.param_space, flat_vals)
        top.append((gl, params))

     top5 = sorted(top, key=lambda x: x[0])[:5]

     print("=== Top 5 Parameter combinations after Generalization-Loss penalty ===")
     for rank, (gl, params) in enumerate(top5, start=1):
        print(f"{rank:>2}. GL = {gl:.4f} → Params: {params}")

     return self.best_params, self.trials

    def evaluate(self) -> dict:
     if self.best_params is None:
        raise ValueError("Call optimize() before evaluate().")
     
     df_is = self.strategy.preprocess_data(self.analyzer.train_df.copy(), self.best_params)
     sig_is = self.strategy.generate_signals(df_is, **self.best_params)
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
        sl_stop=self.best_params.get('sl_pct'),
        tp_stop=self.best_params.get('tp_pct')
    )

     # oos
     self.oos_pfs = []
     for train_df, val_df in self._splits:
        df_val = self.strategy.preprocess_data(val_df.copy(), self.best_params)
        sig_val = self.strategy.generate_signals(df_val, **self.best_params)
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
            sl_stop=self.best_params.get('sl_pct'),
            tp_stop=self.best_params.get('tp_pct')
        )
        self.oos_pfs.append(pf_val)

     self.test_pf = self.oos_pfs[-1] if self.oos_pfs else None


     train_summary = self.s.backtest_summary(self.train_pf, self.timeframe)
     test_summary  = (
        self.s.backtest_summary(self.test_pf, self.timeframe)
        if self.test_pf is not None
        else None
     )

     return {
        'train_pf':       self.train_pf,
        'test_pf':        self.test_pf,
        'train_summary':  train_summary,
        'test_summary':   test_summary,
        'oos_pfs':        self.oos_pfs,
        'trial_metrics':  self.trial_metrics
     }

    def plot_walkforward_summary(self, title: str = "Walk-Forward Summary"):
     """x axis format not correct yet"""
     return _PlotWFOSummary(self).plot(title=title)
    
#
