import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional, List, Callable, Union, Sequence
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .stats import Stats
from .analyzer import Analyzer

class LocalSensitivityAnalyzer:
    """
    Local Sensitivity Analyzer (LSA)

    Evaluates the local sensitivity of a trading strategy by perturbing input parameters 
    either by percentage (relative) or fixed values (absolute). For each tested parameter, 
    the strategy is backtested at baseline, lower, and upper perturbation levels.
    """

    def __init__(
        self,
        analyzer,
        base_params: dict,
        param_perturbations: Dict[str, Union[float, Dict[str, float]]],
        metrics: Union[str, List[str]] = "sharpe_ratio"
    ):
        self.analyzer = analyzer
        self.base_params = base_params
        self.tf = analyzer.timeframe
        self.stats = analyzer.s
        self.metrics = [metrics] if isinstance(metrics, str) else list(metrics)

        self.perturbation_plan = self._expand_perturbations(param_perturbations)

        self.metric_funcs = {
            "sharpe_ratio":  lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[0],
            "sortino_ratio": lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[1],
            "calmar_ratio":  lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[2],
            "total_return":  lambda pf: self.stats._returns(pf)[0],
            "max_drawdown":  lambda pf: self.stats._risk_metrics(self.tf, pf)[0],
            "volatility":    lambda pf: self.stats._risk_metrics(self.tf, pf)[2],
            "profit_factor": lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

        print(f"[LSA] Initialized for metrics: {self.metrics}")
        print(f"[LSA] Perturbation plan: {self.perturbation_plan}")

    def _expand_perturbations(self, perturbations):
        plan = []

        for param, val in perturbations.items():
            if param not in self.base_params:
                continue

            base_val = self.base_params[param]

            if isinstance(val, dict) and "down" in val and "up" in val:
                plan.append((param, val["down"], val["up"], "absolute"))
            elif isinstance(val, (float, int)) and isinstance(base_val, (int, float)):
                down_val = base_val * (1 - val)
                up_val = base_val * (1 + val)
                plan.append((param, down_val, up_val, "relative"))
            else:
                print(f"[LSA] Skipped invalid perturbation format for {param}")

        return plan

    def _run_backtest(self, params: dict):
        full_params = {**self.base_params, **params}
        analyzer = self.analyzer.__class__(
            strategy=self.analyzer.strategy,
            params=full_params,
            full_data=self.analyzer.full_data,
            timeframe=self.tf,
            test_size=0.0,
            init_cash=self.analyzer.init_cash,
            fees=self.analyzer.fees,
            slippage=self.analyzer.slippage,
            sl_stop=full_params.get("sl_stop", full_params.get("sl_pct", None)),
            tp_stop=full_params.get("tp_stop", full_params.get("tp_pct", None)),
            size=full_params.get("size", None)
        )
        return analyzer.pf

    def run(self) -> pd.DataFrame:
        print("\n[LSA] Running sensitivity analysis...\n")
        rows = []

        base_pf = self._run_backtest({})
        base_metrics = {m: self.metric_funcs[m](base_pf) for m in self.metrics}

        for param, down_val, up_val, mode in tqdm(self.perturbation_plan, desc="LSA Parameters"):
            base_val = self.base_params.get(param)
            if base_val is None:
                continue

            if isinstance(base_val, int):
                down_val = round(down_val)
                up_val = round(up_val)

            pct_down = (base_val - down_val) / base_val if base_val != 0 else 0
            pct_up   = (up_val - base_val) / base_val if base_val != 0 else 0

            row = {
                "parameter": param,
                "mode": mode,
                "pct_change": None if mode == "absolute" else (pct_down + pct_up) / 2,
                "baseline": base_val,
                "down_val": down_val,
                "up_val": up_val
            }

            for m in self.metrics:
                row[f"{m}_base"] = base_metrics[m]

            # Run down test
            try:
                pf_down = self._run_backtest({param: down_val})
                for m in self.metrics:
                    val = self.metric_funcs[m](pf_down)
                    row[f"{m}_down"] = val
                    row[f"{m}_delta_down"] = val - base_metrics[m]
            except Exception as e:
                print(f"[ERROR] Down test failed for {param}={down_val}: {e}")
                for m in self.metrics:
                    row[f"{m}_down"] = np.nan
                    row[f"{m}_delta_down"] = np.nan

            # Run up test
            try:
                pf_up = self._run_backtest({param: up_val})
                for m in self.metrics:
                    val = self.metric_funcs[m](pf_up)
                    row[f"{m}_up"] = val
                    row[f"{m}_delta_up"] = val - base_metrics[m]
            except Exception as e:
                print(f"[ERROR] Up test failed for {param}={up_val}: {e}")
                for m in self.metrics:
                    row[f"{m}_up"] = np.nan
                    row[f"{m}_delta_up"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows).set_index(["parameter", "mode"])
        print("\n[LSA] Sensitivity analysis complete.")
        return df

    def plot(self, df: pd.DataFrame, height_per_plot=400, width=1000):
        num_metrics = len(self.metrics)
        total_height = height_per_plot * num_metrics

        fig = make_subplots(
            rows=num_metrics, cols=1,
            shared_xaxes=False,
            subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in self.metrics]
        )

        for i, metric in enumerate(self.metrics, start=1):
            metric_cols = [col for col in df.columns if col.startswith(f"{metric}_")]
            df_metric = df[metric_cols].copy()
            params = df_metric.index.get_level_values("parameter").unique()
            deltas = []
            for param in params:
                param_data = df_metric.loc[param]
                avg_down = -param_data[f"{metric}_delta_down"].mean()
                avg_up = param_data[f"{metric}_delta_up"].mean()
                max_effect = max(abs(avg_down), abs(avg_up))
                deltas.append((param, avg_down, avg_up, max_effect))

            deltas_sorted = sorted(deltas, key=lambda x: x[3], reverse=True)
            params_sorted = [x[0] for x in deltas_sorted]
            avg_down = [x[1] for x in deltas_sorted]
            avg_up = [x[2] for x in deltas_sorted]

            fig.add_trace(go.Bar(
                y=params_sorted,
                x=avg_down,
                name='-Δ',
                orientation='h',
                marker=dict(color='#EF553B'),
                showlegend=(i == 1),
                hovertext=[f"Δ: {x:.3f}" for x in avg_down]
            ), row=i, col=1)

            fig.add_trace(go.Bar(
                y=params_sorted,
                x=avg_up,
                name='+Δ',
                orientation='h',
                marker=dict(color='#00CC96'),
                showlegend=(i == 1),
                hovertext=[f"Δ: {x:.3f}" for x in avg_up]
            ), row=i, col=1)

        fig.update_layout(
            height=total_height,
            width=width,
            title_text="<b>Local Sensitivity Analysis - Tornado Charts</b>",
            template="plotly_dark",
            barmode='relative',
            hoverlabel=dict(bgcolor='#1A1A1A', font_size=13),
            margin=dict(l=120, r=50, t=60, b=40),
        )

        fig.show()

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class Gridsearcher:
    """
    2D-Gridsearch for simple global sensitivity analysis.

    Parameters:
        analyzer: Backtest-Engine (wie in LSA)
        x: dict  {"param": "paramname", "from": a, "to": b, "by": step}
        y: dict  {"param": "paramname", "from": a, "to": b, "by": step}
        metric: str, Zielmetrik (e.g. "sharpe_ratio")
    """

    def __init__(self, analyzer, x: dict, y: dict, metric="sharpe_ratio"):
        self.analyzer = analyzer
        self.x = x
        self.y = y
        self.metric = metric
        self.base_params = analyzer.params
        self.tf = analyzer.timeframe
        self.stats = analyzer.s
        self.df_result = None

        self.metric_funcs = {
            "sharpe_ratio":  lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[0],
            "sortino_ratio": lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[1],
            "calmar_ratio":  lambda pf: self.stats._risk_adjusted_metrics(self.tf, pf)[2],
            "total_return":  lambda pf: self.stats._returns(pf)[0],
            "max_drawdown":  lambda pf: self.stats._risk_metrics(self.tf, pf)[0],
            "volatility":    lambda pf: self.stats._risk_metrics(self.tf, pf)[2],
            "profit_factor": lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

    def _create_grid(self):
        x_vals = np.arange(self.x["from"], self.x["to"] + self.x["by"], self.x["by"])
        y_vals = np.arange(self.y["from"], self.y["to"] + self.y["by"], self.y["by"])
        return x_vals, y_vals

    def _run_backtest(self, override_params):
        full_params = {**self.base_params, **override_params}
        analyzer = self.analyzer.__class__(
            strategy=self.analyzer.strategy,
            params=full_params,
            full_data=self.analyzer.full_data,
            timeframe=self.analyzer.timeframe,
            test_size=0.0,
            init_cash=self.analyzer.init_cash,
            fees=self.analyzer.fees,
            slippage=self.analyzer.slippage,
            sl_stop=full_params.get("sl_stop", full_params.get("sl_pct", None)),
            tp_stop=full_params.get("tp_stop", full_params.get("tp_pct", None)),
            size=full_params.get("size", None)
        )
        return analyzer.pf

    def run(self, max_workers=None):
        x_vals, y_vals = self._create_grid()
        grid = [(xv, yv) for xv in x_vals for yv in y_vals]
        total = len(grid)

        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 2)

        results = []

        def task(xv, yv):
            params = {
                self.x["param"]: xv,
                self.y["param"]: yv
            }
            try:
                pf = self._run_backtest(params)
                score = self.metric_funcs[self.metric](pf)
            except Exception as e:
                print(f"[Error] ({xv}, {yv}) failed: {e}")
                score = np.nan
            return (xv, yv, score)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(task, xv, yv): (xv, yv) for xv, yv in grid}
            pbar = tqdm(total=total, desc="Parallel Gridsearch")
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
            pbar.close()

        self.df_result = pd.DataFrame(results, columns=[self.x["param"], self.y["param"], self.metric])
        return self.df_result

    def plot_heatmap(self, figsize=(8, 8), annot=True, fmt=".2f", cmap="viridis"):
    
     if not hasattr(self, "df_result"):
        raise ValueError("Please run .run() first.")

     import matplotlib.pyplot as plt
     plt.style.use('dark_background')
     import seaborn as sns

     df_pivot = self.df_result.pivot(index=self.y["param"], columns=self.x["param"], values=self.metric)

     plt.figure(figsize=figsize)
     sns.heatmap(
        df_pivot,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={"label": self.metric.replace("_", " ").title()}
     )
     plt.title(f"Gridsearch Heatmap: {self.metric.replace('_', ' ').title()}")
     plt.xlabel(self.x["param"])
     plt.ylabel(self.y["param"])
     plt.tight_layout()
     plt.show()


#



