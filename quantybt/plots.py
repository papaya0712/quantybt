import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import TYPE_CHECKING, Optional

import warnings
warnings.filterwarnings("ignore")
# ============================== Analyzer Plot   ============================== # 
class _PlotBacktest:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.pf       = analyzer.pf
        self.stats    = analyzer.s
        self.tf       = analyzer.timeframe
        self._df      = analyzer.train_df

    def plot_backtest(self, title: str = "Backtest Results") -> go.Figure:
        eq_strat = self.pf.value()
        ts_strat = self._timestamps_for(eq_strat.index)
        try:
            eq_bench = self.pf.benchmark_value()
            ts_bench = self._timestamps_for(eq_bench.index)
        except Exception:
            eq_bench = pd.Series(dtype=float)
            ts_bench = pd.Series(dtype="datetime64[ns]")

        dd_strat = self._drawdown(eq_strat)
        dd_bench = self._drawdown(eq_bench) if not eq_bench.empty else pd.Series(dtype=float)

        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "xy"}], [{"type": "xy"}]],
            subplot_titles=["Equity Curve", "Drawdown Curve [%]"],
            shared_xaxes=True
        )

        fig.add_trace(go.Scatter(x=ts_strat, y=eq_strat.values, mode="lines", name="Strategy"), row=1, col=1)
        if not eq_bench.empty:
            fig.add_trace(go.Scatter(x=ts_bench, y=eq_bench.values, mode="lines", name="Benchmark"), row=1, col=1)

        fig.add_trace(go.Scatter(x=ts_strat, y=dd_strat.values, mode="lines", fill="tozeroy", name="Drawdown"), row=2, col=1)
        if not dd_bench.empty:
            fig.add_trace(go.Scatter(x=ts_bench, y=dd_bench.values, mode="lines", fill="tozeroy", name="Benchmark DD"), row=2, col=1)

        fig.update_layout(
            title=title,
            template="plotly_dark",
            showlegend=True,
            hovermode="x unified",
            height=700,
            width=1100
        )
        fig.update_xaxes(type="date")
        return fig

    def _timestamps_for(self, idx: pd.DatetimeIndex) -> pd.Series:
        df_idx = self._df.index
        if pd.api.types.is_integer_dtype(df_idx.dtype):
            raw = idx.view('int64')
            try:
                ts = self._df['timestamp'].loc[raw]
                return ts
            except Exception:
                pass
        if 'timestamp' in self._df.columns:
            ts = self._df['timestamp'].reindex(idx)
            return ts.fillna(idx)
        return pd.Series(idx, index=idx)

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        if equity.empty:
            return pd.Series(dtype=float)
        return (equity - equity.cummax()) / equity.cummax() * 100

    def _entry_exit_indices(self):
        tr = self.pf.trades
        rets = self.pf.returns()
        n = len(rets)
        if hasattr(tr, 'entry_idx') and hasattr(tr, 'exit_idx'):
            try:
                entries = tr.entry_idx.values.astype(int)
                exits   = tr.exit_idx.values.astype(int)
                exits = np.where(np.isnan(exits), n - 1, exits).astype(int)
                return entries, exits
            except Exception:
                pass
        try:
            rr = tr.records_readable
            for e_col, x_col in [('Entry Timestamp', 'Exit Timestamp'), ('Entry Index', 'Exit Index')]:
                if e_col in rr.columns and x_col in rr.columns:
                    entries = rr[e_col].astype(int).values
                    exits   = rr[x_col].fillna(n - 1).astype(int).values
                    return entries, exits
        except Exception:
            pass
        return np.arange(n, dtype=int), np.arange(n, dtype=int)

    def _trade_returns(self):
        rets = self.pf.returns()
        entries, exits = self._entry_exit_indices()
        mask = np.zeros(len(rets), bool)
        for en, ex in zip(entries, exits):
            mask[en:ex + 1] = True
        return rets[mask]

# ============================== Walkforward plot ============================== # 
class _PlotWFOSummary:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.analyzer = optimizer.analyzer
        self.init_cash = self.analyzer.init_cash

        if not getattr(self.optimizer, 'oos_pfs', None):
            raise RuntimeError("Call AdvancedOptimizer.evaluate() before plotting WFO summary.")

        self.oos_pfs = self.optimizer.oos_pfs
        self._df     = self.analyzer.train_df
        self._prepare_equities()

    def _prepare_equities(self):
        last_value = self.init_cash
        self.equities  = []
        self.timestamps = []

        for pf in self.oos_pfs:
            pf_eq = pf.value()
            eq    = pf_eq / pf_eq.iloc[0] * last_value
            last_value = eq.iloc[-1]

            ts = pd.Series(pf_eq.index, index=pf_eq.index)
            if 'timestamp' in self._df.columns:
                ts = self._df['timestamp'].reindex(pf_eq.index).ffill()

            self.equities.append(eq)
            self.timestamps.append(ts)

    def plot(self, title="Walk-Forward OOS Folds (Individual Plots)") -> go.Figure:
        num_folds = len(self.equities)
        fig = make_subplots(
            rows=num_folds, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"Fold {i+1}" for i in range(num_folds)]
        )

        for i, (eq, ts) in enumerate(zip(self.equities, self.timestamps), start=1):
            fig.add_trace(
                go.Scatter(x=ts, y=eq.values, mode="lines",
                           name=f"Fold {i}",
                           hovertemplate=f"Fold {i}<br>%{{x}}<br>Equity: %{{y:.2f}}"),
                row=i, col=1
            )
            fig.update_xaxes(
                range=[ts.min(), ts.max()],
                type="date",
                row=i, col=1
            )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=300 * num_folds,
            width=1100,
            hovermode="closest",
            showlegend=False
        )
        return fig
    
# ============================== Montecarlo Bootstrapping Plot ============================== # 
if TYPE_CHECKING:
    from .montecarlo import Bootstrapping

class _PlotBootstrapping:
    def __init__(self, mc):
        self.mc = mc

    def plot_histograms(self, mc_results: Optional[pd.DataFrame] = None):
        shown_legends = set()
        if mc_results is None:
            data = self.mc.mc_with_replacement()
            mc_results = pd.DataFrame(data['simulated_stats'])

        stats = ['Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown']
        for stat in stats:
            mc_results[stat] = mc_results[stat][np.isfinite(mc_results[stat])]

        sharpe_q = np.percentile(mc_results['Sharpe'], [5, 50, 95])
        sortino_q = np.percentile(mc_results['Sortino'], [5, 50, 95])
        calmar_q = np.percentile(mc_results['Calmar'], [5, 50, 95])
        maxdd_q = np.percentile(mc_results['MaxDrawdown'], [1, 5, 50, 95])

        bench_ret = self.mc.pf.benchmark_returns()
        bench_stats = self.mc._analyze_series(bench_ret)

        fig = make_subplots(rows=2, cols=2, subplot_titles=stats)

        params = [
            ('Sharpe', sharpe_q, bench_stats['Sharpe'], 1, 1),
            ('Sortino', sortino_q, bench_stats['Sortino'], 1, 2),
            ('Calmar', calmar_q, bench_stats['Calmar'], 2, 1),
            ('MaxDrawdown', maxdd_q, bench_stats['MaxDrawdown'], 2, 2)
        ]

        for name, qs, bench, row, col in params:
            values = mc_results[name].dropna()
            hist_vals, bin_edges = np.histogram(values, bins=40)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            max_y = hist_vals.max()

            hist_trace = go.Bar(
                x=bin_centers,
                y=hist_vals,
                name=f"{name} Distribution",
                marker_color='grey',
                showlegend=False
            )
            fig.add_trace(hist_trace, row=row, col=col)

            if name != 'MaxDrawdown':
                labels = ['5%', '50%', '95%']
                colors = ['green', 'deepskyblue', 'red']
               
                dashes = [None, None, None]  
            else:
                labels = ['1%', '5%', '50%', '95%']
                colors = ['yellow', 'green', 'deepskyblue', 'red']
                dashes = [None, None, None, None]

            for i, val in enumerate(qs):
                legend_label = f"{labels[i]} Quantile"
                showlegend_flag = legend_label not in shown_legends
                if showlegend_flag:
                    shown_legends.add(legend_label)
                q_trace = go.Scatter(
                    x=[val, val], y=[0, max_y], mode='lines',   name=legend_label,  marker=dict(color=colors[i]),  showlegend=showlegend_flag )
                fig.add_trace(q_trace, row=row, col=col)

            
            bench_trace = go.Scatter(
                x=[bench, bench],
                y=[0, max_y],
                mode='lines',

                name=f"Benchmark",
                marker=dict(color='purple'),  
                showlegend=(row == 1 and col == 1)
            )
            fig.add_trace(bench_trace, row=row, col=col)

      
        fig.update_layout(
            height=800,
            template="plotly_dark",
            width=1000,
            title_text="Bootstrapped Metric Distributions",
            legend=dict(orientation='h', yanchor='bottom', y=-0.1)
        )

        return fig   
# ============================== Heatmap - local sensitivity analysis ============================== # 
import holoviews as hv
class _lsa:
    def __init__(self, matrix: pd.DataFrame, title: str = "Parameter Sensitivities"):
        self.df = matrix
        self.title = title

    def heatmap(self) -> hv.HeatMap:
        hv.extension('bokeh')
        hv.renderer('bokeh').theme = 'carbon'    
        data = (
            self.df.filter(like="relative_sensitivity_")
                .rename(columns=lambda c: c.replace("relative_sensitivity_", ""))
                .reset_index()
                .melt(id_vars="parameter", var_name="metric", value_name="rs")
        )

        return hv.HeatMap(
            data,
            kdims=["parameter", "metric"],
            vdims=["rs"]
        ).opts(
            cmap="Plasma",
            colorbar=True,
            invert_yaxis=True,
            width=600,
            height=300,
            tools=["hover"],
            xlabel="Metrik",
            ylabel="Parameter",
            hover_tooltips=[
                ("Parameter", "@parameter"),
                ("Metrik", "@metric"),
                ("RelSens", "@rs{0.000}"),
            ],
            title=self.title,
        )

#
