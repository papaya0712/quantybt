import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import TYPE_CHECKING

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
import holoviews as hv

class _PlotBootstrapping:
    def __init__(self, mc):
        self.mc = mc

    def _align_series(self, sim_eq: pd.DataFrame, bench_eq: pd.Series):
        sim_eq.index = pd.to_datetime(sim_eq.index)
        bench_eq.index = pd.to_datetime(bench_eq.index)
        start = max(sim_eq.index.min(), bench_eq.index.min())
        end = min(sim_eq.index.max(), bench_eq.index.max())
        sim_eq, bench_eq = sim_eq.loc[start:end], bench_eq.loc[start:end]
        idx = sim_eq.index.union(bench_eq.index)
        return sim_eq.reindex(idx).ffill(), bench_eq.reindex(idx).ffill()

    def plot_histograms(self, mc_results: pd.DataFrame = None):
        hv.extension('bokeh')
        hv.renderer('bokeh').theme = 'carbon'

        if mc_results is None:
            data = self.mc.mc_with_replacement()
            mc_results = pd.DataFrame(data['simulated_stats'])

        sharpe_vals = mc_results['Sharpe']
        sortino_vals = mc_results['Sortino']
        calmar_vals = mc_results['Calmar']
        maxdd_vals = mc_results['MaxDrawdown']

        sharpe_q5, sharpe_q50, sharpe_q95         = np.percentile(sharpe_vals,  [5, 50, 95])
        sortino_q5, sortino_q50, sortino_q95      = np.percentile(sortino_vals, [5, 50, 95])
        calmar_q5, calmar_q50, calmar_q95         = np.percentile(calmar_vals,  [5, 50, 95])
        maxdd_q5, maxdd_q50, maxdd_q95, maxdd_q01 = np.percentile(maxdd_vals,   [5,50,95,1])

        bench_ret = self.mc.pf.benchmark_returns()
        bench_stats = self.mc._analyze_series(bench_ret)

        bench_sharpe = bench_stats['Sharpe']
        bench_sortino = bench_stats['Sortino']
        bench_calmar = bench_stats['Calmar']
        bench_maxdd = bench_stats['MaxDrawdown']

        color_q5 = "green"
        color_q50 = "deepskyblue"
        color_q95 = "red"
        color_q01 = "yellow"
        color_bench = "purple"
        bins = 50
        plot_width = 600
        plot_height = 400

        hist_opts = dict(fill_color="lightgrey", bgcolor=None, gridstyle={'grid_line_alpha': 0.3}, width=plot_width, height=plot_height, show_legend=False)

        sharpe_vals = sharpe_vals[np.isfinite(sharpe_vals)]
        sortino_vals = sortino_vals[np.isfinite(sortino_vals)]
        calmar_vals = calmar_vals[np.isfinite(calmar_vals)]
        maxdd_vals = maxdd_vals[np.isfinite(maxdd_vals)]

        sharpe_hist_values, sharpe_bin_edges = np.histogram(sharpe_vals, bins=bins)
        sortino_hist_values, sortino_bin_edges = np.histogram(sortino_vals, bins=bins)
        calmar_hist_values, calmar_bin_edges = np.histogram(calmar_vals, bins=bins)
        maxdd_hist_values, maxdd_bin_edges = np.histogram(maxdd_vals, bins=bins)


        hist_sharpe = hv.Histogram((sharpe_hist_values, sharpe_bin_edges)).opts(title="Sharpe Distribution", xlabel="Sharpe", ylabel="Frequency", **hist_opts)
        hist_sortino = hv.Histogram((sortino_hist_values, sortino_bin_edges)).opts(title="Sortino Distribution", xlabel="Sortino", ylabel="Frequency", **hist_opts)
        hist_calmar = hv.Histogram((calmar_hist_values, calmar_bin_edges)).opts(title="Calmar Distribution", xlabel="Calmar", ylabel="Frequency", **hist_opts)
        hist_maxdd = hv.Histogram((maxdd_hist_values, maxdd_bin_edges)).opts(title="Max Drawdown Distribution", xlabel="MaxDrawdown", ylabel="Frequency", **hist_opts)

        max_y_sharpe = sharpe_hist_values.max()
        max_y_sortino = sortino_hist_values.max()
        max_y_calmar = calmar_hist_values.max()
        max_y_maxdd = maxdd_hist_values.max()

        spikes_sharpe = (
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q5], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q50], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q95], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_sortino = (
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q5], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q50], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q95], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_calmar = (
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q5], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q50], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q95], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_maxdd = (
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q01], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='1th %ile').opts(color=color_q01, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q5], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q50], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q95], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        bench_sh_spike = hv.Spikes(pd.DataFrame({'Sharpe': [bench_sharpe], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_so_spike = hv.Spikes(pd.DataFrame({'Sortino': [bench_sortino], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_ca_spike = hv.Spikes(pd.DataFrame({'Calmar': [bench_calmar], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_dd_spike = hv.Spikes(pd.DataFrame({'MaxDrawdown': [bench_maxdd], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)

        plot_sharpe = (hist_sharpe * spikes_sharpe * bench_sh_spike).opts(show_legend=True, legend_position='top_right')
        plot_sortino = (hist_sortino * spikes_sortino * bench_so_spike).opts(show_legend=True, legend_position='top_right')
        plot_calmar = (hist_calmar * spikes_calmar * bench_ca_spike).opts(show_legend=True, legend_position='top_right')
        plot_maxdd = (hist_maxdd * spikes_maxdd * bench_dd_spike).opts(show_legend=True, legend_position='top_right')

        final_plot = (plot_sharpe + plot_sortino + plot_calmar + plot_maxdd).opts(shared_axes=False)
        return final_plot
    
# ============================== Heatmap - local sensitivity analysis ============================== # 
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
