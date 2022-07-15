import src.custom_plot as cp
import numpy as np


class FigureCollection:
    def __init__(self, system_object):
        self.object = system_object

    def state_population(self, use_unique=True, **kwargs):
        if use_unique:
            kwargs.setdefault('data', self.object.unique_series_converted)
            kwargs.setdefault('xlim', [0, len(self.object.unique_states)])
            kwargs.setdefault('xticks', range(len(self.object.unique_states)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.unique_names, rotation=70))
            kwargs.setdefault('bins', np.arange(0, len(self.object.unique_states) + 1, 1))
        else:
            kwargs.setdefault('data', self.object.state_series)
            kwargs.setdefault('xlim', [0, len(self.object.Joined_States)])
            kwargs.setdefault('xticks', range(len(self.object.Joined_States)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.state_names, rotation=70))
            kwargs.setdefault('bins', np.arange(0, len(self.object.Joined_States) + 1, 1))

        kwargs.setdefault('density', True)
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('log', True)

        fig, ax = cp.universal_figure(type_="hist", **kwargs)

        return fig, ax

    def time_steps(self, only_emitting_transitions=True, **kwargs):
        if only_emitting_transitions:
            kwargs.setdefault('data', self.object.time_step_series[self.object.emitting_mask])
        else:
            kwargs.setdefault('data', self.object.time_step_series)

        kwargs.setdefault('density', True)
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('log', True)

        fig, ax = cp.universal_figure(type_="hist", **kwargs)

        return fig, ax

    def occupation_time(self, total=False, **kwargs):
        if total:
            kwargs.setdefault('data', [np.arange(len(self.object.unique_states)), self.object.occupation_time_total])
            kwargs.setdefault('ylabel', "Total time [s]")
        else:
            kwargs.setdefault('data', [np.arange(len(self.object.unique_states)), self.object.occupation_time_mean])
            kwargs.setdefault('ylabel', "Mean time [s]")

        kwargs.setdefault('yscale', "log")
        kwargs.setdefault('xlabel', None)
        kwargs.setdefault('xticks', range(len(self.object.unique_states)))
        kwargs.setdefault('xticklabels', dict(labels=self.object.unique_names, rotation=70))

        fig, ax = cp.universal_figure(type_="bar", **kwargs)

        return fig, ax

    def emission_events(self, time_series=True, **kwargs):
        if time_series:
            kwargs.setdefault('type_', "line")
            kwargs.setdefault('data', [self.object.pandas_series.index, self.object.pandas_series.values])
            kwargs.setdefault('ylabel', "Emission count")
            kwargs.setdefault('xlabel', "time [s]")
        else:
            kwargs.setdefault('type_', "hist")
            kwargs.setdefault('data', self.object.pandas_series[self.object.pandas_series != 0])
            kwargs.setdefault('ylabel', "PD")
            kwargs.setdefault('xlabel', "Photon counts")
            kwargs.setdefault('density', True)

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def fcs(self, log=True, **kwargs):
        if log:
            kwargs.setdefault('data', [self.object.autocorrelation[0][1:], self.object.autocorrelation[1][1:]])
        else:
            kwargs.setdefault('data', [self.object.pandas_series.index[1:1000], self.object.autocorrelation[1:1000]])

        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', "time [s]")
        kwargs.setdefault('ylabel', "G(t)")

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def on_off(self, on=True, resample="5 ms", display_mean=True, **kwargs):
        if on:
            kwargs.setdefault('data', self.object.on_periods)
            kwargs.setdefault('xlabel', "ON periods [frames à " + resample + "]")
            mean = np.mean(self.object.on_periods)
        else:
            kwargs.setdefault('data', self.object.off_periods)
            kwargs.setdefault('xlabel', "OFF periods [frames à " + resample + "]")
            mean = np.mean(self.object.off_periods)

        kwargs.setdefault('type_', "hist")
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('density', True)

        fig, ax = cp.universal_figure(**kwargs)
        if display_mean:
            ax.text(x=0.7, y=0.85, s=f"mean: {mean:.2f}", transform=ax.transAxes, fontsize=16)

        return fig, ax
