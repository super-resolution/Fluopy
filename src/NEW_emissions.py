import numpy as np
import pandas as pd
from scipy.stats import geom
import src.custom_plot as cp


class Emissions:
    def __init__(self, transition_df, transition_series, time_series, photon_collection_rate, resample, emccd_gain,
                 seed):
        self.emissions = self.get_emissions(transition_df, transition_series)
        self.events = self.get_events(photon_collection_rate, seed)
        self.event_time_points = time_series[self.events]
        self.event_time_series = self.construct_event_time_series(resample, emccd_gain, seed)

    @staticmethod
    def get_emissions(combined_state_transitions_df, transition_series):
        df = combined_state_transitions_df
        emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
        emissions = np.in1d(transition_series, emitting_transition_ids).nonzero()[0]

        return emissions

    def get_events(self, photon_collection_rate, seed=100):
        rng = np.random.default_rng(seed)

        amount_not_detected = round((1 - photon_collection_rate) * self.emissions.shape[0])
        no_events_indices = rng.choice(np.arange(0, self.emissions.shape[0]), size=amount_not_detected, replace=False)
        events = np.delete(self.emissions, no_events_indices)

        return events

    def construct_event_time_series(self, resample, emccd_gain=None, seed=100):
        event_time_points = np.insert(self.event_time_points, 0, 0)
        time_deltas = pd.to_timedelta(event_time_points, unit='s')
        if emccd_gain is not None:
            events = geom.rvs(p=1/emccd_gain, size=event_time_points.shape[0], random_state=seed)
            events = events.astype(float)
        else:
            events = np.ones(shape=event_time_points.shape[0])
        events[0] = 0

        event_time_series = pd.Series(events, index=time_deltas)
        event_time_series = event_time_series.resample(resample).sum()
        time_deltas = event_time_series.index
        in_seconds = time_deltas / np.timedelta64(1, 's')
        event_time_series.index = in_seconds

        return event_time_series

    def plot(self, mode, density=True, include_0=False, **kwargs):
        if mode == 'histogram':
            data = self.event_time_series
            if not include_0:
                data = data[data != 0]
            fig, ax = plot_histogram(data, density, **kwargs)
        else:
            data = [self.event_time_series.index, self.event_time_series.values]
            fig, ax = plot_time_series(data, **kwargs)

        return fig, ax


def plot_histogram(data, density, **kwargs):
    kwargs.setdefault('type_', 'hist')
    kwargs.setdefault('title', 'intensity distribution')
    kwargs.setdefault('xlabel', 'photon count')
    if density:
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)
    else:
        kwargs.setdefault('ylabel', 'Pr')
        kwargs.setdefault('weights', np.ones_like(data) / data.size)

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax


def plot_time_series(data, **kwargs):
    kwargs.setdefault('type_', 'line')
    kwargs.setdefault('title', 'fluorescence trajectory')
    kwargs.setdefault('xlabel', 'time [s]')
    kwargs.setdefault('ylabel', 'photon count')

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax
