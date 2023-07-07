"""
Module emissions
"""
import numpy as np
import pandas as pd
from scipy.stats import geom
import src.figure as fi


class Emissions:
    """
    Container for emission-associated attributes.

    Attributes
    ----------
    simulation : src.simulation.Simulation
        Container for simulation-associated attributes.
    emission_indices : 1-D array_like
        Indices of emitting transitions to apply to simulation.transition_series.
    event_indices : 1-D array_like
        Indices of detected emitting transitions to apply to simulation.transition_series.
    event_time_points : 1-D array_like
        The time points at which emissions are detected.
    event_time_series :
    """
    def __init__(self, simulation, photon_collection_rate=1, resample='5ms', emccd_gain=1, seed=None):
        if simulation.transition_series is None:
            raise ValueError('emissions not available if simulation has not been run.')
        self.simulation = simulation
        self.emission_indices = self.get_emission_indices()
        self.event_indices = self.get_event_indices(photon_collection_rate=photon_collection_rate, seed=seed)
        self.event_time_points = self.simulation.time_series[self.event_indices + 1]
        self.event_time_series = self.construct_event_time_series(resample=resample, emccd_gain=emccd_gain, seed=seed)

    def get_emission_indices(self):
        """
        Get indices to apply to simulation.transition_series to yield emitting transitions.

        Returns
        -------
        emission_indices : 1-D array_like
            Indices of emitting transitions to apply to simulation.transition_series.
        """
        df = self.simulation.transitions.combined_state_transitions_df
        emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
        emissions = np.in1d(self.simulation.transition_series, emitting_transition_ids).nonzero()[0]

        return emissions

    def get_event_indices(self, photon_collection_rate=1, seed=None):
        """
        Alters emission_indices keeping only a relative number (photon_collection_rate) of randomly selected indices.

        Parameters
        ----------
        photon_collection_rate : float
            Number between 0 and 1. Dictates the fraction of kept indices of emissions.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        event_indices : 1-D array_like
            Indices of detected emitting transitions to apply to simulation.transition_series.
        """
        rng = np.random.default_rng(seed)

        amount_not_detected = round((1 - photon_collection_rate) * self.emission_indices.shape[0])
        no_events_indices = rng.choice(np.arange(0, self.emission_indices.shape[0]), size=amount_not_detected,
                                       replace=False)
        event_indices = np.delete(self.emission_indices, no_events_indices)

        return event_indices

    def construct_event_time_series(self, resample='5ms', emccd_gain=None, seed=None):
        """
        Counts events within a time interval (resample).

        Parameters
        ----------
        resample : str
            For possible input values, see https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
        emccd_gain : None, int
            The gain of an EMCCD camera.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        event_time_series : pd.Series
            Contains the time points (increasing by the defined time interval) as index and the number of events (i.e.,
            detected emissions) as values.
        """
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

    def plot(self, mode='time_series', density=True, include_0=False, **kwargs):
        """
        Plot histogram or time series of events.

        Parameters
        ----------
        mode : str
            One of 'histogram', 'time_series'.
        density : bool
            Whether to display the histogram as probability densities. Else, probabilities.
            Only used if mode is 'histogram'.
        include_0 : bool
            Whether to include counts of 0 events.
            Only used if mode is 'histogram'.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if mode == 'histogram':
            data = self.event_time_series
            if not include_0:
                data = data[data != 0]
            axes = plot_histogram(data=data, density=density, **kwargs)
        elif mode == 'time_series':
            data = [self.event_time_series.index, self.event_time_series.values]
            axes = plot_time_series(data=data, **kwargs)
        else:
            raise AttributeError('mode has to be one of "histogram" or "time_series".')

        return axes


def plot_histogram(data, density=True, **kwargs):
    """
    Plot histogram of events.

    Parameters
    ----------
    data : 1-D array_like
    density : bool
        Whether to display the histogram as probability densities. Else, probabilities.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault('type_', 'hist')
    kwargs.setdefault('title', 'intensity distribution')
    kwargs.setdefault('xlabel', 'photon count')
    if density:
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)
    else:
        kwargs.setdefault('ylabel', 'Pr')
        kwargs.setdefault('weights', np.ones_like(data) / data.size)

    axes = fi.universal_figure(data=data, **kwargs)

    return axes


def plot_time_series(data, **kwargs):
    """
    Plot time series of events.

    Parameters
    ----------
    data : 2-D array_like
        Contains x and y data.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault('type_', 'line')
    kwargs.setdefault('title', 'fluorescence trajectory')
    kwargs.setdefault('xlabel', 'time [s]')
    kwargs.setdefault('ylabel', 'photon count')

    ax = fi.universal_figure(data=data, **kwargs)

    return ax
