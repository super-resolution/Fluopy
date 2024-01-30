"""
Module emissions
"""
import numpy as np
import pandas as pd
from scipy.stats import geom, norm
import src.figure as fi
import os


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
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and the number of events (i.e.,
        detected emissions) as values.
    """
    def __init__(self, simulation, photon_collection_rate=1, resample='5ms', emccd_gain=1, seed=None):
        """_summary_

        Parameters
        ----------
        simulation : _type_
            _description_
        photon_collection_rate : int, optional
            _description_, by default 1
        resample : str, optional
            _description_, by default '5ms'
        emccd_gain : int, optional
            _description_, by default 1
        seed : _type_, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
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
        emission_indices = np.in1d(self.simulation.transition_series, emitting_transition_ids).nonzero()[0]

        return emission_indices

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
        if photon_collection_rate > 1 or photon_collection_rate < 0:
            raise ValueError('photon_collection_rate has to be between 0 and 1.')
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
        rng = np.random.default_rng(seed)

        added_end_time = False
        if event_time_points[-1] != self.simulation.time_series[-1]:
            added_end_time = True
            event_time_points = np.append(event_time_points, self.simulation.time_series[-1])

        time_deltas = pd.to_timedelta(event_time_points, unit='s')
        if emccd_gain is not None:
            events = geom.rvs(p=1/emccd_gain, size=event_time_points.shape[0], random_state=rng)
            events = events.astype(float)
        else:
            events = np.ones(shape=event_time_points.shape[0])
        events[0] = 0

        if added_end_time:
            events[-1] = 0

        event_time_series = pd.Series(events, index=time_deltas)
        event_time_series = event_time_series.resample(resample).sum()
        time_deltas = event_time_series.index
        in_seconds = time_deltas / np.timedelta64(1, 's')
        event_time_series.index = in_seconds

        return event_time_series
    

    def add_noise(self, mean, std):
        size = self.event_time_series.size
        variates = norm(mean, std).rvs(size)
        self.event_time_series = self.event_time_series + variates


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
        elif mode == 'cum_events':
            cum_events = self.event_time_series.cumsum()
            cum_events = cum_events / cum_events.max() * 100
            data = [self.event_time_series.index, cum_events.values]
            axes = plot_cumulative_events(data=data, **kwargs)
        else:
            raise AttributeError('mode has to be one of "histogram", "time_series" or "cum_events".')

        return axes

    def save(self, path, name_extension=''):
        """
        Saves event_time_series and event_time_points to a file.

        Parameters
        ----------
        path : str
            Directory where the files shall be stored.
        name_extension : str
            Optional file name extension.

        Returns
        -------
        None
        """
        time_series_file = os.path.join(path, 'event_time_series' + name_extension)
        time_points_file = os.path.join(path, 'event_time_points' + name_extension)
        np.save(time_series_file, self.event_time_series)
        np.save(time_points_file, self.event_time_points)

    @classmethod
    def load(cls, path, name_extension=''):
        """
        Load event_time_series and event_time_points from file.
        Adapted from an unpopular answer of
        https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-implement-multiple-constructors

        Parameters
        ----------
        path : str
            Directory where the files shall be stored.
        name_extension : str
            Optional file name extension.

        Returns
        -------
        obj : src.emissions.Emissions
            Instance of Emissions constructed with existing data.
        """
        obj = cls.__new__(cls)
        obj.event_time_series = np.load(os.path.join(path, 'event_time_series' + name_extension + '.npy'))
        obj.event_time_points = np.load(os.path.join(path, 'event_time_points' + name_extension + '.npy'))
        obj.simulation = 'This attribute has no meaning when instance constructed via load().'
        obj.emission_indices = 'This attribute has no meaning when instance constructed via load().'
        obj.event_indices = 'This attribute has no meaning when instance constructed via load().'
        return obj


def plot_histogram(data, density=True, display_mean=False, **kwargs):
    """
    Plot histogram of events.

    Parameters
    ----------
    data : 1-D array_like
    density : bool
        Whether to display the histogram as probability densities. Else, probabilities.
    display_mean : bool
        Whether to display the mean inside the plot. The unit corresponds to the unit of the x-axis.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault('type_', 'hist')
    kwargs.setdefault('title', 'intensity distribution')
    kwargs.setdefault('xlabel', 'photon count')
    kwargs.setdefault('fontsize', 16)
    if density:
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)
    else:
        kwargs.setdefault('ylabel', 'Pr')
        kwargs.setdefault('weights', np.ones_like(data) / data.size)

    axes = fi.universal_figure(data=data, **kwargs)

    if display_mean:
        mean = np.mean(data)
        axes[0][0].text(x=0.3, y=0.85, s=fr"$\mu = {mean:.2f}$", transform=axes[0][0].transAxes,
                        fontsize=kwargs['fontsize'])

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

    axes = fi.universal_figure(data=data, **kwargs)

    return axes


def plot_cumulative_events(data, **kwargs):
    """
    Plot cumulative events versus time.

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
    kwargs.setdefault('title', 'cum. events')
    kwargs.setdefault('xlabel', 'time [s]')
    kwargs.setdefault('ylabel', '%')
    kwargs.setdefault('ylim', [0, 100])

    axes = fi.universal_figure(data=data, **kwargs)

    return axes
