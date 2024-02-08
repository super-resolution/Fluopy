"""
Module emissions
"""
import numpy as np
import pandas as pd
from scipy.stats import geom, norm, poisson, gamma
import src.figure as fi
from src.simulation import simulate_experiment
import os


class Emissions:
    """
    Container for emission-associated attributes.

    Attributes
    ----------
    event_time_points : 1-D array_like
        The time points at which emissions are detected.
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and the number of events (i.e.,
        detected emissions) as values.
    """
    def __init__(self, photon_collection_rate, frame_time, emccd_gain, seed):
        """
        Parameters
        ----------
        photon_collection_rate : float
            Number between 0 and 1. Dictates the fraction of kept indices of emissions.
        frame_time : str
            For possible input values, see https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
        emccd_gain : None, int
            The gain of an EMCCD camera.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        """
        self.parameters = {'photon_collection_rate': photon_collection_rate, 'frame_time': frame_time, 
                           'emccd_gain': emccd_gain, 'seed': seed}
        self.event_time_points = None
        self.event_time_series = None
    
    def extract(self, simulation):
        """
        Extracts events from a simulation.

        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container for simulation-associated attributes.
        """
        if simulation.transition_series is None:
            raise ValueError('emissions not available if simulation has not been run.')
        emission_indices = self.get_emission_indices(simulation)
        event_indices = self.get_event_indices(emission_indices=emission_indices, 
                                                    photon_collection_rate=self.parameters['photon_collection_rate'], 
                                                    seed=self.parameters['seed'])
        self.event_time_points = simulation.time_series[event_indices + 1]
        self.event_time_series = self.construct_event_time_series(simulation, resample=self.parameters['frame_time'],
                                                                  emccd_gain=self.parameters['emccd_gain'], 
                                                                  seed=self.parameters['seed'])

    def simulate(self, transitions, start_at=None, size=1e5, frames=10, seed=None, 
                 store_time_points=False):
        """
        Simulates events per time.

        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        start_at : None, tuple
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible SingleState values. See
            transitions.single_states.
        size : int
            Size of random_numbers drawn at once.
        frames : int
            Total number of frames to be simulated.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        store_time_points : bool
            Whether to also create an array which contains the time points at which photons are detected.
        """
        if start_at is None:
            start_at = tuple(np.zeros(shape=transitions.fluorophore_system.count, dtype=int))
        elif len(start_at) != transitions.fluorophore_system.count:
            raise ValueError("The number of starting states doesn't match the number of fluorophores.")
        size = int(size)
        df = transitions.combined_state_transitions_df
        emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
        start_index = df[df['final_state'] == start_at].index[0]
        self.event_time_points, self.event_time_series = \
            simulate_experiment(transition_matrix=transitions.transition_matrix, row_sums=transitions.row_sums,
                                emitting_transition_ids=emitting_transition_ids, start_index=start_index,
                                size=size, frames=frames, frame_time=self.parameters['frame_time'], seed=seed, 
                                store_time_points=store_time_points)
    
    def get_emission_indices(self, simulation):
        """
        Get indices to apply to simulation.transition_series to yield emitting transitions.

        Returns
        -------
        emission_indices : 1-D array_like
            Indices of emitting transitions to apply to simulation.transition_series.
        """
        df = simulation.transitions.combined_state_transitions_df
        emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
        emission_indices = np.in1d(simulation.transition_series, emitting_transition_ids).nonzero()[0]

        return emission_indices

    def get_event_indices(self, emission_indices, photon_collection_rate=1, seed=None):
        """
        Alters emission_indices keeping only a relative number (photon_collection_rate) of randomly selected indices.
        Experimentally, this can represent a combination of photon collection and transmission efficiency by the
        microscope or quantum efficiency of the EMCCD.

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

        amount_not_detected = round((1 - photon_collection_rate) * emission_indices.shape[0])
        no_events_indices = rng.choice(np.arange(0, emission_indices.shape[0]), size=amount_not_detected,
                                       replace=False)
        event_indices = np.delete(emission_indices, no_events_indices)

        return event_indices

    def construct_event_time_series(self, simulation, resample='5ms', emccd_gain=None, seed=None):
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
        if event_time_points[-1] != simulation.time_series[-1]:
            added_end_time = True
            event_time_points = np.append(event_time_points, simulation.time_series[-1])

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
    

    def add_emccd_gain(self, emccd_gain, seed):
        rng = np.random.default_rng(seed)
        self.event_time_series.values[:] = gamma.rvs(a=self.event_time_series.values, 
                                                     scale=emccd_gain, random_state=rng)


    def add_gaussian_noise(self, mean, std, seed):
        """
        Add artificial noise to the events. The noise is normal distributed and can represent
        readout noise (insigificant in the case of EMCCD) or background. 

        Parameters
        ----------
        mean : float
            Mean of the normal distributed artificial noise.
        std : float
            Standard deviation of the normal distributed artificial noise.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size
        variates = norm(mean, std).rvs(size, random_state=rng)
        self.event_time_series = self.event_time_series + variates


    def add_poisson_noise(self, rate, seed):
        """
        Add artifical noise to the events. The noise is Poisson distributed and can represent
        dark current noise.

        Parameters
        ----------
        rate : float
            Rate of the Poisson distributed artificial noise.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size
        variates = poisson(rate).rvs(size, random_state=rng)
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
