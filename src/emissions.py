"""
Module emissions
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, poisson, gamma, binom
import src.figure as fi
from src.simulation import simulate_experiment


class Emissions:
    """
    Container for emission-associated attributes.

    Attributes
    ----------
    parameters : dict
        Contains the parameters with which the instance was initialized.
    event_time_points : 1-D array_like
        The time points at which emissions are happening - they may be filtered by
        bandpass, however it is not considered whether they actually end up being
        detected and converted to an electrical voltage.
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and
        the number of events as values. Depending on the applied functions events may
        represent emissions effected by obstacles introduced by the optical path.
    """

    def __init__(self, frame_time="5ms", bandpass=None, seed=1):
        """
        Parameters
        ----------
        frame_time : str
            For possible input values, see
            https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
        bandpass : tuple
            The lowest and highest emission wavelength to be passed by the bandpass
            filter.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        """
        self.parameters = {"frame_time": frame_time, "seed": seed, "bandpass": bandpass}
        self.event_time_points = None
        self.event_time_series = None

    def extract(self, simulation):
        """
        Extracts events from a simulation. The events may be filtered by bandpass.

        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container for simulation-associated attributes.

        Returns
        -------
        None
        """
        if simulation.transition_series is None:
            raise ValueError("emissions not available if simulation has not been run.")
        emission_indices = self.get_emission_indices(
            simulation=simulation,
            bandpass=self.parameters["bandpass"],
            seed=self.parameters["seed"],
        )
        self.event_time_points = simulation.time_series[emission_indices + 1]
        self.construct_event_time_series(
            simulation,
            resample=self.parameters["frame_time"],
        )

    def simulate(
        self,
        transition_set,
        start_at=None,
        size=1e5,
        frames=10,
        store_time_points=False,
        seed=None,
        triplet_1=None,
        triplet_2=None,
        triplet_3=None,
        triplet_4=None,
    ):
        """
        Simulates events per time.

        Parameters
        ----------
        transition_set : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        start_at : None, tuple
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible
            SingleState values. See transition_set.single_states.
        size : int
            Size of random_numbers drawn at once.
        frames : int
            Total number of frames to be simulated.
        store_time_points : bool
            Whether to also create an array which contains the time points at which
            photons are detected.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if start_at is None:
            start_at = tuple(
                np.zeros(shape=transition_set.fluorophore_system.count, dtype=int)
            )
        elif len(start_at) != transition_set.fluorophore_system.count:
            raise ValueError(
                "The number of starting states doesn't match the number of "
                "fluorophores."
            )
        size = int(size)
        emitting_transition_ids = {}
        if self.parameters["bandpass"] is not None:
            processed = []
            data_dir = os.path.join(Path(__file__).parent, "fluorophore_collection")
            for fluorophore in transition_set.fluorophore_system.fluorophores:
                if fluorophore.constants is None:
                    raise ValueError(
                        "bandpass not None but emission data not available for "
                        f"this kind of fluorophore: {fluorophore.name}"
                    )
                if fluorophore.name not in processed:
                    p_passed = get_p_filter(
                        data_dir=data_dir,
                        fluorophore=fluorophore,
                        bandpass=self.parameters["bandpass"],
                    )
                    sub_df = transition_set.transition_df.loc[fluorophore.name]
                    emitting_transitions_f = sub_df[sub_df["photon"]].index.to_numpy()
                    df = transition_set.combined_state_transitions_df
                    emitting_transition_ids_f = df[
                        df["transition_id"].isin(emitting_transitions_f)
                    ].index.to_numpy()
                    for emitting_transition_id in emitting_transition_ids_f:
                        emitting_transition_ids[emitting_transition_id] = p_passed
        else:
            df = transition_set.combined_state_transitions_df
            emitting_transition_ids_ = df.loc[df["photon"] == True].index.to_numpy()
            emitting_transition_ids = {
                identity: 1 for identity in emitting_transition_ids_
            }
        start_index = df[df["final_state"] == start_at].index[0]
        self.event_time_points, self.event_time_series, durations = (
            simulate_experiment(
                transition_matrix=transition_set.transition_matrix,
                row_sums=transition_set.row_sums,
                emitting_transition_ids=emitting_transition_ids,
                start_index=start_index,
                size=size,
                frames=frames,
                frame_time=self.parameters["frame_time"],
                store_time_points=store_time_points,
                seed=seed,
                triplet_1=triplet_1,
                triplet_2=triplet_2,
                triplet_3=triplet_3,
                triplet_4=triplet_4,
            )
        )
        print(f'{np.mean(durations):.2e}')
        print(len(durations))
        return durations


    def get_emission_indices(self, simulation, bandpass, seed):
        """
        Get indices to apply to simulation.transition_series to yield emitting
        transitions.

        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container of simulation-associated attributes and methods.
        bandpass : tuple
            The lowest and highest emission wavelength to be passed by the bandpass
            filter.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        emission_indices : 1-D array_like
            Indices of emitting transitions to apply to simulation.transition_series.
        """
        if bandpass is not None:
            rng = np.random.default_rng(seed)
            processed = []
            collect_emission_indices = []
            data_dir = os.path.join(Path(__file__).parent, "fluorophore_collection")

            for (
                fluorophore
            ) in simulation.transition_set.fluorophore_system.fluorophores:
                if fluorophore.constants is None:
                    raise ValueError(
                        "bandpass not None but emission data not available for "
                        f"this kind of fluorophore: {fluorophore.name}"
                    )
                if fluorophore.name not in processed:
                    p_passed = get_p_filter(
                        data_dir=data_dir, fluorophore=fluorophore, bandpass=bandpass
                    )
                    p_not_passed = 1 - p_passed
                    sub_df = simulation.transition_set.transition_df.loc[
                        fluorophore.name
                    ]
                    emitting_transitions_f = sub_df[sub_df["photon"]].index.to_numpy()
                    df = simulation.transition_set.combined_state_transitions_df
                    emitting_transition_ids_f = df[
                        df["transition_id"].isin(emitting_transitions_f)
                    ].index.to_numpy()
                    emission_indices_f = np.in1d(
                        simulation.transition_series, emitting_transition_ids_f
                    ).nonzero()[0]
                    amount_not_detected = binom.rvs(
                        emission_indices_f.size, p_not_passed, random_state=rng
                    )
                    not_detected_indices = rng.choice(
                        np.arange(0, emission_indices_f.size),
                        size=amount_not_detected,
                        replace=False,
                    )
                    filtered_emission_indices_f = np.delete(
                        emission_indices_f, not_detected_indices
                    )
                    collect_emission_indices.append(filtered_emission_indices_f)
                    processed.append(fluorophore.name)
            emission_indices = np.concatenate(collect_emission_indices)
        else:
            df = simulation.transition_set.combined_state_transitions_df
            emitting_transition_ids = df.loc[df["photon"] == True].index.to_numpy()
            emission_indices = np.in1d(
                simulation.transition_series, emitting_transition_ids
            ).nonzero()[0]

        return emission_indices

    def construct_event_time_series(self, simulation, resample="5ms"):
        """
        Counts events within a time interval (resample).

        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container of simulation-associated attributes and methods.
        resample : str
            For possible input values, see https://pandas.pydata.org/docs/user_guide/
            timeseries.html -> Offset aliases.

        Returns
        -------
        None
        """
        event_time_points = np.insert(self.event_time_points, 0, 0)

        added_end_time = False
        if event_time_points[-1] != simulation.time_series[-1]:
            added_end_time = True
            event_time_points = np.append(event_time_points, simulation.time_series[-1])

        time_deltas = pd.to_timedelta(event_time_points, unit="s")
        events = np.ones(shape=event_time_points.shape[0])
        events[0] = 0

        if added_end_time:
            events[-1] = 0

        event_time_series = pd.Series(events, index=time_deltas, dtype=np.int64)
        event_time_series = event_time_series.resample(resample).sum()
        time_deltas = event_time_series.index
        in_seconds = time_deltas / np.timedelta64(1, "s")
        event_time_series.index = in_seconds

        self.event_time_series = event_time_series

    def add_photon_collection_objective(self, p, seed=None):
        """
        Adds the effect of photon collection of the objective.

        Parameters
        ----------
        p : float
            Between 0 and 1. Probability of photons being collected.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if p > 1 or p < 0:
            raise ValueError("p has to be between 0 and 1.")
        rng = np.random.default_rng(seed)
        nonzero = self.event_time_series.values.nonzero()
        self.event_time_series.values[nonzero] = binom.rvs(
            n=self.event_time_series.values[nonzero], p=p, random_state=rng
        )

    def add_quantum_efficiency(self, p, seed=None):
        """
        Adds the effect of quantum efficiency of the EMCCD.

        Parameters
        ----------
        p : float
            Between 0 and 1. Quantum efficiency of the EMCCD.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if p > 1 or p < 0:
            raise ValueError("p has to be between 0 and 1.")
        rng = np.random.default_rng(seed)
        nonzero = self.event_time_series.values.nonzero()
        self.event_time_series.values[nonzero] = binom.rvs(
            n=self.event_time_series.values[nonzero], p=p, random_state=rng
        )

    def add_transmittance(self, p, seed=None):
        """
        Adds the effect of transmittance of a component of the optical path.

        Parameters
        ----------
        p : float
            Between 0 and 1. Transmittance of the component.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if p > 1 or p < 0:
            raise ValueError("p has to be between 0 and 1.")
        rng = np.random.default_rng(seed)
        nonzero = self.event_time_series.values.nonzero()
        self.event_time_series.values[nonzero] = binom.rvs(
            n=self.event_time_series.values[nonzero], p=p, random_state=rng
        )

    def add_emccd_gain(self, emccd_gain, seed=None):
        """
        Add the effect of the gain of the EMCCD.

        Parameters
        ----------
        emccd_gain : float
            The gain of an EMCCD.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        nonzero = self.event_time_series.values.nonzero()
        self.event_time_series.values[nonzero] = gamma.rvs(
            a=self.event_time_series.values[nonzero], scale=emccd_gain, random_state=rng
        )

    def add_gaussian_noise(self, mean, std, seed=None):
        """
        Add artificial noise to the events. The noise is normal distributed and can
        represent readout noise (insigificant in the case of EMCCD).

        Parameters
        ----------
        mean : float
            Mean of the normal distributed artificial noise.
        std : float
            Standard deviation of the normal distributed artificial noise.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size
        variates = norm(mean, std).rvs(size, random_state=rng)
        variates = variates.astype(int)
        self.event_time_series = self.event_time_series + variates

    def add_poisson_noise(self, rate, seed=None):
        """
        Add Poisson noise to the events. The noise is Poisson distributed and can
        represent dark current noise.

        Parameters
        ----------
        rate : float
            Rate of the Poisson distributed artificial noise.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size
        variates = poisson(rate).rvs(size, random_state=rng)
        self.event_time_series = self.event_time_series + variates

    def apply_threshold(self, threshold):
        """
        Apply a threshold to the events. All events below the threshold are set to 0.

        Parameters
        ----------
        threshold : int
            The minimum value of events to be considered.
        """
        self.event_time_series[self.event_time_series < threshold] = 0

    def plot_cumulative_events(self, **kwargs):
        """
        Plot cumulative events versus time.

        Parameters
        ----------
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        cum_events = self.event_time_series.cumsum()
        cum_events = cum_events / cum_events.max() * 100
        data = [self.event_time_series.index, cum_events.values]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("title", "cum. events")
        kwargs.setdefault("xlabel", "time [s]")
        kwargs.setdefault("ylabel", "%")
        kwargs.setdefault("ylim", [0, 100])

        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_histogram(
        self, density=True, display_mean=False, include_0=False, **kwargs
    ):
        """
        Plot histogram of events.

        Parameters
        ----------
        density : bool
            Whether to display the histogram as probability densities. Else,
            probabilities.
        display_mean : bool
            Whether to display the mean inside the plot. The unit corresponds to the
            unit of the x-axis.
        include_0 : bool
            Whether to include counts of 0 events.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        data = self.event_time_series
        if not include_0:
            data = data[data != 0]

        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("title", "intensity distribution")
        kwargs.setdefault("xlabel", "photon count")
        kwargs.setdefault("fontsize", 16)
        if density:
            kwargs.setdefault("ylabel", "PD")
            kwargs.setdefault("density", True)
        else:
            kwargs.setdefault("ylabel", "Pr")
            kwargs.setdefault("weights", np.ones_like(data) / data.size)

        axes = fi.universal_figure(data=data, **kwargs)

        mean_color = "black"
        if "ylabelcolor" in kwargs:
            mean_color = kwargs["ylabelcolor"]
        if display_mean:
            mean = np.mean(data)
            axes[0][0].text(
                x=0.3,
                y=0.85,
                s=rf"$\mu = {mean:.2f}$",
                transform=axes[0][0].transAxes,
                fontsize=kwargs["fontsize"],
                color=mean_color,
            )

        return axes

    def plot_time_series(self, **kwargs):
        """
        Plot time series of events.

        Parameters
        ----------
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        data = [self.event_time_series.index, self.event_time_series.values]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("title", "fluorescence trajectory")
        kwargs.setdefault("xlabel", "time [s]")
        kwargs.setdefault("ylabel", "photon count")

        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def save(self, path, name_extension=""):
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
        time_series_file = os.path.join(path, "event_time_series" + name_extension)
        time_points_file = os.path.join(path, "event_time_points" + name_extension)
        np.save(time_series_file, self.event_time_series)
        np.save(time_points_file, self.event_time_points)

    @classmethod
    def load(cls, path, name_extension=""):
        """
        Load event_time_series and event_time_points from file.
        Adapted from an unpopular answer of
        https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-
        implement-multiple-constructors

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
        obj.event_time_series = np.load(
            os.path.join(path, "event_time_series" + name_extension + ".npy")
        )
        obj.event_time_points = np.load(
            os.path.join(path, "event_time_points" + name_extension + ".npy")
        )
        return obj


def get_p_filter(data_dir, fluorophore, bandpass):
    """
    Get the probability of a photon emitted by fluorophore passing the bandpass filter.

    Parameters
    ----------
    data_dir : str
        The directory of data files of fluorophores.
    fluorophore : src.fluorophores.Fluorophore
        Contains attributes of a fluorophore.
    bandpass : tuple
        The lowest and highest emission wavelength to be passed by the bandpass filter.

    Returns
    -------
    p_passed : float
        The probability of a photon passing the bandpass filter.
    """

    if bandpass[0] < 200 or bandpass[0] > 1000:
        raise ValueError("The lower bandpass limit has to be between 200 and 1000 nm.")
    if bandpass[1] < 200 or bandpass[1] > 1000:
        raise ValueError("The upper bandpass limit has to be between 200 and 1000 nm.")
    if bandpass[0] >= bandpass[1]:
        raise ValueError(
            "The lower bandpass limit has to be smaller than the upper limit."
        )

    emission_data = pd.read_csv(
        os.path.join(data_dir, fluorophore.constants.data_files, "emission.csv")
    )

    minimum_wavelength = 200

    emissions = emission_data["y"]
    bandpass_low = bandpass[0] - minimum_wavelength
    bandpass_high = bandpass_low + (bandpass[1] - bandpass[0])
    rel_emission = emissions[bandpass_low:bandpass_high] / emissions.sum()
    p_passed = rel_emission.sum()

    return p_passed
