"""
Work with observable photon emission time series.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import binom, gamma, norm, poisson

from . import figure as fi
from .fluorophores import Fluorophore
from .simulation import (
    Simulation,
    eval_floating_point_precision_error,
    simulate_experiment,
)
from .simulation_tcspc import simulate_TCSPC, simulate_TCSPC_detailed
from .transitions import TransitionSet

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.fluopy_types import RandomGeneratorSeed


__all__: list[str] = ["Emissions"]

logger = logging.getLogger(__name__)


class Emissions:
    """
    Container for emission-associated attributes.

    Attributes
    ----------
    parameters : dict[str, Any]
        Contains the parameters with which the instance was initialized.
    event_time_points : 1-D array_like
        The time points at which emissions are happening - they may be filtered by
        bandpass, however it is not considered whether they actually end up being
        detected and converted to an electrical voltage.
        None until extract(), simulate() or tcspc() has been called.
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and
        the number of events as values. Depending on the applied functions events may
        represent emissions effected by obstacles introduced by the optical path.
        None until extract(), simulate() or tcspc() has been called.
    """

    def __init__(
        self,
        frame_time: str = "5ms",
        bandpass: tuple[float, float] | None = None,
        seed: RandomGeneratorSeed = None,
    ) -> None:
        """
        Parameters
        ----------
        frame_time
            For possible input values, see
            https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
        bandpass
            The lowest and highest emission wavelength to be passed by the bandpass
            filter.
        seed
            A seed to initialize the BitGenerator.
        """
        self.parameters = {"frame_time": frame_time, "seed": seed, "bandpass": bandpass}
        self.event_time_points = None
        self.event_time_series = None

    def extract(self, simulation: Simulation) -> None:
        """
        Extracts events from a simulation. The events may be filtered by bandpass.

        Parameters
        ----------
        simulation
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
            simulation=simulation,
            resample=self.parameters["frame_time"],
        )

    def simulate(
        self,
        transition_set: TransitionSet,
        start_at: tuple[int, ...] | None = None,
        size: int = 1e5,
        frames: int = 10,
        store_time_points: bool = False,
    ) -> None:
        """
        Simulates events per time.

        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.
        start_at
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible
            SingleState values. See transition_set.single_states.
        size
            Size of random_numbers drawn at once.
        frames
            Total number of frames to be simulated.
        store_time_points
            Whether to also create an array which contains the time points at which
            photons are detected.

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
        emitting_transition_ids = get_emitting_transition_ids(
            bandpass=self.parameters["bandpass"], transition_set=transition_set
        )
        df = transition_set.combined_state_transitions_df
        start_index = df[df["final_state"] == start_at].index[0]
        self.event_time_points, self.event_time_series = simulate_experiment(
            transition_matrix=transition_set.transition_matrix,
            row_sums=transition_set.row_sums,
            emitting_transition_ids=emitting_transition_ids,
            start_index=start_index,
            size=size,
            frames=frames,
            frame_time=self.parameters["frame_time"],
            store_time_points=store_time_points,
            seed=self.parameters["seed"],
        )

    def tcspc(
        self,
        transition_set: TransitionSet,
        number_pulses: int = 1e4,
        pulse_duration: float = 5e-11,
        time_between_pulses: float = 1e-7,
        excitation_rates: dict[str, float] | None = None,
        size: int = 1e5,
        store_time_points: bool = False,
        details: bool = False,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        Simulation | None,
    ]:
        """
        Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence
        lifetime measurements). The return value lifetimes_DA contains the S1 durations
        of detected emissions when energy transfer is available. This does not
        discriminate between the number or kind of energy transfers. Note that if energy
        transfer is available, the emitting fluorophore could have been the donor even
        if other potential donors exist, because all implemented energy transfers have
        S1 as the donor (e.g., S1|S1|S0 goes to S0|S1|S0 --> S0 potential acceptor,
        first S1 emitted, both S1 could have been donors).
        Also note that energy transfer may have become available during the S1 duration
        of the emitting fluorophore. Also note that the S1 durations are the time
        differences of photon emission to last laser pulse.
        For processes other than S0 excitation that are also dependent on the
        irradiance, the given rates should correspond to the mean irradiance. They will
        not be adjusted to pulsed excitation.

        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.
        number_pulses
            Number of pulses to be simulated.
        pulse_duration
            The duration of a laser pulse in s. This time is used to calculate the
            probability of excitation, other than that it is neglected.
        time_between_pulses
            Time between two pulses in seconds.
        excitation_rates
            Contains the fluorophore names as keys and the excitation rates as values.
            Assumes uniform irradiance over the pulse duration.
            If None, the irradiance used for the excitation rates in transition_set is
            assumed to be the mean irradiance of pulse and no pulse duration.
        size
            Size of random_numbers drawn at once.
        store_time_points
            Whether to store the time points at which photons are detected.
        details
            Whether to additionally return a simulation object.

        Returns
        -------
        lifetimes_DA : 1-D array_like
            Contains the S1 durations of detected emissions when energy
            transfer available.
        lifetimes_D : 1-D array_like
            Contains the S1 durations of detected emissions when energy
            transfer not available.
        lifetimes_all : 1-D array_like
            Contains the S1 durations of all detected emissions.
        simulation_object : fluopy.simulation.Simulation
            Container for simulation-associated attributes and methods. Only returned if
            details is True.
        """
        df = transition_set.transition_df
        exc = [j for _, j in df.index if df.loc[(_, j), "abbreviation"] == "EXC"]
        transition_set = transition_set.adjust_rates(
            change_dict={identity: 0 for identity in exc}, keep_zero_rates=True
        )
        transition_set.finalize()

        if excitation_rates is None:
            logger.warning(
                "The irradiance used initially for excitation rates in\n"
                " transition_set is now assumed to be the mean irradiance of\n"
                " pulse and no pulse duration.",
                stacklevel=2,
            )
            # This assumes that the irradiance used for the excitation rates in
            # transition_set is the mean irradiance of pulse and no pulse duration.
            factor_excitation_rate = time_between_pulses / pulse_duration
            excitation_rates = {}
            for f in transition_set.fluorophore_system.fluorophores:
                exc_rate = df.loc[f.name][df.loc[f.name]["abbreviation"] == "EXC"][
                    "rate"
                ].values[0]
                excitation_rates[f.name] = exc_rate * factor_excitation_rate
        emitting_transition_ids = get_emitting_transition_ids(
            bandpass=self.parameters["bandpass"], transition_set=transition_set
        )
        emit_ids_list = list(emitting_transition_ids.keys())
        df = transition_set.combined_state_transitions_df
        # if fluorophore_ids length is greater than 1, it is an energy transfer
        et_initial_states = (
            df["initial_state"][df["fluorophore_ids"].apply(len) > 1]
        ).values
        # if the initial state is in et_initial_states, the fluorescence occurred
        # while energy transfer was also an option
        et_transition_ids = df.iloc[emit_ids_list][
            df.iloc[emit_ids_list]["initial_state"].isin(et_initial_states)
        ].index.to_numpy()
        if details:
            func = simulate_TCSPC_detailed
            eval_floating_point_precision_error(
                transition_set=transition_set,
                largest_number=number_pulses * time_between_pulses,
            )
        else:
            func = simulate_TCSPC
        return_values = func(
            transition_set=transition_set,
            emitting_transition_ids=emitting_transition_ids,
            et_transition_ids=et_transition_ids,
            number_pulses=number_pulses,
            pulse_duration=pulse_duration,
            time_between_pulses=time_between_pulses,
            excitation_rates=excitation_rates,
            frame_time=self.parameters["frame_time"],
            size=size,
            store_time_points=store_time_points,
            seed=self.parameters["seed"],
        )
        self.event_time_series = return_values[0]
        self.event_time_points = return_values[1]
        lifetimes_DA = return_values[2]
        lifetimes_D = return_values[3]
        lifetimes_all = return_values[4]
        if details:
            return lifetimes_DA, lifetimes_D, lifetimes_all, return_values[5]
        else:
            return lifetimes_DA, lifetimes_D, lifetimes_all, None

    def get_emission_indices(
        self,
        simulation: Simulation,
        bandpass: tuple[float, float] | None,
        seed: RandomGeneratorSeed,
    ) -> npt.NDArray[np.int64]:
        """
        Get indices to apply to simulation.transition_series to yield (detected)
        emitting transitions.

        Parameters
        ----------
        simulation
            Container of simulation-associated attributes and methods.
        bandpass
            The lowest and highest emission wavelength to be passed by the bandpass
            filter.
        seed
            A seed to initialize the BitGenerator.

        Returns
        -------
        emission_indices : npt.NDArray[np.int64]
            Indices of emitting transitions to apply to simulation.transition_series.
        """
        if bandpass is not None:
            rng = np.random.default_rng(seed)
            processed = []
            collect_emission_indices = []
            data_dir = Path(__file__).parent / "fluorophore_spectra"

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
                    emission_indices_f = np.isin(
                        simulation.transition_series, emitting_transition_ids_f
                    ).nonzero()[0]
                    amount_not_detected = binom.rvs(
                        n=emission_indices_f.size, p=p_not_passed, random_state=rng
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
            emitting_transition_ids = df.loc[df["photon"]].index.to_numpy()
            emission_indices = np.isin(
                simulation.transition_series, emitting_transition_ids
            ).nonzero()[0]

        return emission_indices

    def construct_event_time_series(
        self, simulation: Simulation, resample: str = "5ms"
    ) -> None:
        """
        Counts events within a time interval (resample).

        Parameters
        ----------
        simulation
            Container of simulation-associated attributes and methods.
        resample
            For possible input values, see https://pandas.pydata.org/docs/user_guide/
            timeseries.html -> Offset aliases.

        Returns
        -------
        None
        """
        event_time_points = np.insert(arr=self.event_time_points, obj=0, values=0)

        added_end_time = False
        if event_time_points[-1] != simulation.time_series[-1]:
            added_end_time = True
            event_time_points = np.append(event_time_points, simulation.time_series[-1])

        time_deltas = pd.to_timedelta(event_time_points, unit="s")
        events = np.ones(shape=event_time_points.shape[0])
        events[0] = 0

        if added_end_time:
            events[-1] = 0

        event_time_series = pd.Series(events, index=time_deltas, dtype=np.int32)
        event_time_series_r = event_time_series.resample(
            resample, closed="right", label="right"
        ).sum()
        if (
            event_time_series_r.index[-1] > event_time_series.index[-1]
            and event_time_series_r.values[-1] == 0
        ):
            event_time_series_r = event_time_series_r.drop(
                event_time_series_r.index[-1]
            )
        time_deltas = event_time_series_r.index
        in_seconds = time_deltas / np.timedelta64(1, "s")
        in_seconds = np.round(in_seconds, decimals=12)
        event_time_series_r.index = in_seconds

        self.event_time_series = event_time_series_r

    def add_photon_collection_objective(
        self, p: float, seed: RandomGeneratorSeed = None
    ) -> None:
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

    def add_quantum_efficiency(
        self, p: float, seed: RandomGeneratorSeed = None
    ) -> None:
        """
        Adds the effect of quantum efficiency of the EMCCD.

        Parameters
        ----------
        p
            Between 0 and 1. Quantum efficiency of the EMCCD.
        seed
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

    def add_transmittance(self, p: float, seed: RandomGeneratorSeed = None) -> None:
        """
        Adds the effect of transmittance of a component of the optical path.

        Parameters
        ----------
        p
            Between 0 and 1. Transmittance of the component.
        seed
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

    def add_emccd_gain(
        self, emccd_gain: float, seed: RandomGeneratorSeed = None
    ) -> None:
        """
        Add the effect of the gain of the EMCCD.

        Parameters
        ----------
        emccd_gain
            The gain of an EMCCD.
        seed
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

    def add_gaussian_noise(
        self, mean: float, std: float, seed: RandomGeneratorSeed = None
    ) -> None:
        """
        Add artificial noise to the events. The noise is normal distributed and can
        represent readout noise (insignificant in the case of EMCCD).

        Parameters
        ----------
        mean
            Mean of the normal distributed artificial noise.
        std
            Standard deviation of the normal distributed artificial noise.
        seed
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size - 1
        variates = norm(loc=mean, scale=std).rvs(size, random_state=rng)
        variates = variates.astype(np.int32)
        self.event_time_series.values[1:] = self.event_time_series.values[1:] + variates
        self.event_time_series.clip(lower=0, inplace=True)

    def add_poisson_noise(self, rate: float, seed: RandomGeneratorSeed = None) -> None:
        """
        Add Poisson noise to the events. The noise is Poisson distributed and can
        represent dark current noise.

        Parameters
        ----------
        rate
            Rate of the Poisson distributed artificial noise.
        seed
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        size = self.event_time_series.size - 1
        variates = poisson(rate).rvs(size, random_state=rng)
        variates = variates.astype(np.int32)
        self.event_time_series.values[1:] = self.event_time_series.values[1:] + variates

    def apply_threshold(self, threshold: int) -> None:
        """
        Apply a threshold to the events. All events below the threshold are set to 0.

        Parameters
        ----------
        threshold : int
            The minimum value of events to be considered.

        Returns
        -------
        None
        """
        self.event_time_series[self.event_time_series < threshold] = 0

    def plot_cumulative_events(self, **kwargs: Any) -> npt.NDArray[mplAxes]:
        """
        Plot cumulative events versus time.

        Parameters
        ----------
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        cum_events = self.event_time_series.cumsum()
        cum_events = cum_events / cum_events.max()
        data = [self.event_time_series.index, cum_events.values]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xlabel", "Photon arrival time (s)")
        kwargs.setdefault("ylabel", "Cumulative prob.")
        kwargs.setdefault("ylim", [0, 1])

        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_histogram(
        self,
        density: bool = True,
        display_mean: bool = False,
        include_0: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray[mplAxes]:
        """
        Plot histogram of events.

        Parameters
        ----------
        density
            Whether to display the histogram as probability densities. Else,
            probabilities.
        display_mean
            Whether to display the mean inside the plot. The unit corresponds to the
            unit of the x-axis.
        include_0
            Whether to include counts of 0 events.
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        data = self.event_time_series
        if not include_0:
            data = data[data != 0]

        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("xlabel", r"$\frac{photons}{frame}$")
        if density:
            kwargs.setdefault("ylabel", "Prob. density")
            kwargs.setdefault("density", True)
        else:
            kwargs.setdefault("ylabel", "Probability")
            kwargs.setdefault("weights", np.ones_like(data) / data.size)

        axes = fi.universal_figure(data=data, **kwargs)

        mean_color = kwargs.get("ylabelcolor", "black")
        fontsize = kwargs.get("fontsize", 16)
        if display_mean:
            mean = np.mean(data)
            axes[0][0].text(
                x=0.3,
                y=0.85,
                s=rf"$\mu = {mean:.2f}$",
                transform=axes[0][0].transAxes,
                fontsize=fontsize,
                color=mean_color,
            )

        return axes

    def plot_time_series(self, **kwargs: Any) -> npt.NDArray[mplAxes]:
        """
        Plot time series of events.

        Parameters
        ----------
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        data = [self.event_time_series.index, self.event_time_series.values]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xlabel", "Time (s)")
        kwargs.setdefault("ylabel", r"$\frac{photons}{frame}$")

        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def save(self, path: str | Path, name_extension: str = "") -> None:
        """
        Saves event_time_series and event_time_points to a file.

        Parameters
        ----------
        path
            Directory where the files shall be stored.
        name_extension
            Optional file name extension.

        Returns
        -------
        None
        """
        time_series_file = Path(path) / ("event_time_series" + name_extension + ".csv")
        time_points_file = Path(path) / ("event_time_points" + name_extension + ".npy")
        self.event_time_series.to_csv(time_series_file, header=False)
        np.save(time_points_file, self.event_time_points)

    @classmethod
    def load(cls, path: str | Path, name_extension: str = "") -> Emissions:
        """
        Load event_time_series and event_time_points from file.
        Adapted from an unpopular answer of
        https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-
        implement-multiple-constructors

        Parameters
        ----------
        path
            Directory where the files are stored.
        name_extension
            Optional file name extension.

        Returns
        -------
        obj : fluopy.emissions.Emissions
            Instance of Emissions constructed with existing data.
        """
        obj = cls.__new__(cls)
        obj.event_time_series = pd.read_csv(
            Path(path) / ("event_time_series" + name_extension + ".csv"),
            index_col=0,
            header=None,
        )
        obj.event_time_series = pd.Series(
            obj.event_time_series.values.flatten(), index=obj.event_time_series.index
        )
        obj.event_time_series.index.name = None
        obj.event_time_points = np.load(
            Path(path) / ("event_time_points" + name_extension + ".npy"),
            allow_pickle=True,
        )

        return obj


def get_p_filter(
    data_dir: str | Path,
    fluorophore: Fluorophore,
    bandpass: tuple[float, float],
) -> float:
    """
    Get the probability of a photon emitted by fluorophore passing the bandpass filter.

    Parameters
    ----------
    data_dir
        The directory of data files of fluorophores.
    fluorophore
        Contains attributes of a fluorophore.
    bandpass
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
        Path(data_dir) / fluorophore.constants.data_files / "emission.csv"
    )

    minimum_wavelength = 200

    emissions = emission_data["y"]
    bandpass_low = bandpass[0] - minimum_wavelength
    bandpass_high = bandpass_low + (bandpass[1] - bandpass[0])
    rel_emission = emissions[bandpass_low:bandpass_high] / emissions.sum()
    p_passed = rel_emission.sum()

    return p_passed


def get_emitting_transition_ids(
    bandpass: tuple[float, float], transition_set: TransitionSet
) -> dict[int, float]:
    """
    Get a dictionary with ids of emitting transitions as keys and probabilities of
    passing the bandpass filter as values. If bandpass is None, all emitting transitions
    are returned with a probability of 1.

    Parameters
    ----------
    bandpass
        The lowest and highest emission wavelength to be passed by the bandpass filter.
        If bandpass is None, all emitting transitions are returned with a probability of
        1.
    transition_set
        Collection of all relevant transitions and related attributes.

    Returns
    -------
    emitting_transition_ids : dict[int, float]
        Dictionary with ids of emitting transitions as keys and probabilities of passing
        the bandpass filter as values.
        The ids correspond to transition_set.combined_state_transitions_df.
    """
    emitting_transition_ids = {}
    if bandpass is not None:
        processed = []
        data_dir = Path(__file__).parent / "fluorophore_spectra"
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
                    bandpass=bandpass,
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
        emitting_transition_ids_ = df.loc[df["photon"]].index.to_numpy()
        emitting_transition_ids = {identity: 1 for identity in emitting_transition_ids_}

    return emitting_transition_ids
