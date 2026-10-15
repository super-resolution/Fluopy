"""
Work with observable photon emission time series.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import gamma, norm, poisson

from . import plotting
from .fluo_data import Spectrum
from .simulation import (
    Simulation,
    eval_floating_point_precision_error,
    simulate_experiment,
    simulate_experiment_sparse,
)
from .tcspc import simulate_TCSPC, simulate_TCSPC_detailed
from .transitions import TransitionSet

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.fluopy_types import RandomGeneratorSeed


__all__: list[str] = ["DetectionChannel", "Emissions"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionChannel:
    """
    Define how emitted photons are detected in one named channel.

    Attributes
    ----------
    bandpass
        Filter transmission as a scalar, a Spectrum, or a tuple of the lowest and
        highest passed wavelength in nm. A tuple represents unit transmission inside
        the band and zero outside. The default 1.0 passes every wavelength.
    fluorophore_ids
        Physical fluorophore identifiers whose photons may enter the channel. If None,
        photons from every fluorophore may enter it. Restricting fluorophore_ids is
        primarily intended for simulations of fluorophore-specific collection paths.
        For ordinary wavelength-resolved detection, leave it as None because a photon
        may enter any channel whose bandpass accepts its wavelength, regardless of
        which fluorophore emitted it.
    detection_efficiency
        Probability that a photon passing the optical response is detected, as a
        scalar or a Spectrum. This can combine collection losses and detector quantum
        efficiency. Optical routing and filter transmission belong in
        routing_probability and bandpass, respectively.
    routing_probability
        Fraction of photons routed into the channel before detection losses. A scalar
        represents a constant beamsplitter fraction. A Spectrum represents a
        wavelength-dependent routing or optical transmission probability, with
        wavelengths in nm and values between 0 and 1. Values are linearly interpolated
        within the supplied wavelength range and are zero outside it. The response is
        multiplied by the bandpass transmission and detection_efficiency.

    Notes
    -----
    Optical responses of channels accepting photons from the same fluorophore must
    sum to at most 1 at each wavelength, before applying detection_efficiency.
    Each emitted photon can be detected in at most one channel. Multiple optical
    components can be represented by their combined response along each detection path.
    All scalar and spectral probabilities must be between 0 and 1. Spectra use nm,
    linear interpolation, and zero response outside their supplied wavelength range.
    Routing, filtering, and detection responses are multiplied before integration
    against the emission spectrum.

    Examples
    --------
    Two channels with identical bandpasses behind a 50/50 beamsplitter:

    >>> channels = {
    ...     "first": DetectionChannel((650, 700), routing_probability=0.5),
    ...     "second": DetectionChannel((650, 700), routing_probability=0.5),
    ... }

    A wavelength-dependent routing response:

    >>> response = Spectrum.from_arrays([600, 650, 700], [0, 0.5, 1])
    >>> channel = DetectionChannel(routing_probability=response)

    A measured filter response and detector efficiency behind a 50/50 beamsplitter:

    >>> transmission = Spectrum.from_arrays([600, 650, 700], [0, 0.9, 0])
    >>> efficiency = Spectrum.from_arrays([600, 650, 700], [0.6, 0.7, 0.5])
    >>> channel = DetectionChannel(
    ...     bandpass=transmission,
    ...     detection_efficiency=efficiency,
    ...     routing_probability=0.5,
    ... )
    """

    bandpass: float | Spectrum | tuple[float, float] = 1.0
    fluorophore_ids: frozenset[int] | None = None
    detection_efficiency: float | Spectrum = 1.0
    routing_probability: float | Spectrum = 1.0

    def __post_init__(self) -> None:
        """Validate the channel configuration."""
        if isinstance(self.bandpass, tuple):
            lower, upper = self.bandpass
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("bandpass limits must be finite.")
            if lower >= upper:
                raise ValueError(
                    "The lower bandpass limit has to be smaller than the upper limit."
                )
        responses = {
            "routing_probability": self.routing_probability,
            "detection_efficiency": self.detection_efficiency,
        }
        if not isinstance(self.bandpass, tuple):
            responses["bandpass"] = self.bandpass
        for name, response in responses.items():
            if isinstance(response, Spectrum):
                if np.any(response.values > 1):
                    raise ValueError(f"{name} spectrum values must be at most 1.")
            elif not isinstance(response, Real):
                raise TypeError(f"{name} must be a scalar or Spectrum.")
            elif not np.isfinite(response) or not 0 <= response <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1.")


def _get_paired_emission_ids(
    combined_transitions: pd.DataFrame,
    emission_ids: npt.NDArray[np.intp],
) -> list[int]:
    """
    Return emissions with an available paired transition for their active component.

    Parameters
    ----------
    combined_transitions
        Combined-state transitions containing initial states, rates, and fluorophore
        identities.
    emission_ids
        Positional indices of detectable emission transitions.

    Returns
    -------
    list[int]
        Combined-state emission indices whose emitting fluorophore is the first,
        active component of a positive-rate paired transition available from the same
        initial state.
    """
    active_paired_states = {
        (row["initial_state"], row["fluorophore_ids"][0])
        for _, row in combined_transitions.iterrows()
        if row["rate"] > 0 and len(row["fluorophore_ids"]) > 1
    }
    paired_emission_ids: list[int] = []
    for emission_id in emission_ids:
        identity = int(emission_id)
        emission = combined_transitions.iloc[identity]
        if (
            emission["initial_state"],
            emission["fluorophore_ids"][0],
        ) in active_paired_states:
            paired_emission_ids.append(identity)
    return paired_emission_ids


class Emissions:
    """
    Container for emission-associated attributes.

    Attributes
    ----------
    parameters
        Contains the parameters with which the instance was initialized.
    event_time_points
        Detected photon time points grouped by detection channel. Bandpass filtering and
        detection efficiency are represented consistently in event_time_points and
        event_time_series. Frame-based detector processing such as gain, noise, and
        thresholding modifies only event_time_series. None until extract() has been
        called, or until simulate() or tcspc() has been called with time-point storage
        enabled.
    event_time_series
        Contains the time points (increasing by a defined time interval) as index and
        one event-count column per detection channel. Gain, noise, and thresholding are
        frame-based operations and do not modify event_time_points. Internally generated
        series are indexed by the end time of each measured frame. None until extract(),
        simulate() or tcspc() has been called.
        Extraction from a Simulation includes only complete frames; photons in an
        incomplete final frame remain available in event_time_points.
    """

    def __init__(
        self,
        frame_time: str = "5ms",
        channels: Mapping[str, DetectionChannel] | None = None,
        seed: RandomGeneratorSeed = None,
    ) -> None:
        """
        Parameters
        ----------
        frame_time
            For possible input values, see
            https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
        channels
            Named, mutually exclusive detection channels.
        seed
            A seed to initialize the BitGenerator.
        """
        if channels is None:
            channels = {"all": DetectionChannel()}
        if not channels:
            raise ValueError("at least one detection channel is required.")
        if any(not isinstance(name, str) or not name for name in channels):
            raise ValueError("detection channel names must be non-empty strings.")
        if any(
            not isinstance(channel, DetectionChannel) for channel in channels.values()
        ):
            raise TypeError("channels must contain DetectionChannel values.")
        try:
            frame_duration = pd.Timedelta(frame_time)
        except (TypeError, ValueError):
            raise ValueError("frame_time must be a positive duration.") from None
        if pd.isna(frame_duration) or frame_duration <= pd.Timedelta(0):
            raise ValueError("frame_time must be a positive duration.")

        self.channels = dict(channels)
        self.parameters: dict[str, Any] = {
            "frame_time": frame_time,
            "seed": seed,
            "channels": self.channels,
        }
        self.event_time_points: dict[str, npt.NDArray[np.float64]] | None = None
        self.event_time_series: pd.DataFrame | None = None

    def _require_event_time_series(self) -> pd.DataFrame:
        """
        Return event_time_series after checking that it is available.

        Returns
        -------
        pd.DataFrame
            Frame-based signal with one column per detection channel.

        Raises
        ------
        ValueError
            If event_time_series is unavailable.
        """
        if self.event_time_series is None:
            raise ValueError("event time series is unavailable.")
        return self.event_time_series

    def _require_event_time_points(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Return event_time_points after checking that they are available.

        Returns
        -------
        dict[str, npt.NDArray[np.float64]]
            Detected photon arrival times in seconds, grouped by detection channel.

        Raises
        ------
        ValueError
            If event_time_points is unavailable.
        """
        if self.event_time_points is None:
            raise ValueError("event time points are unavailable.")
        return self.event_time_points

    def resolve_channel(self, channel: str | None = None) -> str:
        """
        Resolve an optional channel name.

        Parameters
        ----------
        channel
            Name of the detection channel. If None, the only configured channel is
            selected.

        Returns
        -------
        str
            Resolved detection channel name.
        """
        if channel is None:
            if len(self.channels) != 1:
                raise ValueError(
                    "channel must be specified when multiple channels are configured."
                )
            return next(iter(self.channels))
        if channel not in self.channels:
            raise ValueError(f"unknown detection channel: {channel}.")
        return channel

    def select_event_time_series(self, channel: str | None = None) -> pd.Series[Any]:
        """
        Select the frame-based signal of one detection channel.

        Parameters
        ----------
        channel
            Name of the detection channel. If None, the only configured channel is
            selected.

        Returns
        -------
        pd.Series
            Frame-based signal of the selected channel.
        """
        channel_name = self.resolve_channel(channel)
        return self._require_event_time_series()[channel_name]

    def select_event_time_points(
        self, channel: str | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Select detected photon arrival times of one detection channel.

        Parameters
        ----------
        channel
            Name of the detection channel. If None, the only configured channel is
            selected.

        Returns
        -------
        npt.NDArray[np.float64]
            Photon arrival times of the selected channel.
        """
        channel_name = self.resolve_channel(channel)
        return self._require_event_time_points()[channel_name]

    def extract(self, simulation: Simulation) -> None:
        """
        Extract detected photons from a completed simulation.

        Parameters
        ----------
        simulation
            Container for simulation-associated attributes.
        """
        if simulation.transition_series is None or simulation.time_series is None:
            raise ValueError("emissions not available if simulation has not been run.")
        detection_probabilities = get_detection_probabilities(
            transition_set=simulation.transition_set,
            channels=self.channels,
        )
        transition_series = simulation.transition_series
        emitting = simulation.transition_set.combined_state_transitions_df["photon"]
        emitting_transition_ids = emitting.index[emitting].to_numpy()
        emission_indices = np.flatnonzero(
            np.isin(transition_series, emitting_transition_ids)
        )
        channel_indices = assign_detection_channels(
            transition_ids=transition_series[emission_indices],
            detection_probabilities=detection_probabilities,
            random_numbers=np.random.default_rng(self.parameters["seed"]).random(
                emission_indices.size
            ),
        )
        self.event_time_points = {
            channel_name: simulation.time_series[
                emission_indices[channel_indices == channel_index] + 1
            ]
            for channel_index, channel_name in enumerate(self.channels)
        }
        self.construct_event_time_series(
            simulation=simulation,
            resample=self.parameters["frame_time"],
        )

    def simulate(
        self,
        transition_set: TransitionSet,
        start_at: tuple[int, ...] | None = None,
        size: int = 100_000,
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
        if size <= 0:
            raise ValueError("size must be positive.")
        if frames <= 0:
            raise ValueError("frames must be positive.")
        detection_probabilities = get_detection_probabilities(
            transition_set=transition_set,
            channels=self.channels,
        )
        df = transition_set.combined_state_transitions_df
        start_indices = df.index[df["final_state"] == start_at]
        if start_indices.empty:
            raise ValueError(f"start_at {start_at} is not a valid combined state.")
        start_index = start_indices[0]
        experiment_simulator = (
            simulate_experiment_sparse
            if transition_set.matrix_format == "sparse"
            else simulate_experiment
        )
        self.event_time_points, self.event_time_series = experiment_simulator(
            transition_matrix=transition_set.transition_matrix,
            row_sums=transition_set.row_sums,
            detection_probabilities=detection_probabilities,
            channel_names=tuple(self.channels),
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
        number_pulses: int = 10_000,
        pulse_duration: float = 5e-11,
        time_between_pulses: float = 1e-7,
        excitation_rates: dict[str, float] | None = None,
        size: int = 100_000,
        store_time_points: bool = False,
        details: bool = False,
    ) -> tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        Simulation | None,
    ]:
        """
        Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence
        lifetime measurements). The return value lifetimes_DA contains the S1 durations
        of detected emissions by channel when a positive-rate paired transition is
        available immediately before emission and the emitting fluorophore is its first,
        active component. This does not discriminate between the number or mechanism of
        paired transitions. A paired transition may become available during the S1
        duration of the emitting fluorophore. The S1 durations are the time differences
        between photon emission and the last laser pulse.
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
        lifetimes_DA : dict[str, npt.NDArray[np.float64]]
            S1 durations of detected emissions when a positive-rate paired transition
            involving the emitter as its active component was available immediately
            before emission, grouped by detection channel.
        lifetimes_D : dict[str, npt.NDArray[np.float64]]
            S1 durations of detected emissions when no paired transition was available,
            grouped by detection channel.
        lifetimes_all : dict[str, npt.NDArray[np.float64]]
            S1 durations of all detected emissions, grouped by detection channel.
        simulation_object : fluopy.simulation.Simulation
            Container for simulation-associated attributes and methods. Only returned if
            details is True.
        """
        number_pulses = int(number_pulses)
        size = int(size)
        if number_pulses <= 0:
            raise ValueError("number_pulses must be positive.")
        if size <= 0:
            raise ValueError("size must be positive.")
        if not np.isfinite(pulse_duration) or pulse_duration <= 0:
            raise ValueError("pulse_duration must be positive and finite.")
        if not np.isfinite(time_between_pulses) or time_between_pulses <= 0:
            raise ValueError("time_between_pulses must be positive and finite.")
        if excitation_rates is not None:
            rates = np.asarray(list(excitation_rates.values()), dtype=np.float64)
            if np.any(~np.isfinite(rates)) or np.any(rates < 0):
                raise ValueError(
                    "excitation_rates values must be finite and non-negative."
                )

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
        detection_probabilities = get_detection_probabilities(
            transition_set=transition_set,
            channels=self.channels,
        )
        emit_ids_list = np.flatnonzero(detection_probabilities.sum(axis=1) > 0)
        df = transition_set.combined_state_transitions_df
        paired_emission_ids = _get_paired_emission_ids(df, emit_ids_list)
        if details:
            eval_floating_point_precision_error(
                transition_set=transition_set,
                largest_number=number_pulses * time_between_pulses,
            )
            detailed_result = simulate_TCSPC_detailed(
                transition_set=transition_set,
                detection_probabilities=detection_probabilities,
                channel_names=tuple(self.channels),
                paired_emission_ids=paired_emission_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=self.parameters["frame_time"],
                size=size,
                store_time_points=store_time_points,
                seed=self.parameters["seed"],
            )
            self.event_time_series = detailed_result[0]
            self.event_time_points = detailed_result[1]
            lifetimes_DA = detailed_result[2]
            lifetimes_D = detailed_result[3]
            lifetimes_all = detailed_result[4]
            simulation_object = detailed_result[5]
            return lifetimes_DA, lifetimes_D, lifetimes_all, simulation_object
        else:
            basic_result = simulate_TCSPC(
                transition_set=transition_set,
                detection_probabilities=detection_probabilities,
                channel_names=tuple(self.channels),
                paired_emission_ids=paired_emission_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=self.parameters["frame_time"],
                size=size,
                store_time_points=store_time_points,
                seed=self.parameters["seed"],
            )
            self.event_time_series = basic_result[0]
            self.event_time_points = basic_result[1]
            lifetimes_DA = basic_result[2]
            lifetimes_D = basic_result[3]
            lifetimes_all = basic_result[4]
            return lifetimes_DA, lifetimes_D, lifetimes_all, None

    def construct_event_time_series(
        self, simulation: Simulation, resample: str = "5ms"
    ) -> None:
        """
        Count events within right-closed time intervals labeled by their end time.

        Only frames ending at or before simulation.time_series[-1] are included in
        event_time_series, consistently across all channels. An incomplete final frame
        is discarded even when it contains photons. If no complete frame was observed,
        event_time_series is empty. event_time_points retains all detected photons.

        Parameters
        ----------
        simulation
            Container of simulation-associated attributes and methods.
        resample
            For possible input values, see https://pandas.pydata.org/docs/user_guide/
            timeseries.html -> Offset aliases.

        Notes
        -----
        A simulation ending at 12 ms with 5 ms frames produces rows at 5 and 10 ms.
        Photons arriving after 10 ms remain in event_time_points but are omitted from
        event_time_series. A photon arriving exactly at 10 ms belongs to the frame
        ending at 10 ms.

        """
        event_time_points_by_channel = self._require_event_time_points()
        time_series = simulation.time_series
        if time_series is None:
            raise ValueError("event time series requires a completed simulation.")
        observation_end = pd.to_timedelta(time_series[-1], unit="s")

        collected_series = {}
        for channel_name, channel_time_points in event_time_points_by_channel.items():
            event_time_points = np.insert(arr=channel_time_points, obj=0, values=0)
            added_end_time = False
            if event_time_points[-1] != time_series[-1]:
                added_end_time = True
                event_time_points = np.append(event_time_points, time_series[-1])

            time_deltas = pd.to_timedelta(event_time_points, unit="s")
            events = np.ones(shape=event_time_points.shape[0], dtype=np.int64)
            events[0] = 0

            if added_end_time:
                events[-1] = 0

            event_time_series = pd.Series(events, index=time_deltas)
            event_time_series_r = event_time_series.resample(
                resample, closed="right", label="right"
            ).sum()
            event_time_series_r = event_time_series_r.iloc[1:]
            event_time_series_r = event_time_series_r[
                event_time_series_r.index <= observation_end
            ]
            resampled_index = event_time_series_r.index
            in_seconds = np.asarray(
                resampled_index.to_numpy() / np.timedelta64(1, "s"),
                dtype=np.float64,
            )
            event_time_series_r.index = np.round(in_seconds, decimals=12)
            collected_series[channel_name] = event_time_series_r

        self.event_time_series = pd.DataFrame(collected_series, dtype=np.int64)

    def add_emccd_gain(
        self,
        emccd_gain: float | Mapping[str, float],
        seed: RandomGeneratorSeed = None,
    ) -> None:
        """
        Add EMCCD gain to the frame-based detector signal.

        This method modifies event_time_series but not photon arrival times in
        event_time_points.

        Parameters
        ----------
        emccd_gain
            The gain of an EMCCD. A scalar is applied to every detection channel. A
            mapping assigns a separate gain to each channel and must contain every
            channel in event_time_series.
        seed
            A seed to initialize the BitGenerator.

        """
        event_time_series = self._require_event_time_series()
        gains = _resolve_channel_parameter(
            emccd_gain,
            event_time_series.columns,
            "emccd_gain",
        )
        if np.any(~np.isfinite(gains)) or np.any(gains <= 0):
            raise ValueError("emccd_gain must be positive and finite.")
        rng = np.random.default_rng(seed)
        values = event_time_series.to_numpy(dtype=np.int64, copy=True)
        nonzero = values != 0
        scales = np.broadcast_to(gains, values.shape)
        values[nonzero] = np.asarray(
            gamma.rvs(a=values[nonzero], scale=scales[nonzero], random_state=rng),
            dtype=np.int64,
        )
        event_time_series.iloc[:] = values

    def add_gaussian_noise(
        self,
        mean: float | Mapping[str, float],
        std: float | Mapping[str, float],
        seed: RandomGeneratorSeed = None,
    ) -> None:
        """
        Add normally distributed noise to the frame-based detector signal.

        This can represent readout noise, which is insignificant for an EMCCD. This
        method modifies event_time_series but does not create photon arrival times in
        event_time_points.

        Parameters
        ----------
        mean
            Mean of normally distributed noise events per frame. A scalar is applied
            to every detection channel. A mapping assigns a separate mean to each
            channel and must contain every channel in event_time_series.
        std
            Standard deviation of normally distributed noise events per frame. A
            scalar is applied to every detection channel. A mapping assigns a separate
            standard deviation to each channel and must contain every channel in
            event_time_series.
        seed
            A seed to initialize the BitGenerator.

        """
        event_time_series = self._require_event_time_series()
        means = _resolve_channel_parameter(mean, event_time_series.columns, "mean")
        standard_deviations = _resolve_channel_parameter(
            std,
            event_time_series.columns,
            "std",
        )
        if np.any(~np.isfinite(means)):
            raise ValueError("mean must be finite.")
        if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations < 0):
            raise ValueError("std must be finite and non-negative.")
        rng = np.random.default_rng(seed)
        values = event_time_series.to_numpy(dtype=np.int64)
        variates = norm(loc=means, scale=standard_deviations).rvs(
            size=event_time_series.shape, random_state=rng
        )
        variates = variates.astype(np.int64)
        event_time_series.iloc[:] = values + variates
        event_time_series[event_time_series < 0] = 0

    def add_poisson_noise(
        self,
        rate: float | Mapping[str, float],
        seed: RandomGeneratorSeed = None,
    ) -> None:
        """
        Add Poisson noise to the frame-based detector signal.

        This can represent dark current noise. This method modifies event_time_series
        but does not create photon arrival times in event_time_points.

        Parameters
        ----------
        rate
            Expected number of Poisson-distributed noise events per frame. A scalar is
            applied to every detection channel. A mapping assigns a separate rate to
            each channel and must contain every channel in event_time_series.
        seed
            A seed to initialize the BitGenerator.

        """
        event_time_series = self._require_event_time_series()
        rates = _resolve_channel_parameter(rate, event_time_series.columns, "rate")
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("rate must be finite and non-negative.")
        rng = np.random.default_rng(seed)
        values = event_time_series.to_numpy(dtype=np.int64)
        variates = poisson(rates).rvs(size=event_time_series.shape, random_state=rng)
        variates = variates.astype(np.int64)
        event_time_series.iloc[:] = values + variates

    def apply_threshold(self, threshold: int | Mapping[str, int]) -> None:
        """
        Apply a threshold to the frame-based detector signal.

        Values below the threshold are set to zero in event_time_series without
        modifying event_time_points.

        Parameters
        ----------
        threshold
            The minimum number of events per frame to be considered. A scalar is
            applied to every detection channel. A mapping assigns a separate threshold
            to each channel and must contain every channel in event_time_series.
        """
        event_time_series = self._require_event_time_series()
        thresholds = _resolve_channel_parameter(
            threshold,
            event_time_series.columns,
            "threshold",
        )
        if np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0):
            raise ValueError("threshold must be finite and non-negative.")
        values = event_time_series.to_numpy(copy=True)
        values[values < thresholds] = 0
        event_time_series.iloc[:] = values

    def plot_cumulative_events(
        self, channel: str | None = None, **kwargs: Any
    ) -> mplAxes:
        """
        Plot cumulative events versus time.

        Parameters
        ----------
        channel
            Detection channel to plot. If None, the only configured channel is used.
        kwargs
            fluopy.plotting.plot_data arguments

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        event_time_series = self.select_event_time_series(channel)
        if event_time_series.empty:
            raise ValueError("cumulative events require at least one event.")
        cum_events = event_time_series.cumsum()
        total_events = cum_events.iloc[-1]
        if total_events <= 0:
            raise ValueError("cumulative events require at least one event.")
        cum_events = cum_events / total_events
        data = [event_time_series.index, cum_events.to_numpy()]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xlabel", "Photon arrival time (s)")
        kwargs.setdefault("ylabel", "Cumulative prob.")
        kwargs.setdefault("ylim", [0, 1])

        ax = plotting.plot_data(data=data, **kwargs)

        return ax

    def plot_histogram(
        self,
        density: bool = True,
        display_mean: bool = False,
        include_0: bool = False,
        channel: str | None = None,
        **kwargs: Any,
    ) -> mplAxes:
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
        channel
            Detection channel to plot. If None, the only configured channel is used.
        kwargs
            fluopy.plotting.plot_data arguments

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        data = self.select_event_time_series(channel)
        if not include_0:
            data = data[data != 0]
        if data.empty:
            raise ValueError("histogram requires at least one included event count.")

        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("xlabel", r"$\frac{photons}{frame}$")
        if density:
            kwargs.setdefault("ylabel", "Prob. density")
            kwargs.setdefault("density", True)
        else:
            kwargs.setdefault("ylabel", "Probability")
            kwargs.setdefault("weights", np.ones_like(data) / data.size)

        ax = plotting.plot_data(data=data, **kwargs)

        mean_color = kwargs.get("ylabelcolor", "black")
        fontsize = kwargs.get("fontsize", 16)
        if display_mean:
            mean = np.mean(data)
            ax.text(
                x=0.3,
                y=0.85,
                s=rf"$\mu = {mean:.2f}$",
                transform=ax.transAxes,
                fontsize=fontsize,
                color=mean_color,
            )

        return ax

    def plot_time_series(self, channel: str | None = None, **kwargs: Any) -> mplAxes:
        """
        Plot time series of events.

        Parameters
        ----------
        channel
            Detection channel to plot. If None, the only configured channel is used.
        kwargs
            fluopy.plotting.plot_data arguments

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        event_time_series = self.select_event_time_series(channel)
        times = np.insert(event_time_series.index.to_numpy(), 0, 0)
        counts = np.insert(event_time_series.to_numpy(), 0, 0)
        data = [times, counts]
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xlabel", "Time (s)")
        kwargs.setdefault("ylabel", r"$\frac{photons}{frame}$")

        ax = plotting.plot_data(data=data, **kwargs)

        return ax


def _resolve_channel_parameter(
    value: float | Mapping[str, float],
    channel_names: pd.Index,
    parameter_name: str,
) -> npt.NDArray[np.float64]:
    """
    Return one parameter value per event-time-series channel.

    Parameters
    ----------
    value
        Scalar applied to every channel or mapping containing exactly the channel
        names as keys and their parameter values as values.
    channel_names
        Channel names in event_time_series column order.
    parameter_name
        Name of the parameter used in error messages.

    Returns
    -------
    npt.NDArray[np.float64]
        Parameter values in channel_names order.
    """
    if not isinstance(value, Mapping):
        return np.full(len(channel_names), value, dtype=np.float64)

    missing = [name for name in channel_names if name not in value]
    unexpected = [name for name in value if name not in channel_names]
    if missing or unexpected:
        raise ValueError(
            f"{parameter_name} mapping must contain exactly the event_time_series "
            f"channels; missing={missing}, unexpected={unexpected}."
        )
    return np.asarray([value[name] for name in channel_names], dtype=np.float64)


def get_p_filter(
    emission_spectrum: Spectrum,
    bandpass: tuple[float, float],
) -> float:
    """
    Get the fraction of an emission spectrum passing the bandpass filter.

    Parameters
    ----------
    emission_spectrum
        Emission spectrum to which the bandpass filter is applied.
    bandpass
        The lowest and highest emission wavelength to be passed by the bandpass filter.

    Returns
    -------
    float
        The probability of a photon passing the bandpass filter.
    """

    lower, upper = bandpass

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bandpass limits must be finite.")
    if lower >= upper:
        raise ValueError(
            "The lower bandpass limit has to be smaller than the upper limit."
        )

    total_emission = emission_spectrum.integral()
    if total_emission == 0:
        raise ValueError("emission spectrum has zero total intensity.")

    passed_emission = emission_spectrum.integral(
        lower=lower,
        upper=upper,
    )
    p_passed = passed_emission / total_emission

    return p_passed


def _validate_channel_routing(channels: Sequence[DetectionChannel]) -> None:
    """
    Check that channel optical responses sum to at most 1 at each wavelength.

    Parameters
    ----------
    channels
        Channels accepting photons from the same physical fluorophore. Their optical
        responses combine routing_probability with the bandpass transmission, before
        applying detection_efficiency.
    """
    error_message = (
        "channel routing probabilities must sum to at most 1 at each wavelength "
        "for the same fluorophore."
    )
    unfiltered_probability = sum(
        float(channel.routing_probability) * float(channel.bandpass)
        for channel in channels
        if not isinstance(channel.bandpass, (tuple, Spectrum))
        and not isinstance(channel.routing_probability, Spectrum)
    )
    if unfiltered_probability > 1 + 1e-12:
        raise ValueError(error_message)

    boundaries: list[npt.NDArray[np.float64]] = []
    for channel in channels:
        if isinstance(channel.bandpass, tuple):
            boundaries.append(np.asarray(channel.bandpass, dtype=np.float64))
        for response in (channel.routing_probability, channel.bandpass):
            if isinstance(response, Spectrum):
                boundaries.append(response.wavelengths)
    if not boundaries:
        return
    wavelengths = np.unique(np.concatenate(boundaries))
    coefficients = np.zeros((3, wavelengths.size - 1), dtype=np.float64)
    for channel in channels:
        active = np.ones(wavelengths.size - 1, dtype=bool)
        if isinstance(channel.bandpass, tuple):
            lower, upper = channel.bandpass
            active &= (wavelengths[:-1] >= lower) & (wavelengths[1:] <= upper)
        endpoint_values = []
        for response in (channel.routing_probability, channel.bandpass):
            if isinstance(response, Spectrum):
                active &= (wavelengths[:-1] >= response.wavelengths[0]) & (
                    wavelengths[1:] <= response.wavelengths[-1]
                )
                values = np.interp(wavelengths, response.wavelengths, response.values)
            elif isinstance(response, tuple):
                values = np.ones(wavelengths.size, dtype=np.float64)
            else:
                values = np.full(wavelengths.size, response, dtype=np.float64)
            endpoint_values.append(values)
        routing, transmission = endpoint_values
        routing_change = np.diff(routing)
        transmission_change = np.diff(transmission)
        coefficients[0, active] += (routing[:-1] * transmission[:-1])[active]
        coefficients[1, active] += (
            routing[:-1] * transmission_change + transmission[:-1] * routing_change
        )[active]
        coefficients[2, active] += (routing_change * transmission_change)[active]
    constant, linear, quadratic = coefficients
    peak_positions = np.zeros_like(constant)
    np.divide(-linear, 2 * quadratic, out=peak_positions, where=quadratic < 0)
    peak_positions = np.clip(peak_positions, 0, 1)
    peak_values = constant + peak_positions * (linear + peak_positions * quadratic)
    if (
        np.any(constant > 1 + 1e-12)
        or np.any(coefficients.sum(axis=0) > 1 + 1e-12)
        or np.any(peak_values > 1 + 1e-12)
    ):
        raise ValueError(error_message)


def _get_p_detection(
    emission_spectrum: Spectrum,
    channel: DetectionChannel,
) -> float:
    """
    Integrate the product of channel responses over an emission spectrum.

    Parameters
    ----------
    emission_spectrum
        Relative photon emission values as a function of wavelength in nm.
    channel
        Channel defining routing, filter transmission, and detector efficiency.
        Spectral responses are linearly interpolated and zero outside their range.

    Returns
    -------
    float
        Fraction of emitted photons detected in the channel.
    """
    total_emission = emission_spectrum.integral()
    if total_emission == 0:
        raise ValueError("emission spectrum has zero total intensity.")
    lower = float(emission_spectrum.wavelengths[0])
    upper = float(emission_spectrum.wavelengths[-1])
    boundaries = [emission_spectrum.wavelengths]
    responses = (
        channel.routing_probability,
        channel.bandpass,
        channel.detection_efficiency,
    )
    for response in responses:
        if isinstance(response, Spectrum):
            lower = max(lower, float(response.wavelengths[0]))
            upper = min(upper, float(response.wavelengths[-1]))
            boundaries.append(response.wavelengths)
        elif isinstance(response, tuple):
            lower = max(lower, response[0])
            upper = min(upper, response[1])
    if lower >= upper:
        return 0.0
    boundaries.append(np.asarray([lower, upper]))
    wavelengths = np.unique(np.concatenate(boundaries))
    wavelengths = wavelengths[(wavelengths >= lower) & (wavelengths <= upper)]
    half_widths = np.diff(wavelengths) / 2
    midpoints = (wavelengths[:-1] + wavelengths[1:]) / 2
    nodes, weights = np.polynomial.legendre.leggauss(3)
    sample_wavelengths = midpoints[:, None] + half_widths[:, None] * nodes
    products = np.interp(
        sample_wavelengths, emission_spectrum.wavelengths, emission_spectrum.values
    )
    for response in responses:
        if isinstance(response, Spectrum):
            products *= np.interp(
                sample_wavelengths, response.wavelengths, response.values
            )
        elif not isinstance(response, tuple):
            products *= response
    detected_emission = np.sum(half_widths * (products @ weights))
    return float(np.clip(detected_emission / total_emission, 0, 1))


def get_detection_probabilities(
    transition_set: TransitionSet,
    channels: Mapping[str, DetectionChannel],
) -> npt.NDArray[np.float64]:
    """
    Get the detection probability of each transition in each channel.

    Channels may have overlapping bandpasses when their combined optical routing
    response sums to at most 1 at each wavelength. Routing is checked before applying
    detection_efficiency, separately for each physical fluorophore. Spectral responses
    are integrated against the fluorophore's emission spectrum and cached for all its
    emitting transitions. Each photon is assigned to at most one channel.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes.
    channels
        Named detection channels in output-column order.

    Returns
    -------
    npt.NDArray[np.float64]
        Array with one row per combined transition and one column per channel. Each row
        contains mutually exclusive probabilities and sums to at most one.
    """
    df = transition_set.combined_state_transitions_df
    probabilities = np.zeros((len(df), len(channels)), dtype=np.float64)
    fluorophores = transition_set.fluorophore_system.fluorophores
    probabilities_by_fluorophore: dict[int, npt.NDArray[np.float64]] = {}

    for transition_id in np.flatnonzero(df["photon"].to_numpy()):
        transition = df.iloc[transition_id]
        fluorophore_ids = transition["fluorophore_ids"]
        if len(fluorophore_ids) != 1:
            raise ValueError("an emitting transition must belong to one fluorophore.")
        fluorophore_id = fluorophore_ids[0]
        if fluorophore_id in probabilities_by_fluorophore:
            probabilities[transition_id] = probabilities_by_fluorophore[fluorophore_id]
            continue

        fluorophore = fluorophores[fluorophore_id]
        fluorophore_probabilities = np.zeros(len(channels), dtype=np.float64)

        eligible_channels = [
            channel
            for channel in channels.values()
            if channel.fluorophore_ids is None
            or fluorophore_id in channel.fluorophore_ids
        ]
        _validate_channel_routing(eligible_channels)
        for channel_index, channel in enumerate(channels.values()):
            if channel not in eligible_channels:
                continue
            spectral_response = any(
                isinstance(response, Spectrum)
                for response in (
                    channel.routing_probability,
                    channel.bandpass,
                    channel.detection_efficiency,
                )
            )
            if spectral_response or isinstance(channel.bandpass, tuple):
                constants = fluorophore.constants
                if constants is None or constants.emission_spectrum is None:
                    raise ValueError(
                        "wavelength-dependent detection but emission data not available "
                        f"for this kind of fluorophore: {fluorophore.name}"
                    )
                if spectral_response:
                    p_detected = _get_p_detection(constants.emission_spectrum, channel)
                else:
                    p_detected = (
                        cast(float, channel.routing_probability)
                        * get_p_filter(
                            emission_spectrum=constants.emission_spectrum,
                            bandpass=cast(tuple[float, float], channel.bandpass),
                        )
                        * cast(float, channel.detection_efficiency)
                    )
            else:
                p_detected = (
                    cast(float, channel.routing_probability)
                    * cast(float, channel.bandpass)
                    * cast(float, channel.detection_efficiency)
                )
            fluorophore_probabilities[channel_index] = p_detected

        probabilities_by_fluorophore[fluorophore_id] = fluorophore_probabilities
        probabilities[transition_id] = fluorophore_probabilities

    if np.any(probabilities.sum(axis=1) > 1 + 1e-12):
        raise ValueError(
            "detection probabilities must sum to at most 1 per transition."
        )

    return probabilities


def assign_detection_channels(
    transition_ids: npt.ArrayLike,
    detection_probabilities: npt.ArrayLike,
    random_numbers: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """
    Assign each transition to one channel or to the undetected outcome.

    Parameters
    ----------
    transition_ids
        Combined transition identifier for each candidate photon.
    detection_probabilities
        Detection probability for every combined transition and channel.
    random_numbers
        One uniform random number in the half-open interval [0, 1) per candidate
        photon.

    Returns
    -------
    npt.NDArray[np.int64]
        Channel index for each transition. An index equal to the number of channels
        represents an undetected photon.
    """
    transition_ids_array = np.asarray(transition_ids, dtype=np.int64)
    probabilities = np.asarray(detection_probabilities, dtype=np.float64)
    random_numbers_array = np.asarray(random_numbers, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError("detection_probabilities must be a two-dimensional array.")
    if transition_ids_array.ndim != 1 or random_numbers_array.ndim != 1:
        raise ValueError("transition ids and random numbers must be one-dimensional.")
    if transition_ids_array.shape != random_numbers_array.shape:
        raise ValueError("one random number is required per transition.")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("detection probabilities must be finite and between 0 and 1.")
    if np.any(probabilities.sum(axis=1) > 1 + 1e-12):
        raise ValueError(
            "detection probabilities must sum to at most 1 per transition."
        )
    if np.any(~np.isfinite(random_numbers_array)) or np.any(
        (random_numbers_array < 0) | (random_numbers_array >= 1)
    ):
        raise ValueError(
            "random numbers must be between 0 (inclusive) and 1 (exclusive)."
        )
    if np.any(
        (transition_ids_array < 0) | (transition_ids_array >= len(probabilities))
    ):
        raise ValueError("transition ids are outside the detection probability array.")

    cumulative_probabilities = np.cumsum(probabilities, axis=1)
    return np.sum(
        random_numbers_array[:, np.newaxis]
        >= cumulative_probabilities[transition_ids_array],
        axis=1,
        dtype=np.int64,
    )
