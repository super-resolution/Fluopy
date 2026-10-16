"""
Run simulations that resemble TCSPC experiments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from . import simulation as si
from .simulation import Simulation

if TYPE_CHECKING:
    from .fluopy_types import RandomGeneratorSeed
    from .transitions import TransitionSet


__all__: list[str] = []

logger = logging.getLogger(__name__)


def _prepare_detection_configuration(
    transition_count: int,
    detection_probabilities: npt.ArrayLike,
    channel_names: Sequence[str],
) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
    """
    Validate and accumulate channel-resolved detection probabilities.

    Parameters
    ----------
    transition_count
        Number of combined_state_transitions represented by the probability rows.
    detection_probabilities
        Detection probability for every combined transition and channel. Values must
        be finite and between 0 and 1, and each row must sum to at most 1.
    channel_names
        Unique, non-empty channel names in detection_probabilities column order.

    Returns
    -------
    cumulative_detection_probabilities : npt.NDArray[np.float64]
        Detection probabilities accumulated across channels for each transition.
        Has shape (transition_count, len(channel_names)).
    names : tuple[str, ...]
        Validated channel names in the supplied order.
    """
    probabilities = np.asarray(detection_probabilities, dtype=np.float64)
    if channel_names is None or isinstance(channel_names, str):
        raise ValueError("channel_names must be a sequence of channel names.")
    names = tuple(channel_names)
    if not names:
        raise ValueError("at least one channel name is required.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("channel names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("channel names must be unique.")

    if probabilities.shape != (transition_count, len(names)):
        raise ValueError(
            "detection_probabilities must contain one row per transition and one "
            "column per channel."
        )
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("detection probabilities must be finite and between 0 and 1.")
    if np.any(probabilities.sum(axis=1) > 1 + 1e-12):
        raise ValueError(
            "detection probabilities must sum to at most 1 per transition."
        )
    return np.cumsum(probabilities, axis=1), names


def _prepare_photon_outputs(
    photon_collector: npt.NDArray[np.int64],
    time_stamps: npt.NDArray[np.float64],
    time_points: list[list[float]] | None,
    pulse_delays_paired: list[list[float]],
    pulse_delays_unpaired: list[list[float]],
    pulse_delays_all: list[list[float]],
    channel_names: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    dict[str, npt.NDArray[np.float64]] | None,
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
]:
    """
    Convert collected photons to channel-resolved return values.

    Parameters
    ----------
    photon_collector
        Detected photon counts with one row per frame and one column per channel.
    time_stamps
        End times of the measured frames in seconds.
    time_points
        Detected photon arrival times in seconds, grouped in channel_names order.
        None if time-point storage was disabled.
    pulse_delays_paired
        Pulse-relative delays when a positive-rate paired transition involving the
        emitter as its active component was available, grouped in channel_names order.
    pulse_delays_unpaired
        Pulse-relative delays when no paired transition was available, grouped in
        channel_names order.
    pulse_delays_all
        Pulse-relative delays of all detected emissions, grouped in channel_names order.
        All delays are in seconds and measured from the latest laser pulse.
    channel_names
        Channel names in photon_collector column order and in the order used by the
        photon arrival-time and pulse-delay collections.

    Returns
    -------
    event_time_series : pd.DataFrame
        Detected photon counts by channel, indexed by the frame end times.
    event_time_points : dict[str, npt.NDArray[np.float64]] | None
        Detected photon arrival times by channel. None if time_points is None.
    pulse_delays_paired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of detected emissions with an available paired transition,
        grouped by channel.
    pulse_delays_unpaired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays without an available paired transition, grouped by
        channel.
    pulse_delays_all : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of all detected emissions, grouped by channel.
    """
    event_time_series = pd.DataFrame(
        photon_collector,
        index=time_stamps,
        columns=channel_names,
        dtype=np.int64,
    )
    event_time_points = (
        None
        if time_points is None
        else {
            name: np.asarray(time_points[index], dtype=np.float64)
            for index, name in enumerate(channel_names)
        }
    )
    pulse_delays_paired_output = {
        name: np.asarray(pulse_delays_paired[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }
    pulse_delays_unpaired_output = {
        name: np.asarray(pulse_delays_unpaired[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }
    pulse_delays_all_output = {
        name: np.asarray(pulse_delays_all[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }

    return (
        event_time_series,
        event_time_points,
        pulse_delays_paired_output,
        pulse_delays_unpaired_output,
        pulse_delays_all_output,
    )


def simulate_TCSPC(
    transition_set: TransitionSet,
    detection_probabilities: npt.ArrayLike,
    channel_names: Sequence[str],
    paired_emission_ids: Sequence[int] | None = None,
    number_pulses: int = 100_000,
    pulse_duration: float = 5e-11,
    time_between_pulses: float = 25e-9,
    excitation_rates: dict[str, float] | None = None,
    frame_time: str = "1ms",
    size: int = 100_000,
    store_time_points: bool = False,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    pd.DataFrame,
    dict[str, npt.NDArray[np.float64]] | None,
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
]:
    """
    Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence lifetime
    measurement). Methodically the direct method of the gillespie algorithm.
    The simulation is bound to start at the state configuration where all fluorophores
    are in the ground state. Photon delays are measured from the latest laser pulse,
    rather than from entry into S1. Only detected photons are
    taken into account. The simulation approximates laser pulses to be instantaneous.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes
    detection_probabilities
        Detection probability for every combined transition and channel.
    channel_names
        Channel names in the order used by detection_probabilities.
    paired_emission_ids
        Combined-state transition indices of emissions for which a paired transition
        with the emitting fluorophore as its active component is available.
    number_pulses
        Number of pulses simulated.
    pulse_duration
        The duration of a laser pulse in s. This time is used to calculate the
        probability of excitation, other than that it is neglected.
    time_between_pulses
        The time between 2 laser pulses in s.
    excitation_rates
        Contains the fluorophore names as keys and the excitation rates as values.
        Assumes uniform irradiance over the pulse duration.
    frame_time
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
    size
        Size of random numbers drawn at once.
    store_time_points
        Whether to store the time points at which emissions are detected.
    seed
        A seed to initialize the BitGenerator.
    Returns
    -------
    event_time_series : pd.DataFrame
        Contains measured frames indexed by their end time and the number of detected
        emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
        If store_time_points is False, this will be None.
    pulse_delays_paired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of emissions selected by paired_emission_ids, grouped
        by channel.
    pulse_delays_unpaired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays without an available paired transition, grouped by
        channel.
    pulse_delays_all : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of all detected emissions, grouped by channel.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    start_at, current_transition_index, transition_matrix_non_exc, row_sums_non_exc = (
        transition_set._get_start_configuration(None)
    )
    final_states = transition_set.combined_state_transitions_df["final_state"].tolist()
    cumulative_detection_probabilities, resolved_channel_names = (
        _prepare_detection_configuration(
            transition_count=len(final_states),
            detection_probabilities=detection_probabilities,
            channel_names=channel_names,
        )
    )
    if current_transition_index == len(final_states):
        final_states.append(start_at)
        cumulative_detection_probabilities = np.pad(
            cumulative_detection_probabilities, ((0, 1), (0, 0))
        )

    sparse_transition_rows = None
    if sparse.issparse(transition_matrix_non_exc):
        sparse_transition_rows = si._prepare_sparse_transition_rows(
            transition_matrix_non_exc
        )
    else:
        cumulative_transition_probabilities_non_exc = si._prepare_dense_transition_rows(
            transition_matrix_non_exc
        )
    number_fluorophores = transition_set.fluorophore_system.count
    if excitation_rates is None:
        excitation_rates = {}
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            fluorophore_id
            for fluorophore_id, f in enumerate(
                transition_set.fluorophore_system.fluorophores
            )
            if f.name == fluorophore
        ]
        excitation_rate = excitation_rates[fluorophore]
        excitation_probability = 1 - np.exp(
            -excitation_rate * pulse_duration
        )  # CDF of exponential distribution
        excitation_probabilities[fluorophore_ids] = excitation_probability
    excitable_indices = np.where(excitation_probabilities != 0)[0]
    rng = np.random.default_rng(seed)
    seconds_per_frame = float(pd.Timedelta(frame_time) / np.timedelta64(1, "s"))
    frames = number_pulses * time_between_pulses / seconds_per_frame
    number_frames = int(np.ceil(frames))
    last_frame_end = round(number_frames * seconds_per_frame, 12)
    time_stamps = np.linspace(seconds_per_frame, last_frame_end, number_frames)
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        logger.warning(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(seconds_per_frame / time_between_pulses)):.1e} pulses).",
            stacklevel=2,
        )
    logger.warning(
        f"the last frame (of index {last_frame_end}) has {frames - number_frames + 1:.2e} "
        "times the pulses of other frames.",
        stacklevel=2,
    )
    photon_collector = np.zeros(
        (time_stamps.size, len(resolved_channel_names)), dtype=np.int64
    )
    pulse_delays_unpaired: list[list[float]] = [[] for _ in resolved_channel_names]
    pulse_delays_paired: list[list[float]] = [[] for _ in resolved_channel_names]
    pulse_delays_all: list[list[float]] = [[] for _ in resolved_channel_names]
    pending_transition: tuple[int, float] = (0, np.inf)
    paired_emission_id_set = frozenset(
        () if paired_emission_ids is None else paired_emission_ids
    )
    pulse_number = 0
    excitation_batch_number = 1
    transition_draw_number = 0
    transition_batch_number = 1
    S0 = 0
    S1 = 1
    reuse_pending_transition = False
    excitation_applied = False
    can_continue = True
    excitation_random_numbers = rng.uniform(
        low=0, high=1, size=(size, excitable_indices.size)
    )
    transition_random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    if store_time_points:
        time_points: list[list[float]] | None = [[] for _ in resolved_channel_names]
    else:
        time_points = None

    while pulse_number < number_pulses:
        pulse_number += 1
        if pulse_number >= excitation_batch_number * size:
            excitation_batch_number = int(np.floor(pulse_number / size)) + 1
            excitation_random_numbers = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
        excitation_random_index = pulse_number - (excitation_batch_number - 1) * size
        time = (pulse_number - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(final_states[current_transition_index])
        if not excitation_applied:
            ground_state_positions = np.where(current_states[excitable_indices] == S0)[
                0
            ]
            if ground_state_positions.size != 0:
                can_continue = True
                ground_state_ids = excitable_indices[ground_state_positions]
                excitations = np.where(
                    excitation_random_numbers[
                        excitation_random_index, ground_state_positions
                    ]
                    < excitation_probabilities[ground_state_ids]
                )[0]
                if excitations.size != 0:
                    excited_fluorophore_ids = ground_state_ids[excitations]
                    current_states[excited_fluorophore_ids] = S1
                elif pending_transition[1] < next_pulse_time:
                    reuse_pending_transition = True
                    next_transition = pending_transition[0]
                    time = pending_transition[1]
                else:
                    pulses_until_excitation = rng.geometric(
                        p=excitation_probabilities[ground_state_ids]
                    )
                    first_excitation_positions = np.where(
                        pulses_until_excitation == np.min(pulses_until_excitation)
                    )[0]
                    future_excitation_pulse_index = (
                        pulse_number
                        + pulses_until_excitation[first_excitation_positions[0]]
                        - 1
                    )
                    future_excitation_time = (
                        future_excitation_pulse_index * time_between_pulses
                    )
                    if future_excitation_time < pending_transition[1]:
                        if future_excitation_pulse_index >= number_pulses:
                            break
                        excited_fluorophore_ids = ground_state_ids[
                            first_excitation_positions
                        ]
                        current_states[excited_fluorophore_ids] = S1
                        current_transition_index = final_states.index(
                            tuple(current_states)
                        )
                        pulse_number = future_excitation_pulse_index
                        excitation_applied = True
                        continue
                    else:
                        reuse_pending_transition = True
                        next_transition = pending_transition[0]
                        time = pending_transition[1]
            else:
                if not can_continue:
                    logger.warning(
                        "All fluorophores underwent photobleaching or entered "
                        "another Markov chain absorbing state.",
                        stacklevel=2,
                    )
                    return _prepare_photon_outputs(
                        photon_collector=photon_collector,
                        time_stamps=time_stamps,
                        time_points=time_points,
                        pulse_delays_paired=pulse_delays_paired,
                        pulse_delays_unpaired=pulse_delays_unpaired,
                        pulse_delays_all=pulse_delays_all,
                        channel_names=resolved_channel_names,
                    )
                elif pending_transition[1] != np.inf:
                    reuse_pending_transition = True
                    next_transition = pending_transition[0]
                    time = pending_transition[1]
                can_continue = False

        excitation_applied = False
        if reuse_pending_transition:
            pulse_number = int(np.floor(time / time_between_pulses)) + 1
            if pulse_number > number_pulses:
                break
            last_pulse_time = (pulse_number - 1) * time_between_pulses
            next_pulse_time = pulse_number * time_between_pulses
        current_transition_index = final_states.index(tuple(current_states))
        while time < next_pulse_time:
            if not reuse_pending_transition:
                transition_draw_number += 1
                if transition_draw_number == transition_batch_number * size:
                    transition_batch_number += 1
                    transition_random_numbers = rng.uniform(
                        low=0, high=1, size=(size, 3)
                    )
                transition_random_index = (
                    transition_draw_number - (transition_batch_number - 1) * size
                )
                event_rate = row_sums_non_exc[current_transition_index]
                if event_rate == 0:
                    pending_transition = (0, np.inf)
                    break
                can_continue = True
                transition_time = (
                    -np.log(transition_random_numbers[transition_random_index, 0])
                    / event_rate
                )
                time += transition_time
                if sparse_transition_rows is None:
                    next_transition = np.searchsorted(
                        cumulative_transition_probabilities_non_exc[
                            current_transition_index
                        ],
                        transition_random_numbers[transition_random_index, 1],
                        side="right",
                    )
                else:
                    next_transition = si._sample_sparse_transition(
                        current_transition_index,
                        transition_random_numbers[transition_random_index, 1],
                        *sparse_transition_rows,
                    )
                if time > next_pulse_time:
                    pending_transition = (int(next_transition), float(time))
                    break
            reuse_pending_transition = False
            current_transition_index = next_transition
            cumulative_probabilities = cumulative_detection_probabilities[
                next_transition
            ]
            if cumulative_probabilities[-1] == 0:
                continue
            transition_random_index = (
                transition_draw_number - (transition_batch_number - 1) * size
            )
            channel_index = np.searchsorted(
                cumulative_probabilities,
                transition_random_numbers[transition_random_index, 2],
                side="right",
            )
            if channel_index < len(resolved_channel_names):
                pulse_delay = time - last_pulse_time
                if next_transition in paired_emission_id_set:
                    pulse_delays_paired[channel_index].append(pulse_delay)
                else:
                    pulse_delays_unpaired[channel_index].append(pulse_delay)
                pulse_delays_all[channel_index].append(pulse_delay)
                frame = int(np.ceil(time / seconds_per_frame))
                photon_collector[frame - 1, channel_index] += 1
                if time_points is not None:
                    time_points[channel_index].append(time)

    return _prepare_photon_outputs(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        pulse_delays_paired=pulse_delays_paired,
        pulse_delays_unpaired=pulse_delays_unpaired,
        pulse_delays_all=pulse_delays_all,
        channel_names=resolved_channel_names,
    )


def simulate_TCSPC_detailed(
    transition_set: TransitionSet,
    detection_probabilities: npt.ArrayLike,
    channel_names: Sequence[str],
    paired_emission_ids: Sequence[int] | None = None,
    number_pulses: int = 100_000,
    pulse_duration: float = 5e-11,
    time_between_pulses: float = 25e-9,
    excitation_rates: dict[str, float] | None = None,
    frame_time: str = "1ms",
    size: int = 100_000,
    store_time_points: bool = False,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    pd.DataFrame,
    dict[str, npt.NDArray[np.float64]] | None,
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    Simulation,
]:
    """
    Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence lifetime
    measurement). Methodically the direct method of the gillespie algorithm.
    The simulation is bound to start at the state configuration where all fluorophores
    are in the ground state. Photon delays are measured from the latest laser pulse,
    rather than from entry into S1. This is the detailed
    version of the simulation, meaning that it additionally provides a simulation object
    that contains all transitions, their time points and the corresponding states. The
    simulation approximates laser pulses to be instantaneous. If multiple excitations
    occur at the same time point, the time points are spaced by a minimal amount.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes
    detection_probabilities
        Detection probability for every combined transition and channel.
    channel_names
        Channel names in the order used by detection_probabilities.
    paired_emission_ids
        Combined-state transition indices of emissions for which a paired transition
        with the emitting fluorophore as its active component is available.
    number_pulses
        Number of pulses simulated.
    pulse_duration
        The duration of a laser pulse in s. This time is used to calculate the
        probability of excitation, other than that it is neglected.
    time_between_pulses
        The time between 2 laser pulses in s.
    excitation_rates
        Contains the fluorophore names as keys and the excitation rates as values.
        Assumes uniform irradiance over the pulse duration.
    frame_time
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
    size
        Size of random numbers drawn at once.
    store_time_points
        Whether to store the time points at which emissions are detected.
    seed
        A seed to initialize the BitGenerator.
    Returns
    -------
    event_time_series : pd.DataFrame
        Contains measured frames indexed by their end time and the number of detected
        emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
        If store_time_points is False, this will be None.
    pulse_delays_paired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of emissions selected by paired_emission_ids, grouped
        by channel.
    pulse_delays_unpaired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays without an available paired transition, grouped by
        channel.
    pulse_delays_all : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of all detected emissions, grouped by channel.
    simulation_object : fluopy.simulation.Simulation
        Container for simulation-associated attributes and methods.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    start_at, current_transition_index, transition_matrix_non_exc, row_sums_non_exc = (
        transition_set._get_start_configuration(None)
    )
    final_states = transition_set.combined_state_transitions_df["final_state"].tolist()
    cumulative_detection_probabilities, resolved_channel_names = (
        _prepare_detection_configuration(
            transition_count=len(final_states),
            detection_probabilities=detection_probabilities,
            channel_names=channel_names,
        )
    )
    if current_transition_index == len(final_states):
        final_states.append(start_at)
        cumulative_detection_probabilities = np.pad(
            cumulative_detection_probabilities, ((0, 1), (0, 0))
        )

    sparse_transition_rows = None
    if sparse.issparse(transition_matrix_non_exc):
        sparse_transition_rows = si._prepare_sparse_transition_rows(
            transition_matrix_non_exc
        )
    else:
        cumulative_transition_probabilities_non_exc = si._prepare_dense_transition_rows(
            transition_matrix_non_exc
        )
    number_fluorophores = transition_set.fluorophore_system.count
    if excitation_rates is None:
        excitation_rates = {}
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            fluorophore_id
            for fluorophore_id, f in enumerate(
                transition_set.fluorophore_system.fluorophores
            )
            if f.name == fluorophore
        ]
        excitation_rate = excitation_rates[fluorophore]
        excitation_probability = 1 - np.exp(
            -excitation_rate * pulse_duration
        )  # CDF of exponential distribution
        excitation_probabilities[fluorophore_ids] = excitation_probability
    excitable_indices = np.where(excitation_probabilities != 0)[0]
    rng = np.random.default_rng(seed)
    seconds_per_frame = float(pd.Timedelta(frame_time) / np.timedelta64(1, "s"))
    frames = number_pulses * time_between_pulses / seconds_per_frame
    number_frames = int(np.ceil(frames))
    last_frame_end = round(number_frames * seconds_per_frame, 12)
    time_stamps = np.linspace(seconds_per_frame, last_frame_end, number_frames)
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        logger.warning(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(seconds_per_frame / time_between_pulses)):.1e} pulses).",
            stacklevel=2,
        )
    logger.warning(
        f"the last frame (of index {last_frame_end}) has {frames - number_frames + 1:.2e} "
        "times the pulses of other frames.",
        stacklevel=2,
    )
    photon_collector = np.zeros(
        (time_stamps.size, len(resolved_channel_names)), dtype=np.int64
    )
    time_series: list[float] = [0.0]
    transition_series: list[int] = []
    excitation_series: list[int] = []
    pulse_delays_unpaired: list[list[float]] = [[] for _ in resolved_channel_names]
    pulse_delays_paired: list[list[float]] = [[] for _ in resolved_channel_names]
    pulse_delays_all: list[list[float]] = [[] for _ in resolved_channel_names]
    pending_transition: tuple[int, float] = (0, np.inf)
    paired_emission_id_set = frozenset(
        () if paired_emission_ids is None else paired_emission_ids
    )
    pulse_number = 0
    excitation_batch_number = 1
    transition_draw_number = 0
    transition_batch_number = 1
    S0 = 0
    S1 = 1
    reuse_pending_transition = False
    excitation_applied = False
    can_continue = True
    excitation_random_numbers = rng.uniform(
        low=0, high=1, size=(size, excitable_indices.size)
    )
    transition_random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    if store_time_points:
        time_points: list[list[float]] | None = [[] for _ in resolved_channel_names]
    else:
        time_points = None

    while pulse_number < number_pulses:
        pulse_number += 1
        if pulse_number >= excitation_batch_number * size:
            excitation_batch_number = int(np.floor(pulse_number / size)) + 1
            excitation_random_numbers = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
        excitation_random_index = pulse_number - (excitation_batch_number - 1) * size
        time = (pulse_number - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(final_states[current_transition_index])
        if not excitation_applied:
            ground_state_positions = np.where(current_states[excitable_indices] == S0)[
                0
            ]
            if ground_state_positions.size != 0:
                can_continue = True
                ground_state_ids = excitable_indices[ground_state_positions]
                excitations = np.where(
                    excitation_random_numbers[
                        excitation_random_index, ground_state_positions
                    ]
                    < excitation_probabilities[ground_state_ids]
                )[0]
                if excitations.size != 0:
                    excited_fluorophore_ids = ground_state_ids[excitations]
                    current_states[excited_fluorophore_ids] = S1
                    excitation_series.extend(excited_fluorophore_ids.tolist())
                    time_series.extend([time] * excitations.size)
                elif pending_transition[1] < next_pulse_time:
                    reuse_pending_transition = True
                    next_transition = pending_transition[0]
                    time = pending_transition[1]
                else:
                    pulses_until_excitation = rng.geometric(
                        p=excitation_probabilities[ground_state_ids]
                    )
                    first_excitation_positions = np.where(
                        pulses_until_excitation == np.min(pulses_until_excitation)
                    )[0]
                    future_excitation_pulse_index = (
                        pulse_number
                        + pulses_until_excitation[first_excitation_positions[0]]
                        - 1
                    )
                    future_excitation_time = (
                        future_excitation_pulse_index * time_between_pulses
                    )
                    if future_excitation_time < pending_transition[1]:
                        if future_excitation_pulse_index >= number_pulses:
                            break
                        excited_fluorophore_ids = ground_state_ids[
                            first_excitation_positions
                        ]
                        current_states[excited_fluorophore_ids] = S1
                        excitation_series.extend(excited_fluorophore_ids.tolist())
                        time_series.extend(
                            [future_excitation_time] * first_excitation_positions.size
                        )
                        current_transition_index = final_states.index(
                            tuple(current_states)
                        )
                        pulse_number = future_excitation_pulse_index
                        excitation_applied = True
                        continue
                    else:
                        reuse_pending_transition = True
                        next_transition = pending_transition[0]
                        time = pending_transition[1]
            else:
                if not can_continue:
                    logger.warning(
                        "All fluorophores underwent photobleaching or entered "
                        "another Markov chain absorbing state.",
                        stacklevel=2,
                    )
                    return_values = prepare_return_values(
                        photon_collector=photon_collector,
                        time_stamps=time_stamps,
                        time_points=time_points,
                        pulse_delays_paired=pulse_delays_paired,
                        pulse_delays_unpaired=pulse_delays_unpaired,
                        pulse_delays_all=pulse_delays_all,
                        time_series=time_series,
                        transition_series=transition_series,
                        excitation_series=excitation_series,
                        transition_set=transition_set,
                        channel_names=resolved_channel_names,
                    )

                    return return_values

                elif pending_transition[1] != np.inf:
                    reuse_pending_transition = True
                    next_transition = pending_transition[0]
                    time = pending_transition[1]
                can_continue = False

        excitation_applied = False
        if reuse_pending_transition:
            pulse_number = int(np.floor(time / time_between_pulses)) + 1
            if pulse_number > number_pulses:
                break
            last_pulse_time = (pulse_number - 1) * time_between_pulses
            next_pulse_time = pulse_number * time_between_pulses
        current_transition_index = final_states.index(tuple(current_states))
        while time < next_pulse_time:
            if not reuse_pending_transition:
                transition_draw_number += 1
                if transition_draw_number == transition_batch_number * size:
                    transition_batch_number += 1
                    transition_random_numbers = rng.uniform(
                        low=0, high=1, size=(size, 3)
                    )
                transition_random_index = (
                    transition_draw_number - (transition_batch_number - 1) * size
                )
                event_rate = row_sums_non_exc[current_transition_index]
                if event_rate == 0:
                    pending_transition = (0, np.inf)
                    break
                can_continue = True
                transition_time = (
                    -np.log(transition_random_numbers[transition_random_index, 0])
                    / event_rate
                )
                time += transition_time
                if sparse_transition_rows is None:
                    next_transition = np.searchsorted(
                        cumulative_transition_probabilities_non_exc[
                            current_transition_index
                        ],
                        transition_random_numbers[transition_random_index, 1],
                        side="right",
                    )
                else:
                    next_transition = si._sample_sparse_transition(
                        current_transition_index,
                        transition_random_numbers[transition_random_index, 1],
                        *sparse_transition_rows,
                    )
                if time > next_pulse_time:
                    pending_transition = (int(next_transition), float(time))
                    break
            reuse_pending_transition = False
            current_transition_index = next_transition
            transition_series.append(int(next_transition))
            excitation_series.append(-1)
            time_series.append(time)
            cumulative_probabilities = cumulative_detection_probabilities[
                next_transition
            ]
            if cumulative_probabilities[-1] == 0:
                continue
            transition_random_index = (
                transition_draw_number - (transition_batch_number - 1) * size
            )
            channel_index = np.searchsorted(
                cumulative_probabilities,
                transition_random_numbers[transition_random_index, 2],
                side="right",
            )
            if channel_index < len(resolved_channel_names):
                pulse_delay = time - last_pulse_time
                if next_transition in paired_emission_id_set:
                    pulse_delays_paired[channel_index].append(pulse_delay)
                else:
                    pulse_delays_unpaired[channel_index].append(pulse_delay)
                pulse_delays_all[channel_index].append(pulse_delay)
                frame = int(np.ceil(time / seconds_per_frame))
                photon_collector[frame - 1, channel_index] += 1
                if time_points is not None:
                    time_points[channel_index].append(time)

    return_values = prepare_return_values(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        pulse_delays_paired=pulse_delays_paired,
        pulse_delays_unpaired=pulse_delays_unpaired,
        pulse_delays_all=pulse_delays_all,
        time_series=time_series,
        transition_series=transition_series,
        excitation_series=excitation_series,
        transition_set=transition_set,
        channel_names=resolved_channel_names,
    )

    return return_values


def space_multiple_excitations(time_series: npt.NDArray[np.float64]) -> None:
    """
    If multiple excitations occur at the same time point, the time point is repeated.
    This function creates a minimal space between the same time points. Note that this
    also creates space for transitions whose time was too small to be differentiated
    given the floating point precision. It is necessary to space excitations AND other
    transitions, otherwise a transition could appear at a time point smaller than the
    previous one.
    Mutates the input inplace.

    Parameters
    ----------
    time_series
        Contains the time points at which transitions occur. Transitions may occur at
        the same time point.

    """
    if not np.issubdtype(time_series.dtype, np.floating):
        raise ValueError("time_series must be of float type.")
    indices = np.unique(time_series, return_index=True)[1]
    if indices.size == time_series.size:
        return
    logger.warning(
        "Multiple excitations at the same time point are spaced by a minimal "
        "amount. This also spaces other transitions whose time was too small "
        "to be differentiated given the floating point precision. Hence, "
        "transitions with mean time lower than expected will now have mean "
        "time higher than expected.",
        stacklevel=2,
    )
    # Repeated passes are acceptable because simultaneous excitations are limited by
    # the typically small number of fluorophores, so duplicate groups remain small.
    while indices.size != time_series.size:
        mask = np.ones(time_series.shape, dtype=bool)
        mask[indices] = False
        time_series[mask] = np.nextafter(time_series[mask], float("inf"))
        indices = np.unique(time_series, return_index=True)[1]


def insert_excitations(
    transition_series: npt.NDArray[np.uint32],
    transition_set: TransitionSet,
    excitation_series: npt.NDArray[np.int16],
) -> npt.NDArray[np.uint32]:
    """
    Inserts the indices of the excitation transitions into the transition series in the
    correct order.

    Parameters
    ----------
    transition_series
        Contains the indices of the transitions that occur. Does not include the indices
        of the excitations.
    transition_set
        Collection of all relevant transitions and related attributes.
    excitation_series
        Contains the index of a fluorophore if excitation, -1 if other transition.

    Returns
    -------
    npt.NDArray[np.uint32]
        Contains the indices of the transitions that occur. Includes the indices of the
        excitations.
    """
    transition_series_ad = transition_series.copy()
    df = transition_set.combined_state_transitions_df
    excitations = df[df["abbreviation"] == "EXC"]
    indices_transitions = np.where(excitation_series == -1)[0]  # gets the indices of
    # transitions that are not excitations
    number_fluorophores = transition_set.fluorophore_system.count
    trailing_start = int(indices_transitions[-1]) + 1 if indices_transitions.size else 0
    current_state = (
        np.array(df["final_state"].iloc[transition_series[-1]], dtype=int)
        if transition_series.size
        else np.zeros(number_fluorophores, dtype=int)
    )
    trailing_excitation_ids = np.empty(
        excitation_series.size - trailing_start, dtype=transition_series_ad.dtype
    )
    for i, fluorophore_id in enumerate(excitation_series[trailing_start:]):
        initial_state = tuple(current_state)
        current_state[fluorophore_id] = 1
        final_state = tuple(current_state)
        matching_excitations = excitations[
            (excitations["initial_state"] == initial_state)
            & (excitations["final_state"] == final_state)
        ]
        trailing_excitation_ids[i] = matching_excitations.index[0]

    if indices_transitions.size == 0:
        return trailing_excitation_ids

    diffs = np.diff(indices_transitions)
    diffs = np.insert(arr=diffs, obj=0, values=indices_transitions[0] + 1)
    # if diffs is > 1, there was an excitation
    already_processed = np.array([], dtype=np.int64)

    for i in range(number_fluorophores):
        diff = i + 2
        processing = np.where(diffs >= diff)[0]
        if processing.size == 0:
            break  # no more excitations to process
        # get the index of the last fluorophore that was excited and not already
        # included
        corresponding_excitations = excitation_series[
            indices_transitions[processing] - 1 - i
        ]
        if i != 0:
            insert_at = np.searchsorted(already_processed, processing, side="left")
            # left because it starts from the back, i.e., last excitation is processed
            # first
            already_processed = np.insert(
                arr=already_processed, obj=insert_at, values=processing
            )
            processing += insert_at
        else:
            already_processed = processing

        # so far, the positions of excitations to insert are determined. Now it is about
        # which exact kind of excitation to insert.
        transitions = transition_series_ad[processing]  # the transition that follows
        # the excitation that is to be inserted
        # at i >= 1, the transitions should all be excitations themselves.
        # the initial state of the following transition is the final state of the
        # excitation to be inserted
        final_states = np.array(df["initial_state"].iloc[transitions].values.tolist())
        # the next block of code converts the excited state of the fluorophore to be
        # excited to the ground state such that the initial state of the excitation
        # is defined.
        initial_states = final_states.copy()
        initial_states[
            np.arange(initial_states.shape[0]), corresponding_excitations
        ] = 0

        df2 = pd.DataFrame(
            {
                "initial_state": map(tuple, initial_states.tolist()),
                "final_state": map(tuple, final_states.tolist()),
            }
        )
        # the dataframe will contain an extra column which duplicates the index.
        # This is needed to store these indices in the merged dataframe.
        df2.reset_index(names="original_index", inplace=True)
        merged = excitations.reset_index(names="id").merge(
            df2, on=["initial_state", "final_state"], how="inner"
        )
        # these are the transition ids of the excitations that are to be inserted
        indices = np.array(merged["id"].tolist())
        # this determines the order in processing (the indices where the excitations
        # should be inserted)
        original_indices = np.array(merged["original_index"].tolist())
        transition_series_ad = np.insert(
            arr=transition_series_ad, obj=processing[original_indices], values=indices
        )

    return np.concatenate((transition_series_ad, trailing_excitation_ids))


def get_state_series(
    transition_set: TransitionSet, transition_series: npt.ArrayLike
) -> npt.NDArray[np.int8]:
    """
    Creates a series of states based on the transitions that occur, starting at state 0.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes.
    transition_series
        Contains the indices of the transitions that occur.

    Returns
    -------
    npt.NDArray[np.int8]
        Contains 1-D array_like for each fluorophore representing its state at index i
        corresponding to transition_series[i-1].
    """
    transition_indices = np.asarray(transition_series, dtype=np.uint32)
    fluorophore_count = transition_set.fluorophore_system.count
    start_at = tuple(np.zeros(shape=fluorophore_count, dtype=int))
    final_states = transition_set.combined_state_transitions_df["final_state"]

    state_series = np.empty(
        shape=(fluorophore_count, transition_indices.size + 1),
        dtype=np.int8,
    )
    state_series[:, 0] = start_at

    for fluorophore_index in range(fluorophore_count):
        final_states_fluorophore = np.array(
            [cast(tuple[int, ...], state)[fluorophore_index] for state in final_states],
            dtype=np.int8,
        )
        state_series[fluorophore_index, 1:] = final_states_fluorophore[
            transition_indices
        ]

    return state_series


def prepare_return_values(
    photon_collector: npt.NDArray[np.int64],
    time_stamps: npt.NDArray[np.float64],
    time_points: list[list[float]] | None,
    pulse_delays_paired: list[list[float]],
    pulse_delays_unpaired: list[list[float]],
    pulse_delays_all: list[list[float]],
    time_series: npt.ArrayLike,
    transition_series: npt.ArrayLike,
    excitation_series: npt.ArrayLike,
    transition_set: TransitionSet,
    channel_names: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    dict[str, npt.NDArray[np.float64]] | None,
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    Simulation,
]:
    """
    Prepares the return values for the detailed TCSPC simulation. This includes the
    conversion to numpy arrays, the insertion of excitation transitions, the spacing of
    multiple excitations at the same point in time, and the creation of the state
    series. Subsequently, the Simulation object is created.

    Parameters
    ----------
    photon_collector
        Contains the number of detected emissions at the corresponding time points.
    time_stamps
        Contains the time points (increasing by a defined time interval).
    time_points
        The time points at which emissions are detected.
    pulse_delays_paired
        Pulse-relative delays when a paired transition was available.
    pulse_delays_unpaired
        Pulse-relative delays when no paired transition was available.
    pulse_delays_all
        Pulse-relative delays of all detected emissions.
    time_series
        Contains the time points at which transitions occur. Also includes the time
        points at which excitations occur. Note that if multiple excitations occur at
        the same time point, the time point is repeated.
    transition_series
        Contains the indices of the transitions that occur. Does not include the indices
        of the excitations.
    excitation_series
        Contains the index of a fluorophore if excitation, -1 if other transition.
    transition_set
        Collection of all relevant transitions and related attributes.
    channel_names
        Names of the detection channels.

    Returns
    -------
    event_time_series : pd.DataFrame
        Contains measured frames indexed by their end time and the number of detected
        emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
    pulse_delays_paired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays with an available paired transition, grouped by channel.
    pulse_delays_unpaired : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays without an available paired transition, grouped by
        channel.
    pulse_delays_all : dict[str, npt.NDArray[np.float64]]
        Pulse-relative delays of all detected emissions, grouped by channel.
    simulation_object : fluopy.simulation.Simulation
        Container for simulation-associated attributes and methods.
    """
    (
        event_time_series,
        event_time_points,
        pulse_delays_paired_output,
        pulse_delays_unpaired_output,
        pulse_delays_all_output,
    ) = _prepare_photon_outputs(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        pulse_delays_paired=pulse_delays_paired,
        pulse_delays_unpaired=pulse_delays_unpaired,
        pulse_delays_all=pulse_delays_all,
        channel_names=channel_names,
    )
    time_series_array = np.asarray(time_series, dtype=np.float64)
    transition_series_array = np.asarray(transition_series, dtype=np.uint32)
    excitation_series_array = np.asarray(excitation_series, dtype=np.int16)
    space_multiple_excitations(time_series=time_series_array)
    transition_series_array = insert_excitations(
        transition_series=transition_series_array,
        transition_set=transition_set,
        excitation_series=excitation_series_array,
    )
    state_series = get_state_series(
        transition_set=transition_set, transition_series=transition_series_array
    )

    simulation_object = si.Simulation(transition_set=transition_set)
    simulation_object.time_series = time_series_array
    simulation_object.transition_series = transition_series_array
    simulation_object.state_series = state_series

    return (
        event_time_series,
        event_time_points,
        pulse_delays_paired_output,
        pulse_delays_unpaired_output,
        pulse_delays_all_output,
        simulation_object,
    )
