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
    """Validate and accumulate channel-resolved detection probabilities."""
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
    lifetimes_DA: list[list[float]],
    lifetimes_D: list[list[float]],
    lifetimes_all: list[list[float]],
    channel_names: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    dict[str, npt.NDArray[np.float64]] | None,
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
]:
    """Convert collected photons to channel-resolved return values."""
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
    lifetimes_DA_output = {
        name: np.asarray(lifetimes_DA[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }
    lifetimes_D_output = {
        name: np.asarray(lifetimes_D[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }
    lifetimes_all_output = {
        name: np.asarray(lifetimes_all[index], dtype=np.float64)
        for index, name in enumerate(channel_names)
    }

    return (
        event_time_series,
        event_time_points,
        lifetimes_DA_output,
        lifetimes_D_output,
        lifetimes_all_output,
    )


def simulate_TCSPC(
    transition_set: TransitionSet,
    detection_probabilities: npt.ArrayLike,
    channel_names: Sequence[str],
    et_transition_ids: Sequence[int] | None = None,
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
    are in the ground state. The S1 durations used for fluorescence lifetimes are the
    time differences of photon emission to last laser pulse. Only detected photons are
    taken into account. The simulation approximates laser pulses to be instantaneous.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes
    detection_probabilities
        Detection probability for every combined transition and channel.
    channel_names
        Channel names in the order used by detection_probabilities.
    et_transition_ids
        Contains the combined_state_transition indices that are emissions when energy
        transfer is available.
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
        Contains the time points (increasing by a defined time interval) as index and
        the number of detected emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
        If store_time_points is False, this will be None.
    lifetimes_DA : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        available, grouped by channel.
    lifetimes_D : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        not available, grouped by channel.
    lifetimes_all : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of all detected emissions, grouped by channel.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    transition_matrix_non_exc = transition_set.transition_matrix
    row_sums_non_exc = transition_set.row_sums
    cumulative_detection_probabilities, resolved_channel_names = (
        _prepare_detection_configuration(
            transition_count=transition_matrix_non_exc.shape[1],
            detection_probabilities=detection_probabilities,
            channel_names=channel_names,
        )
    )

    transition_matrix_sorted_indices_non_exc = np.argsort(
        transition_matrix_non_exc, axis=1
    )
    sorted_transition_matrix_non_exc = np.take_along_axis(
        arr=transition_matrix_non_exc,
        indices=transition_matrix_sorted_indices_non_exc,
        axis=1,
    )
    cumsum_sorted_trm_non_exc = np.cumsum(sorted_transition_matrix_non_exc, axis=1)
    number_fluorophores = transition_set.fluorophore_system.count
    if excitation_rates is None:
        excitation_rates = {}
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            f.identity
            for f in transition_set.fluorophore_system.fluorophores
            if f.name == fluorophore and f.identity is not None
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
    time_stamps = np.linspace(
        0, np.ceil(frames) * seconds_per_frame, int(np.ceil(frames)) + 1
    )
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        logger.warning(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(seconds_per_frame / time_between_pulses)):.1e} pulses).",
            stacklevel=2,
        )
    logger.warning(
        f"the last frame (of index {time_stamps[-1]}) has {frames - int(frames):.2e} "
        "times the pulses of other frames.",
        stacklevel=2,
    )
    photon_collector = np.zeros(
        (time_stamps.size, len(resolved_channel_names)), dtype=np.int64
    )
    df = transition_set.combined_state_transitions_df
    current_state_index = df.loc[
        df["final_state"] == tuple(np.zeros(number_fluorophores))
    ].index[0]
    lifetimes_D: list[list[float]] = [[] for _ in resolved_channel_names]
    lifetimes_DA: list[list[float]] = [[] for _ in resolved_channel_names]
    lifetimes_all: list[list[float]] = [[] for _ in resolved_channel_names]
    remember: tuple[int, float] = (0, np.inf)
    if et_transition_ids is None:
        et_transition_ids = []
    i = 0
    j = 1
    k = 0
    m = 1
    S0 = 0
    S1 = 1
    # skip is True if next transition shall be the remembered one without the chance of
    # another transition that is not excitation to come before it
    skip = False
    # checked is True if next excitation pulse that excites a fluorophore happens before
    # a remembered transition
    checked = False
    # not_broken is False if no excitable states AND no possible future transitions
    # (i.e., to differentiate between S0 and B since S0 leads to current_state_lambda=0)
    not_broken = True
    random_numbers_exc = rng.uniform(low=0, high=1, size=(size, excitable_indices.size))
    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    if store_time_points:
        time_points: list[list[float]] | None = [[] for _ in resolved_channel_names]
    else:
        time_points = None

    while i < number_pulses:
        i += 1
        if i >= j * size:
            j = int(np.floor(i / size)) + 1
            random_numbers_exc = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
        time = (i - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(df["final_state"][current_state_index])
        if not checked:
            S0s = np.where(current_states[excitable_indices] == S0)[0]
            # if there are any S0 states, excitations can occur
            if S0s.size != 0:
                not_broken = True
                # multiple excitations can occur within one pulse
                excitations = np.where(
                    random_numbers_exc[i - (j - 1) * size, S0s]
                    < excitation_probabilities[excitable_indices[S0s]]
                )[0]
                # if excitations occur, the fluorophores are set to S1
                if excitations.size != 0:
                    current_states[excitable_indices[S0s[excitations]]] = S1
                # if no excitation happened and if a transition that is remembered could
                # take place, the simulation continues from there
                elif remember[1] < next_pulse_time:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    # here no new calculation of next_pulse_time since the time is smaller
                    # than current next_pulse_time, hence the calculation would result
                    # in the same value
                # if no excitation happened and if a potentially remembered transition
                # could not take place, and if the number of pulses until an excitation
                # event have not been checked immediately before, the number of pulses
                # until the next excitation event is calculated
                else:
                    number_pulses_until_next_excitation = rng.geometric(
                        p=excitation_probabilities[excitable_indices[S0s]]
                    )
                    indices_minimum = np.where(
                        number_pulses_until_next_excitation
                        == np.min(number_pulses_until_next_excitation)
                    )[0]
                    potential_i = (
                        i + number_pulses_until_next_excitation[indices_minimum[0]] - 1
                    )
                    # if the next excitation event(s) is before a remembered transition takes
                    # place, the excitation(s) is realized at the calculated pulse
                    if potential_i * time_between_pulses < remember[1]:
                        current_states[excitable_indices[S0s[indices_minimum]]] = S1
                        current_state_index = df[
                            df["final_state"] == tuple(current_states)
                        ].index[0]
                        i = potential_i
                        checked = True
                        continue
                    # if the next excitation event is after a remembered transition takes
                    # place, the simulation continues from there
                    else:
                        skip = True
                        next_transition = remember[0]
                        time = remember[1]
                        # here the calculation is necessary since the time is larger than
                        # current next_pulse_time
                        i = int(np.floor(time / time_between_pulses))
                        next_pulse_time = (i + 1) * time_between_pulses
            # if there are no S0 states, no excitations can occur. If something is
            # remembered, the simulation continues from there
            else:
                # if not_broken is False, this means that in the previous pulse it was
                # already at this point (all non-excitable) and did not get past the
                # current_state_lambda above 0. This means that the Markov chain has
                # encountered an absorbing state that is not excitable S0.
                if not not_broken:
                    logger.warning(
                        "All fluorophores underwent photobleaching or entered "
                        "another Markov chain absorbing state.",
                        stacklevel=2,
                    )
                    return _prepare_photon_outputs(
                        photon_collector=photon_collector,
                        time_stamps=time_stamps,
                        time_points=time_points,
                        lifetimes_DA=lifetimes_DA,
                        lifetimes_D=lifetimes_D,
                        lifetimes_all=lifetimes_all,
                        channel_names=resolved_channel_names,
                    )
                # note that not_broken will be set to False. The rembered transition will be
                # carried out and after that, if the transition was not to an all-absorbing
                # state (i.e., current_state_lambda != 0), not_broken will be set to True.
                # current_state_lambda could also be 0 due to S0 states, but in that case,
                # not_broken will be set to True since S0s.size != 0.
                elif remember[1] != np.inf:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    # here the calculation can be necessary or result is the same number as
                    # current next_pulse_time
                    i = int(np.floor(time / time_between_pulses))
                    next_pulse_time = (i + 1) * time_between_pulses
                not_broken = False

        checked = False
        current_state_index = df[df["final_state"] == tuple(current_states)].index[0]
        while time < next_pulse_time:
            if not skip:
                k += 1
                if k == m * size:
                    m += 1
                    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
                current_state_lambda = row_sums_non_exc[current_state_index]
                if current_state_lambda == 0:
                    remember = (0, np.inf)
                    break
                not_broken = True
                transition_time = (1 / current_state_lambda) * np.log(
                    1 / random_numbers[k - (m - 1) * size, 0]
                )
                time += transition_time
                sorted_index = np.searchsorted(
                    cumsum_sorted_trm_non_exc[current_state_index],
                    random_numbers[k - (m - 1) * size, 1],
                )

                next_transition = transition_matrix_sorted_indices_non_exc[
                    current_state_index, sorted_index
                ]
                if time > next_pulse_time:
                    remember = (int(next_transition), float(time))
                    break
            skip = False
            current_state_index = next_transition
            cumulative_probabilities = cumulative_detection_probabilities[
                next_transition
            ]
            if cumulative_probabilities[-1] == 0:
                continue
            channel_index = np.searchsorted(
                cumulative_probabilities,
                random_numbers[k - (m - 1) * size, 2],
                side="right",
            )
            if channel_index < len(resolved_channel_names):
                lifetime = time - last_pulse_time
                if next_transition in et_transition_ids:
                    lifetimes_DA[channel_index].append(lifetime)
                else:
                    lifetimes_D[channel_index].append(lifetime)
                lifetimes_all[channel_index].append(lifetime)
                frame = int(np.ceil(time / seconds_per_frame))
                try:
                    photon_collector[frame, channel_index] += 1
                    if time_points is not None:
                        time_points[channel_index].append(time)
                except IndexError:
                    pass

    return _prepare_photon_outputs(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        lifetimes_DA=lifetimes_DA,
        lifetimes_D=lifetimes_D,
        lifetimes_all=lifetimes_all,
        channel_names=resolved_channel_names,
    )


def simulate_TCSPC_detailed(
    transition_set: TransitionSet,
    detection_probabilities: npt.ArrayLike,
    channel_names: Sequence[str],
    et_transition_ids: Sequence[int] | None = None,
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
    are in the ground state. The S1 durations used for fluorescence lifetimes are the
    time differences of photon emission to last laser pulse. This is the detailed
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
    et_transition_ids
        Contains the combined_state_transition indices that are emissions when energy
        transfer is available.
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
        Contains the time points (increasing by a defined time interval) as index and
        the number of detected emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
        If store_time_points is False, this will be None.
    lifetimes_DA : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        available, grouped by channel.
    lifetimes_D : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        not available, grouped by channel.
    lifetimes_all : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of all detected emissions, grouped by channel.
    simulation_object : fluopy.simulation.Simulation
        Container for simulation-associated attributes and methods.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    transition_matrix_non_exc = transition_set.transition_matrix
    row_sums_non_exc = transition_set.row_sums
    cumulative_detection_probabilities, resolved_channel_names = (
        _prepare_detection_configuration(
            transition_count=transition_matrix_non_exc.shape[1],
            detection_probabilities=detection_probabilities,
            channel_names=channel_names,
        )
    )

    transition_matrix_sorted_indices_non_exc = np.argsort(
        transition_matrix_non_exc, axis=1
    )
    sorted_transition_matrix_non_exc = np.take_along_axis(
        arr=transition_matrix_non_exc,
        indices=transition_matrix_sorted_indices_non_exc,
        axis=1,
    )
    cumsum_sorted_trm_non_exc = np.cumsum(sorted_transition_matrix_non_exc, axis=1)
    number_fluorophores = transition_set.fluorophore_system.count
    if excitation_rates is None:
        excitation_rates = {}
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            f.identity
            for f in transition_set.fluorophore_system.fluorophores
            if f.name == fluorophore and f.identity is not None
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
    time_stamps = np.linspace(
        0, np.ceil(frames) * seconds_per_frame, int(np.ceil(frames)) + 1
    )
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        logger.warning(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(seconds_per_frame / time_between_pulses)):.1e} pulses).",
            stacklevel=2,
        )
    logger.warning(
        f"the last frame (of index {time_stamps[-1]}) has {frames - int(frames):.2e} "
        "times the pulses of other frames.",
        stacklevel=2,
    )
    photon_collector = np.zeros(
        (time_stamps.size, len(resolved_channel_names)), dtype=np.int64
    )
    df = transition_set.combined_state_transitions_df
    current_state_index = df.loc[
        df["final_state"] == tuple(np.zeros(number_fluorophores))
    ].index[0]
    time_series: list[float] = [0.0]
    transition_series: list[int] = []
    excitation_series: list[int] = []
    lifetimes_D: list[list[float]] = [[] for _ in resolved_channel_names]
    lifetimes_DA: list[list[float]] = [[] for _ in resolved_channel_names]
    lifetimes_all: list[list[float]] = [[] for _ in resolved_channel_names]
    remember: tuple[int, float] = (0, np.inf)
    if et_transition_ids is None:
        et_transition_ids = []
    i = 0
    j = 1
    k = 0
    m = 1
    S0 = 0
    S1 = 1
    # skip is True if next transition shall be the remembered one wihtout the chance of
    # another transition that is not excitation to come before it
    skip = False
    # checked is True if next excitation pulse that excites a fluorophore happens before
    # a remembered transition
    checked = False
    # not_broken is False if no excitable states AND no possible future transitions
    # (i.e., to differentiate between S0 and B since S0 leads to current_state_lambda=0)
    not_broken = True
    random_numbers_exc = rng.uniform(low=0, high=1, size=(size, excitable_indices.size))
    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    if store_time_points:
        time_points: list[list[float]] | None = [[] for _ in resolved_channel_names]
    else:
        time_points = None

    while i < number_pulses:
        i += 1
        if i >= j * size:
            j = int(np.floor(i / size)) + 1
            random_numbers_exc = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
        time = (i - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(df["final_state"][current_state_index])
        if not checked:
            S0s = np.where(current_states[excitable_indices] == S0)[0]
            # if there are any S0 states, excitations can occur
            if S0s.size != 0:
                not_broken = True
                # multiple excitations can occur within one pulse
                excitations = np.where(
                    random_numbers_exc[i - (j - 1) * size, S0s]
                    < excitation_probabilities[excitable_indices[S0s]]
                )[0]
                # if excitations occur, the fluorophores are set to S1
                if excitations.size != 0:
                    current_states[excitable_indices[S0s[excitations]]] = S1
                    excitation_series.extend(
                        excitable_indices[S0s[excitations]].tolist()
                    )
                    time_series.extend([time] * excitations.size)
                # if no excitation happened and if a transition that is remembered could
                # take place, the simulation continues from there
                elif remember[1] < next_pulse_time:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    # here no new calculation of next_pulse_time since the time is smaller
                    # than current next_pulse_time, hence the calculation would result
                    # in the same value
                # if no excitation happened and if a potentielly remembered transition
                # could not take place, and if the number of pulses until an excitation
                # event have not been checked immidiately before, the number of pulses
                # until the next excitation event is calculated
                else:
                    number_pulses_until_next_excitation = rng.geometric(
                        p=excitation_probabilities[excitable_indices[S0s]]
                    )
                    indices_minimum = np.where(
                        number_pulses_until_next_excitation
                        == np.min(number_pulses_until_next_excitation)
                    )[0]
                    potential_i = (
                        i + number_pulses_until_next_excitation[indices_minimum[0]] - 1
                    )
                    # if the next excitation event(s) is before a remembered transition takes
                    # place, the excitation(s) is realized at the calculated pulse
                    if potential_i * time_between_pulses < remember[1]:
                        current_states[excitable_indices[S0s[indices_minimum]]] = S1
                        excitation_series.extend(
                            excitable_indices[S0s[indices_minimum]].tolist()
                        )
                        time_series.extend(
                            [potential_i * time_between_pulses] * indices_minimum.size
                        )
                        current_state_index = df[
                            df["final_state"] == tuple(current_states)
                        ].index[0]
                        i = potential_i
                        checked = True
                        continue
                    # if the next excitation event is after a remembered transition takes
                    # place, the simulation continues from there
                    else:
                        skip = True
                        next_transition = remember[0]
                        time = remember[1]
                        # here the calculation is necessary since the time is larger than
                        # current next_pulse_time
                        i = int(np.floor(time / time_between_pulses))
                        next_pulse_time = (i + 1) * time_between_pulses
            # if there are no S0 states, no excitations can occur. If something is
            # remembered, the simulation continues from there
            else:
                # if not_broken is False, this means that in the previous pulse it was
                # already at this point (all non-excitable) and did not get past the
                # current_state_lambda above 0. This means that the Markov chain has
                # encountered an absorbing state that is not excitable S0.
                if not not_broken:
                    logger.warning(
                        "All fluorophores underwent photobleaching or entered "
                        "another Markov chain absorbing state.",
                        stacklevel=2,
                    )
                    return_values = prepare_return_values(
                        photon_collector=photon_collector,
                        time_stamps=time_stamps,
                        time_points=time_points,
                        lifetimes_DA=lifetimes_DA,
                        lifetimes_D=lifetimes_D,
                        lifetimes_all=lifetimes_all,
                        time_series=time_series,
                        transition_series=transition_series,
                        excitation_series=excitation_series,
                        transition_set=transition_set,
                        channel_names=resolved_channel_names,
                    )

                    return return_values

                # note that not_broken will be set to False. The rembered transition will be
                # carried out and after that, if the transition was not to an all-absorbing
                # state (i.e., current_state_lambda != 0), not_broken will be set to True.
                # current_state_lambda could also be 0 due to S0 states, but in that case,
                # not_broken will be set to True since S0s.size != 0.
                elif remember[1] != np.inf:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    # here the calculation can be necessary or result is the same number as
                    # current next_pulse_time
                    i = int(np.floor(time / time_between_pulses))
                    next_pulse_time = (i + 1) * time_between_pulses
                not_broken = False

        checked = False
        current_state_index = df[df["final_state"] == tuple(current_states)].index[0]
        while time < next_pulse_time:
            if not skip:
                k += 1
                if k == m * size:
                    m += 1
                    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
                current_state_lambda = row_sums_non_exc[current_state_index]
                if current_state_lambda == 0:
                    remember = (0, np.inf)
                    break
                not_broken = True
                transition_time = (1 / current_state_lambda) * np.log(
                    1 / random_numbers[k - (m - 1) * size, 0]
                )
                time += transition_time
                sorted_index = np.searchsorted(
                    cumsum_sorted_trm_non_exc[current_state_index],
                    random_numbers[k - (m - 1) * size, 1],
                )

                next_transition = transition_matrix_sorted_indices_non_exc[
                    current_state_index, sorted_index
                ]
                if time > next_pulse_time:
                    remember = (int(next_transition), float(time))
                    break
            skip = False
            current_state_index = next_transition
            transition_series.append(int(next_transition))
            excitation_series.append(-1)
            time_series.append(time)
            cumulative_probabilities = cumulative_detection_probabilities[
                next_transition
            ]
            if cumulative_probabilities[-1] == 0:
                continue
            channel_index = np.searchsorted(
                cumulative_probabilities,
                random_numbers[k - (m - 1) * size, 2],
                side="right",
            )
            if channel_index < len(resolved_channel_names):
                lifetime = time - last_pulse_time
                if next_transition in et_transition_ids:
                    lifetimes_DA[channel_index].append(lifetime)
                else:
                    lifetimes_D[channel_index].append(lifetime)
                lifetimes_all[channel_index].append(lifetime)
                frame = int(np.ceil(time / seconds_per_frame))
                try:
                    photon_collector[frame, channel_index] += 1
                    if time_points is not None:
                        time_points[channel_index].append(time)
                except IndexError:
                    pass

    # if checked, the last simulated pulse is past the last existing pulse
    if checked:
        time_series = time_series[:-1]
        excitation_series = excitation_series[:-1]
    return_values = prepare_return_values(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        lifetimes_DA=lifetimes_DA,
        lifetimes_D=lifetimes_D,
        lifetimes_all=lifetimes_all,
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
    logger.warning(
        "Multiple excitations at the same time point are spaced by a minimal "
        "amount. This also spaces other transitions whose time was too small "
        "to be differentiated given the floating point precision. Hence, "
        "transitions with mean time lower than expected will now have mean "
        "time higher than expected.",
        stacklevel=2,
    )
    indices = np.unique(time_series, return_index=True)[1]
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
    if indices_transitions.size == 0:
        current_state = np.zeros(number_fluorophores, dtype=int)
        excitation_ids = np.empty(
            excitation_series.size, dtype=transition_series_ad.dtype
        )
        for i, fluorophore_id in enumerate(excitation_series):
            initial_state = tuple(current_state)
            current_state[fluorophore_id] = 1
            final_state = tuple(current_state)
            matching_excitations = excitations[
                (excitations["initial_state"] == initial_state)
                & (excitations["final_state"] == final_state)
            ]
            excitation_ids[i] = matching_excitations.index[0]
        return excitation_ids

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

    return transition_series_ad


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
    lifetimes_DA: list[list[float]],
    lifetimes_D: list[list[float]],
    lifetimes_all: list[list[float]],
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
    lifetimes_DA
        Contains the S1 durations of detected emissions when energy transfer
        available.
    lifetimes_D
        Contains the S1 durations of detected emissions when energy transfer
        not available.
    lifetimes_all
        Contains the S1 durations of all detected emissions.
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
        Contains the time points (increasing by a defined time interval) as index and
        the number of detected emissions in each channel as columns.
    event_time_points : dict[str, npt.NDArray[np.float64]] or None
        The time points at which emissions are detected, grouped by channel.
    lifetimes_DA : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        available, grouped by channel.
    lifetimes_D : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of detected emissions when energy transfer
        not available, grouped by channel.
    lifetimes_all : dict[str, npt.NDArray[np.float64]]
        Contains the S1 durations of all detected emissions, grouped by channel.
    simulation_object : fluopy.simulation.Simulation
        Container for simulation-associated attributes and methods.
    """
    (
        event_time_series,
        event_time_points,
        lifetimes_DA_output,
        lifetimes_D_output,
        lifetimes_all_output,
    ) = _prepare_photon_outputs(
        photon_collector=photon_collector,
        time_stamps=time_stamps,
        time_points=time_points,
        lifetimes_DA=lifetimes_DA,
        lifetimes_D=lifetimes_D,
        lifetimes_all=lifetimes_all,
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
        lifetimes_DA_output,
        lifetimes_D_output,
        lifetimes_all_output,
        simulation_object,
    )
