"""
Various routines to deal with simulation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from fluopy.simulation import Simulation

__all__: list[str] = []


def get_bleaching_times(simulation: Simulation) -> npt.NDArray[np.float64]:
    """
    Get the times where photobleaching occurred - for each fluorophore, one number will
    be extracted. If no bleaching occurred, the entry will be np.nan. The elements will
    be sorted, np.nan will be at the end.

    Parameters
    ----------
    simulation
        Container for simulation-associated attributes

    Returns
    -------
    npt.NDArray[np.float64]
        Times where photobleaching occurred of shape (n_times,).
    """
    state_series = simulation.state_series
    time_series = simulation.time_series
    if state_series is None or time_series is None:
        raise ValueError("bleaching times require a completed simulation.")
    df = simulation.transition_set.transition_df
    absorbing_final_states = df[df["absorbing"]]["final_state"]
    bleached_state_values = np.unique([x.value for x in absorbing_final_states])
    if len(bleached_state_values) == 1:
        bleached_state = bleached_state_values[0]
    elif len(bleached_state_values) == 0:
        return np.full(state_series.shape[0], fill_value=np.nan)
    else:
        raise NotImplementedError(
            "Multiple bleaching states not yet implemented in " + "this function."
        )

    bleaching_times: list[float] = []
    for fluorophore_states in state_series:
        if fluorophore_states[-1] == bleached_state:
            first_occurence = np.where(fluorophore_states == bleached_state)[0][0]
            time = time_series[first_occurence]
        else:
            time = np.nan
        bleaching_times.append(time)
    bleaching_times_array = np.sort(np.asarray(bleaching_times, dtype=np.float64))

    return bleaching_times_array


def get_delta_bleaching_times(
    bleaching_times: npt.ArrayLike,
) -> list[npt.NDArray[np.float64]]:
    """
    Get the delta times between bleaching events.

    Parameters
    ----------
    bleaching_times
        Times where photobleaching occurred. Each run is a row, each fluorophore a
        column. Each row is sorted, np.nan will be at the end.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        The arrival times of photons between bleaching events. The timer starts at the
        previous bleaching event.
    """
    bleaching_times_array = np.asarray(bleaching_times, dtype=np.float64)
    delta_bleaching_times_all: list[npt.NDArray[np.float64]] = []
    previous_times = np.zeros(bleaching_times_array.shape[0], dtype=np.float64)
    for fluorophore in range(bleaching_times_array.shape[1]):
        bleaching_times_fluo = bleaching_times_array[:, fluorophore]
        delta_bleaching_times = bleaching_times_fluo - previous_times
        delta_bleaching_times = delta_bleaching_times[~np.isnan(delta_bleaching_times)]
        delta_bleaching_times_all.append(delta_bleaching_times)
        previous_times = bleaching_times_fluo

    return delta_bleaching_times_all
