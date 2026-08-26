"""
Various routines to deal with simulation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import emissions as em
from . import formulas as fo
from . import simulation as si

if TYPE_CHECKING:
    from fluopy.emissions import Emissions
    from fluopy.fluopy_types import RandomGeneratorSeed
    from fluopy.simulation import Simulation
    from fluopy.transitions import TransitionSet


__all__: list[str] = []


class EmissionsParameters(TypedDict):
    frame_time: str
    bandpass: tuple[float, float] | None


def emission_post_processing(emis: Emissions, seed: RandomGeneratorSeed) -> None:
    """
    A typical post-processing routine for emission data.

    Parameters
    ----------
    emis
        Container for emission-associated attributes.
    seed
        A seed to initialize the BitGenerator.

    Returns
    -------
    None
    """
    rng = np.random.default_rng(seed)
    photon_collection_rate = fo.calculate_photon_collection_rate(NA=1.45, n1=1.51)
    emis.add_photon_collection_objective(p=photon_collection_rate, seed=rng)
    emis.add_transmittance(p=0.9, seed=rng)  # mirror 90/100
    emis.add_transmittance(p=0.99, seed=rng)  # lens 1
    emis.add_transmittance(p=0.99, seed=rng)  # lens 2
    emis.add_quantum_efficiency(p=0.85, seed=rng)
    emis.add_poisson_noise(rate=0.6, seed=rng)
    emis.apply_threshold(threshold=10)


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
    bleached_state_values = [x.value for x in absorbing_final_states]
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
    delta_bleaching_times_all : list[npt.NDArray[np.float64]]
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


def fingerprint_analysis(
    transition_set: TransitionSet,
    batch_size: int,
    batches: int,
    filepath: str | Path,
    filename: str,
    seed: RandomGeneratorSeed,
    use_memmap: str | Path | None = None,
) -> tuple[
    pd.Series,
    npt.NDArray[np.float64],
    list[list[npt.NDArray[np.float64]]],
]:
    """
    Routine to perform fingerprint analysis. Returns the fingerprint data and the times
    where photobleaching occurred. Each batch is stored as a parquet file. The bleaching
    times are stored as a numpy file (once per function call, not per batch).

    Parameters
    ----------
    transition_set
        All relevant transitions and related attributes.
    batch_size
        Size of each batch.
    batches
        Number of batches.
    filepath
        Path to save the fingerprint data.
    filename
        The name of the file. In the case of single_run data, the name is extended with
        the batch number.
    seed
        A seed to initialize the BitGenerator.
    use_memmap
        If None, the data will be stored in memory. If a string, the data will be stored
        in a memmap file. Default is None.

    Returns
    -------
    fingerprint_data : pd.Series
        Fingerprint data - normalized cumulative emissions.
    bleaching_times : npt.NDArray[np.float64]
        Times where photobleaching occurred. Each run is a row, each fluorophore a
        column). Each row is sorted, np.nan will be at the end.
    delta_times_photons_between_bleaching : list[list[float]]
        The arrival times of photons between bleaching events. The timer starts at the
        previous bleaching event.
    """
    rng = np.random.default_rng(seed)
    fingerprint_data = pd.Series(
        np.zeros(300001),
        np.round(np.linspace(0, 300, 300001), decimals=12),
        dtype=np.int32,
    )
    output_file_bleach = Path(filepath) / f"bleaching_times_{filename}.npy"
    bleaching_times_all_runs: list[npt.NDArray[np.float64]] = []
    delta_times_photons_between_bleaching: list[list[npt.NDArray[np.float64]]] = [
        [] for _ in range(transition_set.fluorophore_system.count)
    ]
    for i in range(batches):
        output_file_run = Path(filepath) / f"single_runs_{filename}_batch_{i}.parquet"
        df: pd.DataFrame | pd.Series[Any] | None = None
        for j in range(batch_size):
            simulation = si.Simulation(transition_set=transition_set)
            simulation.run(
                size=1_000_000, seed=rng, end_time=300, use_memmap=use_memmap
            )
            bleaching_times = get_bleaching_times(simulation=simulation)
            bleaching_times_all_runs.append(bleaching_times)
            emis = em.Emissions(seed=rng, **PARAMS_EMIS)
            emis.extract(simulation=simulation)
            event_time_points = emis.event_time_points
            event_time_series = emis.event_time_series
            if event_time_points is None or event_time_series is None:
                raise RuntimeError("emission extraction did not produce event data.")

            for n in range(transition_set.fluorophore_system.count):
                if n > 0:
                    start = bleaching_times[n - 1]
                else:
                    start = 0
                start_index = np.searchsorted(event_time_points, start)
                if bleaching_times.size > n:
                    end_index = np.searchsorted(event_time_points, bleaching_times[n])
                    delta_times_photons_between_bleaching[n].append(
                        event_time_points[start_index:end_index] - start
                    )
                else:
                    delta_times_photons_between_bleaching[n].append(
                        event_time_points[start_index:] - start
                    )  # the delta, not the actual times
                    break

            emission_post_processing(emis=emis, seed=rng)
            event_time_series = emis.event_time_series
            if event_time_series is None:
                raise RuntimeError("emission processing removed the event time series.")
            event_time_series.name = i * batch_size + j
            if df is None:
                df = event_time_series
            else:
                df = pd.concat([df, event_time_series], axis=1, ignore_index=False)
            fingerprint_data = fingerprint_data + event_time_series
        if df is None:
            raise RuntimeError("batch did not produce emission data.")
        df.to_parquet(output_file_run)
    bleaching_times_array = np.asarray(bleaching_times_all_runs, dtype=np.float64)
    np.save(output_file_bleach, bleaching_times_array)
    fingerprint_data = fingerprint_data.cumsum() / fingerprint_data.sum()

    return (
        cast(pd.Series[Any], fingerprint_data),
        bleaching_times_array,
        delta_times_photons_between_bleaching,
    )


def truncate_fingerprints(
    fingerprint: pd.Series, low: int | None = None, high: int | None = None
) -> pd.Series:
    """
    Truncate the fingerprint data. The data will be normalized again.

    Parameters
    ----------
    fingerprint
        Fingerprint data - normalized cumulative emissions.
    low
        Lower bound for truncation.
    high
        Upper bound for truncation.

    Returns
    -------
    pd.Series
        Truncated fingerprint data - normalized (to [0, 1]) cumulative emissions.
    """
    if low is None:
        low = 0
    if high is None:
        high = -1
    fingerprint = fingerprint.iloc[low:high]
    fingerprint = fingerprint - fingerprint.iloc[0]
    fingerprint = fingerprint / fingerprint.iloc[-1]

    return fingerprint


PARAMS_DSTORM = {
    "irradiance": 2.5,
    "wavelength": 640,
    "dstorm": True,
    "dstorm_parameters": {
        "reducing_agent": "mea",
        "concentration": 100,
        "ph": 7.5,
    },
    "energy_transfer_parameters": {"overwrite": {"off": [1, 1e-4]}, "exclude": ["s0"]},
}


PARAMS_TROLOX = {
    "irradiance": 2.5,
    "wavelength": 640,
    "dstorm": False,
    "energy_transfer_parameters": {"exclude": ["s0"]},
}


PARAMS_EMIS: EmissionsParameters = {
    "frame_time": "1ms",
    "bandpass": (665, 731),
}


PARAMS_PULSE = {
    "time_between_pulses": 1.25e-8,
    "pulse_duration": 5e-11,
}
