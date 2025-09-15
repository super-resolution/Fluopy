"""
Module routines
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def emission_post_processing(emis: Emissions, seed: RandomGeneratorSeed) -> None:
    """
    Post-processing of the emission data.

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
    df = simulation.transition_set.transition_df
    bleached_states = df[df["absorbing"]]["final_state"]
    bleached_states = [x.value for x in bleached_states]
    if len(bleached_states) == 1:
        bleached_state = bleached_states[0]
    elif len(bleached_states) == 0:
        return np.full(simulation.state_series.shape[0], np.nan)
    else:
        raise NotImplementedError(
            "Multiple bleaching states not yet implemented in " + "this function."
        )

    bleaching_times = []
    for state_series in simulation.state_series:
        if state_series[-1] == bleached_state:
            first_occurence = np.where(state_series == bleached_state)[0][0]
            time = simulation.time_series[first_occurence]
        else:
            time = np.nan
        bleaching_times.append(time)
    bleaching_times = np.sort(np.array(bleaching_times))

    return bleaching_times


def get_delta_bleaching_times(
    bleaching_times: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
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
    delta_bleaching_times_all = []
    previous_times = np.zeros_like(bleaching_times.shape[0])
    for fluorophore in range(bleaching_times.shape[1]):
        bleaching_times_fluo = bleaching_times[:, fluorophore]
        delta_bleaching_times = bleaching_times_fluo - previous_times
        delta_bleaching_times = delta_bleaching_times[~np.isnan(delta_bleaching_times)]
        delta_bleaching_times_all.append(delta_bleaching_times)
        previous_times = bleaching_times_fluo

    return delta_bleaching_times_all


def fingerprint_analysis(
    transition_set: TransitionSet,
    batch_size: int,
    batches: int,
    filepath: str | os.PathLike[Any],
    filename: str,
    seed: RandomGeneratorSeed,
    use_memmap: str | os.PathLike[Any] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[list[np.float64]]]:
    """
    Routine to perform fingerprint analysis. Returns the fingerprint data and the times
    where photobleaching occurred. Each run is stored as a parquet file. The bleaching
    times are stored as a numpy file.

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
    fingerprint_data : npt.NDArray[np.float64]
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
    bleaching_times_all_runs = []
    delta_times_photons_between_bleaching = [
        [] for _ in range(transition_set.fluorophore_system.count)
    ]
    for i in range(batches):
        output_file_run = Path(filepath) / f"single_runs_{filename}_batch_{i}.parquet"
        df = None
        for j in range(batch_size):
            simulation = si.Simulation(transition_set)
            simulation.run(size=1e6, seed=rng, end_time=300, use_memmap=use_memmap)
            bleaching_times = get_bleaching_times(simulation)
            bleaching_times_all_runs.append(bleaching_times)
            emis = em.Emissions(seed=rng, **PARAMS_EMIS)
            emis.extract(simulation)

            for n in range(transition_set.fluorophore_system.count):
                if n > 0:
                    start = bleaching_times[n - 1]
                else:
                    start = 0
                start_index = np.searchsorted(emis.event_time_points, start)
                if bleaching_times.size > n:
                    end_index = np.searchsorted(
                        emis.event_time_points, bleaching_times[n]
                    )
                    delta_times_photons_between_bleaching[n].append(
                        emis.event_time_points[start_index:end_index] - start
                    )
                else:
                    delta_times_photons_between_bleaching[n].append(
                        emis.event_time_points[start_index:] - start
                    )  # the delta, not the actual times
                    break

            emission_post_processing(emis, rng)
            emis.event_time_series.name = i * batch_size + j
            if df is None:
                df = emis.event_time_series
            else:
                df = pd.concat([df, emis.event_time_series], axis=1, ignore_index=False)
            fingerprint_data = fingerprint_data + emis.event_time_series
        df.to_parquet(output_file_run)
    bleaching_times_all_runs = np.array(bleaching_times_all_runs)
    np.save(output_file_bleach, bleaching_times_all_runs)
    fingerprint_data = fingerprint_data.cumsum() / fingerprint_data.sum()

    return (
        fingerprint_data,
        bleaching_times_all_runs,
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
    npt.NDArray[np.float64]
        Truncated fingerprint data - normalized cumulative emissions.
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


PARAMS_EMIS = {
    "frame_time": "1ms",
    "bandpass": [665, 731],
}


PARAMS_PULSE = {
    "time_between_pulses": 1.25e-8,
    "pulse_duration": 5e-11,
}
