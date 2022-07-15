import numpy as np
import pandas as pd
import multipletau as mp


def identify_duplices(all_names):
    duplices = []
    for i, joined_state_1 in enumerate(all_names):
        fluos_1 = joined_state_1.split("_")
        for e, joined_state_2 in enumerate(all_names[i+1:]):
            fluos_2 = joined_state_2.split("_")
            if sorted(fluos_2) == sorted(fluos_1):
                indices = [i, e+i+1]
                duplices.append(indices)
    return duplices


def uniques(duplices, state_series, joined_states):
    unique_series = state_series.copy()
    for duplex in duplices:
        indices = np.where(unique_series == duplex[1])  # with duplex[1] the higher index will be replaced
        unique_series[indices] = duplex[0]

    unique_states = np.unique(unique_series)

    unique_joined_states = []
    unique_names = []
    for joined_state in joined_states:
        if joined_state.value in unique_states:
            unique_joined_states.append(joined_state)
            unique_names.append(joined_state.name)

    return unique_series, unique_states, unique_joined_states, unique_names


def occupation_t(time_step_series, unique_series, unique_states):
    total_times = []
    mean_times = []

    for i in unique_states:
        mask = np.where(unique_series == i)[0] + 1
        if mask[-1] == len(time_step_series):
            total = np.inf
            mean = np.inf
        else:
            total = np.sum(time_step_series[mask])
            mean = np.mean(time_step_series[mask])

        total_times.append(total)
        mean_times.append(mean)

    return total_times, mean_times


def convert_unique_states(unique_states, unique_series):
    unique_series_converted = unique_series.copy()
    for i, identity in enumerate(unique_states):
        indices = np.where(unique_series_converted == identity)
        unique_series_converted[indices] = i

    return unique_series_converted


def autocorrelate(pandas_series, normalize=False, log=False, m=2, deltat=5e-3):
    """

    Parameters
    ----------
    pandas_series
    normalize
    log
    m : int
        Defines the number of points on one level, i.e., 1, 2, 4, 8, etc. E.g., m=4 leads to 1,2,3,4; 2,4,6,8; ...
    deltat : float
        The time between each entry of pands_series.

    Returns
    -------
    correl : np.ndarray

    or

    time : np.ndarray
    correl : np.ndarray
    """
    if normalize and not log:
        mean = np.mean(pandas_series.values)
        deviation = pandas_series.values - mean
        correl = np.correlate(deviation, deviation, mode="full")
        correl = correl[correl.size//2:]
        correl = np.divide(correl, np.arange(correl.size, 0, -1))
        correl = correl / mean**2

        return correl

    elif normalize and log:
        correl = mp.autocorrelate(pandas_series.values, m=m, deltat=deltat, normalize=True)
        time, correl = np.transpose(correl)

        return time, correl

    elif log:
        correl = mp.autocorrelate(pandas_serires.values, m=m, deltat=deltat)
        time, correl = np.transpose(correl)

        return time, correl

    else:
        correl = np.correlate(pandas_series.values, pandas_series.values, mode="full")
        correl = correl[correl.size//2:]

        return correl
