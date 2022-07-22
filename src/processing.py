import numpy as np
import multipletau as mp


def identify_duplices(state_names):
    """
    Identifies states that are equal but in different order (e.g., S0_S1 and S1_S0) and stores their unique values as
    pairs.

    Parameters
    ----------
    state_names : Collection
        Contains all state names.

    Returns
    -------
    duplices : list
        Contains lists of two elements, where the first element is the unique value of a state and the second element
        is the unique value of another state that is ordered equal to the first state.
    """
    duplices = []
    for i, joined_state_1 in enumerate(state_names):
        fluos_1 = joined_state_1.split("_")
        for e, joined_state_2 in enumerate(state_names[i+1:]):
            fluos_2 = joined_state_2.split("_")
            if sorted(fluos_2) == sorted(fluos_1):
                indices = [i, e+i+1]
                duplices.append(indices)

    return duplices


def uniques(duplices, state_series, joined_states):
    """
    Removes duplices from state_series and returns other related collections.

    Parameters
    ----------
    duplices : list
        The return value of identify_duplices.
    state_series : np.ndarray
        Contains the consecutive state's unique values.
    joined_states : enum.EnumMeta
        The return value of initialize.state_pairs.

    Returns
    -------
    unique_series : np.ndarray
        Copy of state_series but with every second element of a list of duplices replaced by its first element.
    unique_states : np.ndarray
        Every state that occurs in unique_series.
    unique_joined_states : list
        Contains elements of joined_states if their unique value occurs in unique_states. It is ordered by unique value.
    unique_names : list
        Contains all unique state names (if their unique value occurs in unique_states).
    """
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
    """
    Returns the total and mean occupation time of each (unique) state during a Markov chain.

    Parameters
    ----------
    time_step_series : np.ndarray
        Contains the time step until the corresponding state occurs (starting from the previous state).
    unique_series : np.ndarray
        The first return value of uniques.
    unique_states : np.ndarray
        The second return value of uniques.

    Returns
    -------
    total_times : np.ndarray
        The total occupation time of each of unique_states.
    mean_times : np.ndarray
        The mean occupation time of each of unique_states.
    """
    total_times = np.zeros(shape=unique_states.size)
    mean_times = np.zeros(shape=unique_states.size)

    for i, value in enumerate(unique_states):
        mask = np.where(unique_series == value)[0] + 1
        if mask[-1] == len(time_step_series):
            total = np.inf
            mean = np.inf
        else:
            total = np.sum(time_step_series[mask])
            mean = np.mean(time_step_series[mask])
        total_times[i] = total
        mean_times[i] = mean

    return total_times, mean_times


def convert_unique_states(unique_series, unique_states):
    """
    Downscales each original unique value (listed in unique_states) within unique_series such that the ith largest value
    becomes i.

    Parameters
    ----------
    unique_series : np.ndarray
        The first return value of uniques.
    unique_states : np.ndarray
        The second return value of uniques.

    Returns
    -------
    unique_series_converted : np.ndarray
        Copy of unique_series but each value is replaced by its corresponding ranking number in ascending order.
    """
    unique_series_converted = unique_series.copy()
    for i, identity in enumerate(unique_states):
        indices = np.where(unique_series_converted == identity)
        unique_series_converted[indices] = i

    return unique_series_converted


def autocorrelate(pandas_series, normalize=False, log=False, m=2, deltat=5e-3):
    """
    Autocorrelation of pandas_series.values as exploited in typical fluorescence correlation spectroscopy setups.

    Parameters
    ----------
    pandas_series
    normalize : bool
        Whether to normalize the correlation. Note that this involves more than just dividing by mean squared.
    log : bool
        Whether to compute the autocorrelation on a logarithmic scale. Uses multipletau package.
    m : int
        Defines the number of points on one level, i.e., 1, 2, 4, 8, etc. E.g., m=4 leads to 1,2,3,4; 2,4,6,8; ...
    deltat : float
        The time between each entry of pands_series.

    Returns
    -------
    time : np.ndarray
        Time points that correspond to the autocorrelation values.
    correl : np.ndarray
        Autocorrelation values.
    """
    if normalize and log:
        correl = mp.autocorrelate(pandas_series.values, m=m, deltat=deltat, normalize=True)
        time, correl = np.transpose(correl)

        return time, correl

    elif log and not normalize:
        correl = mp.autocorrelate(pandas_series.values, m=m, deltat=deltat)
        time, correl = np.transpose(correl)

        return time, correl

    elif normalize and not log:
        mean = np.mean(pandas_series.values)
        deviation = pandas_series.values - mean  # delta I(t) (wiki) - fluctuation around the mean value
        correl = np.correlate(deviation, deviation, mode="full")
        correl = correl[correl.size//2:]
        correl = np.divide(correl, np.arange(correl.size, 0, -1))  # averaging
        correl = correl / mean**2  # normalization with mean squared
        time = pandas_series.index.values

        return time, correl

    else:
        correl = np.correlate(pandas_series.values, pandas_series.values, mode="full")
        correl = correl[correl.size//2:]
        time = pandas_series.index.values

        return time, correl
