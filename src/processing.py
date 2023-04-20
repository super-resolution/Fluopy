import numpy as np


def multiple_transitions(joined_transitions, joined_states, single_transitions):
    """
    In case there are multiple transitions between the same two joined states, the selection of which transition has
    happened will be based on the cumulative sum of probabilities and inserting a random number between 0 and 1 (similar
    to the strategy of Gillespie algorithm). This function generates the cumulative sum and a correspondingly sorted
    index array.

    Parameters
    ----------
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), single_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.

    Returns
    -------
    transition_cum_sum : np.ndarray
        Of shape (joined_states.index.size, joined_states.index.size, single_transitions.index.size).
        Contains the cumulative sum of all available transitions of each joined_state combination.
    transition_sorted_indices : np.ndarray
        Of shape (joined_states.index.size, joined_states.index.size, single_transitions.index.size).
        Contains the ids of the single_transitions ordered in a way such that it corresponds to the order of cum_sum.
    """
    uni_dir = joined_states.index.size
    transition_count = single_transitions.index.size
    transitions = np.empty(shape=(uni_dir, uni_dir, transition_count))
    transitions[:] = 0
    for index_pair, transition_id, rate in zip(joined_transitions['joined_states_id'],
                                               joined_transitions['single_transition_id'],
                                               joined_transitions['rate']):
        transitions[index_pair][transition_id] = rate  # the index pair may occur multiple times, the transition id in
        # combination with the index pair is unique

    # normalization to yield probabilities

    row_sums = transitions.sum(axis=2)
    non_zero = np.where(row_sums != 0)
    row_sums = row_sums[:, :, np.newaxis]
    transitions[non_zero] = transitions[non_zero] / row_sums[non_zero]

    # sort indices and cum_sum, similar to Gillespie algorithm

    transition_sorted_indices = np.argsort(transitions, axis=2)
    sorted_transitions = np.take_along_axis(transitions, transition_sorted_indices, axis=2)
    transition_cum_sum = np.cumsum(sorted_transitions, axis=2)

    return transition_cum_sum, transition_sorted_indices


def searchsorted2d(arr_2d, arr_1d):
    """
    From https://stackoverflow.com/questions/56471109/how-to-vectorize-with-numpy-searchsorted-in-a-2d-array. Because
    of some conditions, the code below is largely modified yielding a simpler version.
    Applies the numpy.searchsorted function to a 2D array using a 1D array.

    Parameters
    ----------
    arr_2d : np.ndarray
        A 2D array which resembles the input array. Of shape (x, y).
    arr_1d : np.ndarray
        Values to insert in arr_2d. Of shape (x,).

    Returns
    -------
    result : np.ndarray
        A 1D array of shape (arr_1d.shape[0],) which contains the indices at which elements of arr_1d have to be
        inserted into arr_2d equal to [np.searchsorted(a[i], b[i]) for i in range(arr_1d.shape[0])].
    """
    # Since the cumulative sum of probabilities always adds up to 1 and the numbers in arr_1d are virtually always
    # below 1, the (np.maximum(arr_2d.max(1) - arr_2d.min(1) + 1, arr_1d) + 1) can be simplified to the number 2.
    scale = np.arange(0, 2 * arr_1d.shape[0], 2)

    a_scaled = (arr_2d + scale[:, None]).ravel()  # adds the offset and converts the result into a 1d array
    b_scaled = arr_1d + scale

    # Use np.searchsorted on scaled arrays and then subtract index offsets introduced by flattening
    result = np.searchsorted(a_scaled, b_scaled) - np.arange(scale.shape[0]) * arr_2d.shape[1]

    return result


def generate_transition_series(state_series, transition_cum_sum, transition_sorted_indices, seed=100):
    """
    Generates a transition series based on the current and future joined state series and, in case of multiple possible
    transitions, on probability.

    Parameters
    ----------
    state_series : np.ndarray
        The third return value of gillespie_algorithm.direct_method.
        The simulated consecutive joined states ids.
    transition_cum_sum : np.ndarray
        The first return value of multiple_transitions.
        Of shape (joined_states.index.size, joined_states.index.size, single_transitions.index.size).
        Contains the cumulative sum of all available transitions of each joined_state combination.
    transition_sorted_indices : np.ndarray
        The second return value of multiple_transitions.
        Of shape (joined_states.index.size, joined_states.index.size, single_transitions.index.size).
        Contains the ids of the single_transitions ordered in a way such that it corresponds to the order of cum_sum.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    transition_series : np.ndarray
        Contains the NEXT transition id for each corresponding simulated joined state (except the last).
    """
    rng = np.random.default_rng(seed)
    current_states = state_series[:-1]
    future_states = state_series[1:]
    values_to_insert = rng.uniform(low=0, high=1, size=current_states.shape[0])  # 1d array

    specific_cum_sums = transition_cum_sum[current_states, future_states, :]  # transition_cum_sum is a 3D array,
    # specific_cum_sums is a 2D array of shape (current_states.shape[0], single_transitions.index.size)
    insert_at = searchsorted2d(specific_cum_sums, values_to_insert)  # 1d array

    transition_series = transition_sorted_indices[current_states, future_states, insert_at]

    return transition_series


def convert_single_state_series(number_fluorophores, state_series, joined_states):
    """
    Converts the state series containing the joined states to a single state series for each fluorophore.

    Parameters
    ----------
    number_fluorophores: int
        Number of fluorophores of the system.
    state_series : np.ndarray
        The third return value of gillespie_algorithm.direct_method.
        The simulated consecutive joined states ids.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.

    Returns
    -------
    single_state_series : np.ndarray
        State series for each individual fluorophore (hence, simulated consecutive single states ids).
    """
    single_state_series = np.zeros(shape=(number_fluorophores, state_series.shape[0]))

    for index, row in joined_states.iterrows():
        single_state_ids = row['single_states']
        if index in state_series:
            mask = np.where(state_series == index)[0]
            single_state_id = np.expand_dims(single_state_ids, axis=1)  # converts e.g. [0, 0] to [[0], [0]]
            single_state_series[:, mask] = single_state_id

    return single_state_series


def time_occurrence_statistics(number_fluorophores, single_states, single_transitions, time_series,
                               transition_series, single_state_series):
    """
    Returns two dictionaries containing statistics of single state occupations and transition occurrences (distribution,
    mean and total of lifetime as well as probability densities of occurrences (see keys below for more information)).

    Parameters
    ----------
    number_fluorophores : int
        Number of fluorophores of the system.
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        single state.
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.
    time_series : np.ndarray
        The first return value of gillespie_algorithm.direct_method.
        The simulated time points at which the corresponding joined states occur.
    transition_series : np.ndarray
        The return value of generate_transition_series.
        Contains the NEXT transition id for each corresponding simulated joined state (except the last).
    single_state_series : np.ndarray
        The return value of convert_single_state_series.
        State series for each individual fluorophore (hence, simulated consecutive single states ids).

    Returns
    -------
    single_state_lifetimes : dict
        Keys are
        'single_state_occurrences' (np.ndarray for each fluorophore containing their single state ids),
        'single_state_occurrences_all' (the above summarized for all fluorophores),
        'lifetimes_fluorophore' (np.ndarray for each fluorophore containing their lifetimes corresponding to the single
            states listed in single_state_occurrences),
        'lifetimes_single_states' (list of each fluorophore containing np.ndarray for each single state with lifetimes),
        'lifetimes_single_states_all' (the above summarized for all fluorophores),
        'mean_lifetimes' (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        'mean_lifetimes_all' (the above summarized for all fluorophores),
        'total_lifetimes' (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        'total_lifetimes_all' (the above summarized for all fluorophores).
    transition_lifetimes : dict
        Keys are
        'transition_occurrences' (np.ndarray for each fluorophore containing their transition ids),
        'transition_occurrences_all' (the above summarized for all fluorophores),
        'transition_times' (list for each fluorophores containing np.ndarray for each transition with lifetimes),
        'transition_times_all' (the above summarized for all fluorophores),
        'mean_transition_times' (np.ndarray with axis 0 for the fluorophores and axis 1 for the transition),
        'mean_transition_times_all' (the above summarized for all fluorophores).
    """
    single_states = np.fromiter(single_states.keys(), dtype=int)
    single_state_occurrences = []
    lifetimes_fluorophore = []
    lifetimes_single_states = []
    lifetimes_single_states_all = [np.array([]) for _ in single_states]
    mean_lifetimes = np.zeros(shape=(number_fluorophores, single_states.shape[0]))
    # note that the mean lifetime of a state depends on all outgoing transitions
    total_lifetimes = np.zeros(shape=(number_fluorophores, single_states.shape[0]))

    transition_times = []
    transition_times_all = [np.array([]) for _ in single_transitions.index]
    mean_transition_times = np.zeros(shape=(number_fluorophores, single_transitions.index.size))
    transition_occurrences = []

    for i in range(number_fluorophores):
        current_fluorophore_series = single_state_series[i]
        diffs = np.diff(current_fluorophore_series)
        changes_at = np.where(diffs != 0)[0]
        initial_single_states = current_fluorophore_series[changes_at]
        transitions = transition_series[changes_at]
        single_state_occurrence = np.append(initial_single_states, current_fluorophore_series[np.max(changes_at) + 1])
        changed = changes_at + 1  # state 0 to state 1 gives entry of 1 in
        # diff at index 0, but the corresponding time is at index 1, hence +1
        total_times = time_series[changed]
        time_intervals = np.diff(total_times)
        time_intervals = np.insert(time_intervals, 0, total_times[0])
        single_state_occurrences.append(single_state_occurrence)
        lifetimes_fluorophore.append(time_intervals)
        transition_occurrences.append(transitions)

        time_intervals_states = []
        for j, state in enumerate(single_states):
            time_intervals_state = time_intervals[np.where(initial_single_states == state)]
            if time_intervals_state.size == 0:
                mean = np.nan
                total = np.nan
            else:
                mean = np.mean(time_intervals_state)
                total = np.sum(time_intervals_state)
            time_intervals_states.append(time_intervals_state)
            mean_lifetimes[i, j] = mean
            total_lifetimes[i, j] = total
            lifetimes_single_states_all[j] = np.concatenate([lifetimes_single_states_all[j], time_intervals_state])
        lifetimes_single_states.append(time_intervals_states)

        time_intervals_transitions = []
        for transition in single_transitions.index:
            time_intervals_transition = time_intervals[np.where(transitions == transition)]
            if time_intervals_transition.size == 0:
                mean = np.nan
            else:
                mean = np.mean(time_intervals_transition)
            time_intervals_transitions.append(time_intervals_transition)
            mean_transition_times[i, transition] = mean
            transition_times_all[transition] = np.concatenate([transition_times_all[transition],
                                                              time_intervals_transition])
        transition_times.append(time_intervals_transitions)

    mean_lifetimes_all = np.array([np.mean(x) if x.size > 0 else np.nan for x in lifetimes_single_states_all])
    # mean has to be computed from whole population (e.g., mean_lifetimes_all != np.mean(mean_lifetimes)) because means
    # of subpopulations often arise using different sample sizes n. If n are known, one could weight the subpopulation
    # means with 1/n before summation, and after summation divide by the sum of all n.
    total_lifetimes_all = np.sum(total_lifetimes, axis=0)
    mean_transition_times_all = np.array([np.mean(x) if x.size > 0 else np.nan for x in transition_times_all])
    single_state_occurrences_all = np.concatenate(single_state_occurrences)
    transition_occurrences_all = transition_series

    single_state_lifetimes = {'single_state_occurrences': single_state_occurrences,
                              'single_state_occurrences_all': single_state_occurrences_all,
                              'lifetimes_fluorophore': lifetimes_fluorophore,
                              'lifetimes_single_states': lifetimes_single_states,
                              'lifetimes_single_states_all': lifetimes_single_states_all,
                              'mean_lifetimes': mean_lifetimes,
                              'mean_lifetimes_all': mean_lifetimes_all,
                              'total_lifetimes': total_lifetimes,
                              'total_lifetimes_all': total_lifetimes_all}
    transition_lifetimes = {'transition_occurrences': transition_occurrences,
                            'transition_occurrences_all': transition_occurrences_all,
                            'transition_times': transition_times,
                            'transition_times_all': transition_times_all,
                            'mean_transition_times': mean_transition_times,
                            'mean_transition_times_all': mean_transition_times_all}

    return single_state_lifetimes, transition_lifetimes
