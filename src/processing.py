import numpy as np


def multiple_transitions(joined_transitions, joined_states, single_transitions):
    """


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
    transition_sorted_indices : np.ndarray
    """
    uni_dir = joined_states.index.size
    transition_count = single_transitions.index.size
    x = np.empty(shape=(uni_dir, uni_dir, transition_count))
    x[:] = 0
    for index_pair, transition_id, rate in zip(joined_transitions['joined_states_id'],
                                               joined_transitions['single_transition_id'],
                                               joined_transitions['rate']):
        x[index_pair][transition_id] =   # the index pair may occur multiple times, the transition id in combination
        # with the index pair is unique.

    # normalization to yield probabilities

    row_sums = x.sum(axis=2)
    non_zero = np.where(row_sums != 0)
    row_sums = row_sums[:, :, np.newaxis]
    x[non_zero] = x[non_zero] / row_sums[non_zero]

    # sort indices and cumsum, similar to Gillespie algorithm

    transition_sorted_indices = np.argsort(x, axis=2)
    sorted_x = np.take_along_axis(x, transition_sorted_indices, axis=2)
    transition_cum_sum = np.cumsum(sorted_x, axis=2)

    return transition_cum_sum, transition_sorted_indices


def searchsorted2d(a, b):
    """
    from https://stackoverflow.com/questions/56471109/how-to-vectorize-with-numpy-searchsorted-in-a-2d-array

    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    # Inputs : a is (m,n) 2D array and b is (m,) 1D array.
    # Finds np.searchsorted(a[i], b[i])) in a vectorized way by
    # scaling/offsetting both inputs and then using searchsorted

    # Get scaling offset and then scale inputs
    s = np.r_[0, (np.maximum(a.max(1)-a.min(1)+1, b)+1).cumsum()[:-1]]
    a_scaled = (a+s[:, None]).ravel()
    b_scaled = b+s

    # Use searchsorted on scaled ones and then subtract offsets
    return np.searchsorted(a_scaled, b_scaled)-np.arange(len(s))*a.shape[1]


def generate_transition_series(state_series, cum_sum_x, x_sorted_indices):
    """

    Parameters
    ----------
    state_series
    cum_sum_x
    x_sorted_indices

    Returns
    -------
    transition_series : np.ndarray
    Contains the next transition id for each corresponding state (except the last).
    """
    current_states = state_series[:-1]
    future_states = state_series[1:]
    values_to_insert = np.random.uniform(low=0, high=1, size=len(current_states))

    specific_cum_sums = cum_sum_x[current_states, future_states, :]
    insert_at = searchsorted2d(specific_cum_sums, values_to_insert)
    transition_series = x_sorted_indices[current_states, future_states, insert_at]
    # vllt noch np.int als dtype
    return transition_series


def convert_single_state_series(number, state_series, joined_states):
    """
    Converts the state series containing the joined states to a single state series for each fluorophore.

    Parameters
    ----------
    number: int
        Number of fluorophores of the system.
    state_series : np.ndarray
        The consecutive state's unique values.
    state_ids : Collection
        Contains all state's identification numbers.
    single_state_ids : dict
        Contains the joined_states as keys and np.ndarray as values. The values contain the single_state indices
        in the correct order. E.g., the key 'S0_S1_S0' will have the value [0, 1, 0].

    Returns
    -------
    single_state_series : np.ndarray
        State series for each individual fluorophore (hence, single_states).
    """
    single_state_series = np.zeros(shape=(number, len(state_series)))

    for index, row in joined_states.iterrows():
        single_state_ids = row['single_states']
        if index in state_series:
            mask = np.where(state_series == index)[0]
            single_state_id = np.expand_dims(single_state_ids, axis=1)  # converts example from documentation
            # to [[0], [1], [0]]
            single_state_series[:, mask] = single_state_id

    return single_state_series


def occupation_time_single_states(number, single_transitions, time_series, transition_series, single_state_series,
                                  single_states):
    """
    Returns two dictionaries (see list of keys below). The content is the following:
        single_state_occurences -
        lifetimes_fluorophore -
        mean_lifetimes -
        total_lifetimes -
        -----------------
        transition_occurences -
        transition_times -
        mean_transition_times -


    Returns (inter alia) the time intervals between state changes of each individual fluorophore - in consecutive order,
    in consecutive order AND assigned to each initial state (i.e., lifetimes_single_states) or the mean lifetimes of
    each state.

    Note that the mean lifetime of a state depends on all outgoing transitions.

    Parameters
    ----------
    number : int
        Number of fluorophores of the system.
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.
    time_series : np.ndarray
        The time points at which the corresponding state occurs.
    transition_series : np.ndarray
        Contains the next transition id for each corresponding state (except the last).
    single_state_series : np.ndarray
        State series for each individual fluorophore (hence, single_states).
    single_states : iterable object
        Contains elements of type str describing each state a single fluorophore can occupy.

    Returns
    -------
    single_state_lifetimes : dict
        Keys are
        single_state_occurrences (np.ndarray for each fluorophore containing their single state ids),
        single_state_occurrences_all (the above summarized for all fluorophores),
        lifetimes_fluorophore (np.ndarray for each fluorophore containing their lifetimes corresponding to the states
            listed in single_state_occurrences),
        lifetimes_single_states (list for each fluorophore containing np.ndarray for each single state with lifetimes),
        lifetimes_single_states_all (the above summarized for all fluorophores),
        mean_lifetimes (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        mean_lifetimes_all (the above summarized for all fluorophores),
        total_lifetimes (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        total_lifetimes_all (the above summarized for all fluorophores).
    transition_lifetimes : dict
        Keys are
        transition_occurrences (np.ndarray for each fluorophore containing their transition ids),
        transition_occurrences_all (the above summarized for all fluorophores),
        transition_times (list for each fluorophores containing np.ndarray for each transition with lifetimes),
        transition_times_all (the above summarized for all fluorophores),
        mean_transition_times (np.ndarray with axis 0 for the fluorophores and axis 1 for the transition),
        mean_transition_times_all (the above summarized for all fluorophores).
    """
    single_states = np.arange(0, len(single_states))
    single_state_occurrences = []
    lifetimes_fluorophore = []
    lifetimes_single_states = []
    lifetimes_single_states_all = [np.array([]) for _ in range(len(single_states))]
    mean_lifetimes = np.zeros(shape=(number, len(single_states)))
    total_lifetimes = np.zeros(shape=(number, len(single_states)))

    transition_times = []
    transition_times_all = [np.array([]) for _ in range(single_transitions.index.size)]
    mean_transition_times = np.zeros(shape=(number, single_transitions.index.size))
    transition_occurrences = []

    for i in range(number):
        current_fluorophore_series = single_state_series[i]
        diffs = np.diff(current_fluorophore_series)
        changes_at = np.where(diffs != 0)[0]
        initial_states = current_fluorophore_series[changes_at]
        transitions = transition_series[changes_at]
        single_state_occurrence = np.append(initial_states, current_fluorophore_series[np.max(changes_at) + 1])
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
            time_intervals_state = time_intervals[np.where(initial_states == state)]
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
        for transition in range(single_transitions.index.size):
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


def occupation_time_prediction(rates):
    return 'not implemented yet'
