import numpy as np


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
        for e, joined_state_2 in enumerate(state_names[i + 1:]):
            fluos_2 = joined_state_2.split("_")
            if sorted(fluos_2) == sorted(fluos_1):
                indices = [i, e + i + 1]
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


def convert_single_state_series(number, state_series, state_ids, single_state_ids):
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
    single_state_ids = list(single_state_ids.values())
    single_state_series = np.zeros(shape=(number, len(state_series)))
    for i, value in enumerate(state_ids):
        if value in state_series:
            mask = np.where(state_series == value)[0]
            single_state_id = np.expand_dims(single_state_ids[value], axis=1)  # converts example from documentation
            # to [[0], [1], [0]]
            single_state_series[:, mask] = single_state_id

    return single_state_series


def occupation_time_single_states(number, rates, time_series, transition_series, single_state_series, single_states):
    """
    Returns (inter alia) the time intervals between state changes of each individual fluorophore - in consecutive order,
    in consecutive order AND assigned to each initial state (i.e., lifetimes_single_states) or the mean lifetimes of
    each state.

    Note that the mean lifetime of a state depends on all outgoing transitions.

    Parameters
    ----------
    number : int
        Number of fluorophores of the system.
    rates : dict
        The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
        the value [k, name_of_transition] assigned to it.
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
    transition_times_all = [np.array([]) for _ in range(len(rates))]
    mean_transition_times = np.zeros(shape=(number, len(rates)))
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
        for transition in range(len(rates)):
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
