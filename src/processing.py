"""Contains functions that work on simulated series but not exclusively on emissions."""
import numpy as np
from scipy.stats import expon


def multiple_transitions(joined_transitions, joined_states, unique_transitions):
    """
    In case there are multiple transitions between the same two joined states, the selection of which transition has
    happened will be based on the cumulative sum of probabilities and inserting a random number between 0 and 1 (similar
    to the strategy of Gillespie algorithm). This function generates the cumulative sum and a correspondingly sorted
    index array.

    Parameters
    ----------
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), unique_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.

    Returns
    -------
    transition_cum_sum : np.ndarray
        Of shape (joined_states.index.size, joined_states.index.size, unique_transitions.index.size).
        Contains the cumulative sum of all available transitions of each joined_state combination.
    transition_sorted_indices : np.ndarray
        Of shape (joined_states.index.size, joined_states.index.size, unique_transitions.index.size).
        Contains the ids of the unique_transitions ordered in a way such that it corresponds to the order of cum_sum.
    """
    uni_dir = joined_states.index.size
    transition_count = unique_transitions.index.size
    transitions = np.empty(shape=(uni_dir, uni_dir, transition_count))
    transitions[:] = 0
    for index_pair, transition_id, rate in zip(joined_transitions['joined_states_id'],
                                               joined_transitions['unique_transition_id'],
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


def _searchsorted2d(arr_2d, arr_1d):
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
        Of shape (joined_states.index.size, joined_states.index.size, unique_transitions.index.size).
        Contains the cumulative sum of all available transitions of each joined_state combination.
    transition_sorted_indices : np.ndarray
        The second return value of multiple_transitions.
        Of shape (joined_states.index.size, joined_states.index.size, unique_transitions.index.size).
        Contains the ids of the unique_transitions ordered in a way such that it corresponds to the order of cum_sum.
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
    # specific_cum_sums is a 2D array of shape (current_states.shape[0], unique_transitions.index.size)
    insert_at = _searchsorted2d(arr_2d=specific_cum_sums, arr_1d=values_to_insert)  # 1d array

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


def predict_occurrences(single_states, unique_transitions, mean_lifetimes):
    """
    Predict the occurrences of single states and transitions. Numbers are given as probabilities (between 0 and 1).

    Parameters
    ----------
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        single state.
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.
    mean_lifetimes : np.ndarray
        The first return value of predict_lifetimes.
        Contains mean lifetimes. The order corresponds to single_states.

    Returns
    -------
    lin_eq_result : np.ndarray
        Contains probabilities of single states. The order corresponds to single_states.
    transition_occurrences : np.ndarray
        Contains probabilities of transitions. The order corresponds to unique_transitions.
    """
    if unique_transitions['name'].str.contains('__').any():
        raise ValueError('Energy transfers not implemented yet.')

    number_states = len(single_states)
    i_s0, i_s1, i_t1, i_cis, i_off = 0, 1, 2, 3, 4
    for i, value in single_states.items():
        if value == 'S0':
            i_s0 = i
        elif value == 'S1':
            i_s1 = i
        elif value == 'T1':
            i_t1 = i
        elif value == 'Cis':
            i_cis = i
        elif value == 'OFF':
            i_off = i
        else:
            raise ValueError('State not implemented yet.')
    b = np.zeros(shape=number_states)
    b[0] = 1

    # sum of all states has to equal number of steps
    a_arrays = np.zeros(shape=(number_states, number_states))
    a_arrays[0][:] = 1  # all states

    # S1 has to occur as often as S0
    a_arrays[1][i_s0] = 1  # S0
    a_arrays[1][i_s1] = -1  # S1

    if number_states > 2:  # assumes S0, S1, T1, ...
        s1_transitions = unique_transitions[unique_transitions['name'].str.startswith('S1')]
        s1_direct_deexcitation = s1_transitions[s1_transitions['name'] == 'S1_S0']
        s1_indirect_deexcitation = s1_transitions[s1_transitions['name'] != 'S1_S0']

        t1_transitions = unique_transitions[unique_transitions['name'].str.startswith('T1')]
        t1_direct_deexcitation = t1_transitions[t1_transitions['name'] == 'T1_S0']
        t1_indirect_deexcitation = t1_transitions[t1_transitions['name'] != 'T1_S0']

        # T1 coverage
        s1_nuniques = s1_indirect_deexcitation['name'].nunique()
        a_arrays[2][i_s0] = -1  # s0
        a_arrays[2][i_t1] = s1_direct_deexcitation['rate'].sum() / s1_indirect_deexcitation[
                'rate'].sum() + 1  # +1 adds the population of the ith state
        if s1_nuniques == 2:
            a_arrays[2][i_cis] = s1_direct_deexcitation['rate'].sum() / s1_indirect_deexcitation[
                'rate'].sum() + 1

        # Cis coverage
        if s1_nuniques == 2:
            s1_uniques = s1_indirect_deexcitation['name'].unique()
            rate_1 = s1_indirect_deexcitation[s1_indirect_deexcitation['name'] == s1_uniques[0]]['rate'].sum()
            rate_2 = s1_indirect_deexcitation[s1_indirect_deexcitation['name'] != s1_uniques[0]]['rate'].sum()
            a_arrays[i_cis][i_t1] = -1  # T1
            a_arrays[i_cis][i_cis] = rate_1 / rate_2  # Cis
        elif s1_nuniques > 2:
            raise ValueError('Only two alternative singlet deexcitation pathways implemented.')

        # OFF coverage
        t1_nuniques = t1_indirect_deexcitation['name'].nunique()
        if t1_nuniques == 1:
            a_arrays[i_off][i_off] = -1  # OFF
            a_arrays[i_off][i_t1] = t1_indirect_deexcitation['rate'].sum() / (t1_direct_deexcitation['rate'].sum() +
                                                                              t1_indirect_deexcitation[
                                                                               'rate'].sum())  # T1
        elif t1_nuniques > 1:
            raise ValueError('Only one alternative triplet deexcitation pathway implemented.')

    lin_eq_result = np.linalg.solve(a_arrays, b)
    ###################################################################################################################
    transition_occurrences = np.zeros(unique_transitions.index.size)
    for index, row in unique_transitions.iterrows():
        source = row['name'].split('_')[0]
        if source == 'S0':
            i = i_s0
        elif source == 'S1':
            i = i_s1
        elif source == 'T1':
            i = i_t1
        elif source == 'Cis':
            i = i_cis
        elif source == 'OFF':
            i = i_off
        else:
            raise ValueError('State not implemented yet.')
        state_occurrences = lin_eq_result[i]
        total_rate = 1 / mean_lifetimes[i]
        current_rate = row['rate']
        transition_occurrence = state_occurrences * current_rate / total_rate
        transition_occurrences[index] = transition_occurrence

    return lin_eq_result, transition_occurrences


def predict_lifetimes(single_states, unique_transitions):
    """
    Predict the lifetimes of single states and times to transition.

    Parameters
    ----------
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        single state.
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.

    Returns
    -------
    mean_lifetimes : np.ndarray
        Contains mean lifetimes. The order corresponds to single_states.
    lifetimes_single_states : list
        Contains scipy distribution classes (scipy.stats._distn_infrastructure.rv_continuous_frozen).
    mean_transition_times : np.ndarray
        Contains mean time to transition. The order corresponds to unique_transitions.
    transition_times : np.ndarray
        Contains scipy distribution classes (scipy.stats._distn_infrastructure.rv_continuous_frozen).
    """
    mean_lifetimes = np.zeros(len(single_states))
    lifetimes_single_states = []
    mean_transition_times = np.zeros(unique_transitions.index.size)
    transition_times = np.empty(unique_transitions.index.size, dtype=np.object)
    for i, state in enumerate(single_states.values()):
        total_rate = 0
        associated_transitions = []
        for index, row in unique_transitions.iterrows():
            source = row['name'].split('_')[0]
            if source == state:
                total_rate += row['rate']
                associated_transitions.append(index)

        mean_lifetime = 1 / total_rate
        mean_lifetimes[i] = mean_lifetime
        mean_transition_times[associated_transitions] = mean_lifetime
        lifetime_pdf = expon(scale=mean_lifetime)
        lifetimes_single_states.append(lifetime_pdf)
        transition_times[associated_transitions] = lifetime_pdf

    return mean_lifetimes, lifetimes_single_states, mean_transition_times, transition_times


def time_occurrence_statistics(number_fluorophores, single_states, unique_transitions, time_series,
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
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.
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
        'singe_state_occurrences_pred' (the prediction of the above),
        'lifetimes_fluorophore' (np.ndarray for each fluorophore containing their lifetimes corresponding to the single
            states listed in single_state_occurrences),
        'lifetimes_single_states' (list of each fluorophore containing np.ndarray for each single state with lifetimes),
        'lifetimes_single_states_all' (the above summarized for all fluorophores),
        'lifetimes_single_states_pred' (the prediction of the above),
        'mean_lifetimes' (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        'mean_lifetimes_all' (the above summarized for all fluorophores),
        'mean_lifetimes_pred' (the prediction of the above),
        'total_lifetimes' (np.ndarray with axis 0 for the fluorophores and axis 1 for the single states),
        'total_lifetimes_all' (the above summarized for all fluorophores),
        'total_lifetimes_pred' (the prediction of the above).
    transition_lifetimes : dict
        Keys are
        'transition_occurrences' (np.ndarray for each fluorophore containing their transition ids),
        'transition_occurrences_all' (the above summarized for all fluorophores),
        'transition_occurrences_pred' (the prediction of the above),
        'transition_times' (list for each fluorophores containing np.ndarray for each transition with lifetimes),
        'transition_times_all' (the above summarized for all fluorophores),
        'transition_times_pred' (the prediction of the above),
        'mean_transition_times' (np.ndarray with axis 0 for the fluorophores and axis 1 for the transition),
        'mean_transition_times_all' (the above summarized for all fluorophores),
        'mean_transition_times_pred' (the prediction of the above).
    """
    single_state_ids = np.fromiter(single_states.keys(), dtype=int)
    single_state_occurrences = []
    lifetimes_fluorophore = []
    lifetimes_single_states = []
    lifetimes_single_states_all = [np.array([]) for _ in single_state_ids]
    mean_lifetimes = np.zeros(shape=(number_fluorophores, single_state_ids.shape[0]))
    # note that the mean lifetime of a state depends on all outgoing transitions
    total_lifetimes = np.zeros(shape=(number_fluorophores, single_state_ids.shape[0]))

    transition_times = []
    transition_times_all = [np.array([]) for _ in unique_transitions.index]
    mean_transition_times = np.zeros(shape=(number_fluorophores, unique_transitions.index.size))
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
        for j, state in enumerate(single_state_ids):
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
        for transition in unique_transitions.index:
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
    ###################################################################################################################
    if unique_transitions['name'].str.contains('__').any():
        mean_lifetimes_pred, lifetimes_single_states_pred, mean_transition_times_pred = None, None, None
        transition_times_pred, single_state_occurrences_pred, transition_occurrences_pred = None, None, None
        total_lifetimes_pred = None
    else:
        mean_lifetimes_pred, lifetimes_single_states_pred, mean_transition_times_pred, transition_times_pred = \
            predict_lifetimes(single_states=single_states, unique_transitions=unique_transitions)
        single_state_occurrences_pred, transition_occurrences_pred = \
            predict_occurrences(single_states=single_states, unique_transitions=unique_transitions,
                                mean_lifetimes=mean_lifetimes_pred)
        total_lifetimes_pred = single_state_occurrences_pred * mean_lifetimes_pred * time_series.shape[0]
    ###################################################################################################################
    single_state_lifetimes = {'single_state_occurrences': single_state_occurrences,
                              'single_state_occurrences_all': single_state_occurrences_all,
                              'single_state_occurrences_pred': single_state_occurrences_pred,
                              'lifetimes_fluorophore': lifetimes_fluorophore,
                              'lifetimes_single_states': lifetimes_single_states,
                              'lifetimes_single_states_all': lifetimes_single_states_all,
                              'lifetimes_single_states_pred': lifetimes_single_states_pred,
                              'mean_lifetimes': mean_lifetimes,
                              'mean_lifetimes_all': mean_lifetimes_all,
                              'mean_lifetimes_pred': mean_lifetimes_pred,
                              'total_lifetimes': total_lifetimes,
                              'total_lifetimes_all': total_lifetimes_all,
                              'total_lifetimes_pred': total_lifetimes_pred}
    transition_lifetimes = {'transition_occurrences': transition_occurrences,
                            'transition_occurrences_all': transition_occurrences_all,
                            'transition_occurrences_pred': transition_occurrences_pred,
                            'transition_times': transition_times,
                            'transition_times_all': transition_times_all,
                            'transition_times_pred': transition_times_pred,
                            'mean_transition_times': mean_transition_times,
                            'mean_transition_times_all': mean_transition_times_all,
                            'mean_transition_times_pred': mean_transition_times_pred}

    return single_state_lifetimes, transition_lifetimes
