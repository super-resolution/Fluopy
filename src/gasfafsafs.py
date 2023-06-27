import numpy as np

def time_occurrence_statistics(number_fluorophores, single_states, unique_transitions, time_series,
                               transition_series, single_state_series):

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

