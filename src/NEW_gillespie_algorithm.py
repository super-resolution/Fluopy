import numpy as np


def direct_method_steps(transition_matrix, row_sums, combined_state_transitions, start_at, n_steps, seed):
    rng = np.random.default_rng(seed)

    time_step_series = np.empty(n_steps + 1)
    time_step_series[0] = 0
    transition_series = np.empty(n_steps, dtype=np.int64)

    # a random index (in this case the first index) at which the final state of a transition equals start_at
    current_state_index = combined_state_transitions[combined_state_transitions['final_state'] == start_at].index[0]

    random_numbers = rng.uniform(low=0, high=1, size=(n_steps, 2))

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, aixs=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    for i in range(n_steps):

        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:
            time_step_series = time_step_series[:i + 1]
            transition_series = transition_series[:i + 1]
            break

        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index], random_numbers[i, 1])

        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = current_state_index = next_transition

        time_step_series[i + 1] = transition_time

    time_series = np.cumsum(time_step_series)

    return time_series, time_step_series, transition_series
