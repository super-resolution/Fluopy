import numpy as np


class Simulation:
    def __init__(self, transitions):
        if transitions.transition_matrix is None:
            raise ValueError('simulation not available if transitions not finalized.')
        self.transitions = transitions
        self.time_series = None
        self.time_step_series = None
        self.transition_series = None
        self.state_series = None

    def run(self, start_at=None, size=int(1e5), end_time=None, seed=None):
        if start_at is None:
            start_at = tuple(np.zeros(shape=self.transitions.fluorophore_system.count, dtype=int))
        if end_time is None:
            self.time_series, self.time_step_series, self.transition_series = \
                direct_method_steps(self.transitions.transition_matrix, self.transitions.row_sums,
                                    self.transitions.combined_state_transitions_df, start_at, size, seed)
        else:
            self.time_series, self.time_step_series, self.transition_series = \
                direct_method_time(self.transitions.transition_matrix, self.transitions.row_sums,
                                   self.transitions.combined_state_transitions_df, start_at, size, end_time, seed)

        final_states = self.transitions.combined_state_transitions_df['final_state']

        self.state_series = np.empty(shape=(len(final_states[0]), self.time_series.size), dtype=np.int64)
        self.state_series[:, 0] = start_at
        print(self.state_series)
        for i, _ in enumerate(final_states[0]):
            final_states_fluorophore = final_states.map(lambda x: x[i]).to_numpy()
            self.state_series[i][1:] = final_states_fluorophore[self.transition_series]


def direct_method_steps(transition_matrix, row_sums, combined_state_transitions, start_at, n_steps, seed):
    rng = np.random.default_rng(seed)

    time_step_series = np.empty(n_steps + 1)
    time_step_series[0] = 0
    transition_series = np.empty(n_steps, dtype=np.int64)

    # a random index (in this case the first index) at which the final state of a transition equals start_at
    current_state_index = combined_state_transitions[combined_state_transitions['final_state'] == start_at].index[0]

    random_numbers = rng.uniform(low=0, high=1, size=(n_steps, 2))

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
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


def direct_method_time(transition_matrix, row_sums, combined_state_transitions, start_at, size, end_time, seed):
    rng = np.random.default_rng(seed)

    current_state_index = combined_state_transitions[combined_state_transitions['final_state'] == start_at].index[0]

    time_series = [0]
    time_step_series = [0]
    transition_series = [current_state_index]

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    random_numbers = rng.uniform(low=0, high=1, size=(size, 2))

    if end_time is not None:
        i = 0
        j = 1
        while time_series[-1] < end_time:
            current_state_lambda = row_sums[current_state_index]

            if current_state_lambda == 0:
                break

            transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i - (j - 1) * size + 1, 0])

            sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index],
                                           random_numbers[i - (j - 1) * size + 1, 1])

            next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

            transition_series.append(next_transition)
            time_step_series.append(transition_time)
            time_series.append(time_series[-1] + transition_time)

            i += 1
            if i == j * size - 1:
                random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
                j += 1
            current_state_index = next_transition

    transition_series = np.array(transition_series)
    time_series = np.array(time_series)
    time_step_series = np.array(time_step_series)

    return time_series, time_step_series, transition_series
