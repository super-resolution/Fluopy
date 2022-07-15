import numpy as np

# Given an initial state x, all possible future states have the same exponentially distributed time tau as a variable -
# this is due to the more states become available (no matter their rate constants), the state x will be occupied for
# less time. This means - even if a future state y has a very low rate constant - if there are other possible
# states with high rate constants, y will only occur if it happens rather quick - therefore, the individual rate
# constants give an insight in the mean occupation time only if it is the only rate constant.


def direct_method_py(row_sums, initial_row_vector, transition_matrix, n_steps, seed):
    rng = np.random.default_rng(seed)

    current_state = initial_row_vector

    time_step_series = np.zeros(n_steps + 1)
    state_series = np.zeros(n_steps + 1)

    state_series[0] = np.where(current_state == 1)[0][0]

    random_numbers = rng.uniform(low=0, high=1, size=(n_steps, 2))

    # the following block is to generate a sorted transition matrix to later check at which index the randomly drawn
    # number should be inserted
    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    for i in range(n_steps):
        current_state_index = np.where(current_state == 1)
        current_state_lambda = row_sums[current_state_index]
        if current_state_lambda == 0:  # there is no possible transition going from the current state
            time_step_series = time_step_series[:i + 1]
            state_series = state_series[:i + 1]
            break

        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])  # resembles inverse transform
        # sampling using the quantile function of the exponential distribution
        # the exponential distribution is used since it is the time between events in a Poisson point process
        # meaning that events occur continuously and independently at a constant average rate

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index][0], random_numbers[i, 1])  # get the
        # index of the sorted value list
        true_index = transition_matrix_sorted_indices[current_state_index, sorted_index]  # use the previous index to
        # get the original index of the sorted value

        current_state = np.zeros(shape=row_sums.shape)
        current_state[true_index] = 1

        state_series[i + 1] = np.where(current_state == 1)[0][0]
        time_step_series[i + 1] = transition_time

    time_series = np.cumsum(time_step_series)

    return time_series, time_step_series, state_series


def simulation_tau_leaping(initial_row_vector, transition_rate_matrix, tau, n_steps=100, seed=100):
    rng = np.random.default_rng(seed)

    current_state = initial_row_vector

    time_step_series = np.zeros(n_steps + 1)

    state_series = [current_state]

    for i in range(n_steps):
        current_state_index = np.where(current_state > 0)
        current_state_transition_rates = transition_rate_matrix[current_state_index]
        current_state_players = current_state[current_state_index]
        current_state_numbers = np.expand_dims(current_state_players, axis=1)
        r_j = current_state_numbers * current_state_transition_rates
        # since in this case (in contrast to direct method of the here applied circumstances) the population of
        # states is often != 1, the propensities are no longer equal to the rate constants (a_j = k * X_i, where a_j
        # is propensity of reaction/transition j, k is rate constant (unit s^-1), X_i is population of state i; true
        # for first order reactions only!)
        if np.sum(current_state_transition_rates) == 0:
            break

        k_j = rng.poisson(lam=r_j*tau, size=r_j.shape)
        # the poisson distribution expresses the probability of a given number of events occurring in a fixed interval
        # of time (or space) if these events occur with a known constant mean rate and independently of the time since
        # the last event

        origin_state_change = -np.sum(k_j, axis=1)
        new_state_change = np.sum(k_j, axis=0)
        # Note that state change vectors v_ij are not necessary in this case, since each event decreases the origin by 1
        # and increases the destination by 1

        total_state_change = new_state_change.copy()
        total_state_change[current_state_index] += origin_state_change
        current_state = np.sum([current_state, total_state_change], axis=0)  # the tau-leaping approximation
        time_step_series[i] = tau
        state_series.append(current_state)

    return current_state, state_series, time_step_series
