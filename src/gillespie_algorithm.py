import numpy as np

# Given an initial state x, all possible future states have the same exponentially distributed time tau as a variable -
# this is due to the more states become available (no matter their rate constants), the state x will be occupied for
# less time. This means - even if a future state y has a very low rate constant - if there are other possible
# states with high rate constants, y will only occur if it happens rather quick - therefore, the individual rate
# constants give an insight in the mean occupation time only if it is the only rate constant.


def direct_method(transition_matrix, row_sums, n_steps, seed):
    """
    The direct method of the gillespie algorithm (i.e., the stochastic simulation algorithm). Note that in this version,
    the propensities are equal to the rate constants, because the occupied state's population is assumed to be always 1
    (the initial row vector contains only zeros except a single one). Additionally, the state change vector is also
    trivial, since each transition leads to a decrease in the current state by 1 and an increase in the following state
    by 1.

    Parameters
    ----------
    transition_matrix : np.ndarray
        The first return value of initialize.transition_matrices.
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        The second return value of initialize.transition_matrices.
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of all transition rates of a
        state.
    n_steps : int
        Maximum number of simulation steps. If the Markov chain reaches an absorbing state, the simulation stops early.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    time_series : np.ndarray
        The time points at which the corresponding state occurs.
    time_step_series : np.ndarray
        The time step until the corresponding state occurs (starting from the previous state). Therefore, the lifetime
        of a (joined!) state of index n is the time step of time_step_series[n+1].
    state_series : np.ndarray
        The consecutive state's (if number > 1 joined_state's) unique values.
    """
    rng = np.random.default_rng(seed)

    current_state = np.zeros(shape=row_sums.shape[0])
    current_state[0] = 1

    time_step_series = np.empty(n_steps + 1)
    time_step_series[0] = 0
    state_series = np.empty(n_steps + 1, dtype=np.int64)

    current_state_index = state_series[0] = np.where(current_state == 1)[0][0]

    random_numbers = rng.uniform(low=0, high=1, size=(n_steps, 2))

    # the following block is to generate a sorted transition matrix to later check at which index the randomly drawn
    # number should be inserted
    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    future_state = None
    for i in range(n_steps):
        if i > 0:
            current_state_index = future_state
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:  # there is no possible transition going from the current state
            time_step_series = time_step_series[:i + 1]
            state_series = state_series[:i + 1]
            break

        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])  # resembles inverse transform
        # sampling using the quantile function of the exponential distribution
        # the exponential distribution is used since it is the time between events in a Poisson point process
        # meaning that events occur continuously and independently at a constant average rate

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index], random_numbers[i, 1])  # get the
        # index of the sorted value list
        future_state = transition_matrix_sorted_indices[current_state_index, sorted_index]  # use the previous index to
        # get the original index of the sorted value

        state_series[i + 1] = future_state
        time_step_series[i + 1] = transition_time

    time_series = np.cumsum(time_step_series)

    return time_series, time_step_series, state_series
