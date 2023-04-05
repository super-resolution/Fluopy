cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

# WARNING I: this implementation once was faster but due to some simplifications it is now even slower than the python
# implementation
# WARNING II: this implementation always returns transition_series as None
# to read the documentation, check out gillespie_algorithm.direct_method_py

def direct_method_cy(row_sums: np.ndarray, initial_row_vector: np.ndarray, transition_matrix: np.ndarray,
                     n_steps: int, seed: int):
    rng = np.random.default_rng(seed)

    cdef np.ndarray current_state = initial_row_vector

    cdef np.ndarray time_step_series = np.empty(n_steps + 1, dtype=np.float64)
    time_step_series[0] = 0
    cdef np.ndarray state_series = np.empty(n_steps + 1, dtype=np.int64)

    cdef np.ndarray random_numbers = rng.uniform(low=0, high=1, size=(n_steps, 2))

    # the following block is to generate a sorted transition matrix to later check at which index the randomly drawn
    # number should be inserted
    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    cdef long long [:, :] transition_matrix_sorted_indices_view = transition_matrix_sorted_indices
    cdef double [:, :] cumsum_sorted_trm_view = cumsum_sorted_trm
    cdef double [:] time_step_series_view = time_step_series
    cdef double [:, :] random_numbers_view = random_numbers
    cdef double [:] row_sums_view = row_sums
    cdef long long [:] state_series_view = state_series
    cdef int i
    cdef np.float64_t current_state_lambda
    cdef double transition_time
    cdef int current_state_index = np.where(current_state == 1)[0][0]
    cdef np.int64_t sorted_index
    cdef int future_state
    state_series[0] = current_state_index

    for i in range(n_steps):
        if i > 0:
            current_state_index = future_state
        current_state_lambda = row_sums_view[current_state_index]
        if current_state_lambda == 0:
            time_step_series_view = time_step_series_view[:i + 1]
            state_series_view = state_series_view[:i + 1]
            break

        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers_view[i, 0])
        time_step_series_view[i + 1] = transition_time
        sorted_index = np.searchsorted(cumsum_sorted_trm_view[current_state_index], random_numbers_view[i, 1])
        future_state = transition_matrix_sorted_indices_view[current_state_index, sorted_index]

        state_series_view[i + 1] = future_state

    time_step_series = np.asarray(time_step_series_view, dtype=np.float64)
    state_series = np.asarray(state_series_view)
    time_series = np.cumsum(time_step_series)
    transition_series = None

    return time_series, time_step_series, state_series, transition_series
