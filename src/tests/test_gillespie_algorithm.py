import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import gillespie_algorithm as ga
import ssa_cython as ga_cy


def test_direct_method_py():
    # first part ensures that the cython and python implementation produce same results
    row_sums = np.array([2, 7.6, 4])
    initial_row_vector = np.array([1, 0, 0])
    transition_matrix = np.array([[0.4, 0, 0.6], [6.5/7.6, 0, 1.1/7.6], [0, 1, 0]])
    n_steps = 1000
    seed = 100
    time_series, time_step_series, state_series = ga.direct_method_py(row_sums, initial_row_vector, transition_matrix,
                                                                      n_steps, seed)
    time_series_2, time_step_series_2, state_series_2 = ga_cy.direct_method_cy(row_sums, initial_row_vector,
                                                                               transition_matrix, n_steps, seed)
    assert np.array_equal(time_series, time_series_2)
    assert np.array_equal(time_step_series, time_step_series_2)
    assert np.array_equal(state_series, state_series_2)

    # second part ensures that the implementations produce the expected result at seed of 99
    # The goal result was captured by a run of the function in a configuration which was likely to perform
    # correctly
    row_sums = np.array([1e6, 7.6, 4])
    initial_row_vector = np.array([1, 0, 0])
    transition_matrix = np.array([[0, 0.4, 0.6], [6.5/7.6, 0, 1.1/7.6], [0, 1, 0]])
    n_steps = 20
    seed = 99
    time_series, time_step_series, state_series = ga.direct_method_py(row_sums, initial_row_vector, transition_matrix,
                                                                      n_steps, seed)
    goal_time_step_series = np.array([0.00000000e+00, 6.81157990e-07, 1.67398702e-01, 6.39855943e-02,
                                      1.24901647e-06, 1.90076402e-01, 9.48618448e-03, 1.17299644e-06,
                                      1.72148343e-01, 6.58944361e-08, 3.35980245e-02, 3.02116056e-01,
                                      1.04409784e-01, 8.32179317e-07, 1.33873639e-01, 5.55360249e-02,
                                      1.64154924e-06, 1.81334854e-01, 5.44805819e-01, 6.88653695e-02,
                                      2.51222563e-07])
    # explanation: The first state is state 0 since the initial row vector is 1 only at index 0. State 0's transition
    # rates result in a large sum (1e6 in row_sums), therefore the state is usually occupied for only a short period of
    # time. This can be seen in goal_time_step_series - the smallest numbers describe the transition times from state 0
    # to another state. It can also be seen that state 2 always transitions to state 1. This is due to its entry in the
    # transition matrix has a value above 0 only at index 1. The other two states transition to both other states,
    # respectively. Here, the transition from state 1 to state 0 occurs more often than to state 2 since the point
    # probability for the latter is much smaller.
    goal_state_series = np.array([0., 2., 1., 0.,
                                  2., 1., 0., 1.,
                                  0., 1., 2., 1.,
                                  0., 2., 1., 0.,
                                  1., 2., 1., 0.,
                                  2.])

    assert np.array_equal(time_series, np.cumsum(time_step_series))  # time_series is the cumsum of time_step_series
    assert np.allclose(time_step_series, goal_time_step_series)
    assert np.array_equal(state_series, goal_state_series)
