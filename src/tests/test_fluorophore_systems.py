import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import fluorophore_systems as fc
import initialize as ini


def test_init_fluorophoresystem():
    system = fc.FluorophoreSystem(2, 1, ("S0", "S1", "T1", "R", "B"), {"k_S0_S1": 0.4, "k_S1_T1": 5.8, "k_T1_S0": 1e-2,
                                                                       "k_T1_R": 13, "k_R_S0": 0.3},
                                  induction_rate=4)
    assert system.number == 2
    assert system.distances == 1
    assert system.states == ("S0", "S1", "T1", "R", "B")
    assert system.rates == {"k_S0_S1": 0.4, "k_S1_T1": 5.8, "k_T1_S0": 1e-2, "k_T1_R": 13, "k_R_S0": 0.3}
    assert system.state_names == ["S0_S0", "S0_S1", "S0_T1", "S0_R", "S0_B", "S1_S0", "S1_S1", "S1_T1", "S1_R", "S1_B",
                                  "T1_S0", "T1_S1", "T1_T1", "T1_R", "T1_B", "R_S0", "R_S1", "R_T1", "R_R", "R_B",
                                  "B_S0", "B_S1", "B_T1", "B_R", "B_B"]
    for joined_state, expected in zip(system.Joined_States, ini.state_pairs(system.number, system.states)):
        assert joined_state.name == expected.name
        assert joined_state.value == expected.value
    assert system.transitions == ini.transition_pairs(system.Joined_States)
    goal_assigned_rate_dict = ini.transition_rate_dict(system.rates, system.transitions)
    goal_assigned_rate_dict = ini.induction(goal_assigned_rate_dict, system.transitions, 4, system.states)
    assert system.assigned_rate_dict == goal_assigned_rate_dict
    assert np.array_equal(system.initial_row_vector, ini.initial_row_vector(system.transitions))
    _, goal_transition_matrix, goal_row_sums = ini.transition_matrices(system.assigned_rate_dict, system.transitions)
    assert np.array_equal(system.transition_matrix, goal_transition_matrix)
    assert np.array_equal(system.row_sums, goal_row_sums)
    assert not system.time_series
    assert not system.time_step_series
    assert not system.state_series
    assert not system.duplices
    assert not system.unique_series
    assert not system.unique_states
    assert not system.unique_joined_states
    assert not system.unique_names
    assert not system.unique_transitions
    assert not system.unique_series_converted
    assert not system.occupation_time_total
    assert not system.occupation_time_mean

