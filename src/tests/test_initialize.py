import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
from enum import Enum
import initialize as ini


def test_recursion():
    combinations = [["t", "t"], ["t", "e"], ["t", "s"], ["t", "t"], ["e", "t"], ["e", "e"], ["e", "s"], ["e", "t"],
                    ["s", "t"], ["s", "e"], ["s", "s"], ["s", "t"], ["t", "t"], ["t", "e"], ["t", "s"], ["t", "t"]]
    generator = ini.recursion(2, 2, ("t", "e", "s", "t"))
    for item_1, item_2 in zip(generator, combinations):
        assert item_1 == item_2


def test_state_pairs():
    goal_values = np.arange(0, 21)
    goal_names = ["B_B_B", "B_B_A", "B_B_Z", "B_A_B", "B_A_A", "B_A_Z", "B_Z_B", "B_Z_A", "B_Z_Z",
                  "A_B_B", "A_B_A", "A_B_Z", "A_A_B", "A_A_A", "A_A_Z", "A_Z_B", "A_Z_A", "A_Z_Z",
                  "Z_B_B", "Z_B_A", "Z_B_Z", "Z_A_B", "Z_A_A", "Z_A_Z", "Z_Z_B", "Z_Z_A", "Z_Z_Z"]
    joined_states = ini.state_pairs(3, ("B", "A", "Z"))
    for enum_item, goal_value, goal_name in zip(joined_states, goal_values, goal_names):
        assert enum_item.value == goal_value
        assert enum_item.name == goal_name


def test_transition_pairs():
    joined_states = Enum("Joined_States", ["T_E_S_T", "T_S_E_T", "E_S_S_T"], start=0)
    goal_transitions = {"T_E_S_T__T_E_S_T": (0, 0), "T_E_S_T__T_S_E_T": (0, 1), "T_E_S_T__E_S_S_T": (0, 2),
                        "T_S_E_T__T_E_S_T": (1, 0), "T_S_E_T__T_S_E_T": (1, 1), "T_S_E_T__E_S_S_T": (1, 2),
                        "E_S_S_T__T_E_S_T": (2, 0), "E_S_S_T__T_S_E_T": (2, 1), "E_S_S_T__E_S_S_T": (2, 2)}
    transitions = ini.transition_pairs(joined_states)
    assert transitions == goal_transitions


def test_initial_row_vector():
    transitions = {"T_E_S_T__T_E_S_T": (0, 0), "T_E_S_T__T_S_E_T": (0, 1), "T_E_S_T__E_S_S_T": (0, 2),
                   "T_S_E_T__T_E_S_T": (1, 0), "T_S_E_T__T_S_E_T": (1, 1), "T_S_E_T__E_S_S_T": (1, 2),
                   "E_S_S_T__T_E_S_T": (2, 0), "E_S_S_T__T_S_E_T": (2, 1), "E_S_S_T__E_S_S_T": (2, 2)}
    goal_vector = np.array([1, 0, 0])
    vector = ini.initial_row_vector(transitions)
    assert np.array_equal(vector, goal_vector)


def test_rate_assignment():
    transitions = {"T_T__T_T": (0, 0), "T_T__T_E": (0, 1), "T_T__T_S": (0, 2),
                   "T_E__T_T": (1, 0), "T_E__T_E": (1, 1), "T_E__T_S": (1, 2),
                   "T_S__T_T": (2, 0), "T_S__T_E": (2, 1), "T_S__T_S": (2, 2)}
    goal_assigned_rate_dict = {"T_E__T_T": 6.5}
    goal_rate_name_dict = {(1, 0): "letter switching"}
    assigned_rate_dict, rate_name_dict = ini.rate_assignment(dict(), dict(), transitions, "E", "T", 6.5,
                                                             "letter switching")
    assert assigned_rate_dict == goal_assigned_rate_dict
    assert rate_name_dict == goal_rate_name_dict


def test_transition_rate_dict():
    rates = {"k_E_T": [6.5, "E exchange"], "k_S_E": [4, "S exchange"]}
    transitions = {"T_T__T_T": (0, 0), "T_T__T_E": (0, 1), "T_T__T_S": (0, 2),
                   "T_E__T_T": (1, 0), "T_E__T_E": (1, 1), "T_E__T_S": (1, 2),
                   "T_S__T_T": (2, 0), "T_S__T_E": (2, 1), "T_S__T_S": (2, 2)}
    goal_assigned_rate_dict = {"T_E__T_T": 6.5, "T_S__T_E": 4}
    goal_rate_name_dict = {(1, 0): "E exchange", (2, 1): "S exchange"}
    assigned_rate_dict, rate_name_dict = ini.transition_rate_dict(rates, transitions)
    assert assigned_rate_dict == goal_assigned_rate_dict
    assert rate_name_dict == goal_rate_name_dict


def test_induction():
    assigned_rate_dict = {"S0_S0__S0_S1": 2, "S0_S0__S1_S0": 2}
    transitions = {"S0_S0__S0_S0": (0, 0), "S0_S0__S0_S1": (0, 1), "S0_S0__S1_S0": (0, 2),
                   "R_S1__S0_S0": (4, 0), "R_S1__S0_S1": (4, 1)}
    rate_name_dict = {(0, 1): "hi", (0, 2): "hello"}
    goal_assigned_rate_dict = {"S0_S0__S0_S1": 2, "S0_S0__S1_S0": 2, "R_S1__S0_S0": 9.7}
    goal_rate_name_dict = {(0, 1): "hi", (0, 2): "hello", (4, 0): "induction"}
    assigned_rate_dict, rate_name_dict = ini.induction(assigned_rate_dict, rate_name_dict, transitions, 9.7,
                                                       ("S0", "S1", "T1", "R", "B"))
    assert assigned_rate_dict == goal_assigned_rate_dict
    assert rate_name_dict == goal_rate_name_dict


def test_absorbing_states():
    rate_name_dict = {(0, 1): "h", (3, 0): "b", (5, 2): "g", (6, 1): "i"}
    state_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    absorb_states = ini.absorbing_states(rate_name_dict, state_ids)
    goal_absorb_states = [1, 2, 4, 7, 8]
    assert absorb_states == goal_absorb_states


def test_transition_matrices():
    assigned_rate_dict = {"T_E__T_T": 6.5, "T_S__T_E": 4, "T_E__T_S": 1.1}
    transitions = {"T_T__T_T": (0, 0), "T_T__T_E": (0, 1), "T_T__T_S": (0, 2),
                   "T_E__T_T": (1, 0), "T_E__T_E": (1, 1), "T_E__T_S": (1, 2),
                   "T_S__T_T": (2, 0), "T_S__T_E": (2, 1), "T_S__T_S": (2, 2)}
    goal_transition_rate_matrix = np.array([[0, 0, 0], [6.5, 0, 1.1], [0, 4, 0]])
    goal_transition_matrix = np.array([[0, 0, 0], [6.5/7.6, 0, 1.1/7.6], [0, 1, 0]])
    goal_row_sums = np.array([0, 7.6, 4])
    transition_rate_matrix, transition_matrix, row_sums = ini.transition_matrices(assigned_rate_dict, transitions)
    assert np.array_equal(transition_rate_matrix, goal_transition_rate_matrix)
    assert np.array_equal(transition_matrix, goal_transition_matrix)
    assert np.array_equal(row_sums, goal_row_sums)
