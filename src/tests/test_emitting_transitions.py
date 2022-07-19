import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import emitting_transitions as et


test_transitions = {"S1__S0": (7, 8), "R__S0": (9, 8), "S0__S1": (8, 7),
    "S0_S1__S0_S0": (1, 0), "S1_S1__S0_S0": (3, 0), "R_R__R_S0": (10, 15),
    "S0_S1_S0__S0_S0_S0": (110, 109), "S1_S1_S0__S0_S0_S0": (111, 109), "S0_S0_R__S0_S0_S0": (112, 109)}
goal_emitting_transitions = ["S1__S0", "S0_S1__S0_S0", "S0_S1_S0__S0_S0_S0"]
goal_emitting_transition_indices = [(7, 8), (1, 0), (110, 109)]


def test_identify_emitting_transitions():
    emitting_transitions, emitting_transition_indices = et.identify_emitting_transitions(test_transitions)
    assert emitting_transitions == goal_emitting_transitions
    assert emitting_transition_indices == goal_emitting_transition_indices
