import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import Xenergy_transfer as et


def test_off_state_rescue():
    assigned_rate_dict = {"S0_S0__S0_S1": 2, "S0_S0__S1_S0": 2}
    transitions = {"S0_S0__S0_S0": (0, 0), "S0_S0__S0_S1": (0, 1), "S0_S0__S1_S0": (0, 2),
                   "R_S1__S0_S0": (4, 0), "R_S1__S0_S1": (4, 1)}
    rate_name_dict = {(0, 1): "hi", (0, 2): "hello"}
    goal_assigned_rate_dict = {"S0_S0__S0_S1": 2, "S0_S0__S1_S0": 2, "R_S1__S0_S0": 9.7}
    goal_rate_name_dict = {(0, 1): "hi", (0, 2): "hello", (4, 0): "induction"}
    assigned_rate_dict, rate_name_dict = et.off_state_rescue(assigned_rate_dict, rate_name_dict, transitions, 9.7,
                                                       ("S0", "S1", "T1", "R", "B"))
    assert assigned_rate_dict == goal_assigned_rate_dict
    assert rate_name_dict == goal_rate_name_dict


def test_fret():
    assigned_rate_dict = {}