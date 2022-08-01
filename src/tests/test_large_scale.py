import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import large_scale as ls


def test_multiple_simulations():
    class_args = {"number": 2, "distances": 1, "rates": {"k_S0_S1": [2, "excitation"], "k_S1_S0": [0.5, "emission"]}}
    simulate_args = {"n_steps": 1000, "base": "cy"}
    emitting_args = None
    object_collector = ls.multiple_simulations(2, "Jablonski", class_args, simulate_args, emitting_args, 100)
    assert not np.array_equal(object_collector[0].state_series, object_collector[1].state_series)
