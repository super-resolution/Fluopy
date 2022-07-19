import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import emitting_transitions as et


test_transitions = {"S1__S0": (1, 0), "R__S0": (3, 0), "S0__S1": (0, 1),
    "S0_S1__S0_S0": (1, 0), "S1_S1__S0_S0": (2, 0), "R_R__R_S0": (10, 15),
    "S0_S1_S0__S0_S0_S0": (110, 109), "S1_S1_S0__S0_S0_S0": (111, 109), "S0_S0_R__S0_S0_S0": (112, 109)}
goal_emitting_transitions = ["S1__S0", "S0_S1__S0_S0", "S0_S1_S0__S0_S0_S0"]
goal_emitting_transition_indices = [(1, 0), (1, 0), (110, 109)]


def test_identify_emitting_transitions():
    emitting_transitions, emitting_transition_indices = et.identify_emitting_transitions(test_transitions)
    assert emitting_transitions == goal_emitting_transitions
    assert emitting_transition_indices == goal_emitting_transition_indices


test_arr = np.array([0, 1, 2, 3, 7, 8, 9, 0, 1, 0, 1, 1003, 1004, 2000])
test_seq = np.array([0, 1])
goal_mask = np.array([True, False, False, False, False, False, False, True, False, True, False, False, False])
test_arr_2 = np.array([0, 1, 3, 7, 1001, 5, 0, 3, 7, 1000, 2, 3, 7, 1001])
test_seq_2 = np.array([3, 7, 1001])
goal_mask_2 = np.array([False, False, True, False, False, False, False, False, False, False, False, True])


def test_search_sequence():
    mask = et.search_sequence(test_arr, test_seq)
    assert np.array_equal(mask, goal_mask)
    assert mask.size == test_arr.size - test_seq.size + 1

    mask_2 = et.search_sequence(test_arr_2, test_seq_2)
    assert np.array_equal(mask_2, goal_mask_2)
    assert mask_2.size == test_arr_2.size - test_seq_2.size + 1


test_state_series = np.array([0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 5, 1])
test_emitting_transition_indices = [(1, 0), (5, 0), (10, 0), (15, 0), (20, 0)]
goal_mask_emit = np.array([False, False, True, False, True, False, False, False, True, False, False, False])
test_state_series_2 = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 4, 5, 6, 1, 4, 0, 10, 1, 0])
test_emitting_transition_indices_2 = [(1, 0), (5, 6), (9, 8), (9, 1)]
goal_mask_emit_2 = np.array([False, True, False, False, True, False, False, False, True, False, False, True, False, False,
                        False, False, False, True])


def test_emitter_mask():
    mask = et.emitter_mask(test_state_series, test_emitting_transition_indices)
    assert np.array_equal(mask, goal_mask_emit)
    mask_2 = et.emitter_mask(test_state_series_2, test_emitting_transition_indices_2)
    assert np.array_equal(mask_2, goal_mask_emit_2)


test_emitting_mask = np.array([False, True, False, True, False, True, False, False, True, False, False, True])
test_photon_collection = 1
test_seed = 1
goal_collection_mask = np.array([False, True, False, True, False, True, False, False, True, False, False, True])
test_emitting_mask_2 = np.array([False, True, False, True, False, True, False, True, False, True, False, True,
                                 False, True, False, True, False, True, False, True])
test_photon_collection_2 = 0.4
goal_true_count = 4


def test_detected_emissions():
    collection_mask = et.detected_emissions(test_emitting_mask, test_photon_collection, test_seed)
    assert np.array_equal(collection_mask, goal_collection_mask)
    collection_mask_2 = et.detected_emissions(test_emitting_mask_2, test_photon_collection_2, test_seed)
    assert goal_true_count == len(np.nonzero(collection_mask_2[0]))


test_events_at = np.array([1e-3, 2e-3, 3e-3, 10e-3, 12e-3, 18e-3, 19e-3, 20e-3])
test_unit = "s"
test_resample = "5ms"
goal_series = np.array([3, 0, 2, 2, 1])


def test_pandas_event_time_series():
    series = et.pandas_event_time_series(test_events_at, test_unit, test_resample)
    assert np.array_equal(goal_series, series.values)
