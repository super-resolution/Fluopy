import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import pandas as pd
import emitting_transitions as et


def test_identify_emitting_transitions():
    transitions = {"S1__S0": (1, 0), "R__S0": (3, 0), "S0__S1": (0, 1), "S0_S1__S0_S0": (1, 0), "S1_S1__S0_S0": (2, 0),
                   "R_R__R_S0": (10, 15), "S0_S1_S0__S0_S0_S0": (110, 109), "S1_S1_S0__S0_S0_S0": (111, 109),
                   "S0_S0_R__S0_S0_S0": (112, 109)}
    goal_emitting_transitions = ["S1__S0", "S0_S1__S0_S0", "S0_S1_S0__S0_S0_S0"]
    goal_emitting_transition_indices = [(1, 0), (1, 0), (110, 109)]
    emitting_transitions, emitting_transition_indices = et.identify_emitting_transitions(transitions)
    assert emitting_transitions == goal_emitting_transitions
    assert emitting_transition_indices == goal_emitting_transition_indices


def test_search_sequence():
    arr = np.array([0, 1, 2, 3, 7, 8, 9, 0, 1, 0, 1, 1003, 1004, 2000])
    seq = np.array([0, 1])
    goal_mask = np.array([True, False, False, False, False, False, False, True, False, True, False, False, False])
    mask = et.search_sequence(arr, seq)
    assert np.array_equal(mask, goal_mask)
    assert mask.size == arr.size - seq.size + 1

    arr_2 = np.array([0, 1, 3, 7, 1001, 5, 0, 3, 7, 1000, 2, 3, 7, 1001])
    seq_2 = np.array([3, 7, 1001])
    goal_mask_2 = np.array([False, False, True, False, False, False, False, False, False, False, False, True])
    mask_2 = et.search_sequence(arr_2, seq_2)
    assert np.array_equal(mask_2, goal_mask_2)
    assert mask_2.size == arr_2.size - seq_2.size + 1


def test_emitter_mask():
    state_series = np.array([0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 5, 1])
    emitting_transition_indices = [(1, 0), (5, 0), (10, 0), (15, 0), (20, 0)]
    goal_mask = np.array([False, False, True, False, True, False, False, False, True, False, False, False])
    mask = et.emitter_mask(state_series, emitting_transition_indices)
    assert np.array_equal(mask, goal_mask)
    state_series_2 = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 4, 5, 6, 1, 4, 0, 10, 1, 0])
    emitting_transition_indices_2 = [(1, 0), (5, 6), (9, 8), (9, 1)]
    goal_mask_2 = np.array([False, True, False, False, True, False, False, False, True, False, False, True, False,
                                 False, False, False, False, True])
    mask_2 = et.emitter_mask(state_series_2, emitting_transition_indices_2)
    assert np.array_equal(mask_2, goal_mask_2)


def test_detected_emissions():
    emitting_mask = np.array([False, True, False, True, False, True, False, False, True, False, False, True])
    goal_collection_mask = np.array([False, True, False, True, False, True, False, False, True, False, False, True])
    collection_mask = et.detected_emissions(emitting_mask, 1, 1)
    assert np.array_equal(collection_mask, goal_collection_mask)
    emitting_mask_2 = np.array([False, True, False, True, False, True, False, True, False, True, False, True,
                                False, True, False, True, False, True, False, True])
    collection_mask_2 = et.detected_emissions(emitting_mask_2, 0.4, 1)
    assert len(np.nonzero(collection_mask_2)[0]) == 4


def test_pandas_event_time_series():
    events_at = np.array([1e-3, 2e-3, 3e-3, 10e-3, 12e-3, 18e-3, 19e-3, 20e-3])
    goal_series = np.array([3, 0, 2, 2, 1])
    series = et.pandas_event_time_series(events_at, "s", "5ms")
    assert np.array_equal(goal_series, series.values)


def test_blink_statistics():
    pandas_series = pd.Series(np.array([0, 5, 0, 4, 3, 1, 0, 0, 5, 6]), index=np.arange(0, 10e-3, 1e-3))
    goal_on_periods = np.array([5, 2])
    goal_off_periods = np.array([3])
    goal_on_periods_frames = np.array([0, 8])
    goal_off_periods_frames = np.array([5])
    on_periods, off_periods, on_periods_frames, off_periods_frames = \
        et.blink_statistics(pandas_series, 2, 1, False)
    assert np.array_equal(goal_on_periods, on_periods)
    assert np.array_equal(goal_off_periods, off_periods)
    assert np.array_equal(goal_off_periods_frames, off_periods_frames)
    assert np.array_equal(goal_on_periods_frames, on_periods_frames)


def test_frac_int_time():
    series = pd.Series(np.array([0, 5, 0, 4, 3, 1, 0, 0, 5, 6]), index=np.arange(0, 10e-3, 1e-3))
    arrival_time_rel = et.frac_int_time(series, 0.6)
    assert round(arrival_time_rel, 6) == round(8e-3/9e-3, 6)
