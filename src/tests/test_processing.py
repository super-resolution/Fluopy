import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import pandas as pd
from enum import Enum
import processing as pc


def test_identify_duplices():
    collection = ["T_Z_U", "Z_U_T", "Z_U_I", "U_T_Z", "T_E_S_T", "T_T_S_S", "T_T_S", "S_E_T_T"]
    goal = [[0, 1], [0, 3], [1, 3], [4, 7]]
    duplices = pc.identify_duplices(collection)
    assert duplices == goal


def test_uniques():
    duplices = [[0, 5], [0, 6], [1, 7], [4, 8]]
    state_series = np.array([0, 3, 2, 0, 1, 7, 0, 4, 3, 2, 4, 8, 3, 8])
    joined_states = Enum("Joined_States", ["A", "B", "C", "D", "E", "F", "G", "H", "I"], start=0)
    goal_unique_series = np.array([0, 3, 2, 0, 1, 1, 0, 4, 3, 2, 4, 4, 3, 4])
    goal_unique_states = np.array([0, 1, 2, 3, 4])
    goal_unique_names = ["A", "B", "C", "D", "E"]
    unique_series, unique_states, unique_joined_states, unique_names = pc.uniques(duplices, state_series, joined_states)
    assert np.array_equal(unique_series, goal_unique_series)
    assert np.array_equal(unique_states, goal_unique_states)
    for unique_joined_state, goal_unique_name, goal_unique_state in zip(unique_joined_states, goal_unique_names,
                                                                        goal_unique_states):
        assert unique_joined_state.name == goal_unique_name
        assert unique_joined_state.value == goal_unique_state
    assert unique_names == goal_unique_names


def test_occupation_t():
    time_step_series = np.array([0, 2, 0.5, 0.1, 7, 10, 0.4, 0.002, 7, 7.1, 0.8])
    unique_series = np.array([0, 3, 5, 2, 7, 0, 2, 3, 0, 7, 10])
    unique_states = np.array([0, 2, 3, 5, 7, 10])
    goal_total_times = np.array([9.5, 7.002, 7.5, 0.1, 10.8, np.inf])
    goal_mean_times = np.array([9.5/3, 7.002/2, 7.5/2, 0.1/1, 10.8/2, np.inf])
    total_times, mean_times = pc.occupation_t(time_step_series, unique_series, unique_states)
    assert np.array_equal(goal_total_times, total_times)
    assert np.array_equal(goal_mean_times, mean_times)


def test_convert_unique_states():
    unique_series = np.array([6, 8, 11, 6, 4, 100])
    unique_states = np.array([4, 6, 8, 11, 100])
    goal_unique_series_converted = np.array([1, 2, 3, 1, 0, 4])
    unique_series_converted = pc.convert_unique_states(unique_series, unique_states)
    assert np.array_equal(unique_series_converted, goal_unique_series_converted)


def test_autocorrelate():
    pandas_series = pd.Series(np.array([0, 5, 0, 4, 3]), index=np.arange(0, 5e-3, 1e-3))
    goal_time = np.arange(0, 5e-3, 1e-3)
    goal_correl = np.array([50, 12, 20, 15, 0])  # 50 = 0*0 + 5*5 + 0*0 + 4*4 + 3*3; 12 = 0*5 + 5*0 + 0*4 + 4*3; ...
    time, correl = pc.autocorrelate(pandas_series, normalize=False, log=False)
    assert np.array_equal(time, goal_time)
    assert np.array_equal(correl, goal_correl)
    pandas_series = pd.Series(np.array([0, 5, 0, 4, 3]), index=np.arange(0, 5e-3, 1e-3))
    goal_time = np.arange(0, 5e-3, 1e-3)
    dev_array = np.array([-2.4, 2.6, -2.4, 1.6, 0.6])  # the array minus the mean (=12/5 = 2.4)
    cor_array = np.array([21.2, -15.36, 8.48, -2.28, -1.44])  # the correlation array
    aver_array = np.array([21.2/5, -15.36/4, 8.48/3, -2.28/2, -1.44])  # average with number of sums during correlation
    goal_correl = aver_array / 2.4**2  # normalization
    time, correl = pc.autocorrelate(pandas_series, normalize=True, log=False)
    assert np.array_equal(time, goal_time)
    assert np.allclose(correl, goal_correl)
