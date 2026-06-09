from copy import deepcopy

import numpy as np
import pandas as pd

from fluopy.routines import (
    emission_post_processing,
    get_bleaching_times,
    get_delta_bleaching_times,
    truncate_fingerprints,
)


def test_emission_post_processing(em_large):
    emission = deepcopy(em_large)
    emission_post_processing(emis=emission, seed=1)
    assert not emission.event_time_series.equals(em_large.event_time_series)


def test_get_bleaching_times(sim_tr_set_1f_bl):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_1f_bl)
    assert np.isnan(bleaching_times)


def test_get_bleaching_times_2(sim_tr_set_2f_diff):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_2f_diff)
    assert np.all(np.isnan(bleaching_times))


def test_get_delta_bleaching_times():
    bleaching_times = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    deltas = get_delta_bleaching_times(bleaching_times=bleaching_times)
    deltas_expected = np.array(
        [
            [1, 10],
            [1, 10],
            [1, 10],
        ]
    )
    assert np.array_equal(deltas, deltas_expected)


def test_truncate_fingerprints():
    fingerprint = pd.Series([1, 2, 3, 4, 5])
    new_fingerprint = truncate_fingerprints(fingerprint=fingerprint, low=1, high=4)
    fingerprint_expected = pd.Series([0, 0.5, 1], index=[1, 2, 3])
    assert new_fingerprint.equals(fingerprint_expected)
