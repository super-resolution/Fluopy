from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluopy.routines import (
    get_bleaching_times,
    get_delta_bleaching_times,
)
from fluopy.transitions import SingleState


def test_get_bleaching_times(sim_tr_set_1f_bl):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_1f_bl)
    assert np.isnan(bleaching_times)


def test_get_bleaching_times_2(sim_tr_set_2f_diff):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_2f_diff)
    assert np.all(np.isnan(bleaching_times))


@pytest.mark.parametrize(
    "state_series, time_series",
    [(None, np.array([0.0])), (np.array([[0]]), None)],
)
def test_get_bleaching_times_requires_completed_simulation(state_series, time_series):
    simulation = SimpleNamespace(
        state_series=state_series,
        time_series=time_series,
    )

    with pytest.raises(ValueError, match="completed simulation"):
        get_bleaching_times(simulation)


@pytest.mark.parametrize("bleaching_transitions", [1, 3])
def test_get_bleaching_times_with_shared_destination(bleaching_transitions):
    transition_df = pd.DataFrame(
        {
            "absorbing": [True] * bleaching_transitions + [False],
            "final_state": [SingleState.B] * bleaching_transitions + [SingleState.S0],
        }
    )
    ground = SingleState.S0.value
    bleached = SingleState.B.value
    simulation = SimpleNamespace(
        transition_set=SimpleNamespace(transition_df=transition_df),
        state_series=np.array(
            [
                [ground, ground, bleached, bleached],
                [ground, bleached, bleached, bleached],
                [ground, ground, ground, ground],
            ],
            dtype=np.int8,
        ),
        time_series=np.array([0.0, 2.0, 5.0, 8.0]),
    )

    bleaching_times = get_bleaching_times(simulation)

    np.testing.assert_array_equal(bleaching_times, [2.0, 5.0, np.nan])


def test_get_bleaching_times_with_distinct_destinations():
    transition_df = pd.DataFrame(
        {
            "absorbing": [True, True],
            "final_state": [SingleState.B, SingleState.OFF],
        }
    )
    simulation = SimpleNamespace(
        transition_set=SimpleNamespace(transition_df=transition_df),
        state_series=np.array([[SingleState.S0.value, SingleState.B.value]]),
        time_series=np.array([0.0, 1.0]),
    )

    with pytest.raises(NotImplementedError, match="Multiple bleaching states"):
        get_bleaching_times(simulation)


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
