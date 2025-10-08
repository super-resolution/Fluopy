import logging

import numpy as np
import pytest

from fluopy import simulation_tcspc as si
from fluopy import transitions as tr

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "dirname, emitting_transition_ids, et_transition_ids, \
                         number_pulses, pulse_duration, time_between_pulses, \
                         excitation_rates, frame_time, store_time_points, \
                         expected",
    [
        [
            "tr_set_1f",
            {1: 1},
            [],
            4.1e4,
            5e-11,
            25e-9,
            {"testfluo_1": 1e11},
            "1ms",
            True,
            0,
        ],
        [
            "tr_set_1f_bl",
            {1: 1},
            [],
            4.1e6,
            5e-11,
            25e-9,
            {"testfluo_1": 1e11},
            "1ms",
            False,
            1,
        ],
        [
            "tr_set_1f",
            {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1},
            [10, 11],
            8e4,
            5e-11,
            1e-8,
            {"testfluo_1": 1e9},
            "1ms",
            True,
            2,
        ],
        [
            "tr_set_bl_et_2f_same",
            {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1},
            [10, 11],
            4e4,
            5e-11,
            1e-1,
            {"testfluo_1": 1e11},
            "1ms",
            True,
            3,
        ],
        [
            "tr_set_bl_et_2f_same",
            {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1},
            [10, 11],
            4e4,
            5e-11,
            1e-9,
            {"testfluo_1": 1e11},
            "1ms",
            True,
            4,
        ],
        [
            "tr_set_bl_et_2f_diff",
            {
                4: 1,
                5: 1,
                6: 1,
                7: 1,
            },
            [4],
            4e4,
            5e-11,
            1e-1,
            {"testfluo_1": 1e11, "testfluo_2": 1e1},
            "1ms",
            True,
            5,
        ],
        [
            "tr_set_1f",
            {1: 1},
            [],
            4e4,
            5e-11,
            1e-1,
            {"testfluo_1": 1e12},
            "1ms",
            True,
            6,
        ],
        [
            "tr_set_bl_et_2f_diff",
            {2: 1, 3: 1, 6: 0.5, 7: 0.5},
            [4],
            4e4,
            5e-11,
            1e-1,
            {"testfluo_1": 1e12, "testfluo_2": 1.4e10},
            "1ms",
            True,
            7,
        ],
    ],
)
def test_simulate_TCSPC(
    dirname,
    emitting_transition_ids,
    et_transition_ids,
    number_pulses,
    pulse_duration,
    time_between_pulses,
    excitation_rates,
    frame_time,
    store_time_points,
    request,
    expected,
    caplog,
):
    rng = np.random.default_rng(42)
    tr_set = request.getfixturevalue(dirname)

    # this tests a standard case of a single fluorophore
    if expected == 0:
        assert tr.SingleState.S1.value == 1
        assert tr.SingleState.S0.value == 0
        tr_set = tr_set.adjust_rates({0: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert (
                "the last frame (of index 0.002) has 2.50e-02 times the pulses of other frames."
                in caplog.text
            )
        caplog.clear()

        assert (
            event_time_series.values.sum()
            == event_time_points.size
            == lifetimes_DA.size + lifetimes_D.size
            == lifetimes_all.size
        )
        assert (
            event_time_series.size
            == 3
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) + 1
        )
        assert lifetimes_DA.size == 0
        np.testing.assert_allclose(lifetimes_D.mean(), 1.0405e-09, rtol=1e-2)

    # this tests for the warning of bleaching
    elif expected == 1:
        tr_set = tr_set.adjust_rates({0: 0}, keep_zero_rates=True)
        tr_set.finalize()

        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert (
                "the last frame (of index 0.103) has 5.00e-01 times the pulses of other frames."
                in caplog.text
            )
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert event_time_series.values.sum() == lifetimes_D.size
        assert event_time_points is None
        assert lifetimes_DA.size == 0
        np.testing.assert_allclose(lifetimes_D.mean(), 9.96e-10, rtol=2e-2)

    # this tests for the warning of not enough pulses
    elif expected == 2:
        tr_set = tr_set.adjust_rates({0: 0, 8: 2e10}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert (
                "Not enough laser pulses to completely simulate a single frame "
                "(requires at least 1.0e+05 pulses)."
            ) in caplog.text
            assert (
                "the last frame (of index 0.001) has 8.00e-01 times the pulses of other "
                "frames."
            ) in caplog.text
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
        )

    # this tests whether homoFRET doesn't change the observed lifetime and other details
    # see comments within
    elif expected == 3:
        tr_set = tr_set.filter_by_identity([9])  # filter triplet et
        tr_set = tr_set.adjust_rates({0: 0, 8: 2e10}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert (
                "the last frame (of index 4000.0) has 0.00e+00 times the pulses of other frames."
                in caplog.text
            )
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
        )
        np.testing.assert_allclose(
            np.mean(np.concatenate([lifetimes_DA, lifetimes_D])), 9.96e-10, rtol=1.1e-2
        )
        # the excitation rate is high enough to ensure that at each pulse, all
        # all fluorophores are excited. The lifetime of only donor consists mostly of
        # S1/S1 deexcitations, meaning that the lifetime is the reciprocal of the
        # combined rates.
        np.testing.assert_allclose(np.mean(lifetimes_D), 0.5e-9, rtol=1e-1)
        # the between-pulse-time is high enough to ensure that (almost) no S1 states
        # are carried over to the next pulse. Since each pulse excites all fluorophores,
        # the lifetime of the donor/acceptor pair 'adds' the time of the first
        # deexciation.
        # Hence, one fluorophore will deexcite under S1/S1 condition, the other under
        # S1/S0 condition, so its 50%-50%.
        np.testing.assert_allclose(np.mean(lifetimes_DA), 1.5e-9, rtol=1e-1)

    # this tests for the importance of a long-enough time-between-pulses to ensure
    # that the fluorophore's excitations are not carried over to the next pulse
    elif expected == 4:
        tr_set = tr_set.adjust_rates({0: 0, 8: 2e10}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert "Not enough laser pulses to completely" in caplog.text
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
        )
        # the values they are tested against are much smaller than the actual
        # fluorescence lifetime, but the time_between_pulses is too short and the
        # measured lifetime always starts at the previous pulse.
        # This is especially visible for homoFRET chains
        np.testing.assert_allclose(np.mean(lifetimes_D), 4.4e-10, rtol=1e-1)
        np.testing.assert_allclose(np.mean(lifetimes_DA), 5.9e-10, rtol=1e-1)

    elif expected == 5:
        # the energy transfer of the second to the first fluorophore is set to 0
        tr_set = tr_set.adjust_rates({0: 0, 8: 1e10, 9: 0, 15: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert "the last frame (of index" in caplog.text
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
        )
        # the excitation probability of the second fluorophore is nearly 0,
        # so all photon emissions happen while energy transfer is possible
        assert lifetimes_D.size == 0
        # the lifetime of the donor when acceptor is available (which is virtually
        # always the case) is lowered by the energy transfer.
        np.testing.assert_allclose(np.mean(lifetimes_DA), 8.9e-11, rtol=1e-1)
        # the emissions do not discriminate between the two fluorophores, so to ensure
        # that only the donor lifetime is measured, the fluorescence of the acceptor
        # should not pass the bandpass filter (here, the number of photons passing it
        # are set to 0).

    # test whether the number of pulses is the same as the number of photons if system
    # is 1 fluorophore and has excitation and emission probability of 1 and the
    # between-pulse-time is long enough
    elif expected == 6:
        tr_set = tr_set.adjust_rates(
            {0: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}, keep_zero_rates=True
        )
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert "the last frame (of index" in caplog.text
        caplog.clear()

        assert event_time_series.size == int(
            np.ceil(number_pulses * time_between_pulses / 1e-3) + 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
            == 4e4
        )

        # this tests for half the number of photons if half the excitation probability
        excitation_rates = {
            "testfluo_1": 1.4e10
        }  # corresponds to excitation probability of ~0.5
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert "the last frame (of index" in caplog.text
        caplog.clear()

        np.testing.assert_allclose(event_time_series.values.sum(), 2e4, rtol=1e-1)

    # here, the excitation probability of one fluorophore is 1, the other 0.5, and the
    # probability of emission detection is 1 and 0.5. No energy transfers possible.
    # Hence, the expected number of emissions is the number of pulses plus a quarter
    # if fluorescence is the only deexcitation pathway.
    elif expected == 7:
        tr_set = tr_set.filter_by_identity([2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15])
        tr_set = tr_set.adjust_rates({0: 0, 2: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with caplog.at_level(logging.WARNING):
            (
                event_time_series,
                event_time_points,
                lifetimes_DA,
                lifetimes_D,
                lifetimes_all,
            ) = si.simulate_TCSPC(
                transition_set=tr_set,
                emitting_transition_ids=emitting_transition_ids,
                et_transition_ids=et_transition_ids,
                number_pulses=number_pulses,
                pulse_duration=pulse_duration,
                time_between_pulses=time_between_pulses,
                excitation_rates=excitation_rates,
                frame_time=frame_time,
                store_time_points=store_time_points,
                seed=rng,
            )
            assert "the last frame (of index" in caplog.text
        caplog.clear()

        np.testing.assert_allclose(event_time_series.values.sum(), 4e4 + 1e4, rtol=1e-1)


def test_simulate_TCSPC_detailed(request, caplog):
    rng = np.random.default_rng(42)
    transition_set = request.getfixturevalue("tr_set_bl_et_2f_diff")
    transition_set = transition_set.filter_by_identity(
        [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
    )
    transition_set = transition_set.adjust_rates({0: 0, 2: 0}, keep_zero_rates=True)
    transition_set.finalize()
    emitting_transition_ids = {2: 1, 3: 1, 6: 0.5, 7: 0.5}
    et_transition_ids = [4]
    number_pulses = 4e4
    pulse_duration = 5e-11
    time_between_pulses = 1e-1
    excitation_rates = {"testfluo_1": 1e12, "testfluo_2": 1.4e10}
    frame_time = "1ms"
    store_time_points = True
    with caplog.at_level(logging.WARNING):
        return_values = si.simulate_TCSPC_detailed(
            transition_set=transition_set,
            emitting_transition_ids=emitting_transition_ids,
            et_transition_ids=et_transition_ids,
            number_pulses=number_pulses,
            pulse_duration=pulse_duration,
            time_between_pulses=time_between_pulses,
            excitation_rates=excitation_rates,
            frame_time=frame_time,
            store_time_points=store_time_points,
            seed=rng,
        )
        assert "the last frame (of index" in caplog.text
    caplog.clear()

    # test whether each transition is followed by a correct transition
    df = transition_set.combined_state_transitions_df
    arr = return_values[-1].transition_series
    curr_indices = arr[:-1]
    next_indices = arr[1:]
    final_states = df["final_state"].to_numpy()
    initial_states = df["initial_state"].to_numpy()
    is_match = final_states[curr_indices] == initial_states[next_indices]
    assert is_match.all()
    # test whether the simulation attributes are correct
    assert (
        return_values[-1].transition_series.size
        == return_values[-1].state_series.shape[1] - 1
    )
    assert return_values[-1].state_series.shape == (2, 119971)
    assert (
        return_values[-1].transition_series.size
        == return_values[-1].time_series.size - 1
    )
    # test whether the spacing of time series was successfull
    assert (
        np.unique(return_values[-1].time_series).size
        == return_values[-1].time_series.size
    )


def test_space_multiple_excitations(caplog):
    time_series = np.array([0, 0, 1, 1, 1, 2], dtype=np.float64)
    assert np.unique(time_series).size != time_series.size
    with caplog.at_level(logging.WARNING):
        time_series_adjusted = si.space_multiple_excitations(time_series)
        assert "Multiple excitations at the same time point" in caplog.text
    caplog.clear()

    assert np.unique(time_series_adjusted).size == time_series_adjusted.size


def test_insert_excitations(request):
    excitation_series = np.array(
        [1, 2, 0, -1, -1, -1, 2, -1, 1, -1, -1, 2, 1, -1, -1, -1]
    )
    transition_series = np.array([250, 241, 446, 446, 80, 120, 81, 398, 122])
    transition_set = request.getfixturevalue("tr_set_bl_et_3f")
    transition_series_adj = si.insert_excitations(
        transition_series, transition_set, excitation_series
    )
    transition_series_exp = np.array(
        [0, 347, 9, 250, 241, 446, 346, 446, 0, 80, 120, 346, 2, 81, 398, 122]
    )
    np.testing.assert_array_equal(transition_series_adj, transition_series_exp)


def test_get_state_series(request):
    transition_set = request.getfixturevalue("tr_set_bl_et_3f")
    transition_series = np.array(
        [0, 347, 9, 250, 241, 446, 346, 446, 0, 80, 120, 346, 2, 81, 398, 122]
    )
    state_series = si.get_state_series(transition_set, transition_series)
    state_series_exp = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3, 0, 0, 1, 3, 3, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 3, 3],
        ]
    )
    np.testing.assert_array_equal(state_series, state_series_exp)
