import os
import re
import pytest
import pandas as pd
import numpy as np
import src.prediction as pr
import src.transitions as tr
import src.simulation as si


def test_direct_method_steps(tr_set_1f):
    time_series, transition_series = si.direct_method_steps(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        seed=1,
        use_memmap=None,
    )
    exp_time_series = np.array(
        [
            0.00000000e00,
            6.69779121e-10,
            3.33705195e-07,
            3.34870488e-07,
            3.67386152e-07,
            3.67984729e-07,
            4.16647630e-07,
            4.17757106e-07,
            6.22956746e-07,
            6.24966350e-07,
            8.98761261e-07,
        ]
    )
    exp_transition_series = np.array([6, 0, 6, 0, 1, 0, 6, 0, 6, 0])
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_direct_method_time(tr_set_1f):
    time_series, transition_series = si.direct_method_time(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        end_time=1e-6,
        seed=1,
        use_memmap=None,
    )
    exp_time_series = np.array(
        [
            0.00000000e00,
            6.69779104e-10,
            3.33705188e-07,
            3.34870480e-07,
            3.67386146e-07,
            3.67984722e-07,
            4.16647625e-07,
            4.17757101e-07,
            6.22956748e-07,
            6.24966353e-07,
            8.98761274e-07,
            8.99048470e-07,
            1.00000000e-06,
        ]
    )
    exp_transition_series = np.array([6, 0, 6, 0, 1, 0, 6, 0, 6, 0, 1])
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_approximation(pred_tr_set_1f):
    time_series, transition_series = si.approximation(
        prediction=pred_tr_set_1f, size=20, seed=1
    )
    exp_time_series = np.array(
        [
            0.00000000e00,
            5.10917683e-09,
            5.57154084e-09,
            1.36951508e-07,
            1.37640248e-07,
            3.16717287e-07,
            3.17831801e-07,
            3.57483429e-07,
            3.58382213e-07,
            6.46776479e-07,
            6.49616447e-07,
            6.85487720e-07,
            6.87518573e-07,
            8.22713899e-07,
            8.23879737e-07,
            8.46019946e-07,
            8.48668970e-07,
            1.00736073e-06,
            1.00768934e-06,
        ]
    )
    exp_transition_series = np.array(
        [0, 1, 0, 1, 0, 6, 0, 6, 0, 6, 0, 6, 0, 6, 0, 1, 0, 6], dtype=np.int64
    )
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


@pytest.mark.parametrize(
    "store_time_points, emitting_transition_ids, expected",
    [[False, {1: 1}, None], [True, {1: 0.9}, ""]],
)
def test_simulate_experiment(
    tr_set_1f, store_time_points, emitting_transition_ids, expected
):
    event_time_points, event_time_series = si.simulate_experiment(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        emitting_transition_ids=emitting_transition_ids,
        start_index=0,
        size=10,
        frames=10,
        frame_time="1ms",
        store_time_points=store_time_points,
        seed=1,
    )
    if expected is None:
        assert event_time_points is None
        exp_event_time_series = pd.Series(
            np.array([388, 702, 442, 87, 436, 558, 73, 435, 344, 229], dtype=np.int64),
            index=np.array(
                [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
            ),
        )
        pd.testing.assert_series_equal(event_time_series, exp_event_time_series)
    else:
        exp_event_time_series = pd.Series(
            np.array([347, 639, 403, 85, 394, 500, 66, 383, 313, 205], dtype=np.int64),
            index=np.array(
                [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
            ),
        )
        assert event_time_points.size == event_time_series.values.sum()
        pd.testing.assert_series_equal(event_time_series, exp_event_time_series)


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
):
    tr_set = request.getfixturevalue(dirname)

    # this tests a standard case of a single fluorophore
    if expected == 0:
        assert tr.SingleState.S1.value == 1
        assert tr.SingleState.S0.value == 0
        tr_set = tr_set.adjust_rates({0: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "the last frame (of index 1) has 2.50e-02 times the pulses of other "
                "frames."
            ),
        ):
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
        )
        assert (
            event_time_series.values.sum()
            == event_time_points.size
            == lifetimes_DA.size + lifetimes_D.size
        )
        assert event_time_series.size == 2
        assert lifetimes_DA.size == 0
        np.testing.assert_allclose(lifetimes_D.mean(), 9.96e-10, rtol=1e-2)

    # this tests for the warning of bleaching
    elif expected == 1:
        tr_set = tr_set.adjust_rates({0: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
        )
        assert str(record[0].message) == (
            "the last frame (of index 102) has 5.00e-01 times the pulses of other "
            "frames."
        )
        assert str(record[1].message) == (
            "All fluorophores underwent photobleaching or entered another Markov "
            "chain absorbing state."
        )
        assert event_time_series.values.sum() == lifetimes_D.size
        assert event_time_points is None
        assert lifetimes_DA.size == 0
        np.testing.assert_allclose(lifetimes_D.mean(), 9.96e-10, rtol=2e-2)

    # this tests for the warning of not enough pulses
    elif expected == 2:
        tr_set = tr_set.adjust_rates({0: 0, 8: 2e10}, keep_zero_rates=True)
        tr_set.finalize()
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
        )
        assert str(record[0].message) == (
            "Not enough laser pulses to completely simulate a single frame "
            "(requires at least 1.0e+05 pulses)."
        )
        assert str(record[1].message) == (
            "the last frame (of index 0) has 8.00e-01 times the pulses of other "
            "frames."
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
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "the last frame (of index 3999999) has 0.00e+00 times the pulses of "
                "other frames."
            ),
        ):
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
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
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
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
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
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
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        assert (
            event_time_series.index[-1]
            == int(np.ceil(number_pulses * time_between_pulses / 1e-3)) - 1
        )
        assert (
            event_time_series.values.sum()
            == lifetimes_D.size + lifetimes_DA.size
            == event_time_points.size
        )
        assert event_time_series.values.sum() == 4e4

        # this tests for half the number of photons if half the excitation probability
        excitation_rates = {
            "testfluo_1": 1.4e10
        }  # corresponds to excitation probability of ~0.5
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        np.testing.assert_allclose(event_time_series.values.sum(), 2e4, rtol=1e-1)

    # here, the excitation probability of one fluorophore is 1, the other 0.5, and the
    # probability of emission detection is 1 and 0.5. No energy transfers possible.
    # Hence, the expected number of emissions is the number of pulses plus a quarter
    # if fluorescence is the only deexcitation pathway.
    elif expected == 7:
        tr_set = tr_set.filter_by_identity([2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15])
        tr_set = tr_set.adjust_rates({0: 0, 2: 0}, keep_zero_rates=True)
        tr_set.finalize()
        with pytest.warns(UserWarning) as record:
            event_time_series, event_time_points, lifetimes_DA, lifetimes_D = (
                si.simulate_TCSPC(
                    transition_set=tr_set,
                    emitting_transition_ids=emitting_transition_ids,
                    et_transition_ids=et_transition_ids,
                    number_pulses=number_pulses,
                    pulse_duration=pulse_duration,
                    time_between_pulses=time_between_pulses,
                    excitation_rates=excitation_rates,
                    frame_time=frame_time,
                    store_time_points=store_time_points,
                    seed=1,
                )
            )
        np.testing.assert_allclose(event_time_series.values.sum(), 4e4 + 1e4, rtol=1e-1)


def test_simulation(tr_set_1f):
    simulation = si.Simulation(transition_set=tr_set_1f)
    assert simulation.transition_set == tr_set_1f
    assert simulation.time_series is None
    assert simulation.transition_series is None
    assert simulation.state_series is None
    assert simulation.memmap_path is None
    with pytest.raises(
        ValueError, match="simulation not available if transition_set not finalized"
    ):
        tr_set_1f_new = tr_set_1f.adjust_rates(change_dict={6: 1e6})
        simulation = si.Simulation(transition_set=tr_set_1f_new)


# also contains the test for simulation.delete_memmaps()
@pytest.mark.parametrize("use_memmap", [[None], [""]])
@pytest.mark.parametrize(
    "end_time, exp_time_series, exp_transition_series, exp_state_series",
    [
        [
            None,
            np.array(
                [
                    0.00000000e00,
                    1.15167403e-07,
                    1.17104237e-07,
                    3.17474333e-07,
                    3.17663434e-07,
                    4.20587581e-07,
                    4.20870590e-07,
                    6.11643129e-07,
                    6.12836509e-07,
                    9.58384640e-07,
                    9.59976949e-07,
                ]
            ),
            np.array([0, 6, 0, 6, 0, 6, 0, 6, 0, 1]),
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]),
        ],
        [
            1e-6,
            np.array(
                [
                    0.00000000e00,
                    1.15167401e-07,
                    1.17104235e-07,
                    3.17474331e-07,
                    3.17663432e-07,
                    4.20587582e-07,
                    4.20870591e-07,
                    6.11643130e-07,
                    6.12836509e-07,
                    9.58384629e-07,
                    9.59976939e-07,
                    1.00000000e-06,
                ]
            ),
            np.array([0, 6, 0, 6, 0, 6, 0, 6, 0, 1]),
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]),
        ],
    ],
)
def test_simulation_run(
    end_time,
    use_memmap,
    exp_time_series,
    exp_transition_series,
    exp_state_series,
    tr_set_1f,
):
    if use_memmap is not None:
        memmap_path = os.path.join(os.path.dirname(__file__), "temp_data")
    else:
        memmap_path = None

    size = 10
    simulation = si.Simulation(transition_set=tr_set_1f)
    simulation.run(
        start_at=(0,), size=size, end_time=end_time, seed=1, use_memmap=memmap_path
    )
    np.testing.assert_array_almost_equal(simulation.time_series, exp_time_series)
    np.testing.assert_array_equal(simulation.transition_series, exp_transition_series)
    np.testing.assert_array_equal(simulation.state_series, exp_state_series)

    if end_time is None:
        assert simulation.time_series.size == size + 1
        assert simulation.transition_series.size == size
        assert simulation.state_series.shape[1] == size + 1
    else:
        assert simulation.time_series.size == simulation.transition_series.size + 2
        assert simulation.state_series.shape[1] == simulation.transition_series.size + 1
        assert simulation.time_series[-1] == end_time

    if use_memmap is not None:
        assert simulation.memmap_path == memmap_path
        assert len(simulation.time_series.base) == exp_time_series.size * 64 / 8
        assert (
            len(simulation.transition_series.base)
            == exp_transition_series.size * 32 / 8
        )
        # 8 bit per number (* 8), result in byte (/ 8)
        assert (
            len(simulation.state_series.base)
            == simulation.state_series.shape[1]
            * 8
            / 8
            * simulation.state_series.shape[0]
        )

        assert os.path.isfile(os.path.join(memmap_path, "state_series"))
        assert os.path.isfile(os.path.join(memmap_path, "time_series"))
        assert os.path.isfile(os.path.join(memmap_path, "transition_series"))
        simulation.delete_memmaps()
        assert not os.path.isfile(os.path.join(memmap_path, "state_series"))
        assert not os.path.isfile(os.path.join(memmap_path, "time_series"))
        assert not os.path.isfile(os.path.join(memmap_path, "transition_series"))
        assert not hasattr(simulation, "transition_series")
        assert not hasattr(simulation, "time_series")
        assert not hasattr(simulation, "state_series")

    else:
        assert simulation.memmap_path is None

    with pytest.raises(
        ValueError,
        match="The number of starting states doesn't match the number of fluorophores.",
    ):
        simulation.run(
            start_at=(0, 1), size=size, end_time=None, seed=1, use_memmap=None
        )


@pytest.mark.parametrize(
    "dirname, expected",
    [
        ["tr_set_1f", "ValueError1"],
        ["tr_set_1f", ""],
        ["tr_set_2f_diff", "ValueError2"],
    ],
)
def test_simulation_approximate(dirname, request, expected):
    tr_set = request.getfixturevalue(dirname)
    pred = pr.Prediction(transition_set=tr_set)
    size = 10
    if expected == "ValueError1":
        tr_set_new = tr_set.adjust_rates(change_dict={6: 1e6})
        tr_set_new.finalize()
        simulation = si.Simulation(transition_set=tr_set_new)
        with pytest.raises(
            ValueError,
            match="prediction is based on different transition_set than simulation.",
        ):
            simulation.approximate(prediction=pred, size=size, seed=1)
    else:
        simulation = si.Simulation(transition_set=tr_set)
        if expected == "ValueError2":
            with pytest.raises(
                ValueError,
                match="approximation only available to single fluorophore systems.",
            ):
                simulation.approximate(prediction=pred, size=size, seed=1)
        else:
            simulation.approximate(prediction=pred, size=size, seed=1)
            exp_time_series = np.array(
                [
                    0.00000000e00,
                    9.24297413e-07,
                    9.24796052e-07,
                    9.87802584e-07,
                    9.88354005e-07,
                    1.00819032e-06,
                    1.00822003e-06,
                    1.31769214e-06,
                    1.31845621e-06,
                ]
            )
            exp_transition_series = np.array([0, 1, 0, 6, 0, 6, 0, 6], dtype=np.int64)
            exp_state_series = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64)
            np.testing.assert_array_almost_equal(
                simulation.time_series, exp_time_series
            )
            np.testing.assert_array_equal(
                simulation.transition_series, exp_transition_series
            )
            np.testing.assert_array_equal(simulation.state_series, exp_state_series)
