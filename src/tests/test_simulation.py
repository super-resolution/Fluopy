import os
import pytest
import pandas as pd
import numpy as np
import src.prediction as pr
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
            np.array(
                [0, 388, 702, 442, 87, 436, 558, 73, 435, 344, 229], dtype=np.int64
            ),
            index=np.array(
                [
                    0.0,
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                    0.005,
                    0.006,
                    0.007,
                    0.008,
                    0.009,
                    0.01,
                ]
            ),
        )
        pd.testing.assert_series_equal(event_time_series, exp_event_time_series)
    else:
        exp_event_time_series = pd.Series(
            np.array(
                [0, 347, 639, 403, 85, 394, 500, 66, 383, 313, 205], dtype=np.int64
            ),
            index=np.array(
                [
                    0.0,
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                    0.005,
                    0.006,
                    0.007,
                    0.008,
                    0.009,
                    0.01,
                ]
            ),
        )
        assert event_time_points.size == event_time_series.values.sum()
        pd.testing.assert_series_equal(event_time_series, exp_event_time_series)


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
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
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
            with pytest.warns(
                UserWarning, match="Floating point precision error warning"
            ):
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
