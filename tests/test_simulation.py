import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fluopy import prediction as pr
from fluopy import simulation as si


class TestSimualation:

    def test_init_empty(self):
        with pytest.raises(TypeError):
            si.Simulation()

    def test_init(self, tr_set_1f_bl):
        simulation = si.Simulation(transition_set=tr_set_1f_bl)
        assert simulation.time_series is None
        assert simulation.state_series is None
        assert simulation.transition_series is None
        assert simulation.memmap_path is None

    def test_run(self, tr_set_1f_bl):
        rng = np.random.default_rng(42)
        simulation = si.Simulation(transition_set=tr_set_1f_bl)
        simulation.run(size=1000, seed=rng)
        assert simulation.time_series.shape == (1001,)
        assert simulation.state_series.shape == (1, 1001)
        assert simulation.transition_series.shape == (1000,)
        assert simulation.memmap_path is None


def test_direct_method_steps(tr_set_1f):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.direct_method_steps(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        seed=rng,
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
    exp_transition_series = np.array([6, 0, 6, 0, 6, 0, 6, 0, 1, 0])
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_direct_method_steps_with_memmap(tr_set_1f, tmp_path):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.direct_method_steps(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        seed=rng,
        use_memmap=tmp_path,
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
    exp_transition_series = np.array([6, 0, 6, 0, 6, 0, 6, 0, 1, 0])
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_direct_method_time(tr_set_1f):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.direct_method_time(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        end_time=1e-6,
        seed=rng,
        use_memmap=None,
    )
    exp_time_series = np.array(
        [
            0.00000000e0,
            2.56240192e-10,
            2.64705453e-8,
            2.88331209e-8,
            7.57644173e-8,
            7.78192550e-8,
            2.48408814e-7,
            2.48849080e-7,
            3.88686227e-7,
            3.89275763e-7,
            4.21806267e-7,
            4.22083224e-7,
            4.27196942e-7,
            4.27447478e-7,
            5.58476170e-7,
            5.60345094e-7,
            6.11016612e-7,
            6.12138006e-7,
            7.42125463e-7,
            7.44166288e-7,
            9.99200862e-7,
            1.00000000e-6,
        ]
    )
    exp_transition_series = np.array(
        [6, 0, 6, 0, 6, 0, 6, 0, 1, 0, 6, 0, 1, 0, 6, 0, 6, 0, 6, 0]
    )
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_direct_method_time_with_memmap(tr_set_1f, tmp_path):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.direct_method_time(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        start_index=0,
        size=10,
        end_time=1e-6,
        seed=rng,
        use_memmap=tmp_path,
    )
    exp_time_series = np.array(
        [
            0.00000000e0,
            2.56240192e-10,
            2.64705453e-8,
            2.88331209e-8,
            7.57644173e-8,
            7.78192550e-8,
            2.48408814e-7,
            2.48849080e-7,
            3.88686227e-7,
            3.89275763e-7,
            4.21806267e-7,
            4.22083224e-7,
            4.27196942e-7,
            4.27447478e-7,
            5.58476170e-7,
            5.60345094e-7,
            6.11016612e-7,
            6.12138006e-7,
            7.42125463e-7,
            7.44166288e-7,
            9.99200862e-7,
            1.00000000e-6,
        ]
    )
    exp_transition_series = np.array(
        [6, 0, 6, 0, 6, 0, 6, 0, 1, 0, 6, 0, 1, 0, 6, 0, 6, 0, 6, 0]
    )
    np.testing.assert_array_almost_equal(time_series, exp_time_series)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_first_reaction_method(tr_set_bl_et_2f_diff):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.first_reaction_method(
        transition_matrix=tr_set_bl_et_2f_diff.transition_matrix,
        row_sums=tr_set_bl_et_2f_diff.row_sums,
        combined_state_transitions_df=tr_set_bl_et_2f_diff.combined_state_transitions_df,
        include_kap_sq=True,
        minimum_rate=1e3,
        start_index=0,
        size=4,
        seed=rng,
        use_memmap=None,
    )
    exp_time_series = np.array(
        [
            0.00000000000000e0,
            2.51211829116471e-10,
            2.52237265362804e-10,
            2.53171254576898e-10,
            2.53177307732260e-10,
        ]
    )
    exp_transition_series = np.array([32, 63, 32, 63])
    np.testing.assert_array_almost_equal(time_series, exp_time_series, decimal=14)
    np.testing.assert_array_equal(transition_series, exp_transition_series)

    time_series, transition_series = si.first_reaction_method(
        transition_matrix=tr_set_bl_et_2f_diff.transition_matrix,
        row_sums=tr_set_bl_et_2f_diff.row_sums,
        combined_state_transitions_df=tr_set_bl_et_2f_diff.combined_state_transitions_df,
        include_kap_sq=False,
        minimum_rate=1e3,
        start_index=0,
        size=4,
        seed=rng,
        use_memmap=None,
    )
    exp_time_series = np.array(
        [
            0.00000000000000e0,
            2.29979230104149e-13,
            4.33853663715442e-13,
            1.49257871145080e-12,
            1.76534034244079e-12,
        ]
    )
    exp_transition_series = np.array([32, 63, 32, 63])
    np.testing.assert_array_almost_equal(time_series, exp_time_series, decimal=14)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_first_reaction_method_with_memmap(tr_set_bl_et_2f_diff, tmp_path):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.first_reaction_method(
        transition_matrix=tr_set_bl_et_2f_diff.transition_matrix,
        row_sums=tr_set_bl_et_2f_diff.row_sums,
        combined_state_transitions_df=tr_set_bl_et_2f_diff.combined_state_transitions_df,
        minimum_rate=1e3,
        start_index=0,
        size=4,
        seed=rng,
        use_memmap=tmp_path,
    )
    exp_time_series = np.array(
        [
            0.00000000000000e0,
            2.51211829116471e-10,
            2.52237265362804e-10,
            2.53171254576898e-10,
            2.53177307732260e-10,
        ]
    )
    exp_transition_series = np.array([32, 63, 32, 63])
    np.testing.assert_array_almost_equal(time_series, exp_time_series, decimal=14)
    np.testing.assert_array_equal(transition_series, exp_transition_series)


def test_approximation(pred_tr_set_1f):
    rng = np.random.default_rng(42)
    time_series, transition_series = si.approximation(
        prediction=pred_tr_set_1f, size=20, seed=rng
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
        [0, 1, 0, 6, 0, 6, 0, 1, 0, 6, 0, 6, 0, 6, 0, 1, 0, 6], dtype=np.int64
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
    rng = np.random.default_rng(42)
    event_time_points, event_time_series = si.simulate_experiment(
        transition_matrix=tr_set_1f.transition_matrix,
        row_sums=tr_set_1f.row_sums,
        emitting_transition_ids=emitting_transition_ids,
        start_index=0,
        size=10,
        frames=10,
        frame_time="1ms",
        store_time_points=store_time_points,
        seed=rng,
    )
    if expected is None:
        assert event_time_points is None
        exp_event_time_series = pd.Series(
            np.array(
                [0, 768, 511, 666, 249, 206, 323, 207, 468, 356, 456], dtype=np.int64
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
                [0, 705, 461, 596, 224, 189, 298, 190, 419, 325, 412], dtype=np.int64
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

    tr_set_1f_new = tr_set_1f.adjust_rates(change_dict={6: 1e6})
    simulation = si.Simulation(transition_set=tr_set_1f_new)
    assert simulation


# also contains the test for simulation.delete_memmaps()
@pytest.mark.parametrize("use_memmap", [[None], ["tmp_path"]])
@pytest.mark.parametrize(
    "end_time, kap_sq_var, exp_time_series, exp_transition_series, exp_state_series",
    [
        [
            None,
            False,
            np.array(
                [
                    0.00000000000e0,
                    4.40600729235e-8,
                    4.42125274724e-8,
                    4.50453456369e-7,
                    4.50726394735e-7,
                    8.04052296605e-7,
                    8.05044394342e-7,
                    8.80747400872e-7,
                    8.81560651839e-7,
                    9.82930309396e-7,
                    9.83119497061e-7,
                ]
            ),
            np.array([0, 6, 0, 6, 0, 6, 0, 1, 0, 6]),
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]),
        ],
        [
            1e-6,
            False,
            np.array(
                [
                    0.00000000000e0,
                    4.40600740169e-8,
                    4.42125285628e-8,
                    4.50453465361e-7,
                    4.50726403722e-7,
                    8.04052314341e-7,
                    8.05044412115e-7,
                    8.80747419909e-7,
                    8.81560670868e-7,
                    9.82930330960e-7,
                    9.83119518629e-7,
                    1.00000000000e-6,
                ]
            ),
            np.array([0, 6, 0, 6, 0, 6, 0, 1, 0, 6]),
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]),
        ],
        [
            None,
            True,
            np.array(
                [
                    0.00000000000e0,
                    1.11581833551e-7,
                    1.12361178140e-7,
                    1.66769968502e-7,
                    1.66845595444e-7,
                    3.13505893572e-7,
                    3.14175598141e-7,
                    3.36006188810e-7,
                    3.40509502976e-7,
                    4.17729425876e-7,
                    4.18007488477e-7,
                ]
            ),
            np.array([0, 6, 0, 6, 0, 6, 0, 1, 0, 6]),
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]),
        ],
        [
            1e4,
            True,
            None,
            None,
            None,
        ],
    ],
)
def test_simulation_run(
    end_time,
    kap_sq_var,
    use_memmap,
    exp_time_series,
    exp_transition_series,
    exp_state_series,
    tr_set_1f,
    tmp_path,
    caplog,
):
    rng = np.random.default_rng(42)
    if use_memmap is not None:
        memmap_path = tmp_path
    else:
        memmap_path = None

    size = 10
    simulation = si.Simulation(transition_set=tr_set_1f)
    if kap_sq_var and end_time is not None:
        with pytest.raises(
            ValueError,
            match="end_time is not None but kap_sq_var is True. Not implemented.",
        ):
            simulation.run(
                start_at=(0,),
                size=size,
                end_time=end_time,
                kap_sq_var=kap_sq_var,
                seed=rng,
                use_memmap=memmap_path,
            )
        return

    with caplog.at_level(logging.WARNING):
        simulation.run(
            start_at=(0,),
            size=size,
            end_time=end_time,
            kap_sq_var=kap_sq_var,
            seed=rng,
            use_memmap=memmap_path,
        )
        assert "Floating point precision error warning" in caplog.text
    caplog.clear()

    np.testing.assert_array_almost_equal(
        simulation.time_series, exp_time_series, decimal=11
    )
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
        assert isinstance(simulation.time_series, np.memmap)
        assert isinstance(simulation.transition_series, np.memmap)
        assert isinstance(simulation.state_series, np.memmap)
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
        # convert to pathlib
        assert (Path(memmap_path) / "state_series").is_file()
        assert (Path(memmap_path) / "time_series").is_file()
        assert (Path(memmap_path) / "transition_series").is_file()
        simulation.delete_memmaps()
        assert not (Path(memmap_path) / "state_series").is_file()
        assert not (Path(memmap_path) / "time_series").is_file()
        assert not (Path(memmap_path) / "transition_series").is_file()
        assert not hasattr(simulation, "transition_series")
        assert not hasattr(simulation, "time_series")
        assert not hasattr(simulation, "state_series")

    else:
        assert simulation.memmap_path is None
        assert not isinstance(simulation.time_series, np.memmap)
        assert not isinstance(simulation.transition_series, np.memmap)
        assert not isinstance(simulation.state_series, np.memmap)

    with pytest.raises(
        ValueError,
        match="The number of starting states doesn't match the number of fluorophores.",
    ):
        simulation.run(
            start_at=(0, 1), size=size, end_time=None, seed=rng, use_memmap=None
        )


@pytest.mark.parametrize(
    "dirname, expected",
    [
        ["tr_set_1f", "ValueError1"],
        ["tr_set_1f", ""],
        ["tr_set_2f_diff", "ValueError2"],
    ],
)
def test_simulation_approximate(dirname, request, expected, caplog):
    rng = np.random.default_rng(42)
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
            simulation.approximate(prediction=pred, size=size, seed=rng)
    else:
        simulation = si.Simulation(transition_set=tr_set)
        if expected == "ValueError2":
            with pytest.raises(
                ValueError,
                match="approximation only available to single fluorophore systems.",
            ):
                simulation.approximate(prediction=pred, size=size, seed=rng)
        else:
            with caplog.at_level(logging.WARNING):
                simulation.approximate(prediction=pred, size=size, seed=rng)
                assert "Floating point precision error warning" in caplog.text
            caplog.clear()

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
            exp_transition_series = np.array([0, 6, 0, 6, 0, 6, 0, 1], dtype=np.int64)
            exp_state_series = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64)
            np.testing.assert_array_almost_equal(
                simulation.time_series, exp_time_series
            )
            np.testing.assert_array_equal(
                simulation.transition_series, exp_transition_series
            )
            np.testing.assert_array_equal(simulation.state_series, exp_state_series)
