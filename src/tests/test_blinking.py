import pytest
import numpy as np
import pandas as pd
import src.blinking as bl


@pytest.mark.parametrize(
    "event_time_series, threshold, memory, exp_on_periods, exp_off_periods, "
    "exp_on_periods_frames, exp_off_periods_frames",
    [
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 0, 0])),
            0,
            0,
            np.array([4, 1]),
            np.array([1]),
            np.array([1, 6]),
            np.array([5]),
        ],
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 0, 0])),
            2,
            0,
            np.array([3, 1]),
            np.array([1]),
            np.array([2, 6]),
            np.array([5]),
        ],
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 0, 0])),
            0,
            1,
            np.array([6]),
            np.array([]),
            np.array([1]),
            np.array([]),
        ],
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 5])),
            0,
            0,
            np.array([4]),
            np.array([1]),
            np.array([1]),
            np.array([5]),
        ],
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 0, 0])),
            0,
            2,
            np.array([6]),
            np.array([]),
            np.array([1]),
            np.array([]),
        ],
        [
            pd.Series(np.array([0, 1, 3, 4, 7, 1, 4, 5])),
            0,
            0,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        ],
        [
            pd.Series(np.array([1, 2, 3, 0, 4, 5, 0, 3])),
            0,
            0,
            np.array([3, 2]),
            np.array([1, 1]),
            np.array([0, 4]),
            np.array([3, 6]),
        ],
        [
            pd.Series(np.array([1, 2, 3, 0, 4, 5, 0, 3])),
            1,
            0,
            np.array([2, 2]),
            np.array([1, 1]),
            np.array([1, 4]),
            np.array([3, 6]),
        ],
        [
            pd.Series(np.array([1, 2, 3, 0, 0, 5, 0, 3, 0])),
            1,
            1,
            np.array([2, 3]),
            np.array([2]),
            np.array([1, 5]),
            np.array([3]),
        ],
    ],
)
def test_get_blinking_statistics(
    event_time_series,
    threshold,
    memory,
    exp_on_periods,
    exp_off_periods,
    exp_on_periods_frames,
    exp_off_periods_frames,
):
    (
        on_periods,
        off_periods,
        on_periods_frames,
        off_periods_frames,
    ) = bl.get_blinking_statistics(
        event_time_series=event_time_series, threshold=threshold, memory=memory
    )
    np.testing.assert_array_equal(on_periods, exp_on_periods)
    np.testing.assert_array_equal(off_periods, exp_off_periods)
    np.testing.assert_array_equal(on_periods_frames, exp_on_periods_frames)
    np.testing.assert_array_equal(off_periods_frames, exp_off_periods_frames)


def test_get_off_statistics(sim_dstorm):
    on_off_times, on_off_values = bl.get_off_statistics(
        simulation=sim_dstorm, index=0
    )
    exp_on_off_values = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    exp_on_off_times = np.array(
        [
            0.00000000e00,
            8.98296370e-03,
            8.98296370e-03,
            5.57004121e01,
            5.57004121e01,
            5.58517447e01,
            5.58517447e01,
        ]
    )
    np.testing.assert_array_almost_equal(on_off_times, exp_on_off_times)
    np.testing.assert_array_equal(on_off_values, exp_on_off_values)


def test_get_analytical_off_statistics():
    off_frames = np.array([2, 5, 10])
    off_periods = np.array([1, 2, 10])
    frame_time = "5ms"
    on_off_times, on_off_values = bl.get_analytical_off_statistics(
        off_frames=off_frames, off_periods=off_periods, frame_time=frame_time
    )
    exp_on_off_times = np.array(
        [
            0.0,
            0.01,
            0.01,
            0.015,
            0.015,
            0.025,
            0.025,
            0.035,
            0.035,
            0.05,
            0.05,
            0.1,
            0.1,
        ]
    )
    exp_on_off_values = np.array(
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(on_off_times, exp_on_off_times)
    np.testing.assert_array_equal(on_off_values, exp_on_off_values)


def test_blinking(em_large):
    blink = bl.Blinking(emissions=em_large)
    assert blink.emissions == em_large
    exp_on_periods = np.array([1, 2, 3])
    np.testing.assert_array_equal(blink.on_periods, exp_on_periods)
    exp_off_periods = np.array([21, 6])
    np.testing.assert_array_equal(blink.off_periods, exp_off_periods)
    exp_on_periods_frames = np.array([0, 22, 30])
    np.testing.assert_array_equal(blink.on_periods_frames, exp_on_periods_frames)
    exp_off_periods_frames = np.array([1, 24])
    np.testing.assert_array_equal(blink.off_periods_frames, exp_off_periods_frames)
