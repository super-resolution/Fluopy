from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluopy import blinking as bl
from fluopy.transitions import SingleState


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


@pytest.mark.parametrize(
    "states, times, expected_times, expected_values",
    [
        pytest.param(
            [
                SingleState.S0,
                SingleState.OFF,
                SingleState.OFF,
                SingleState.S0,
                SingleState.S0,
            ],
            [0, 1, 2, 3, 4],
            [0, 1, 1, 3, 3, 4],
            [1, 1, 0, 0, 1, 1],
            id="completed-final-off",
        ),
        pytest.param(
            [SingleState.S0, SingleState.OFF, SingleState.OFF],
            [0, 1, 2, 4],
            [0, 1, 1, 4],
            [1, 1, 0, 0],
            id="unfinished-off-with-end-time",
        ),
        pytest.param(
            [SingleState.OFF, SingleState.OFF, SingleState.S0],
            [0, 1, 2, 4],
            [0, 2, 2, 4],
            [0, 0, 1, 1],
            id="starts-off",
        ),
        pytest.param(
            [SingleState.S0, SingleState.OFF, SingleState.OFF2, SingleState.S0],
            [0, 1, 2, 3, 4],
            [0, 1, 1, 3, 3, 4],
            [1, 1, 0, 0, 1, 1],
            id="off-subtype-change",
        ),
        pytest.param(
            [SingleState.OFF, SingleState.OFF2, SingleState.OFF],
            [0, 1, 2, 4],
            [0, 4],
            [0, 0],
            id="entirely-off",
        ),
    ],
)
def test_get_off_statistics_controlled(states, times, expected_times, expected_values):
    simulation = SimpleNamespace(
        state_series=np.array([[state.value for state in states]], dtype=np.int8),
        time_series=np.array(times, dtype=np.float64),
    )

    on_off_times, on_off_values = bl.get_off_statistics(simulation=simulation, index=0)

    np.testing.assert_array_equal(on_off_times, expected_times)
    np.testing.assert_array_equal(on_off_values, expected_values)
    assert on_off_values.dtype == np.int8


def test_get_blinking_statistics_without_events():
    statistics = bl.get_blinking_statistics(pd.Series([0, 0, 0]))

    assert all(values.size == 0 for values in statistics)
    assert all(values.dtype == np.int64 for values in statistics)


@pytest.mark.parametrize(
    "simulation, index, message",
    [
        (
            SimpleNamespace(state_series=None, time_series=None),
            0,
            "completed simulation",
        ),
        (
            SimpleNamespace(
                state_series=np.array([[SingleState.S0.value]], dtype=np.int8),
                time_series=np.array([0.0, 1.0]),
            ),
            1,
            "2 fluorophores",
        ),
        (
            SimpleNamespace(
                state_series=np.array([[SingleState.S0.value]], dtype=np.int8),
                time_series=np.array([0.0, 1.0]),
            ),
            0,
            "no photophysical OFF states",
        ),
    ],
)
def test_get_off_statistics_rejects_invalid_inputs(simulation, index, message):
    with pytest.raises(ValueError, match=message):
        bl.get_off_statistics(simulation=simulation, index=index)


def test_get_analytical_off_statistics():
    off_frames = np.array([2, 5, 10])
    off_periods = np.array([1, 2, 10])
    on_frames = np.array([0, 4, 7])
    frame_time = "5ms"
    on_off_times, on_off_values = bl.get_analytical_off_statistics(
        off_frames=off_frames,
        off_periods=off_periods,
        on_frames=on_frames,
        frame_time=frame_time,
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


def test_blinking_requires_extracted_emissions():
    with pytest.raises(ValueError, match="require extracted emissions"):
        bl.Blinking(emissions=SimpleNamespace(event_time_series=None))


@pytest.mark.parametrize(
    "mode, artist_attribute",
    [
        ("on_histogram", "patches"),
        ("off_histogram", "patches"),
        ("on_boxplot", "lines"),
        ("off_boxplot", "lines"),
        ("on_frame_series", "lines"),
        ("off_frame_series", "lines"),
    ],
)
def test_blinking_plot_modes(em_large, mode, artist_attribute):
    blink = bl.Blinking(emissions=em_large)

    ax = blink.plot(mode=mode)

    assert len(getattr(ax, artist_attribute)) > 0


def test_blinking_rejects_unknown_plot_mode(em_large):
    blink = bl.Blinking(emissions=em_large)

    with pytest.raises(ValueError, match="mode unknown unknown"):
        blink.plot(mode="unknown")


@pytest.mark.parametrize("plotter", [bl.plot_histogram, bl.plot_boxplot])
def test_period_plot_requires_frame_duration_for_time_axis(plotter):
    with pytest.raises(ValueError, match="sec_per_frame is required"):
        plotter([1, 2], as_time="ms")


@pytest.mark.parametrize("plotter", [bl.plot_histogram, bl.plot_boxplot])
def test_period_plot_rejects_unknown_time_unit(plotter):
    with pytest.raises(ValueError, match="unit not implemented"):
        plotter([1, 2], as_time="minute", sec_per_frame=0.1)


def test_histogram_time_conversion_and_probability_weights():
    ax = bl.plot_histogram([1, 2], density=False, as_time="ms", sec_per_frame=0.01)

    assert ax.get_xlabel() == "OFF period (ms)"
    assert ax.get_ylabel() == "Probability"
    assert ax.texts[0].get_text() == r"$\mu = 15.00$"
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(1)
