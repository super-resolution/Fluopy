import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fluopy import emissions as em
from fluopy import fluo_data as fd


def assert_all_channel_equal(emis, expected):
    pd.testing.assert_series_equal(
        emis.select_event_time_series("all"), expected.rename("all")
    )


def test_emissions():
    frame_time = "5ms"
    channels = {"red": em.DetectionChannel(bandpass=(650, 700))}
    rng = np.random.default_rng(1)
    emis = em.Emissions(frame_time=frame_time, channels=channels, seed=rng)
    assert emis.parameters["frame_time"] == frame_time
    assert emis.parameters["channels"] == channels
    assert emis.parameters["seed"] == rng
    assert emis.channels == channels
    assert emis.event_time_points is None
    assert emis.event_time_series is None


@pytest.mark.parametrize("detection_efficiency", [-0.1, 1.1, np.nan])
def test_detection_channel_rejects_invalid_detection_efficiency(
    detection_efficiency,
):
    with pytest.raises(ValueError, match="detection_efficiency"):
        em.DetectionChannel(detection_efficiency=detection_efficiency)


@pytest.mark.parametrize("bandpass", [(700, 650), (650, np.inf)])
def test_detection_channel_rejects_invalid_bandpass(bandpass):
    with pytest.raises(ValueError, match="bandpass"):
        em.DetectionChannel(bandpass=bandpass)


def test_emissions_requires_channels():
    with pytest.raises(ValueError, match="at least one"):
        em.Emissions(channels={})

    with pytest.raises(ValueError, match="non-empty strings"):
        em.Emissions(channels={"": em.DetectionChannel()})

    with pytest.raises(TypeError, match="DetectionChannel"):
        em.Emissions(channels={"detector": object()})


def test_emissions_requires_available_event_data():
    emis = em.Emissions()

    with pytest.raises(ValueError, match="event time series is unavailable"):
        emis._require_event_time_series()
    with pytest.raises(ValueError, match="event time points are unavailable"):
        emis._require_event_time_points()


def test_emissions_selects_channel_data():
    emis = em.Emissions(
        channels={
            "green": em.DetectionChannel(),
            "red": em.DetectionChannel(),
        }
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 1], "red": [0, 2]}, dtype=np.int64
    )
    emis.event_time_points = {
        "green": np.array([0.1]),
        "red": np.array([0.2, 0.3]),
    }

    pd.testing.assert_series_equal(
        emis.select_event_time_series("red"),
        pd.Series([0, 2], name="red", dtype=np.int64),
    )
    np.testing.assert_array_equal(
        emis.select_event_time_points("green"), np.array([0.1])
    )


def test_emissions_channel_selection_requires_unambiguous_channel():
    emis = em.Emissions(
        channels={
            "green": em.DetectionChannel(),
            "red": em.DetectionChannel(),
        }
    )

    with pytest.raises(ValueError, match="channel must be specified"):
        emis.resolve_channel()
    with pytest.raises(ValueError, match="unknown detection channel"):
        emis.resolve_channel("blue")


def test_emissions_extract_requires_completed_simulation():
    simulation = SimpleNamespace(transition_series=None, time_series=None)

    with pytest.raises(ValueError, match="simulation has not been run"):
        em.Emissions().extract(simulation)


@pytest.mark.parametrize(
    "dirname, frame_time, channel, expected",
    [
        ["sim_tr_set_1f_bl", "10us", em.DetectionChannel((650, 680)), 77],
        ["sim_tr_set_et_2f_diff", "5ms", em.DetectionChannel(), 2],
    ],
)
def test_emissions_extract(dirname, request, frame_time, channel, expected, caplog):
    rng = np.random.default_rng(1)
    emis = em.Emissions(frame_time=frame_time, channels={"detector": channel}, seed=rng)
    with caplog.at_level(logging.WARNING):
        simulation = request.getfixturevalue(dirname)
        assert "Floating point precision error warning" in caplog.text
    caplog.clear()

    emis.extract(simulation=simulation)
    assert emis.event_time_points is not None
    assert emis.event_time_points["detector"].size == expected
    if frame_time == "10us":
        exp_event_time_series = pd.Series(
            # fmt: off
                np.array(
                    [
                        0, 12, 5, 4, 8, 7, 3, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 5, 4,
                        7,
                ],
                dtype=np.int64,
            ),
            # fmt: on
            index=np.linspace(0, 0.00045, 46),
        )
    else:
        exp_event_time_series = pd.Series(
            np.array([0, 2], dtype=np.int64), index=[0.0, 0.005]
        )
    assert emis.event_time_series is not None
    pd.testing.assert_series_equal(
        emis.event_time_series["detector"],
        exp_event_time_series.rename("detector"),
    )


def test_emissions_extract_orders_filtered_time_points(sim_tr_set_2f_diff):
    emis = em.Emissions(channels={"detector": em.DetectionChannel((650, 700))}, seed=1)

    emis.extract(simulation=sim_tr_set_2f_diff)

    assert emis.event_time_points is not None
    assert np.all(np.diff(emis.event_time_points["detector"]) >= 0)


def test_assign_detection_channels():
    detection_probabilities = np.array([[0.25, 0.5], [0.0, 1.0]])
    channel_indices = em.assign_detection_channels(
        transition_ids=[0, 0, 0, 1],
        detection_probabilities=detection_probabilities,
        random_numbers=[0.1, 0.5, 0.9, 0.1],
    )

    np.testing.assert_array_equal(channel_indices, [0, 1, 2, 1])


@pytest.mark.parametrize(
    "transition_ids, detection_probabilities, random_numbers, match",
    [
        ([0], [0.5], [0.1], "two-dimensional"),
        ([[0]], [[0.5]], [[0.1]], "one-dimensional"),
        ([0], [[0.5]], [0.1, 0.2], "one random number"),
        ([0], [[np.nan]], [0.1], "finite and between"),
        ([0], [[0.6, 0.6]], [0.1], "sum to at most 1"),
        ([0], [[0.5]], [np.nan], "random numbers"),
        ([1], [[0.5]], [0.1], "outside"),
    ],
)
def test_assign_detection_channels_validates_inputs(
    transition_ids, detection_probabilities, random_numbers, match
):
    with pytest.raises(ValueError, match=match):
        em.assign_detection_channels(
            transition_ids=transition_ids,
            detection_probabilities=detection_probabilities,
            random_numbers=random_numbers,
        )


def test_emissions_extract_assigns_each_photon_at_most_once(sim_tr_set_2f_diff):
    emis = em.Emissions(
        channels={
            "green": em.DetectionChannel((500, 650)),
            "red": em.DetectionChannel((650, 800)),
        },
        seed=1,
    )
    emis.extract(sim_tr_set_2f_diff)

    assert emis.event_time_points is not None
    green = emis.event_time_points["green"]
    red = emis.event_time_points["red"]
    assert np.intersect1d(green, red).size == 0
    assert emis.event_time_series is not None
    assert emis.event_time_series["green"].sum() == green.size
    assert emis.event_time_series["red"].sum() == red.size


def test_generated_event_time_series_starts_with_boundary(em_tr_set_1f_bl):
    assert em_tr_set_1f_bl.event_time_series.index[0] == 0
    assert (em_tr_set_1f_bl.event_time_series.iloc[0] == 0).all()


def test_construct_event_time_series_requires_completed_simulation():
    emis = em.Emissions()
    emis.event_time_points = {"all": np.array([0.001])}

    with pytest.raises(ValueError, match="requires a completed simulation"):
        emis.construct_event_time_series(simulation=SimpleNamespace(time_series=None))


def test_construct_event_time_series_drops_bin_after_simulation_end():
    emis = em.Emissions()
    emis.event_time_points = {"all": np.array([0.001])}
    simulation = SimpleNamespace(time_series=np.array([0.0, 0.011]))

    emis.construct_event_time_series(simulation=simulation, resample="5ms")

    assert emis.event_time_series.index[-1] == pytest.approx(0.01)
    np.testing.assert_array_equal(emis.event_time_series["all"], [0, 1, 0])


def test_emissions_simulate(tr_set_1f_bl):
    rng = np.random.default_rng(1)
    emis = em.Emissions(
        frame_time="100us",
        channels={"red": em.DetectionChannel((650, 700))},
        seed=rng,
    )
    emis.simulate(
        transition_set=tr_set_1f_bl,
        start_at=None,
        size=1e3,
        frames=10,
        store_time_points=True,
    )
    assert emis.event_time_points is not None
    assert emis.event_time_points["red"].size == 204
    exp_event_time_series = pd.Series(
        np.array([0, 80, 0, 0, 0, 0, 0, 16, 51, 7, 50], dtype=np.int64),
        index=np.linspace(0, 0.001, 11),
    )
    assert emis.event_time_series is not None
    pd.testing.assert_series_equal(
        emis.event_time_series["red"], exp_event_time_series.rename("red")
    )


def test_emissions_simulate_routes_photons_to_distinct_channels(tr_set_1f_bl):
    emis = em.Emissions(
        frame_time="100us",
        channels={
            "green": em.DetectionChannel((500, 650)),
            "red": em.DetectionChannel((650, 800)),
        },
        seed=1,
    )

    emis.simulate(
        transition_set=tr_set_1f_bl,
        size=1000,
        frames=10,
        store_time_points=True,
    )

    assert emis.event_time_points is not None
    assert emis.event_time_series is not None
    assert list(emis.event_time_series.columns) == ["green", "red"]
    green = emis.event_time_points["green"]
    red = emis.event_time_points["red"]
    assert np.intersect1d(green, red).size == 0
    assert green.size == emis.event_time_series["green"].sum()
    assert red.size == emis.event_time_series["red"].sum()


def test_emissions_simulate_requires_one_start_state_per_fluorophore(tr_set_1f_bl):
    with pytest.raises(ValueError, match="number of starting states"):
        em.Emissions().simulate(
            transition_set=tr_set_1f_bl,
            start_at=(0, 1),
            frames=1,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "dirname, excitation_rates, expected",
    [
        ["tr_set_1f_bl", {"testfluo_1": 1e5}, 0],
        ["tr_set_bl_et_2f_diff", {"testfluo_1": 1e10, "testfluo_2": 1e11}, 1],
        ["tr_set_bl_et_2f_same", {"testfluo_1": 1e10}, 1],
    ],
)
def test_emissions_tcspc(dirname, request, excitation_rates, expected, caplog):
    rng = np.random.default_rng(1)
    tr_set = request.getfixturevalue(dirname)
    if expected == 1:
        ets = [j for h, j in tr_set.transition_df.index if "dist" in h]
        tr_set = tr_set.adjust_rates({identity: 1e10 for identity in ets})
        tr_set.finalize()
    emis = em.Emissions(
        frame_time="100us",
        channels={"red": em.DetectionChannel((650, 700))},
        seed=rng,
    )
    with caplog.at_level(logging.WARNING):
        lifetimes_DA, _, _, _ = emis.tcspc(
            transition_set=tr_set,
            number_pulses=1e4,
            pulse_duration=1e-9,
            time_between_pulses=1e-6,
            excitation_rates=excitation_rates,
            size=1e3,
            store_time_points=True,
        )
        assert "the last frame" in caplog.text
    caplog.clear()

    if expected == 1:
        assert lifetimes_DA["red"].size > 0
    else:
        assert lifetimes_DA["red"].size == 0


def test_emissions_tcspc_parameters(tr_set_bl_et_2f_diff):
    rng = np.random.default_rng(1)
    tr_set = tr_set_bl_et_2f_diff.adjust_rates(
        {8: 1e10, 15: 1e10}, keep_zero_rates=True
    )
    tr_set.finalize()
    emis = em.Emissions(
        frame_time="1ms",
        channels={
            "donor": em.DetectionChannel((650, 700), fluorophore_ids=frozenset({0})),
            "acceptor": em.DetectionChannel((650, 700), fluorophore_ids=frozenset({1})),
        },
        seed=rng,
    )
    with patch("fluopy.emissions.simulate_TCSPC") as mock_tcspc:
        empty = np.array([], dtype=np.float64)
        mock_tcspc.return_value = (
            pd.DataFrame({"donor": [0], "acceptor": [0]}, dtype=np.int64),
            {"donor": empty, "acceptor": empty},
            {"donor": empty, "acceptor": empty},
            {"donor": empty, "acceptor": empty},
            {"donor": empty, "acceptor": empty},
        )
        emis.tcspc(
            transition_set=tr_set,
            number_pulses=1e4,
            pulse_duration=1e-9,
            time_between_pulses=1e-6,
            excitation_rates={"testfluo_1": 1e5, "testfluo_2": 1e6},
            size=1e3,
            store_time_points=True,
        )
        args, kwargs = mock_tcspc.call_args
        np.testing.assert_array_equal(
            kwargs["et_transition_ids"], np.array([4, 38, 40])
        )
        expected = np.zeros_like(kwargs["detection_probabilities"])
        expected[[4, 5, 6, 7], 0] = 0.6820037131347214
        expected[[38, 39, 40, 41, 42], 1] = 0.5847564420110373
        np.testing.assert_allclose(kwargs["detection_probabilities"], expected)
        assert kwargs["channel_names"] == ("donor", "acceptor")
        assert list(emis.event_time_series.columns) == ["donor", "acceptor"]


def test_emissions_tcspc_details_infers_excitation_rates(tr_set_1f_bl, caplog):
    emis = em.Emissions(seed=1)
    event_time_series = pd.DataFrame({"all": [0, 1]}, index=[0.0, 0.005])
    event_time_points = {"all": np.array([0.001])}
    lifetimes_da = {"all": np.array([1.0])}
    lifetimes_d = {"all": np.array([2.0])}
    lifetimes_all = {"all": np.array([1.0, 2.0])}
    simulation_object = object()
    excitation_rate = tr_set_1f_bl.transition_df.loc[
        tr_set_1f_bl.transition_df["abbreviation"] == "EXC", "rate"
    ].iloc[0]

    with (
        patch(
            "fluopy.emissions.simulate_TCSPC_detailed",
            return_value=(
                event_time_series,
                event_time_points,
                lifetimes_da,
                lifetimes_d,
                lifetimes_all,
                simulation_object,
            ),
        ) as mock_tcspc,
        caplog.at_level(logging.WARNING),
    ):
        result = emis.tcspc(
            transition_set=tr_set_1f_bl,
            number_pulses=2,
            pulse_duration=1e-9,
            time_between_pulses=1e-6,
            excitation_rates=None,
            details=True,
        )

    assert "assumed to be the mean irradiance" in caplog.text
    assert result[3] is simulation_object
    np.testing.assert_array_equal(result[0]["all"], lifetimes_da["all"])
    actual_rate = mock_tcspc.call_args.kwargs["excitation_rates"]["testfluo_1"]
    assert actual_rate == pytest.approx(excitation_rate * 1000)


def test_emissions_apply_threshold():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [0, 1, 3, 2]}, dtype=np.int64)

    emis.apply_threshold(3)

    pd.testing.assert_frame_equal(
        emis.event_time_series,
        pd.DataFrame({"all": [0, 0, 3, 0]}, dtype=np.int64),
    )


def test_emissions_add_emccd_gain(em_large):
    rng = np.random.default_rng(1)
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series_prev)
    em_large.add_emccd_gain(emccd_gain=10, seed=rng)
    # fmt: off
    exp_values = np.array(
        [
            492113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 497653, 75361, 0, 0, 0, 0, 0, 0, 664801,
            760421, 358760, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        dtype=np.int64)
    # fmt: on
    exp_event_time_series = pd.Series(exp_values, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series)


def test_emissions_add_gaussian_noise(em_large):
    rng = np.random.default_rng(1)
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series_prev)
    em_large.add_gaussian_noise(mean=10, std=5, seed=rng)
    # fmt: off
    exp_values = np.array(
        [
            49135, 11, 14, 11, 3, 14, 12, 7, 12, 11, 11, 10, 12, 6, 9, 7, 12, 10, 8, 6,
            8, 10, 49700, 7474, 15, 0, 0, 9, 7, 11, 66630, 75962, 35875, 8, 20, 13, 13,
            7, 1, 10, 10, 3, 6, 9, 5, 9, 10, 10, 7, 12, 14, 11, 5, 13, 7, 14, 4, 14, 9,
            3, 8, 10, 11, 5, 4, 10, 7, 11, 13, 1, 11, 16, 8, 5, 13, 11, 14, 8, 2, 9, 7,
            13, 10, 1, 4, 14, 13, 6, 9, 12, 12, 14, 11, 9, 8, 15, 0, 9, 10, 2
        ],
        dtype=np.int64,
    )
    # fmt: on
    exp_event_time_series = pd.Series(exp_values, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series)


def test_emissions_add_poisson_noise(em_large):
    rng = np.random.default_rng(1)
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series_prev)
    em_large.add_poisson_noise(rate=10, seed=rng)
    # fmt: off
    exp_values = np.array(
        [
            49135, 8, 13, 10, 13, 8, 8, 6, 7, 12, 10, 10, 11, 10, 11, 11, 14, 10, 7, 7,
            14, 8, 49703, 7472, 7, 6, 15, 11, 12, 13, 66633, 75955, 35880, 12, 9, 8, 5,
            9, 11, 14, 13, 10, 7, 7, 12, 9, 12, 11, 12, 9, 5, 8, 8, 12, 4, 7, 10, 8, 10,
            18, 8, 6, 7, 11, 10, 13, 11, 9, 20, 12, 14, 13, 9, 9, 6, 11, 13, 11, 13, 10,
            10, 12, 6, 6, 8, 3, 7, 17, 16, 5, 5, 7, 8, 12, 11, 9, 6, 7, 10, 8
        ],
        np.int64)
    # fmt: on
    exp_event_time_series = pd.Series(exp_values, index=exp_index)
    assert_all_channel_equal(em_large, exp_event_time_series)


def test_emissions_add_poisson_noise_is_frame_only_for_multiple_channels():
    emis = em.Emissions(
        channels={
            "green": em.DetectionChannel(),
            "red": em.DetectionChannel(),
        }
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 1, 3], "red": [0, 2, 4]}, dtype=np.int64
    )
    emis.event_time_points = {
        "green": np.array([0.1]),
        "red": np.array([0.2, 0.3]),
    }
    original_time_points = {
        name: time_points.copy() for name, time_points in emis.event_time_points.items()
    }

    emis.add_poisson_noise(rate=0.6, seed=1)

    expected = pd.DataFrame({"green": [0, 1, 4], "red": [0, 3, 4]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)
    assert emis.event_time_points is not None
    for name, time_points in original_time_points.items():
        np.testing.assert_array_equal(emis.event_time_points[name], time_points)


@pytest.mark.parametrize("counts", [[], [0, 0]])
def test_plot_cumulative_events_without_events(counts):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": counts}, dtype=np.int64)

    with pytest.raises(ValueError, match="require at least one event"):
        emis.plot_cumulative_events()


def test_plot_histogram_without_included_counts():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [0, 0]}, dtype=np.int64)

    with pytest.raises(ValueError, match="requires at least one included event count"):
        emis.plot_histogram(include_0=False)


@pytest.mark.parametrize(
    "plot_method",
    ["plot_cumulative_events", "plot_histogram", "plot_time_series"],
)
def test_emissions_plot_methods(plot_method):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [0, 1, 2]}, index=[0.0, 0.005, 0.01])

    ax = getattr(emis, plot_method)()

    assert ax.has_data()


def test_emissions_plot_selects_channel():
    emis = em.Emissions(
        channels={
            "green": em.DetectionChannel(),
            "red": em.DetectionChannel(),
        }
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 1, 0], "red": [0, 2, 3]}, index=[0.0, 0.005, 0.01]
    )

    ax = emis.plot_time_series(channel="red")

    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 2, 3])
    with pytest.raises(ValueError, match="channel must be specified"):
        emis.plot_time_series()


def test_emissions_histogram_probability_and_mean():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [0, 1, 3]}, dtype=np.int64)

    ax = emis.plot_histogram(
        density=False,
        display_mean=True,
        include_0=True,
    )

    assert ax.get_ylabel() == "Probability"
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(1)
    assert ax.texts[0].get_text() == r"$\mu = 1.33$"


def test_save_and_load(request, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        em_tr_set_1f_bl = request.getfixturevalue("em_tr_set_1f_bl")
        assert "Floating point precision error warning" in caplog.text
    caplog.clear()

    em_tr_set_1f_bl.save(path=tmp_path, name_extension="_test_extension")
    assert (Path(tmp_path) / "event_time_series_test_extension.csv").is_file()
    assert (Path(tmp_path) / "event_time_points_test_extension.npy").is_file()
    emis = em.Emissions.load(path=tmp_path, name_extension="_test_extension")
    assert isinstance(emis.event_time_points, dict)
    assert isinstance(emis.event_time_series, pd.DataFrame)
    assert em_tr_set_1f_bl.event_time_points is not None
    assert em_tr_set_1f_bl.event_time_series is not None
    np.testing.assert_array_equal(
        emis.event_time_points["all"], em_tr_set_1f_bl.event_time_points["all"]
    )
    pd.testing.assert_frame_equal(
        emis.event_time_series, em_tr_set_1f_bl.event_time_series
    )
    assert emis.parameters == {
        "frame_time": "5ms",
        "seed": None,
        "channels": {"all": em.DetectionChannel()},
    }
    (Path(tmp_path) / "event_time_series_test_extension.csv").unlink(missing_ok=True)
    (Path(tmp_path) / "event_time_points_test_extension.npy").unlink(missing_ok=True)
    assert not (Path(tmp_path) / "event_time_series_test_extension.csv").is_file()
    assert not (Path(tmp_path) / "event_time_points_test_extension.npy").is_file()


def test_save_and_load_without_event_time_points(tmp_path):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame(
        {"all": [0, 2]}, index=[0.0, 0.005], dtype=np.int64
    )

    emis.save(path=tmp_path)
    loaded = em.Emissions.load(path=tmp_path)

    pd.testing.assert_frame_equal(loaded.event_time_series, emis.event_time_series)
    assert loaded.event_time_points is None
    assert not (tmp_path / "event_time_points.npy").is_file()


@pytest.mark.parametrize(
    "bandpass, expected",
    [
        [(650, 700), 0.6820037131347214],
        [(450, 400), "ValueError"],
        [(200, 1000), 1.0],
    ],
)
def test_get_p_filter(bandpass, expected):
    emission_spectrum = fd.testfluo_1.emission_spectrum
    assert emission_spectrum is not None
    if expected == "ValueError":
        with pytest.raises(
            ValueError,
            match=("The lower bandpass limit has to be smaller than the upper limit."),
        ):
            p_passed = em.get_p_filter(
                emission_spectrum=emission_spectrum,
                bandpass=bandpass,
            )
    else:
        p_passed = em.get_p_filter(
            emission_spectrum=emission_spectrum,
            bandpass=bandpass,
        )
        assert p_passed == pytest.approx(expected)


@pytest.mark.parametrize("bandpass", [(np.nan, 700), (650, np.inf)])
def test_get_p_filter_non_finite_bandpass(bandpass):
    emission_spectrum = fd.Spectrum(
        wavelengths=[500, 600],
        values=[0, 1],
    )
    with pytest.raises(
        ValueError,
        match="bandpass limits must be finite.",
    ):
        em.get_p_filter(
            emission_spectrum=emission_spectrum,
            bandpass=bandpass,
        )


def test_get_p_filter_with_in_memory_spectrum():
    emission_spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[0, 1, 0],
    )

    p_passed = em.get_p_filter(
        emission_spectrum=emission_spectrum,
        bandpass=(505, 515),
    )

    assert p_passed == pytest.approx(0.75)


def test_get_p_filter_zero_emission_spectrum():
    emission_spectrum = fd.Spectrum(
        wavelengths=[500, 600],
        values=[0, 0],
    )

    with pytest.raises(
        ValueError,
        match="emission spectrum has zero total intensity.",
    ):
        em.get_p_filter(
            emission_spectrum=emission_spectrum,
            bandpass=(500, 600),
        )


def test_get_p_filter_without_spectral_overlap():
    emission_spectrum = fd.Spectrum(
        wavelengths=[500, 600],
        values=[0, 1],
    )

    p_passed = em.get_p_filter(
        emission_spectrum=emission_spectrum,
        bandpass=(700, 800),
    )

    assert p_passed == 0


def test_get_detection_probabilities_uses_each_fluorophore_spectrum(
    tr_set_bl_et_2f_diff,
):
    channels = {
        "first": em.DetectionChannel((650, 700), fluorophore_ids=frozenset({0})),
        "second": em.DetectionChannel(
            (650, 700),
            fluorophore_ids=frozenset({1}),
            detection_efficiency=0.5,
        ),
    }

    probabilities = em.get_detection_probabilities(
        transition_set=tr_set_bl_et_2f_diff, channels=channels
    )

    expected = np.zeros_like(probabilities)
    expected[[4, 5, 6, 7], 0] = 0.6820037131347214
    expected[[38, 39, 40, 41, 42], 1] = 0.5847564420110373 * 0.5
    np.testing.assert_allclose(probabilities, expected)


def test_get_detection_probabilities_rejects_overlapping_channels(
    tr_set_bl_et_2f_diff,
):
    channels = {
        "first": em.DetectionChannel((600, 700)),
        "second": em.DetectionChannel((650, 750)),
    }

    with pytest.raises(ValueError, match="must not overlap"):
        em.get_detection_probabilities(tr_set_bl_et_2f_diff, channels)


def test_get_detection_probabilities_requires_spectral_data_for_bandpass():
    transition_set = SimpleNamespace(
        combined_state_transitions_df=pd.DataFrame(
            {"photon": [True], "fluorophore_ids": [[0]]}
        ),
        fluorophore_system=SimpleNamespace(
            fluorophores=[SimpleNamespace(name="unknown", constants=None)]
        ),
    )

    with pytest.raises(ValueError, match="emission data not available"):
        em.get_detection_probabilities(
            transition_set=transition_set,
            channels={"detector": em.DetectionChannel((500, 600))},
        )
