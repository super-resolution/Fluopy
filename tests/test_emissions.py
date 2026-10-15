import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fluopy import emissions as em
from fluopy import fluo_data as fd
from fluopy import fluorophores as fl
from fluopy import simulation as si
from fluopy import transitions as tr


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


@pytest.mark.parametrize("routing_probability", [-0.1, 1.1, np.nan, np.inf])
def test_detection_channel_rejects_invalid_routing_probability(routing_probability):
    with pytest.raises(ValueError, match="routing_probability"):
        em.DetectionChannel(routing_probability=routing_probability)


def test_detection_channel_rejects_invalid_routing_spectrum():
    response = fd.Spectrum([500, 600], [0, 1.1])

    with pytest.raises(ValueError, match="routing_probability spectrum"):
        em.DetectionChannel(routing_probability=response)


def test_detection_channel_requires_scalar_or_spectrum_routing():
    with pytest.raises(TypeError, match="scalar or Spectrum"):
        em.DetectionChannel(routing_probability=[0.5, 0.5])


@pytest.mark.parametrize("name", ["bandpass", "detection_efficiency"])
@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_detection_channel_rejects_invalid_scalar_response(name, value):
    with pytest.raises(ValueError, match=name):
        em.DetectionChannel(**{name: value})


@pytest.mark.parametrize("name", ["bandpass", "detection_efficiency"])
def test_detection_channel_rejects_invalid_spectral_response(name):
    with pytest.raises(ValueError, match=f"{name} spectrum"):
        em.DetectionChannel(**{name: fd.Spectrum([500, 600], [0, 1.1])})

    for value in ("invalid", None):
        with pytest.raises(TypeError, match=name):
            em.DetectionChannel(**{name: value})


def test_emissions_requires_channels():
    with pytest.raises(ValueError, match="at least one"):
        em.Emissions(channels={})

    with pytest.raises(ValueError, match="non-empty strings"):
        em.Emissions(channels={"": em.DetectionChannel()})

    with pytest.raises(TypeError, match="DetectionChannel"):
        em.Emissions(channels={"detector": object()})


@pytest.mark.parametrize("frame_time", ["0ms", "-1ms", "invalid"])
def test_emissions_requires_positive_frame_time(frame_time):
    with pytest.raises(ValueError, match="frame_time must be a positive duration"):
        em.Emissions(frame_time=frame_time)


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
        ["sim_tr_set_1f_bl", "10us", em.DetectionChannel((650, 680)), 65],
        ["sim_tr_set_et_2f_diff", "5ms", em.DetectionChannel(), 2],
    ],
)
def test_emissions_extract(dirname, request, frame_time, channel, expected):
    rng = np.random.default_rng(1)
    emis = em.Emissions(frame_time=frame_time, channels={"detector": channel}, seed=rng)
    with pytest.warns(
        si.FloatingPointPrecisionWarning,
        match="Floating point precision error warning",
    ):
        simulation = request.getfixturevalue(dirname)

    emis.extract(simulation=simulation)
    assert emis.event_time_points is not None
    assert emis.event_time_points["detector"].size == expected
    if frame_time == "10us":
        exp_event_time_series = pd.Series(
            # fmt: off
                np.array(
                    [
                        1, 0, 0, 5, 9, 7, 1, 3, 0, 2, 6, 4, 6, 8, 3, 4,
                ],
                dtype=np.int64,
            ),
            # fmt: on
            index=np.linspace(0.00001, 0.00016, 16),
        )
    else:
        exp_event_time_series = pd.Series(
            dtype=np.int64, index=np.array([], dtype=np.float64)
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
        frame_time="10us",
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
    last_frame_end = emis.event_time_series.index[-1]
    assert emis.event_time_series["green"].sum() == np.count_nonzero(
        green <= last_frame_end
    )
    assert emis.event_time_series["red"].sum() == np.count_nonzero(
        red <= last_frame_end
    )


def test_generated_event_time_series_starts_with_first_frame_end(sim_tr_set_1f_bl):
    emis = em.Emissions(frame_time="10us", seed=1)
    emis.extract(sim_tr_set_1f_bl)

    assert emis.event_time_series.index[0] == pytest.approx(0.00001)


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
    np.testing.assert_array_equal(emis.event_time_series["all"], [1, 0])


def test_construct_event_time_series_preserves_photon_at_simulation_end():
    emis = em.Emissions()
    emis.event_time_points = {"all": np.array([0.001, 0.01])}
    simulation = SimpleNamespace(time_series=np.array([0.0, 0.01]))

    emis.construct_event_time_series(simulation=simulation, resample="5ms")

    np.testing.assert_array_equal(emis.event_time_series["all"], [1, 1])


@pytest.mark.parametrize(
    "end_time, frame_ends, red_counts, green_counts",
    [
        (0.004, [], [], []),
        (0.012, [0.005, 0.01], [2, 0], [1, 1]),
        (0.015, [0.005, 0.01, 0.015], [2, 0, 1], [1, 1, 0]),
    ],
)
def test_construct_event_time_series_keeps_only_complete_frames(
    end_time, frame_ends, red_counts, green_counts
):
    emis = em.Emissions(
        channels={
            "red": em.DetectionChannel(),
            "green": em.DetectionChannel(),
            "empty": em.DetectionChannel(),
        }
    )
    red = np.array([0.002, 0.005, 0.012])
    green = np.array([0.003, 0.01])
    emis.event_time_points = {
        "red": red[red <= end_time],
        "green": green[green <= end_time],
        "empty": np.array([], dtype=np.float64),
    }
    original_time_points = {
        name: points.copy() for name, points in emis.event_time_points.items()
    }
    simulation = SimpleNamespace(time_series=np.array([0.0, end_time]))

    emis.construct_event_time_series(simulation=simulation, resample="5ms")

    expected = pd.DataFrame(
        {"red": red_counts, "green": green_counts, "empty": np.zeros(len(frame_ends))},
        index=np.asarray(frame_ends, dtype=np.float64),
        dtype=np.int64,
    )
    pd.testing.assert_frame_equal(emis.event_time_series, expected)
    for name, points in original_time_points.items():
        np.testing.assert_array_equal(emis.event_time_points[name], points)


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
    assert emis.event_time_points["red"].size == 493
    exp_event_time_series = pd.Series(
        np.array([17, 49, 64, 27, 56, 66, 51, 43, 73, 47], dtype=np.int64),
        index=np.linspace(0.0001, 0.001, 10),
    )
    assert emis.event_time_series is not None
    pd.testing.assert_series_equal(
        emis.event_time_series["red"], exp_event_time_series.rename("red")
    )


def test_emissions_simulate_sparse_backend_matches_dense(tr_set_1f_bl):
    sparse_transition_set = tr_set_1f_bl.with_matrix_format("sparse")
    kwargs = {
        "frame_time": "100us",
        "channels": {"red": em.DetectionChannel((650, 700))},
        "seed": 42,
    }
    dense_emissions = em.Emissions(**kwargs)
    sparse_emissions = em.Emissions(**kwargs)

    dense_emissions.simulate(tr_set_1f_bl, size=100, frames=2, store_time_points=True)
    sparse_emissions.simulate(
        sparse_transition_set, size=100, frames=2, store_time_points=True
    )

    pd.testing.assert_frame_equal(
        sparse_emissions.event_time_series, dense_emissions.event_time_series
    )
    assert sparse_emissions.event_time_points is not None
    assert dense_emissions.event_time_points is not None
    np.testing.assert_array_equal(
        sparse_emissions.event_time_points["red"],
        dense_emissions.event_time_points["red"],
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


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"size": 0}, "size must be positive"),
        ({"size": -1}, "size must be positive"),
        ({"frames": 0}, "frames must be positive"),
        ({"frames": -1}, "frames must be positive"),
        ({"start_at": (99,)}, "not a valid combined state"),
    ],
)
def test_emissions_simulate_rejects_invalid_boundaries(tr_set_1f_bl, kwargs, message):
    with pytest.raises(ValueError, match=message):
        em.Emissions().simulate(transition_set=tr_set_1f_bl, **kwargs)


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"number_pulses": 0}, "number_pulses must be positive"),
        ({"size": 0}, "size must be positive"),
        ({"pulse_duration": 0}, "pulse_duration must be positive and finite"),
        (
            {"time_between_pulses": 0},
            "time_between_pulses must be positive and finite",
        ),
        (
            {"excitation_rates": {"testfluo_1": -1}},
            "excitation_rates values must be finite and non-negative",
        ),
        (
            {"excitation_rates": {"testfluo_1": np.nan}},
            "excitation_rates values must be finite and non-negative",
        ),
    ],
)
def test_emissions_tcspc_rejects_invalid_boundaries(
    tr_set_1f_bl, details, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        em.Emissions().tcspc(
            transition_set=tr_set_1f_bl,
            details=details,
            **kwargs,
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
            kwargs["paired_emission_ids"], np.array([4, 38, 40])
        )
        expected = np.zeros_like(kwargs["detection_probabilities"])
        expected[[4, 5, 6, 7], 0] = 0.6820037131347214
        expected[[38, 39, 40, 41, 42], 1] = 0.5847564420110373
        np.testing.assert_allclose(kwargs["detection_probabilities"], expected)
        assert kwargs["channel_names"] == ("donor", "acceptor")
        assert list(emis.event_time_series.columns) == ["donor", "acceptor"]


def test_get_paired_emission_ids_ignores_zero_rate_transition():
    transitions = pd.DataFrame(
        {
            "initial_state": [(1, 0), (1, 0)],
            "fluorophore_ids": [[0], [0, 1]],
            "rate": [1.0, 0.0],
        }
    )

    paired_emission_ids = em._get_paired_emission_ids(
        transitions, np.array([0], dtype=np.intp)
    )

    assert paired_emission_ids == []


def test_get_paired_emission_ids_requires_emitter_as_active_component():
    transitions = pd.DataFrame(
        {
            "initial_state": [(1, 0, 1), (1, 0, 1), (1, 0, 1)],
            "fluorophore_ids": [[0], [2], [0, 1]],
            "rate": [1.0, 1.0, 1.0],
        }
    )

    paired_emission_ids = em._get_paired_emission_ids(
        transitions, np.array([0, 1], dtype=np.intp)
    )

    assert paired_emission_ids == [0]


def test_emissions_tcspc_details_infers_excitation_rates(tr_set_1f_bl, caplog):
    emis = em.Emissions(seed=1)
    event_time_series = pd.DataFrame({"all": [1]}, index=[0.005])
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


def test_emissions_accepts_zero_threshold():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [0, 1]}, dtype=np.int64)
    expected = emis.event_time_series.copy()

    emis.apply_threshold(0)

    pd.testing.assert_frame_equal(emis.event_time_series, expected)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf])
def test_emissions_rejects_invalid_threshold(threshold):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [1]}, dtype=np.int64)

    with pytest.raises(ValueError, match="threshold must be finite and non-negative"):
        emis.apply_threshold(threshold)


def test_emissions_applies_channel_specific_thresholds():
    emis = em.Emissions(
        channels={"green": em.DetectionChannel(), "red": em.DetectionChannel()}
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 2, 4], "red": [0, 2, 4]}, dtype=np.int64
    )

    emis.apply_threshold({"green": 3, "red": 2})

    expected = pd.DataFrame({"green": [0, 0, 4], "red": [0, 2, 4]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


@pytest.mark.parametrize(
    "threshold",
    [
        {"green": 2},
        {"green": 2, "red": 2, "blue": 2},
    ],
)
def test_emissions_requires_complete_channel_parameter_mappings(threshold):
    emis = em.Emissions(
        channels={"green": em.DetectionChannel(), "red": em.DetectionChannel()}
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 2], "red": [0, 2]}, dtype=np.int64
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        emis.apply_threshold(threshold)


def test_emissions_adds_scalar_emccd_gain():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame(
        {"all": [0, 2, 4]},
        index=[0.1, 0.2, 0.3],
        dtype=np.int64,
    )

    emis.add_emccd_gain(emccd_gain=2, seed=1)

    expected = pd.DataFrame(
        {"all": [0, 4, 8]},
        index=[0.1, 0.2, 0.3],
        dtype=np.int64,
    )
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


def test_emissions_adds_channel_specific_emccd_gain():
    emis = em.Emissions(
        channels={"green": em.DetectionChannel(), "red": em.DetectionChannel()}
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 2, 4], "red": [0, 3, 5]}, dtype=np.int64
    )

    emis.add_emccd_gain({"green": 2, "red": 5}, seed=1)

    expected = pd.DataFrame({"green": [0, 4, 11], "red": [0, 16, 18]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


@pytest.mark.parametrize("emccd_gain", [0, -1, np.nan, np.inf])
def test_emissions_rejects_invalid_emccd_gain(emccd_gain):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [1]}, dtype=np.int64)

    with pytest.raises(ValueError, match="emccd_gain must be positive and finite"):
        emis.add_emccd_gain(emccd_gain)


def test_emissions_adds_scalar_gaussian_noise_and_clips_negative_counts():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame(
        {"all": [0, 1, 3]},
        index=[0.1, 0.2, 0.3],
        dtype=np.int64,
    )

    emis.add_gaussian_noise(mean=-2, std=0, seed=1)

    expected = pd.DataFrame(
        {"all": [0, 0, 1]},
        index=[0.1, 0.2, 0.3],
        dtype=np.int64,
    )
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


def test_emissions_adds_gaussian_noise_with_nonzero_standard_deviation():
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame(
        {"all": [10, 10, 10, 10]},
        dtype=np.int64,
    )

    emis.add_gaussian_noise(mean=0, std=5, seed=1)

    expected = pd.DataFrame({"all": [11, 14, 11, 4]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


def test_emissions_adds_channel_specific_gaussian_noise():
    emis = em.Emissions(
        channels={"green": em.DetectionChannel(), "red": em.DetectionChannel()}
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 1, 1], "red": [0, 1, 1]}, dtype=np.int64
    )

    emis.add_gaussian_noise(
        mean={"green": 2, "red": 5},
        std={"green": 0, "red": 0},
        seed=1,
    )

    expected = pd.DataFrame({"green": [2, 3, 3], "red": [5, 6, 6]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


@pytest.mark.parametrize("mean", [np.nan, np.inf, -np.inf])
def test_emissions_rejects_nonfinite_gaussian_mean(mean):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [1]}, dtype=np.int64)

    with pytest.raises(ValueError, match="mean must be finite"):
        emis.add_gaussian_noise(mean=mean, std=0)


@pytest.mark.parametrize("std", [-1, np.nan, np.inf])
def test_emissions_rejects_invalid_gaussian_standard_deviation(std):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [1]}, dtype=np.int64)

    with pytest.raises(ValueError, match="std must be finite and non-negative"):
        emis.add_gaussian_noise(mean=0, std=std)


def test_emissions_adds_channel_specific_poisson_noise():
    emis = em.Emissions(
        channels={"green": em.DetectionChannel(), "red": em.DetectionChannel()}
    )
    emis.event_time_series = pd.DataFrame(
        {"green": [0, 1, 1], "red": [0, 1, 1]}, dtype=np.int64
    )

    emis.add_poisson_noise({"green": 0, "red": 4}, seed=1)

    expected = pd.DataFrame({"green": [0, 1, 1], "red": [5, 4, 6]}, dtype=np.int64)
    pd.testing.assert_frame_equal(emis.event_time_series, expected)


@pytest.mark.parametrize("rate", [-1, np.nan, np.inf])
def test_emissions_rejects_invalid_poisson_rate(rate):
    emis = em.Emissions()
    emis.event_time_series = pd.DataFrame({"all": [1]}, dtype=np.int64)

    with pytest.raises(ValueError, match="rate must be finite and non-negative"):
        emis.add_poisson_noise(rate)


def test_emissions_adds_scalar_poisson_noise_without_changing_time_points():
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

    expected = pd.DataFrame({"green": [0, 2, 4], "red": [1, 2, 5]}, dtype=np.int64)
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
    emis.event_time_series = pd.DataFrame({"all": [1, 2]}, index=[0.005, 0.01])

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
        {"green": [1, 0], "red": [2, 3]}, index=[0.005, 0.01]
    )

    ax = emis.plot_time_series(channel="red")

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 0.005, 0.01])
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
    monkeypatch,
):
    original_get_p_filter = em.get_p_filter
    filtered_bandpasses = []

    def count_filter_calls(emission_spectrum, bandpass):
        filtered_bandpasses.append(bandpass)
        return original_get_p_filter(emission_spectrum, bandpass)

    monkeypatch.setattr(em, "get_p_filter", count_filter_calls)
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
    assert filtered_bandpasses == [(650, 700), (650, 700)]


def test_get_detection_probabilities_rejects_overlapping_channels(
    tr_set_bl_et_2f_diff,
):
    channels = {
        "first": em.DetectionChannel((600, 700)),
        "second": em.DetectionChannel((650, 750)),
    }

    with pytest.raises(ValueError, match="routing probabilities must sum to at most 1"):
        em.get_detection_probabilities(tr_set_bl_et_2f_diff, channels)


@pytest.mark.parametrize(
    "channel",
    [
        em.DetectionChannel((500, 600)),
        em.DetectionChannel(routing_probability=fd.Spectrum([500, 600], [0, 1])),
        em.DetectionChannel(bandpass=fd.Spectrum([500, 600], [0, 1])),
        em.DetectionChannel(detection_efficiency=fd.Spectrum([500, 600], [0, 1])),
    ],
)
def test_get_detection_probabilities_requires_spectral_data(channel):
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
            channels={"detector": channel},
        )


def test_get_detection_probabilities_rejects_paired_emission():
    transition_set = SimpleNamespace(
        combined_state_transitions_df=pd.DataFrame(
            {"photon": [True], "fluorophore_ids": [[0, 1]]}
        ),
        fluorophore_system=SimpleNamespace(fluorophores=[]),
    )

    with pytest.raises(ValueError, match="must belong to one fluorophore"):
        em.get_detection_probabilities(
            transition_set,
            {"detector": em.DetectionChannel()},
        )


def test_get_detection_probabilities_rejects_unfiltered_shared_channels(
    tr_set_1f,
):
    channels = {
        "first": em.DetectionChannel(),
        "second": em.DetectionChannel((600, 650)),
    }

    with pytest.raises(ValueError, match="routing probabilities must sum to at most 1"):
        em.get_detection_probabilities(tr_set_1f, channels)


def test_get_detection_probabilities_rejects_sum_above_one(tr_set_1f, monkeypatch):
    monkeypatch.setattr(em, "get_p_filter", lambda **_: 0.6)
    channels = {
        "first": em.DetectionChannel((500, 550)),
        "second": em.DetectionChannel((600, 650)),
    }

    with pytest.raises(ValueError, match="sum to at most 1"):
        em.get_detection_probabilities(tr_set_1f, channels)


@pytest.fixture
def routing_transition_set():
    constants = fd.FluorophoreData(emission_spectrum=fd.Spectrum([500, 600], [0, 1]))
    fluorophore = fl.Fluorophore("routing", [0, 0], constants)
    system = fl.FluorophoreSystem([fluorophore])
    return tr.TransitionSet(
        {
            "routing": [
                tr.Transition(tr.TransitionType.EXCITATION, 1, [0]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [0]),
            ]
        },
        system,
    )


def test_get_detection_probabilities_allows_beamsplitter_overlap(
    routing_transition_set,
):
    channels = {
        "first": em.DetectionChannel(
            (500, 600), detection_efficiency=0.5, routing_probability=0.5
        ),
        "second": em.DetectionChannel(
            (500, 600), detection_efficiency=0.5, routing_probability=0.5
        ),
    }

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0, 0], [0.25, 0.25]])


@pytest.mark.parametrize("transmission", [1.0, 0.5])
def test_get_detection_probabilities_routes_without_spectral_data(transmission):
    transition_set = SimpleNamespace(
        combined_state_transitions_df=pd.DataFrame(
            {"photon": [True], "fluorophore_ids": [[0]]}
        ),
        fluorophore_system=SimpleNamespace(
            fluorophores=[SimpleNamespace(name="unknown", constants=None)]
        ),
    )
    channels = {
        "first": em.DetectionChannel(transmission, routing_probability=0.6),
        "second": em.DetectionChannel(transmission, routing_probability=0.4),
    }

    probabilities = em.get_detection_probabilities(transition_set, channels)

    np.testing.assert_allclose(probabilities, np.array([[0.6, 0.4]]) * transmission)


@pytest.mark.parametrize("transmission", [1.0, 0.8])
@pytest.mark.parametrize("detection_efficiency", [1.0, 0.1])
def test_get_detection_probabilities_rejects_excess_constant_routing(
    routing_transition_set, transmission, detection_efficiency
):
    channels = {
        name: em.DetectionChannel(
            bandpass=transmission,
            routing_probability=0.5,
            detection_efficiency=detection_efficiency,
        )
        for name in ("first", "second", "third")
    }

    with pytest.raises(ValueError, match="at each wavelength"):
        em.get_detection_probabilities(routing_transition_set, channels)


@pytest.mark.parametrize(
    "bandpass, expected",
    [(1.0, 2 / 3), ((525, 575), 13 / 48), ((700, 800), 0), ((450, 650), 2 / 3)],
)
def test_get_detection_probabilities_integrates_spectral_routing(
    routing_transition_set, bandpass, expected
):
    response = fd.Spectrum([500, 525, 600], [0, 0.25, 1])
    channels = {"detector": em.DetectionChannel(bandpass, routing_probability=response)}

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0], [expected]])


def test_get_detection_probabilities_allows_complementary_routing(
    routing_transition_set,
):
    channels = {
        "transmitted": em.DetectionChannel(
            routing_probability=fd.Spectrum([500, 600], [0, 1])
        ),
        "reflected": em.DetectionChannel(
            routing_probability=fd.Spectrum([500, 550, 600], [1, 0.5, 0])
        ),
    }

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0, 0], [2 / 3, 1 / 3]])


def test_get_detection_probabilities_does_not_extrapolate_routing(
    routing_transition_set,
):
    channels = {
        "detector": em.DetectionChannel(
            routing_probability=fd.Spectrum([525, 575], [1, 1])
        )
    }

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0], [0.5]])


def test_get_detection_probabilities_allows_touching_bandpasses(routing_transition_set):
    channels = {
        "first": em.DetectionChannel((500, 550)),
        "second": em.DetectionChannel((550, 600)),
    }

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0, 0], [0.25, 0.75]])


def test_spectral_routing_rejects_zero_emission_spectrum():
    with pytest.raises(ValueError, match="zero total intensity"):
        em._get_p_detection(
            fd.Spectrum([500, 600], [0, 0]),
            em.DetectionChannel(routing_probability=fd.Spectrum([500, 600], [0, 1])),
        )


@pytest.mark.parametrize(
    "response_names, expected",
    [
        (("bandpass",), 2 / 3),
        (("detection_efficiency",), 2 / 3),
        (("bandpass", "detection_efficiency"), 1 / 2),
        (("routing_probability", "bandpass", "detection_efficiency"), 2 / 5),
    ],
)
def test_get_detection_probabilities_integrates_product_of_spectral_responses(
    routing_transition_set, response_names, expected
):
    grids = {
        "routing_probability": ([500, 525, 600], [0, 0.25, 1]),
        "bandpass": ([500, 550, 600], [0, 0.5, 1]),
        "detection_efficiency": ([500, 575, 600], [0, 0.75, 1]),
    }
    responses = {name: fd.Spectrum(*grids[name]) for name in response_names}
    channels = {"detector": em.DetectionChannel(**responses)}

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0], [expected]])


@pytest.mark.parametrize("name", ["bandpass", "detection_efficiency"])
@pytest.mark.parametrize("limits, expected", [([525, 575], 0.5), ([700, 800], 0)])
def test_get_detection_probabilities_does_not_extrapolate_spectral_responses(
    routing_transition_set, name, limits, expected
):
    channels = {"detector": em.DetectionChannel(**{name: fd.Spectrum(limits, [1, 1])})}

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0], [expected]])


def test_get_detection_probabilities_combines_bandpass_and_spectral_efficiency(
    routing_transition_set,
):
    channels = {
        "detector": em.DetectionChannel(
            (525, 575),
            routing_probability=0.5,
            detection_efficiency=fd.Spectrum([500, 600], [0, 1]),
        )
    }

    probabilities = em.get_detection_probabilities(routing_transition_set, channels)

    np.testing.assert_allclose(probabilities, [[0], [13 / 96]])


@pytest.mark.parametrize("baseline", [0.75, 0.8])
@pytest.mark.parametrize("spectral_efficiency", [False, True])
def test_get_detection_probabilities_checks_optical_maxima_between_knots(
    routing_transition_set, baseline, spectral_efficiency
):
    efficiency = fd.Spectrum([500, 600], [0.1, 0.1]) if spectral_efficiency else 0.1
    channels = {
        "first": em.DetectionChannel(
            bandpass=fd.Spectrum([500, 600], [1, 0]),
            routing_probability=fd.Spectrum([500, 600], [0, 1]),
            detection_efficiency=efficiency,
        ),
        "second": em.DetectionChannel(
            bandpass=baseline, detection_efficiency=efficiency
        ),
    }

    if baseline > 0.75:
        with pytest.raises(ValueError, match="at each wavelength"):
            em.get_detection_probabilities(routing_transition_set, channels)
    else:
        probabilities = em.get_detection_probabilities(routing_transition_set, channels)
        np.testing.assert_allclose(probabilities, [[0, 0], [1 / 60, 0.075]])


@pytest.mark.parametrize("detection_efficiency", [1, 0.1])
def test_get_detection_probabilities_rejects_pointwise_excess_routing(
    routing_transition_set, detection_efficiency
):
    channels = {
        "first": em.DetectionChannel(
            detection_efficiency=detection_efficiency,
            routing_probability=fd.Spectrum([500, 550, 600], [0, 0.6, 0]),
        ),
        "second": em.DetectionChannel(
            detection_efficiency=detection_efficiency, routing_probability=0.6
        ),
    }

    with pytest.raises(ValueError, match="at each wavelength"):
        em.get_detection_probabilities(routing_transition_set, channels)


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize("method", ["extract", "simulate", "tcspc", "tcspc_detailed"])
@pytest.mark.parametrize("spectral_response", [False, True])
def test_beamsplitter_routes_each_photon_at_most_once(
    routing_transition_set, matrix_format, method, spectral_response
):
    transition_set = routing_transition_set.with_matrix_format(matrix_format)
    if spectral_response:
        efficiency = fd.Spectrum([500, 600], [1, 1])
        channels = {
            "first": em.DetectionChannel(
                fd.Spectrum([500, 600], [0, 1]), detection_efficiency=efficiency
            ),
            "second": em.DetectionChannel(
                fd.Spectrum([500, 600], [1, 0]), detection_efficiency=efficiency
            ),
        }
    else:
        channels = {
            "first": em.DetectionChannel(routing_probability=0.5),
            "second": em.DetectionChannel(routing_probability=0.5),
        }
    emissions = em.Emissions(
        frame_time="1s",
        channels=channels,
        seed=42,
    )
    if method == "extract":
        simulation = si.Simulation(transition_set)
        simulation.run(size=200, seed=1)
        emissions.extract(simulation)
        expected_times = simulation.time_series[1:][simulation.transition_series == 1]
        detected_times = np.sort(
            np.concatenate(list(emissions.event_time_points.values()))
        )
        np.testing.assert_array_equal(detected_times, expected_times)
    elif method == "simulate":
        emissions.simulate(transition_set, frames=30, size=50, store_time_points=True)
    else:
        emissions.tcspc(
            transition_set,
            number_pulses=30,
            pulse_duration=0.1,
            time_between_pulses=1,
            excitation_rates={"routing": 100},
            size=50,
            store_time_points=True,
            details=method == "tcspc_detailed",
        )

    first = emissions.event_time_points["first"]
    second = emissions.event_time_points["second"]
    assert first.size + second.size > 0
    assert np.intersect1d(first, second).size == 0
    last_frame_end = emissions.event_time_series.index[-1]
    for name, times in emissions.event_time_points.items():
        assert emissions.event_time_series[name].sum() == np.count_nonzero(
            times <= last_frame_end
        )
