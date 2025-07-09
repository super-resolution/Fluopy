import logging
import os

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from fluopy import emissions as em
from fluopy import fluorophores as fl


@pytest.mark.parametrize(
    "bandpass, expected",
    [
        [(650, 700), 0.685702066268812],
        [(100, 750), "ValueError1"],
        [(200, 1001), "ValueError2"],
        [(450, 400), "ValueError3"],
        [(200, 1000), 1.0],
    ],
)
def test_get_p_filter(bandpass, expected):
    data_dir = os.path.join(
        Path(__file__).parents[1], "src", "fluopy", "fluorophore_spectra"
    )
    fluorophore = fl.Fluorophore(name="testfluo_1", position=[0, 0])
    if expected == "ValueError1":
        with pytest.raises(
            ValueError,
            match="The lower bandpass limit has to be between 200 and 1000 nm.",
        ):
            em.get_p_filter(
                data_dir=data_dir, fluorophore=fluorophore, bandpass=bandpass
            )
    elif expected == "ValueError2":
        with pytest.raises(
            ValueError,
            match="The upper bandpass limit has to be between 200 and 1000 nm.",
        ):
            em.get_p_filter(
                data_dir=data_dir, fluorophore=fluorophore, bandpass=bandpass
            )
    elif expected == "ValueError3":
        with pytest.raises(
            ValueError,
            match="The lower bandpass limit has to be smaller than the upper limit.",
        ):
            em.get_p_filter(
                data_dir=data_dir, fluorophore=fluorophore, bandpass=bandpass
            )
    else:
        p_passed = em.get_p_filter(
            data_dir=data_dir, fluorophore=fluorophore, bandpass=bandpass
        )
        assert p_passed == expected


@pytest.mark.parametrize(
    "bandpass, expected",
    [
        [None, {4: 1, 5: 1, 6: 1, 7: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1}],
        [
            (650, 700),
            {
                4: 0.685702066268812,
                5: 0.685702066268812,
                6: 0.685702066268812,
                7: 0.685702066268812,
                38: 0.5859441764799607,
                39: 0.5859441764799607,
                40: 0.5859441764799607,
                41: 0.5859441764799607,
                42: 0.5859441764799607,
            },
        ],
    ],
)
def test_get_emitting_transition_ids(bandpass, expected, tr_set_bl_et_2f_diff):
    emitting_transition_ids = em.get_emitting_transition_ids(
        bandpass=bandpass, transition_set=tr_set_bl_et_2f_diff
    )
    assert emitting_transition_ids == expected


def test_emissions():
    frame_time = "5ms"
    bandpass = None
    seed = 1
    emis = em.Emissions(frame_time=frame_time, bandpass=bandpass, seed=seed)
    assert emis.parameters["frame_time"] == frame_time
    assert emis.parameters["bandpass"] == bandpass
    assert emis.parameters["seed"] == seed
    assert emis.event_time_points is None
    assert emis.event_time_series is None


# test_emissions_extract also tests for...
# ...get_emission_indices()
# ...construct_event_time_series()
@pytest.mark.parametrize(
    "dirname, frame_time, bandpass, expected",
    [
        ["sim_tr_set_1f_bl", "10us", (650, 680), 83],
        ["sim_tr_set_et_2f_diff", "5ms", None, 2],
    ],
)
def test_emissions_extract(dirname, request, frame_time, bandpass, expected, caplog):
    emis = em.Emissions(frame_time=frame_time, bandpass=bandpass, seed=1)
    with caplog.at_level(logging.WARNING):
        simulation = request.getfixturevalue(dirname)
        assert "Floating point precision error warning" in caplog.text
    caplog.clear()

    emis.extract(simulation=simulation)
    assert emis.event_time_points.size == expected
    if frame_time == "10us":
        exp_event_time_series = pd.Series(
            # fmt: off
            np.array(
                [
                    0, 11, 4, 5, 9, 9, 7, 10, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 3, 5, 
                    7,
                ],
                dtype=np.int32,
            ),
            # fmt: on
            index=np.linspace(0, 0.00045, 46),
        )
    else:
        exp_event_time_series = pd.Series(
            np.array([0, 2], dtype=np.int32), index=[0.0, 0.005]
        )
    pd.testing.assert_series_equal(emis.event_time_series, exp_event_time_series)


def test_emissions_simulate(tr_set_1f_bl):
    emis = em.Emissions(frame_time="100us", bandpass=(650, 700), seed=1)
    emis.simulate(
        transition_set=tr_set_1f_bl,
        start_at=None,
        size=1e3,
        frames=10,
        store_time_points=True,
    )
    assert emis.event_time_points.size == 205
    exp_event_time_series = pd.Series(
        np.array([0, 80, 0, 0, 0, 0, 0, 16, 52, 7, 50], dtype=np.int64),
        index=np.linspace(0, 0.001, 11),
    )
    pd.testing.assert_series_equal(emis.event_time_series, exp_event_time_series)


@pytest.mark.parametrize(
    "dirname, excitation_rates, expected",
    [
        ["tr_set_1f_bl", {"testfluo_1": 1e5}, 0],
        ["tr_set_bl_et_2f_diff", {"testfluo_1": 1e10, "testfluo_2": 1e11}, 1],
        ["tr_set_bl_et_2f_same", {"testfluo_1": 1e10}, 1],
    ],
)
def test_emissions_tcspc(dirname, request, excitation_rates, expected, caplog):
    tr_set = request.getfixturevalue(dirname)
    if expected == 1:
        ets = [j for h, j in tr_set.transition_df.index if "dist" in h]
        tr_set = tr_set.adjust_rates({identity: 1e10 for identity in ets})
        tr_set.finalize()
    emis = em.Emissions(frame_time="100us", bandpass=(650, 700), seed=1)
    with caplog.at_level(logging.WARNING):
        lifetimes_DA, _, _ = emis.tcspc(
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
        assert lifetimes_DA.size > 0
    else:
        assert lifetimes_DA.size == 0


def test_emissions_tcspc_parameters(tr_set_bl_et_2f_diff):
    tr_set = tr_set_bl_et_2f_diff.adjust_rates(
        {8: 1e10, 15: 1e10}, keep_zero_rates=True
    )
    tr_set.finalize()
    emis = em.Emissions(frame_time="1ms", bandpass=(650, 700), seed=1)
    with patch("fluopy.emissions.simulate_TCSPC") as mock_tcspc:
        mock_tcspc.return_value = (0, 0, 0, 0, 0)
        emis.tcspc(
            transition_set=tr_set,
            number_pulses=1e4,
            pulse_duration=1e-9,
            time_between_pulses=1e-6,
            excitation_rates={"testfluo_1": 1e5, "testfluo_2": 1e6},
            size=1e3,
            store_time_points=True,
        )
        emitting_transition_ids = {
            4: 0.685702066268812,
            5: 0.685702066268812,
            6: 0.685702066268812,
            7: 0.685702066268812,
            38: 0.5859441764799607,
            39: 0.5859441764799607,
            40: 0.5859441764799607,
            41: 0.5859441764799607,
            42: 0.5859441764799607,
        }
        args, kwargs = mock_tcspc.call_args
        np.testing.assert_array_equal(
            kwargs["et_transition_ids"], np.array([4, 38, 40])
        )
        assert kwargs["emitting_transition_ids"] == emitting_transition_ids


@pytest.mark.parametrize(
    "p, expected",
    [[0.7, ""], [1.0, "no change"], [1.1, "ValueError"], [-0.1, "ValueError"]],
)
def test_emissions_add_photon_collection_objective(p, em_large, expected):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )

    if expected == "ValueError":
        with pytest.raises(ValueError, match="p has to be between 0 and 1."):
            em_large.add_photon_collection_objective(p=p, seed=10)
    elif expected == "no change":
        em_large.add_photon_collection_objective(p=p, seed=10)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series_prev
        )
    else:
        em_large.add_photon_collection_objective(p=p, seed=10)
        # fmt: off
        exp_values = np.array(
            [
                34176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                34842, 5175, 0, 0, 0, 0, 0, 0, 46614, 53267, 25163, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
            dtype=np.int64)
        # fmt: on

        exp_event_time_series = pd.Series(exp_values, index=exp_index)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series
        )


@pytest.mark.parametrize(
    "p, expected",
    [[0.7, ""], [1.0, "no change"], [1.1, "ValueError"], [-0.1, "ValueError"]],
)
def test_emissions_add_quantum_efficiency(em_large, p, expected):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )

    if expected == "ValueError":
        with pytest.raises(ValueError, match="p has to be between 0 and 1."):
            em_large.add_quantum_efficiency(p=p, seed=1)
    elif expected == "no change":
        em_large.add_quantum_efficiency(p=p, seed=1)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series_prev
        )
    else:
        em_large.add_quantum_efficiency(p=p, seed=1)
        # fmt: off
        exp_values = np.array(
            [
                34454, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                34953, 5221, 0, 0, 0, 0, 0, 0, 46702, 52968, 25294, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], 
            dtype=np.int64)
        # fmt: on

        exp_event_time_series = pd.Series(exp_values, index=exp_index)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series
        )


@pytest.mark.parametrize(
    "p, expected",
    [[0.7, ""], [1.0, "no change"], [1.1, "ValueError"], [-0.1, "ValueError"]],
)
def test_emissions_add_transmittance(em_large, p, expected):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )

    if expected == "ValueError":
        with pytest.raises(ValueError, match="p has to be between 0 and 1."):
            em_large.add_transmittance(p=p, seed=1)
    elif expected == "no change":
        em_large.add_transmittance(p=p, seed=1)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series_prev
        )
    else:
        em_large.add_transmittance(p=p, seed=1)
        # fmt: off
        exp_values = np.array(
            [
                34454, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                34953, 5221, 0, 0, 0, 0, 0, 0, 46702, 52968, 25294, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], 
            dtype=np.int64)
        # fmt: on

        exp_event_time_series = pd.Series(exp_values, index=exp_index)
        pd.testing.assert_series_equal(
            em_large.event_time_series, exp_event_time_series
        )


def test_emissions_add_emccd_gain(em_large):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )
    em_large.add_emccd_gain(emccd_gain=10, seed=1)
    # fmt: off
    exp_values = np.array(
        [
            492113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 497653, 75361, 0, 0, 0, 0, 0, 0, 664801,
            760421, 358760, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        dtype=np.int64)
    # fmt: on
    exp_event_time_series = pd.Series(exp_values, index=exp_index)
    pd.testing.assert_series_equal(em_large.event_time_series, exp_event_time_series)


def test_emissions_add_gaussian_noise(em_large):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )
    em_large.add_gaussian_noise(mean=10, std=5, seed=1)
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
    pd.testing.assert_series_equal(em_large.event_time_series, exp_event_time_series)


def test_emissions_add_poisson_noise(em_large):
    # fmt: off
    exp_values_prev = np.array(
        [
            49135, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 49692, 7458, 0, 0, 0, 0, 0, 0, 66619, 75942,
            35871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 
        dtype=np.int64)
    # fmt: on
    exp_index = np.linspace(0, 9.9, 100)
    exp_event_time_series_prev = pd.Series(exp_values_prev, index=exp_index)
    pd.testing.assert_series_equal(
        em_large.event_time_series, exp_event_time_series_prev
    )
    em_large.add_poisson_noise(rate=10, seed=1)
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
    pd.testing.assert_series_equal(em_large.event_time_series, exp_event_time_series)


def test_save_and_load(request, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        em_tr_set_1f_bl = request.getfixturevalue("em_tr_set_1f_bl")
        assert "Floating point precision error warning" in caplog.text
    caplog.clear()

    em_tr_set_1f_bl.save(path=tmp_path, name_extension="_test_extension")
    assert os.path.isfile(os.path.join(tmp_path, "event_time_series_test_extension.csv"))
    assert os.path.isfile(os.path.join(tmp_path, "event_time_points_test_extension.npy"))
    emis = em.Emissions.load(path=tmp_path, name_extension="_test_extension")
    assert type(emis.event_time_points) is np.ndarray
    assert type(emis.event_time_series) is pd.Series
    os.remove(os.path.join(tmp_path, "event_time_series_test_extension.csv"))
    os.remove(os.path.join(tmp_path, "event_time_points_test_extension.npy"))
    assert not os.path.isfile(
        os.path.join(tmp_path, "event_time_series_test_extension.csv")
    )
    assert not os.path.isfile(
        os.path.join(tmp_path, "event_time_points_test_extension.npy")
    )
