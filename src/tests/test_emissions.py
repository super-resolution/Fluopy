import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import src.emissions as em
import src.fluorophores as fl


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
    data_dir = os.path.join(Path(__file__).parent.parent, "fluorophore_collection")
    fluorophore = fl.Fluorophore(name="cy5", position=[0, 0])
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
def test_emissions_extract(dirname, request, frame_time, bandpass, expected):
    emis = em.Emissions(frame_time=frame_time, bandpass=bandpass, seed=1)
    simulation = request.getfixturevalue(dirname)
    emis.extract(simulation=simulation)
    assert emis.event_time_points.size == expected
    if frame_time == "10us":
        exp_event_time_series = pd.Series(
            # fmt: off
            np.array(
                [
                    11, 4, 5, 9, 9, 7, 10, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 3, 5, 7,
                ],
                dtype=np.int64,
            ),
            # fmt: on
            index=np.linspace(0, 0.00044, 45),
        )
    else:
        exp_event_time_series = pd.Series([expected], index=[0.0])
    pd.testing.assert_series_equal(emis.event_time_series, exp_event_time_series)


def test_emissions_simulate(tr_set_1f_bl):
    emis = em.Emissions(frame_time="100us", bandpass=(650, 700), seed=1)
    emis.simulate(
        transition_set=tr_set_1f_bl,
        start_at=None,
        size=1e3,
        frames=10,
        store_time_points=True,
        seed=1,
    )
    assert emis.event_time_points.size == 306
    exp_event_time_series = pd.Series(
        np.array([80, 0, 0, 0, 9, 80, 79, 47, 11, 0], dtype=np.int64),
        index=np.linspace(0, 0.0009, 10),
    )
    pd.testing.assert_series_equal(emis.event_time_series, exp_event_time_series)


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
        with pytest.raises(ValueError, match="rate has to be between 0 and 1."):
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
        with pytest.raises(ValueError, match="rate has to be between 0 and 1."):
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
    exp_values = np.array(
        [
            49146.72792096032,
            14.108090717505792,
            11.652185380916936,
            3.484213841978195,
            14.526779333365589,
            12.231872861820056,
            7.315233823198574,
            12.905590520981765,
            11.822861980930378,
            11.47066248327763,
            10.142111206578983,
            12.733564933062235,
            6.317729564991666,
            9.185450260034736,
            7.589403436600109,
            12.994231063173139,
            10.198610537408294,
            8.537716245174558,
            6.090457688215789,
            8.714038796905646,
            10.040710902591718,
            8.621985473503148,
            49708.47031907199,
            7473.033621576529,
            -3.555812394829843,
            0.5549337701616359,
            9.126139539724191,
            7.889047942118232,
            11.068214987493056,
            11.086609655112818,
            66639.58919377526,
            75946.43989618654,
            35879.11197496436,
            20.21385803746165,
            13.233514981009234,
            13.315316861881309,
            7.429968141562686,
            1.7596241457217374,
            10.837323721113705,
            10.545070439107738,
            3.8632397287771294,
            6.583866691097189,
            9.639781601363863,
            5.2762418846961125,
            9.508650160738913,
            10.477415137347272,
            10.177931185277428,
            7.468541708428426,
            12.968740358929114,
            14.455834771411642,
            11.604241522832819,
            5.908848863048465,
            13.658261418927204,
            7.492799907664738,
            14.395803091439927,
            4.641062915612779,
            14.572336015643906,
            9.899682726922597,
            3.756255548327923,
            8.430502640165761,
            10.270511393857719,
            11.363956695822269,
            5.089059375295111,
            4.463134764174034,
            10.997922664235404,
            7.66625191560099,
            11.177528058651125,
            13.797597612391897,
            1.756063168245257,
            11.271940582588087,
            16.123234837678663,
            8.512365778147634,
            5.94592708381215,
            13.761219135897964,
            11.267232581040707,
            14.479415353887802,
            8.273921449743602,
            2.5909086313889436,
            9.449946176443746,
            7.770859234943839,
            13.87661911023787,
            10.96816424188577,
            1.8457538378244944,
            4.024184599484001,
            14.418945182936277,
            13.398825087089232,
            6.798783170457556,
            9.994756017163597,
            12.22786776888093,
            12.342021679236389,
            14.381210980571751,
            11.28242813610781,
            9.52585830515751,
            8.705759676060772,
            15.278714002666256,
            -1.2542713753926886,
            9.306723374543314,
            10.1650005199203,
            2.8732551956490617,
            11.664068065690234,
        ],
        dtype=np.int64
    )
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
            49143, 13, 10, 13, 8,
            8, 6, 7, 12, 10, 10, 11, 10, 11, 11, 14, 10, 7, 7, 14, 8, 11, 49706,
            7465, 6, 15, 11, 12, 13, 14, 66632, 75951, 35883, 9, 8, 5, 9, 11, 14,
            13, 10, 7, 7, 12, 9, 12, 11, 12, 9, 5, 8, 8, 12, 4, 7, 10, 8, 10, 18, 8,
            6, 7, 11, 10, 13, 11, 9, 20, 12, 14, 13, 9, 9, 6, 11, 13, 11, 13, 10,
            10, 12, 6, 6, 8, 3, 7, 17, 16, 5, 5, 7, 8, 12, 11, 9, 6, 7, 10, 8, 13
        ],
        np.int64)
    # fmt: on
    exp_event_time_series = pd.Series(exp_values, index=exp_index)
    pd.testing.assert_series_equal(em_large.event_time_series, exp_event_time_series)


def test_save_and_load(em_tr_set_1f_bl):
    path = os.path.join(os.path.dirname(__file__), "temp_data")
    em_tr_set_1f_bl.save(path=path, name_extension="_test_extension")
    assert os.path.isfile(os.path.join(path, "event_time_series_test_extension.npy"))
    assert os.path.isfile(os.path.join(path, "event_time_points_test_extension.npy"))
    emis = em.Emissions.load(path=path, name_extension="_test_extension")
    assert type(emis.event_time_points) is np.ndarray
    assert type(emis.event_time_series) is np.ndarray
    os.remove(os.path.join(path, "event_time_series_test_extension.npy"))
    os.remove(os.path.join(path, "event_time_points_test_extension.npy"))
    assert not os.path.isfile(
        os.path.join(path, "event_time_series_test_extension.npy")
    )
    assert not os.path.isfile(
        os.path.join(path, "event_time_points_test_extension.npy")
    )
