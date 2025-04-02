import pytest
import warnings
import numpy as np
import pandas as pd
import src.analysis as an
import src.transitions as tr


# test_analysis_# includes testing of...
# ...check_absorbing()
# ...get_transition_occurrences()
# ...get_state_occurrences()
# ...get_lifetimes()
# ...infer_stats()


# test with 1 fluorophore, with bleaching
def test_analysis_1(request):
    with pytest.warns(
        UserWarning,
        match="absorbing states have a lifetime of inf and a frequency / occupation "
        "of 0. Absorbing transitions have a frequency of 0.",
    ):
        pred_bl = request.getfixturevalue("pred_tr_set_1f_bl")
        pred_bl_2 = request.getfixturevalue("pred_tr_set_1f_bl_2")
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
        sim_tr_set_1f_bl = request.getfixturevalue("sim_tr_set_1f_bl")
    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    assert analysis.simulation == sim_tr_set_1f_bl
    exp_freq_trans = np.array([0.496, 0.147, 0.001, 0.001, 0.008, 0.008, 0.339, 0.0])
    np.testing.assert_array_almost_equal(analysis.frequency_transitions, exp_freq_trans)
    exp_freq_states = {
        "testfluo_1": np.array([0.4955045, 0.4955045, 0.000999, 0.00799201, 0.0])
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_array_almost_equal(freq, exp_freq_states[fluorophore])
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for fluorophore, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            1.75134964e-07,
            8.60868941e-10,
            1.09073262e-09,
            3.09550669e-04,
            7.35362906e-10,
            5.75112962e-06,
            9.43759362e-10,
            np.nan,
        ]
    )
    np.testing.assert_array_almost_equal(
        analysis.mean_transition_times, exp_mean_trans_times
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.75134964e-07, 9.16072311e-10, 3.09550669e-04, 5.75112962e-06, np.nan]
        )
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_array_almost_equal(
            mean_lifetimes, exp_mean_lifetimes[fluorophore]
        )
    exp_state_occ = {
        "testfluo_1": np.array([0.19614058, 0.00102595, 0.6989477, 0.10388577, 0.0])
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_array_almost_equal(state_occ, exp_state_occ[fluorophore])
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_frequency_transitions(prediction=pred_bl_2)
    analysis.plot_frequency_transitions(prediction=pred_bl)
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_frequency_states(prediction=pred_bl_2)
    analysis.plot_frequency_states(prediction=pred_bl)
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_mean_transition_times(prediction=pred_bl_2)
    analysis.plot_mean_transition_times(prediction=pred_bl)
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_mean_lifetimes(prediction=pred_bl_2)
    analysis.plot_mean_lifetimes(prediction=pred_bl)
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_state_occupations(prediction=pred_bl_2)
    analysis.plot_state_occupations(prediction=pred_bl)
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_lifetime_distributions(
            fluorophore="testfluo_1", state_identity=0, prediction=pred_bl_2
        )
    analysis.plot_lifetime_distributions(
        fluorophore="testfluo_1", state_identity=0, prediction=pred_bl
    )
    with pytest.warns(
        UserWarning,
        match="prediction is based on different TransitionSet than simulation.",
    ):
        analysis.plot_transition_time_distributions(
            fluorophore="testfluo_1", transition_id=0, prediction=pred_bl_2
        )
    analysis.plot_transition_time_distributions(
        fluorophore="testfluo_1", transition_id=0, prediction=pred_bl
    )


# test with 2 fluorophores, with energy transfer
def test_analysis_2(request):
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
        sim_tr_set_et_2f_diff = request.getfixturevalue("sim_tr_set_et_2f_diff")
    analysis = an.Analysis(simulation=sim_tr_set_et_2f_diff)
    assert analysis.simulation == sim_tr_set_et_2f_diff
    exp_freq_trans = np.array(
        [
            5.137426e-04,
            2.568713e-04,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            2.568713e-04,
            9.989725e-01,
            5.137426e-04,
            2.568713e-04,
            0.000000e00,
            0.000000e00,
            2.568713e-04,
            9.989725e-01,
            0.000000e00,
        ]
    )
    np.testing.assert_array_almost_equal(analysis.frequency_transitions, exp_freq_trans)
    exp_freq_states = {
        "testfluo_1": np.array([0.50006424, 0.49993576, 0.0, 0.0]),
        "testfluo_2": np.array([0.50006424, 0.49993576, 0.0]),
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_array_almost_equal(freq, exp_freq_states[fluorophore])
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for fluorophore, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            8.446330e-08,
            3.778371e-13,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.315027e-14,
            4.772330e-13,
            1.380437e-07,
            4.950885e-13,
            np.nan,
            np.nan,
            1.008086e-14,
            4.636361e-13,
            np.nan,
        ]
    )
    np.testing.assert_array_almost_equal(
        analysis.mean_transition_times, exp_mean_trans_times
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array([1.14838204e-10, 4.77090756e-13, np.nan, np.nan]),
        "testfluo_2": np.array([1.14847159e-10, 4.68138143e-13, np.nan]),
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_array_almost_equal(
            mean_lifetimes, exp_mean_lifetimes[fluorophore]
        )
    exp_state_occ = {
        "testfluo_1": np.array([0.99586379, 0.00413621, 0.0, 0.0]),
        "testfluo_2": np.array([0.995981, 0.004019, 0.0]),
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_array_almost_equal(state_occ, exp_state_occ[fluorophore])


# test with 2 fluorophores, without energy transfer
def test_analysis_3(request):
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
        sim_tr_set_2f_diff = request.getfixturevalue("sim_tr_set_2f_diff")
    analysis = an.Analysis(simulation=sim_tr_set_2f_diff)
    assert analysis.simulation == sim_tr_set_2f_diff
    exp_freq_trans = np.array(
        [
            0.49180328,
            0.1147541,
            0.0,
            0.0,
            0.01639344,
            0.01639344,
            0.36065574,
            0.5,
            0.30283912,
            0.00157729,
            0.00157729,
            0.19400631,
        ]
    )
    np.testing.assert_array_almost_equal(analysis.frequency_transitions, exp_freq_trans)
    exp_freq_states = {
        "testfluo_1": np.array([0.49318801, 0.49046322, 0.0, 0.01634877]),
        "testfluo_2": np.array([0.4992126, 0.4992126, 0.0015748]),
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_array_almost_equal(freq, exp_freq_states[fluorophore])
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for fluorophore, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            1.75200755e-07,
            1.02682822e-09,
            np.nan,
            np.nan,
            4.97850718e-10,
            1.01301802e-05,
            8.40610810e-10,
            2.73973283e-07,
            2.96493537e-09,
            4.91345055e-11,
            4.79878872e-06,
            2.84417298e-09,
        ]
    )
    np.testing.assert_array_almost_equal(
        analysis.mean_transition_times, exp_mean_trans_times
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.75200755e-07, 8.72636203e-10, np.nan, 1.01301802e-05]
        ),
        "testfluo_2": np.array([2.73973283e-07, 2.90870254e-09, 4.79878872e-06]),
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_array_almost_equal(
            mean_lifetimes, exp_mean_lifetimes[fluorophore]
        )
    exp_state_occ = {
        "testfluo_1": np.array([0.3422721, 0.00169536, 0.0, 0.65603253]),
        "testfluo_2": np.array([0.93820002, 0.00996062, 0.05183936]),
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_array_almost_equal(state_occ, exp_state_occ[fluorophore])


def test_get_fluorescence_lifetimes(request):
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
        sim_tr_set_1f_bl = request.getfixturevalue("sim_tr_set_1f_bl")
        sim_tr_set_2f_diff = request.getfixturevalue("sim_tr_set_2f_diff")
    assert tr.SingleState.S1.value == 1  # if it fails, check
    # analysis.get_fluorescence_lifetimes
    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    fluorescence_lifetimes = analysis.get_fluorescence_lifetimes()
    np.testing.assert_array_almost_equal(
        fluorescence_lifetimes[:12],
        np.array(
            [
                1.93683425e-09,
                1.89101373e-10,
                2.83008866e-10,
                1.19337973e-09,
                1.59230928e-09,
                7.23212712e-10,
                6.13916751e-10,
                6.61515620e-10,
                4.89384977e-10,
                7.77973574e-10,
                5.22660359e-10,
                6.71603162e-10,
            ]
        ),
    )
    with pytest.raises(
        ValueError, match="fluorophore wrong_name not found in lifetime_distributions."
    ):
        analysis.get_fluorescence_lifetimes(fluorophore="wrong_name")
    analysis = an.Analysis(simulation=sim_tr_set_2f_diff)
    with pytest.raises(
        ValueError,
        match="if multiple fluorophores are present, fluorophore must be specified.",
    ):
        analysis.get_fluorescence_lifetimes()
    analysis.get_fluorescence_lifetimes("testfluo_1")


def test_get_emitting_transition_lifetimes(request):
    with pytest.warns(UserWarning, match="Floating point precision error warning"):
        sim_tr_set_1f_bl = request.getfixturevalue("sim_tr_set_1f_bl")
        sim_tr_set_2f_diff = request.getfixturevalue("sim_tr_set_2f_diff")
    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    exp_fluorescence_lifetimes = analysis.get_emitting_transition_lifetimes()
    np.testing.assert_array_almost_equal(
        exp_fluorescence_lifetimes[:12],
        np.array(
            [
                1.93683425e-09,
                1.89101373e-10,
                2.83008866e-10,
                1.19337973e-09,
                1.59230928e-09,
                7.23212712e-10,
                6.13916751e-10,
                6.61515620e-10,
                4.89384977e-10,
                7.77973574e-10,
                5.22660359e-10,
                6.71603162e-10,
            ]
        ),
    )
    with pytest.raises(
        ValueError, match="fluorophore wrong_name not found in transition dataframe."
    ):
        analysis.get_emitting_transition_lifetimes(fluorophore="wrong_name")
    analysis = an.Analysis(simulation=sim_tr_set_2f_diff)
    with pytest.raises(
        ValueError,
        match="if multiple fluorophores are present, fluorophore must be specified.",
    ):
        analysis.get_emitting_transition_lifetimes()
    analysis.get_emitting_transition_lifetimes("testfluo_1")


def test_no_diff_dist():
    arrays = [
        [
            "Cy5",
            "Cy5",
            "Cy5",
            "Cy5_dist_1",
            "Cy5_dist_1",
            "Cy5_dist_2",
            "Cy5_dist_2",
            "Cy5_dist_3",
            "Cy5_dist_3",
            "H",
            "H",
            "H",
            "H_dist_2",
            "H_dist_2",
            "H_dist_1",
            "H_dist_1",
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ]
    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Group", "Number"])

    data = np.random.randn(16, 3)
    df = pd.DataFrame(data, index=index, columns=["X", "Y", "Z"])

    df2, dict2, dict2_vals = an.no_diff_dist(df, ["Cy5", "H"])
    assert df2.shape == (10, 3)
    assert df2.index.get_level_values(0).unique().tolist() == [
        "Cy5",
        "Cy5_dist_1",
        "H",
        "H_dist_2",
    ]
    dict2_exp = {
        np.int64(3): pd.Index([5, 7], dtype="int64", name="Number"),
        np.int64(4): pd.Index([6, 8], dtype="int64", name="Number"),
        np.int64(8): pd.Index([14], dtype="int64", name="Number"),
        np.int64(9): pd.Index([15], dtype="int64", name="Number"),
    }
    assert dict2.keys() == dict2_exp.keys()
    for key in dict2_exp:
        pd.testing.assert_index_equal(dict2[key], dict2_exp[key])
    dict2_vals_exp = np.array([5, 7, 6, 8, 14 ,15])
    np.testing.assert_array_equal(dict2_vals, dict2_vals_exp)
