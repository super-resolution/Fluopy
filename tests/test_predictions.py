import pytest
import numpy as np
import scipy.stats as stats
from fluopy import prediction as pr


@pytest.mark.parametrize(
    "drop_transitions, exp_Q",
    [
        [np.array([0, 1]), np.array([[0.0, 0.8], [0.4, 0.0]])],
        [0, np.array([[0.0, 0.05, 0.05], [0.1, 0.0, 0.8], [0.3, 0.4, 0.0]])],
    ],
)
def test_get_Q(drop_transitions, exp_Q):
    P = np.array(
        [
            [0, 0.3, 0.4, 0.3],
            [0.9, 0, 0.05, 0.05],
            [0.1, 0.1, 0, 0.8],
            [0.3, 0.3, 0.4, 0],
        ]
    )
    Q = pr.get_Q(P=P, drop_transitions=drop_transitions)
    np.testing.assert_array_equal(Q, exp_Q)


def test_get_I_t():
    Q = np.array([[0.0, 0.05, 0.05], [0.1, 0.0, 0.8], [0.3, 0.4, 0.0]])
    I_t = pr.get_I_t(Q=Q)
    exp_I_t = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(I_t, exp_I_t)


def test_get_N():
    Q = np.array([[0.0, 0.05, 0.05], [0.1, 0.0, 0.8], [0.3, 0.4, 0.0]])
    I_t = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    N = pr.get_N(I_t=I_t, Q=Q)
    exp_N = np.array(
        [
            [1.05263158, 0.10835913, 0.13931889],
            [0.52631579, 1.5247678, 1.24613003],
            [0.52631579, 0.64241486, 1.54024768],
        ]
    )
    np.testing.assert_array_almost_equal(N, exp_N)


# test_prediction_# includes testing of...
# ...predict_transition_occurrences()
# ...predict_transition_occurrences_abs()
# ...predict_state_occurrences()
# ...predict_lifetimes()
# ...infer_stats()


# test with 3 fluorophores
def test_prediction_1(tr_set_bl_et_3f):
    with pytest.raises(
        ValueError, match="prediction not available for more than 2 " "fluorophores"
    ):
        prediction = pr.Prediction(transition_set=tr_set_bl_et_3f)


# test with not finalized transition_set
def test_prediction_2(tr_set_bl_et_2f_diff):
    tr_set_new = tr_set_bl_et_2f_diff.filter_by_identity(remove_list=[4])
    prediction = pr.Prediction(transition_set=tr_set_new)
    assert prediction


# test with 2 different fluorophores, energy transfer, bleaching
def test_prediction_3(tr_set_bl_et_2f_diff):
    with pytest.warns(UserWarning) as record:
        prediction = pr.Prediction(transition_set=tr_set_bl_et_2f_diff)
        assert str(record[0].message) == (
            "prediction accuracy of energy transfers more difficult to tune. Only "
            "frequencies available, lifetimes and occupations not available."
        )
        assert str(record[1].message) == (
            "absorbing states have a lifetime of inf and a frequency / occupation "
            "of 0. Absorbing transitions have a frequency of 0."
        )
    assert prediction.energy_transfer
    assert prediction.absorbing_chain
    assert prediction.transition_set == tr_set_bl_et_2f_diff
    exp_freq_trans = np.array(
        [
            5.132559e-04,
            1.632239e-04,
            5.017624e-07,
            5.016621e-07,
            1.209066e-05,
            1.209066e-05,
            4.287167e-04,
            0.000000e00,
            9.988696e-01,
            3.720376e-03,
            1.993971e-03,
            9.969856e-06,
            9.96975588e-06,
            1.31934422e-03,
            0.00000000e00,
            9.92639978e-01,
            3.06391128e-04,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [5.015829e-01, 4.984108e-01, 2.518078e-07, 6.067659e-06, 0.000000e00]
        ),
        "testfluo_2": np.array([4.984165e-01, 5.015785e-01, 4.987747e-06, 0.000000e00]),
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    assert prediction.transition_time_distributions is None
    assert prediction.lifetime_distributions is None
    assert prediction.mean_transition_times is None
    assert prediction.mean_lifetimes is None
    assert prediction.state_occupations is None


# test with 2 same fluorophores, energy transfer, bleaching
def test_prediction_4(tr_set_bl_et_2f_same):
    with pytest.warns(UserWarning) as record:
        prediction = pr.Prediction(transition_set=tr_set_bl_et_2f_same)
        assert str(record[0].message) == (
            "prediction accuracy of energy transfers more difficult to tune. Only "
            "frequencies available, lifetimes and occupations not available."
        )
        assert str(record[1].message) == (
            "absorbing states have a lifetime of inf and a frequency / occupation "
            "of 0. Absorbing transitions have a frequency of 0."
        )
    assert prediction.energy_transfer
    assert prediction.absorbing_chain
    assert prediction.transition_set == tr_set_bl_et_2f_same
    exp_freq_trans = np.array(
        [
            2.402425e-05,
            4.406643e-06,
            1.354635e-08,
            1.354364e-08,
            3.264180e-07,
            3.264180e-07,
            1.157429e-05,
            0.000000e00,
            9.999516e-01,
            7.703353e-06,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [6.666603e-01, 3.333395e-01, 9.031066e-09, 2.176161e-07, 0.000000e00]
        )
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    assert prediction.transition_time_distributions is None
    assert prediction.lifetime_distributions is None
    assert prediction.mean_transition_times is None
    assert prediction.mean_lifetimes is None
    assert prediction.state_occupations is None


# test with 2 different fluorophores, energy transfer, no bleaching
def test_prediction_5(tr_set_et_2f_diff):
    with pytest.warns(
        UserWarning,
        match="prediction accuracy of energy transfers more difficult to tune. Only "
        "frequencies available, lifetimes and occupations not available.",
    ):
        prediction = pr.Prediction(transition_set=tr_set_et_2f_diff)
    assert prediction.energy_transfer
    assert not prediction.absorbing_chain
    assert prediction.transition_set == tr_set_et_2f_diff
    exp_freq_trans = np.array(
        [
            4.298562e-04,
            1.407102e-04,
            4.325535e-07,
            4.325535e-07,
            1.042297e-05,
            1.042297e-05,
            3.695831e-04,
            9.990381e-01,
            8.609659e-04,
            2.769130e-04,
            1.384565e-06,
            1.384565e-06,
            1.832241e-04,
            9.983679e-01,
            3.082213e-04,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [5.001878e-01, 4.998067e-01, 2.164534e-07, 5.215744e-06]
        ),
        "testfluo_2": np.array([4.998092e-01, 5.001901e-01, 6.926155e-07]),
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    assert prediction.transition_time_distributions is None
    assert prediction.lifetime_distributions is None
    assert prediction.mean_transition_times is None
    assert prediction.mean_lifetimes is None
    assert prediction.state_occupations is None


# test with 2 different fluorophores, no energy transfer, no bleaching
def test_prediction_6(tr_set_2f_diff):
    prediction = pr.Prediction(transition_set=tr_set_2f_diff)
    assert not prediction.energy_transfer
    assert not prediction.absorbing_chain
    assert prediction.transition_set == tr_set_2f_diff
    exp_freq_trans = np.array(
        [
            4.94846177e-01,
            1.33608468e-01,
            4.10722327e-04,
            4.10722327e-04,
            9.89692354e-03,
            9.89692354e-03,
            3.50930063e-01,
            4.99251123e-01,
            2.99550674e-01,
            1.49775337e-03,
            1.49775337e-03,
            1.98202696e-01,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [4.94846177e-01, 4.94846177e-01, 4.10722327e-04, 9.89692354e-03]
        ),
        "testfluo_2": np.array([0.49925112, 0.49925112, 0.00149775]),
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    for distr in prediction.transition_time_distributions:
        assert isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)
    for fluorophore, distr_col in prediction.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)

    exp_mean_trans_times = np.array(
        [
            1.71948334e-07,
            1.00000000e-09,
            1.00000000e-09,
            2.00000000e-04,
            1.00000000e-09,
            9.12888721e-06,
            1.00000000e-09,
            2.79534464e-07,
            3.00000000e-09,
            3.00000000e-09,
            1.00000000e-05,
            3.00000000e-09,
        ]
    )
    np.testing.assert_allclose(
        prediction.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.71948334e-07, 1.00000000e-09, 2.00000000e-04, 9.12888721e-06]
        ),
        "testfluo_2": np.array([2.79534464e-07, 3.00000000e-09, 1.00000000e-05]),
    }
    for fluorophore, means in prediction.mean_lifetimes.items():
        np.testing.assert_allclose(means, exp_mean_lifetimes[fluorophore], rtol=1e-6)
    exp_state_occ = {
        "testfluo_1": np.array([0.32970227, 0.00191745, 0.31829664, 0.35008363]),
        "testfluo_2": np.array([0.89441164, 0.00959894, 0.09598941]),
    }
    for fluorophore, occ in prediction.state_occupations.items():
        np.testing.assert_allclose(occ, exp_state_occ[fluorophore], rtol=1e-6)


# test with 1 fluorophore, with bleaching
def test_prediction_7(tr_set_1f_bl):
    with pytest.warns(
        UserWarning,
        match="absorbing states have a lifetime of inf and a frequency / occupation "
        "of 0. Absorbing transitions have a frequency of 0.",
    ):
        prediction = pr.Prediction(transition_set=tr_set_1f_bl)
    assert not prediction.energy_transfer
    assert prediction.absorbing_chain
    assert prediction.transition_set == tr_set_1f_bl
    exp_freq_trans = np.array(
        [
            4.94846218e-01,
            1.33608479e-01,
            4.10722361e-04,
            4.10640233e-04,
            9.89692435e-03,
            9.89692435e-03,
            3.50930092e-01,
            0.00000000e00,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [
                4.94846136e-01,
                4.94846218e-01,
                4.10722361e-04,
                9.89692435e-03,
                0.00000000e00,
            ]
        )
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-6)
    for distr in prediction.transition_time_distributions:
        assert isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)
    for fluorophore, distr_col in prediction.lifetime_distributions.items():
        for distr in distr_col:
            assert (
                isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)
                or distr == np.inf
            )
    exp_mean_trans_times = np.array(
        [
            1.71948334e-07,
            1.00000000e-09,
            1.00000000e-09,
            1.99960008e-04,
            1.00000000e-09,
            9.12888721e-06,
            1.00000000e-09,
            1.99960008e-04,
        ]
    )
    np.testing.assert_allclose(
        prediction.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.71948334e-07, 1.00000000e-09, 1.99960008e-04, 9.12888721e-06, np.inf]
        )
    }
    for fluorophore, means in prediction.mean_lifetimes.items():
        np.testing.assert_allclose(means, exp_mean_lifetimes[fluorophore], rtol=1e-6)
    exp_state_occ = {
        "testfluo_1": np.array([0.32972322, 0.00191757, 0.31825327, 0.35010594, 0.0])
    }
    for fluorophore, occ in prediction.state_occupations.items():
        np.testing.assert_allclose(occ, exp_state_occ[fluorophore], rtol=1e-6)


# test with 1 fluorophore, no bleaching
def test_prediction_8(tr_set_1f):
    prediction = pr.Prediction(transition_set=tr_set_1f)
    assert not prediction.energy_transfer
    assert not prediction.absorbing_chain
    assert prediction.transition_set == tr_set_1f
    exp_freq_trans = np.array(
        [
            4.94846177e-01,
            1.33608468e-01,
            4.10722327e-04,
            4.10722327e-04,
            9.89692354e-03,
            9.89692354e-03,
            3.50930063e-01,
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array(
            [4.94846177e-01, 4.94846177e-01, 4.10722327e-04, 9.89692354e-03]
        )
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-6)
    for distr in prediction.transition_time_distributions:
        assert isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)
    for fluorophore, distr_col in prediction.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, stats._distn_infrastructure.rv_continuous_frozen)
    exp_mean_trans_times = np.array(
        [
            1.71948334e-07,
            1.00000000e-09,
            1.00000000e-09,
            2.00000000e-04,
            1.00000000e-09,
            9.12888721e-06,
            1.00000000e-09,
        ]
    )
    np.testing.assert_allclose(
        prediction.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.71948334e-07, 1.00000000e-09, 2.00000000e-04, 9.12888721e-06]
        )
    }
    for fluorophore, means in prediction.mean_lifetimes.items():
        np.testing.assert_allclose(means, exp_mean_lifetimes[fluorophore], rtol=1e-6)
    exp_state_occ = {
        "testfluo_1": np.array([0.32970227, 0.00191745, 0.31829664, 0.35008363])
    }
    for fluorophore, occ in prediction.state_occupations.items():
        np.testing.assert_allclose(occ, exp_state_occ[fluorophore], rtol=1e-6)
