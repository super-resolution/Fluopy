import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluopy import analysis as an
from fluopy import prediction as pr
from fluopy import simulation as si
from fluopy import transitions as tr

# test_analysis_# includes testing of...
# ...is_absorbing()
# ...get_transition_occurrences()
# ...get_state_occurrences()
# ...get_lifetimes()
# ...infer_stats()


@pytest.fixture()
def analysis_1f(sim_tr_set_1f_bl):
    return an.Analysis(simulation=sim_tr_set_1f_bl)


def test_analysis_requires_completed_simulation():
    simulation = SimpleNamespace(
        transition_series=None,
        state_series=None,
        time_series=None,
    )

    with pytest.raises(ValueError, match="simulation has not been run"):
        an.Analysis(simulation=simulation)


def test_analysis_warns_when_absorbing_state_was_reached(
    sim_tr_set_1f_bl, monkeypatch, caplog
):
    monkeypatch.setattr(an.Analysis, "is_absorbing", lambda self: True)

    with caplog.at_level(logging.WARNING):
        an.Analysis(simulation=sim_tr_set_1f_bl)

    assert "absolute state and transition frequency of 1" in caplog.text


# test with 1 fluorophore, with bleaching
def test_analysis_1(tr_set_1f_bl, tr_set_1f_bl_2, caplog):
    with caplog.at_level(logging.WARNING):
        pred_bl = pr.Prediction(transition_set=tr_set_1f_bl)
        pred_bl_2 = pr.Prediction(transition_set=tr_set_1f_bl_2)
        assert (
            "absorbing states have a lifetime of inf and a frequency / occupation "
            "of 0. Absorbing transitions have a frequency of 0."
        ) in caplog.text
    caplog.clear()

    with pytest.warns(
        si.FloatingPointPrecisionWarning,
        match="highest possible rate",
    ):
        sim_tr_set_1f_bl = si.Simulation(transition_set=tr_set_1f_bl)
        sim_tr_set_1f_bl.run(size=1000, seed=np.random.default_rng(1))

    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    assert analysis.simulation == sim_tr_set_1f_bl
    exp_freq_trans = np.array([0.495, 0.126, 0.0, 0.0, 0.01, 0.01, 0.359, 0.0])
    np.testing.assert_allclose(
        analysis.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array([0.4955045, 0.49450549, 0.0, 0.00999001, 0.0])
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-6)
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for _, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            1.69117671e-07,
            9.62991915e-10,
            np.nan,
            np.nan,
            9.05998220e-10,
            8.11900555e-06,
            9.44551601e-10,
            np.nan,
        ]
    )
    np.testing.assert_allclose(
        analysis.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.69117671e-07, 9.48466643e-10, np.nan, 8.11900555e-06, np.nan]
        )
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_allclose(
            mean_lifetimes, exp_mean_lifetimes[fluorophore], rtol=1e-6
        )
    exp_state_occ = {
        "testfluo_1": np.array(
            [0.5062093063386812, 0.0028389856535901, 0.0, 0.4909517080077286, 0.0]
        )
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_allclose(state_occ, exp_state_occ[fluorophore], rtol=1e-6)

    with caplog.at_level(logging.WARNING):
        analysis.plot_frequency_transitions(prediction=pred_bl_2)
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_frequency_transitions(prediction=pred_bl)

    with caplog.at_level(logging.WARNING):
        analysis.plot_frequency_states(prediction=pred_bl_2)
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_frequency_states(prediction=pred_bl)

    with caplog.at_level(logging.WARNING):
        analysis.plot_mean_transition_times(prediction=pred_bl_2)
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_mean_transition_times(prediction=pred_bl)

    with caplog.at_level(logging.WARNING):
        analysis.plot_mean_lifetimes(prediction=pred_bl_2)
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_mean_lifetimes(prediction=pred_bl)

    with caplog.at_level(logging.WARNING):
        analysis.plot_state_occupations(prediction=pred_bl_2)
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_state_occupations(prediction=pred_bl)

    with caplog.at_level(logging.WARNING):
        analysis.plot_lifetime_distributions(
            fluorophore="testfluo_1", state_identity=0, prediction=pred_bl_2
        )
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_lifetime_distributions(
        fluorophore="testfluo_1", state_identity=0, prediction=pred_bl
    )

    with caplog.at_level(logging.WARNING):
        analysis.plot_transition_time_distributions(
            fluorophore="testfluo_1", transition_id=0, prediction=pred_bl_2
        )
        assert "prediction uses a different TransitionSet object" in caplog.text
    caplog.clear()

    analysis.plot_transition_time_distributions(
        fluorophore="testfluo_1", transition_id=0, prediction=pred_bl
    )


def test_is_absorbing_is_determined_per_fluorophore(caplog, capsys):
    analysis = an.Analysis.__new__(an.Analysis)
    analysis.simulation = SimpleNamespace(
        transition_set=SimpleNamespace(
            absorbing_states={0: np.array([2]), 1: np.array([], dtype=np.int64)},
            states_by_value={2: SimpleNamespace(name="terminal")},
        )
    )
    analysis.state_series = np.array([[0, 2], [2, 1]], dtype=np.int64)

    with caplog.at_level(logging.INFO):
        reached_absorbing_state = analysis.is_absorbing()

    assert reached_absorbing_state
    assert (
        "fluorophore 0 has reached the Markovian absorbing state terminal"
        in caplog.text
    )
    assert capsys.readouterr().out == ""


def test_analysis_without_state_changes(tr_set_1f):
    simulation = SimpleNamespace(
        transition_set=tr_set_1f,
        transition_series=np.array([], dtype=np.int64),
        state_series=np.array([[0]], dtype=np.int64),
        time_series=np.array([0.0]),
    )

    analysis = an.Analysis(simulation=simulation)

    initial_state_index = np.where(tr_set_1f.single_states["testfluo_1"] == 0)[0][0]
    expected_state_frequencies = np.zeros(tr_set_1f.single_states["testfluo_1"].size)
    expected_state_frequencies[initial_state_index] = 1
    np.testing.assert_array_equal(
        analysis.frequency_states["testfluo_1"], expected_state_frequencies
    )
    assert all(
        distribution.size == 0
        for distribution in analysis.transition_time_distributions
    )
    assert all(
        distribution.size == 0
        for distribution in analysis.lifetime_distributions["testfluo_1"]
    )
    assert np.isnan(analysis.mean_transition_times).all()
    assert np.isnan(analysis.mean_lifetimes["testfluo_1"]).all()
    np.testing.assert_array_equal(
        analysis.state_occupations["testfluo_1"],
        np.zeros(expected_state_frequencies.size),
    )


def test_self_transition_resets_event_time_but_not_state_lifetime(
    tr_set_self_cycle,
):
    combined_transitions = tr_set_self_cycle.combined_state_transitions_df
    combined_ids = {
        transition_id: combined_transitions.index[
            combined_transitions["transition_id"] == transition_id
        ][0]
        for transition_id in range(3)
    }
    simulation = SimpleNamespace(
        transition_set=tr_set_self_cycle,
        transition_series=np.array(
            [combined_ids[0], combined_ids[0], combined_ids[1], combined_ids[2]]
        ),
        state_series=np.array([[0, 0, 0, 1, 0]]),
        time_series=np.array([0.0, 1.0, 2.0, 4.0, 7.0]),
    )
    baseline_simulation = SimpleNamespace(
        transition_set=tr_set_self_cycle,
        transition_series=np.array([combined_ids[1], combined_ids[2]]),
        state_series=np.array([[0, 1, 0]]),
        time_series=np.array([0.0, 4.0, 7.0]),
    )

    analysis = an.Analysis(simulation)
    baseline = an.Analysis(baseline_simulation)

    np.testing.assert_array_equal(analysis.transition_time_distributions[0], [1.0, 1.0])
    np.testing.assert_array_equal(analysis.transition_time_distributions[1], [2.0])
    np.testing.assert_array_equal(analysis.transition_time_distributions[2], [3.0])
    np.testing.assert_array_equal(
        analysis.lifetime_distributions["testfluo_1"][0], [4.0]
    )
    np.testing.assert_array_equal(
        analysis.lifetime_distributions["testfluo_1"][1], [3.0]
    )
    np.testing.assert_array_equal(
        analysis.mean_lifetimes["testfluo_1"],
        baseline.mean_lifetimes["testfluo_1"],
    )
    np.testing.assert_array_equal(
        analysis.state_occupations["testfluo_1"],
        baseline.state_occupations["testfluo_1"],
    )


def test_self_transition_only_analysis(tr_set_self_cycle):
    transition_set = tr_set_self_cycle.remove_transitions([1, 2])
    simulation = SimpleNamespace(
        transition_set=transition_set,
        transition_series=np.array([0, 0]),
        state_series=np.array([[0, 0, 0]]),
        time_series=np.array([0.0, 0.5, 1.0]),
    )

    analysis = an.Analysis(simulation)

    np.testing.assert_array_equal(analysis.transition_time_distributions[0], [0.5, 0.5])
    assert analysis.lifetime_distributions["testfluo_1"][0].size == 0
    assert np.isnan(analysis.mean_lifetimes["testfluo_1"][0])
    np.testing.assert_array_equal(analysis.state_occupations["testfluo_1"], [1.0])


@pytest.mark.parametrize("self_rate", [0, 9])
@pytest.mark.parametrize(
    "observation_end, expected", [(2.0, [0.75, 0.25]), (3.0, [5 / 6, 1 / 6])]
)
def test_state_occupations_include_unvisited_molecules_and_final_residence(
    flu_sys_2xcy5, self_rate, observation_end, expected
):
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(tr.TransitionType.EXCITATION, 1, [0, 1]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [0, 1]),
                tr.Transition(
                    tr.TransitionType(
                        "SELF", tr.SingleState.S0, tr.SingleState.S0, False
                    ),
                    self_rate,
                    [0, 1],
                ),
            ]
        },
        flu_sys_2xcy5,
        keep_zero_rates=True,
    )
    if self_rate:
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        states = np.array([[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
        transition_ids = [2, 0, 2, 1]
    else:
        times = [0.0, 1.0, 2.0]
        states = np.array([[0, 1, 0], [0, 0, 0]])
        transition_ids = [0, 1]
    combined_transitions = transition_set.combined_state_transitions_df
    simulated_transitions = [
        combined_transitions.index[
            (combined_transitions["transition_id"] == transition_id)
            & (combined_transitions["initial_state"] == tuple(initial_state))
        ][0]
        for transition_id, initial_state in zip(transition_ids, states.T[:-1])
    ]
    if observation_end > times[-1]:
        times.append(observation_end)
    simulation = SimpleNamespace(
        transition_set=transition_set,
        transition_series=np.asarray(simulated_transitions),
        state_series=states,
        time_series=np.asarray(times),
    )

    analysis = an.Analysis(simulation)

    np.testing.assert_allclose(analysis.state_occupations["testfluo_1"], expected)


def test_energy_transfer_time_is_recorded_only_for_donor_with_equal_source_states():
    energy_transfer_label = "D: A, A: B, dist: 1"
    transition_df = pd.DataFrame(
        {"initial_state": [SimpleNamespace(donor=SimpleNamespace(value=1))]},
        index=pd.MultiIndex.from_tuples([(energy_transfer_label, 0)]),
    )
    combined_transition_df = pd.DataFrame(
        {"transition_id": [0], "fluorophore_ids": [[0, 1]]}
    )
    transition_set = SimpleNamespace(
        transition_df=transition_df,
        combined_state_transitions_df=combined_transition_df,
        single_states={"A": np.array([0, 1]), "B": np.array([1, 2])},
        fluorophore_system=SimpleNamespace(
            fluorophores=[SimpleNamespace(name="A"), SimpleNamespace(name="B")]
        ),
    )
    analysis = an.Analysis.__new__(an.Analysis)
    analysis.simulation = SimpleNamespace(transition_set=transition_set)
    analysis.transition_series = np.array([0], dtype=np.int64)
    analysis.state_series = np.array([[1, 0], [1, 2]], dtype=np.int64)
    analysis.time_series = np.array([0.0, 5.0])

    transition_times, lifetimes = analysis.get_lifetimes()

    np.testing.assert_array_equal(transition_times[0], np.array([5.0]))
    np.testing.assert_array_equal(lifetimes["A"][1], np.array([5.0]))
    np.testing.assert_array_equal(lifetimes["B"][0], np.array([5.0]))


def test_transition_frequencies_are_zero_without_observed_transitions(tr_set_1f):
    analysis = an.Analysis.__new__(an.Analysis)
    analysis.simulation = SimpleNamespace(transition_set=tr_set_1f)
    analysis.transition_series = np.array([], dtype=np.int64)

    frequencies = analysis.get_transition_occurrences()

    np.testing.assert_array_equal(frequencies, np.zeros(7))


def test_plot_rejects_incompatible_transition_dimensions(sim_tr_set_1f_bl):
    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    prediction = SimpleNamespace(
        transition_set=sim_tr_set_1f_bl.transition_set,
        frequency_transitions=np.zeros(analysis.frequency_transitions.size - 1),
    )

    with pytest.raises(
        ValueError,
        match="prediction and simulation have incompatible transition dimensions",
    ):
        analysis.plot_frequency_transitions(prediction=prediction)


@pytest.mark.parametrize(
    "method, arguments, message",
    [
        (
            "plot_mean_transition_times",
            {},
            "predicted mean_transition_times not available",
        ),
        (
            "plot_mean_lifetimes",
            {},
            "predicted lifetime_distributions not available",
        ),
        (
            "plot_state_occupations",
            {},
            "predicted state_occupations not available",
        ),
        (
            "plot_lifetime_distributions",
            {"fluorophore": "testfluo_1", "state_identity": 0},
            "predicted lifetime_distributions not available",
        ),
        (
            "plot_transition_time_distributions",
            {"fluorophore": "testfluo_1", "transition_id": 0},
            "predicted transition_time_distributions not available",
        ),
    ],
)
def test_analysis_plots_reject_energy_transfer_prediction(
    analysis_1f, method, arguments, message
):
    prediction = SimpleNamespace(
        transition_set=analysis_1f.simulation.transition_set,
        paired_transitions=True,
    )

    with pytest.raises(ValueError, match=message):
        getattr(analysis_1f, method)(prediction=prediction, **arguments)


@pytest.mark.parametrize(
    "method, attribute, arguments, message",
    [
        (
            "plot_mean_transition_times",
            "mean_transition_times",
            {},
            "predicted mean transition times are unavailable",
        ),
        (
            "plot_mean_lifetimes",
            "mean_lifetimes",
            {},
            "predicted mean lifetimes are unavailable",
        ),
        (
            "plot_state_occupations",
            "state_occupations",
            {},
            "predicted state occupations are unavailable",
        ),
        (
            "plot_lifetime_distributions",
            "lifetime_distributions",
            {"fluorophore": "testfluo_1", "state_identity": 0},
            "predicted lifetime distributions are unavailable",
        ),
        (
            "plot_transition_time_distributions",
            "transition_time_distributions",
            {"fluorophore": "testfluo_1", "transition_id": 0},
            "predicted transition-time distributions are unavailable",
        ),
    ],
)
def test_analysis_plots_reject_unavailable_prediction_statistics(
    analysis_1f, method, attribute, arguments, message
):
    prediction = SimpleNamespace(
        transition_set=analysis_1f.simulation.transition_set,
        paired_transitions=False,
    )
    setattr(prediction, attribute, None)

    with pytest.raises(ValueError, match=message):
        getattr(analysis_1f, method)(prediction=prediction, **arguments)


@pytest.mark.parametrize(
    "method, prediction_attributes, message",
    [
        (
            "plot_frequency_states",
            {"frequency_states": {"testfluo_1": np.array([1.0])}},
            "incompatible state dimensions",
        ),
        (
            "plot_mean_transition_times",
            {"paired_transitions": False, "mean_transition_times": np.array([1.0])},
            "incompatible transition dimensions",
        ),
        (
            "plot_mean_lifetimes",
            {
                "paired_transitions": False,
                "mean_lifetimes": {"testfluo_1": np.array([1.0])},
            },
            "incompatible state dimensions",
        ),
        (
            "plot_state_occupations",
            {
                "paired_transitions": False,
                "state_occupations": {"testfluo_1": np.array([1.0])},
            },
            "incompatible state dimensions",
        ),
    ],
)
def test_analysis_plots_reject_incompatible_prediction_dimensions(
    analysis_1f, method, prediction_attributes, message
):
    prediction = SimpleNamespace(
        transition_set=analysis_1f.simulation.transition_set,
        **prediction_attributes,
    )

    with pytest.raises(ValueError, match=message):
        getattr(analysis_1f, method)(prediction=prediction)


def test_analysis_transition_plots_can_collapse_transfer_distances():
    transition_df = pd.DataFrame(
        {
            "abbreviation": ["A", "ET1", "ET2", "ET1", "ET2", "B"],
            "initial_state": ["S0", "S1_S0", "S1_S0", "S1_S0", "S1_S0", "S1"],
            "final_state": ["S1", "S0_S1", "S0_S1", "S0_S1", "S0_S1", "S0"],
            "photon": [False, False, False, False, False, True],
            "transition_type": [
                SimpleNamespace(mechanism=mechanism)
                for mechanism in [None, "FRET", "PET", "FRET", "PET", None]
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", 0),
                ("D: A, A: B, dist: 1", 1),
                ("D: A, A: B, dist: 1", 2),
                ("D: A, A: B, dist: 2", 3),
                ("D: A, A: B, dist: 2", 4),
                ("B", 5),
            ]
        ),
    )
    transition_set = SimpleNamespace(
        transition_df=transition_df,
        single_states={"A": np.array([0]), "B": np.array([0])},
    )
    analysis = an.Analysis.__new__(an.Analysis)
    analysis.simulation = SimpleNamespace(transition_set=transition_set)
    analysis.frequency_transitions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    analysis.transition_time_distributions = [
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        np.array([4.0, 6.0]),
        np.array([5.0, 7.0]),
        np.array([], dtype=np.float64),
    ]
    analysis.mean_transition_times = np.array([1.0, 2.0, 3.0, 5.0, 6.0, np.nan])

    frequency_ax = analysis.plot_frequency_transitions(diff_dist=False, yscale="linear")
    mean_ax = analysis.plot_mean_transition_times(diff_dist=False, yscale="linear")

    np.testing.assert_allclose(
        [patch.get_height() for patch in frequency_ax.patches],
        [0.1, 0.6, 0.8, 0.6],
    )
    np.testing.assert_allclose(
        [patch.get_height() for patch in mean_ax.patches],
        [1.0, 4.0, 5.0, np.nan],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "method, values",
    [
        ("plot_frequency_transitions", "frequency_transitions"),
        ("plot_mean_transition_times", "mean_transition_times"),
    ],
)
def test_analysis_transition_plots_use_transition_set_schema(
    sim_tr_set_et_2f_diff, method, values
):
    analysis = an.Analysis(sim_tr_set_et_2f_diff)

    ax = getattr(analysis, method)(diff_dist=False, yscale="linear")

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        getattr(analysis, values),
        equal_nan=True,
    )


# test with 2 fluorophores, with energy transfer
def test_analysis_2(tr_set_et_2f_diff):
    with pytest.warns(
        si.FloatingPointPrecisionWarning,
        match="highest possible rate",
    ):
        sim_tr_set_et_2f_diff = si.Simulation(transition_set=tr_set_et_2f_diff)
        sim_tr_set_et_2f_diff.run(
            size=1000,
            end_time=5e-7,
            seed=np.random.default_rng(1),
        )

    analysis = an.Analysis(simulation=sim_tr_set_et_2f_diff)
    assert analysis.simulation == sim_tr_set_et_2f_diff
    exp_freq_trans = np.array(
        [
            5.137426e-04,
            5.137426e-04,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            5.137426e-04,
            9.984588e-01,
            7.706139e-04,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            2.568713e-04,
            9.989725e-01,
            0.000000e00,
        ]
    )
    np.testing.assert_allclose(
        analysis.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array([0.50006424, 0.49993576, 0.0, 0.0]),
        "testfluo_2": np.array([0.50006424, 0.49993576, 0.0]),
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-6)
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for _, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            1.38043664e-07,
            2.69331086e-13,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            5.30726868e-13,
            4.89662687e-13,
            6.27453880e-08,
            np.nan,
            np.nan,
            np.nan,
            3.54344052e-13,
            4.51943853e-13,
            np.nan,
        ]
    )
    np.testing.assert_allclose(
        analysis.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array([1.19784225e-10, 4.89570542e-13, np.nan, np.nan]),
        "testfluo_2": np.array([1.19852792e-10, 4.51918763e-13, np.nan]),
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_allclose(
            mean_lifetimes, exp_mean_lifetimes[fluorophore], rtol=1e-6
        )
    exp_state_occ = {
        "testfluo_1": np.array([0.9961901620390533, 0.0038098379609468, 0.0, 0.0]),
        "testfluo_2": np.array([0.9964840720202729, 0.0035159279797271, 0.0]),
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_allclose(state_occ, exp_state_occ[fluorophore], rtol=1e-6)


# test with 2 fluorophores, without energy transfer
def test_analysis_3(tr_set_2f_diff):
    with pytest.warns(
        si.FloatingPointPrecisionWarning,
        match="highest possible rate",
    ):
        sim_tr_set_2f_diff = si.Simulation(transition_set=tr_set_2f_diff)
        sim_tr_set_2f_diff.run(size=1000, seed=np.random.default_rng(1))

    analysis = an.Analysis(simulation=sim_tr_set_2f_diff)
    assert analysis.simulation == sim_tr_set_2f_diff
    exp_freq_trans = np.array(
        [
            0.495,
            0.155,
            0.0,
            0.0,
            0.01,
            0.01,
            0.33,
            0.5,
            0.32,
            0.0,
            0.0,
            0.18,
        ]
    )
    np.testing.assert_allclose(
        analysis.frequency_transitions, exp_freq_trans, rtol=1e-6
    )
    exp_freq_states = {
        "testfluo_1": np.array([0.49625935, 0.49376559, 0.0, 0.00997506]),
        "testfluo_2": np.array([0.50083195, 0.49916805, 0.0]),
    }
    for fluorophore, freq in analysis.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-6)
    for time_distribution in analysis.transition_time_distributions:
        assert isinstance(time_distribution, np.ndarray)
    for _, distr_col in analysis.lifetime_distributions.items():
        for distr in distr_col:
            assert isinstance(distr, np.ndarray)
    exp_mean_trans_times = np.array(
        [
            1.89149553e-07,
            9.56660616e-10,
            np.nan,
            np.nan,
            1.21315873e-09,
            9.32848389e-06,
            9.71552909e-10,
            2.46557592e-07,
            3.06333031e-09,
            np.nan,
            np.nan,
            2.49423001e-09,
        ]
    )
    np.testing.assert_allclose(
        analysis.mean_transition_times, exp_mean_trans_times, rtol=1e-6
    )
    exp_mean_lifetimes = {
        "testfluo_1": np.array(
            [1.89149553e-07, 9.71770592e-10, np.nan, 9.32848389e-06]
        ),
        "testfluo_2": np.array([2.46557592e-07, 2.85845420e-09, np.nan]),
    }
    for fluorophore, mean_lifetimes in analysis.mean_lifetimes.items():
        np.testing.assert_allclose(
            mean_lifetimes, exp_mean_lifetimes[fluorophore], rtol=1e-6
        )
    exp_state_occ = {
        "testfluo_1": np.array(
            [0.4996348975778782, 0.0025669132844381, 0.0, 0.4977981891376837]
        ),
        "testfluo_2": np.array([0.9885597701926258, 0.0114402298073743, 0.0]),
    }
    for fluorophore, state_occ in analysis.state_occupations.items():
        np.testing.assert_allclose(state_occ, exp_state_occ[fluorophore], rtol=1e-6)


def test_get_fluorescence_lifetimes(tr_set_1f_bl, tr_set_2f_diff):
    with pytest.warns(
        si.FloatingPointPrecisionWarning,
        match="highest possible rate",
    ):
        sim_tr_set_1f_bl = si.Simulation(transition_set=tr_set_1f_bl)
        sim_tr_set_1f_bl.run(size=1000, seed=np.random.default_rng(1))
        sim_tr_set_2f_diff = si.Simulation(transition_set=tr_set_2f_diff)
        sim_tr_set_2f_diff.run(size=1000, seed=np.random.default_rng(1))

    assert tr.SingleState.S1.value == 1  # if it fails, check
    # analysis.get_fluorescence_lifetimes
    analysis = an.Analysis(simulation=sim_tr_set_1f_bl)
    fluorescence_lifetimes = analysis.get_fluorescence_lifetimes()
    np.testing.assert_allclose(
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
                4.72422934e-10,
                3.22910609e-09,
                4.44213999e-10,
                1.74494447e-10,
                1.91106997e-09,
            ]
        ),
        rtol=1e-6,
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


def test_get_emitting_transition_lifetimes():
    transition_df = pd.DataFrame(
        {"photon": [True, False, True]},
        index=pd.MultiIndex.from_tuples(
            [("A", 0), ("A", 1), ("B", 2)], names=["fluorophore", "identity"]
        ),
    )
    analysis = an.Analysis.__new__(an.Analysis)
    analysis.simulation = SimpleNamespace(
        transition_set=SimpleNamespace(
            single_states={"A": np.array([0, 1]), "B": np.array([0])},
            transition_df=transition_df,
        )
    )
    analysis.transition_time_distributions = [
        np.array([1.0, 2.0]),
        np.array([3.0]),
        np.array([4.0, 5.0]),
    ]

    np.testing.assert_array_equal(
        analysis.get_emitting_transition_lifetimes("A"), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        analysis.get_emitting_transition_lifetimes("B"), [4.0, 5.0]
    )

    analysis.simulation.transition_set.single_states = {"A": np.array([0, 1])}
    np.testing.assert_array_equal(
        analysis.get_emitting_transition_lifetimes(), [1.0, 2.0]
    )
    analysis.simulation.transition_set.single_states["B"] = np.array([0])

    with pytest.raises(
        ValueError, match="fluorophore wrong_name not found in transition dataframe."
    ):
        analysis.get_emitting_transition_lifetimes(fluorophore="wrong_name")

    with pytest.raises(
        ValueError,
        match="if multiple fluorophores are present, fluorophore must be specified.",
    ):
        analysis.get_emitting_transition_lifetimes()


def test_no_diff_dist():
    arrays = [
        [
            "Cy5",
            "Cy5",
            "Cy5",
            "D: Cy5, A: H, dist: 1",
            "D: Cy5, A: H, dist: 1",
            "D: Cy5, A: H, dist: 2",
            "D: Cy5, A: H, dist: 2",
            "D: Cy5, A: H, dist: 3",
            "D: Cy5, A: H, dist: 3",
            "H",
            "H",
            "H",
            "D: H, A: Cy5, dist: 2",
            "D: H, A: Cy5, dist: 2",
            "D: H, A: Cy5, dist: 1",
            "D: H, A: Cy5, dist: 1",
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ]
    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Group", "Number"])

    rng = np.random.default_rng(42)
    data = rng.standard_normal((16, 3))
    df = pd.DataFrame(data, index=index, columns=["X", "Y", "Z"])
    df["abbreviation"] = [
        "A",
        "B",
        "C",
        "ET1",
        "ET2",
        "ET1",
        "ET2",
        "ET1",
        "ET2",
        "D",
        "E",
        "F",
        "ET1",
        "ET2",
        "ET1",
        "ET2",
    ]
    df["initial_state"] = "initial"
    df["final_state"] = "final"
    df["photon"] = False
    df["transition_type"] = [
        SimpleNamespace(mechanism=mechanism)
        for mechanism in [None] * 3 + ["FRET"] * 6 + [None] * 3 + ["FRET"] * 4
    ]

    df2, dict2, dict2_vals = an.no_diff_dist(df, ["Cy5", "H"])
    assert df2.shape[0] == 10
    assert df2.index.get_level_values(0).unique().tolist() == [
        "Cy5",
        "D: Cy5, A: H, dist: 1",
        "H",
        "D: H, A: Cy5, dist: 2",
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
    dict2_vals_exp = np.array([5, 7, 6, 8, 14, 15])
    np.testing.assert_array_equal(dict2_vals, dict2_vals_exp)


def test_no_diff_dist_keeps_single_distance_pair():
    index = pd.MultiIndex.from_tuples(
        [("Cy5", 0), ("D: Cy5, A: H, dist: 1", 1)],
        names=["Group", "Number"],
    )
    transition_df = pd.DataFrame(
        {
            "rate": [1.0, 2.0],
            "abbreviation": ["EXC", "FRET"],
            "initial_state": ["S0", "S1_S0"],
            "final_state": ["S1", "S0_S1"],
            "photon": [False, False],
            "transition_type": [tr.TransitionType.EXCITATION, tr.TransitionType.FRET],
        },
        index=index,
    )

    collapsed, discarded_by_position, discarded = an.no_diff_dist(
        transition_df, ["Cy5", "H"]
    )

    pd.testing.assert_frame_equal(
        collapsed.reset_index(drop=True), transition_df.reset_index(drop=True)
    )
    assert (
        collapsed.index.get_level_values(0).tolist()
        == index.get_level_values(0).tolist()
    )
    assert discarded_by_position == {}
    np.testing.assert_array_equal(discarded, np.array([], dtype=np.int64))


@pytest.mark.parametrize("second_pet_rate", [0, 2])
def test_no_diff_dist_matches_retained_types_across_distances(
    flu_sys_2xcy5_1xatto643, second_pet_rate
):
    pet = tr.TransitionType(
        "FRET", tr.PairedState.S1_S0, tr.PairedState.S0_S1, False, "PET"
    )
    transition_set = tr.TransitionSet(
        {
            "D: testfluo_1, A: testfluo_2, dist: 2.0": [
                tr.Transition(tr.TransitionType.FRET, 1, [(0, 2)]),
                tr.Transition(pet, 2, [(0, 2)]),
            ],
            "D: testfluo_1, A: testfluo_2, dist: 1.0": [
                tr.Transition(pet, second_pet_rate, [(1, 2)]),
                tr.Transition(tr.TransitionType.FRET, 9, [(1, 2)]),
            ],
        },
        flu_sys_2xcy5_1xatto643,
        keep_zero_rates=True,
    )

    collapsed, discarded_by_position, discarded = an.no_diff_dist(
        transition_set.transition_df, ["testfluo_1", "testfluo_2"]
    )

    assert collapsed["transition_type"].tolist() == [tr.TransitionType.FRET, pet]
    assert collapsed.index.get_level_values(0).tolist() == [
        "D: testfluo_1, A: testfluo_2, dist: 2.0",
        "D: testfluo_1, A: testfluo_2, dist: 2.0",
    ]
    pd.testing.assert_index_equal(
        discarded_by_position[0],
        pd.Index([3], name="identity"),
    )
    pd.testing.assert_index_equal(
        discarded_by_position[1], pd.Index([2], name="identity")
    )
    np.testing.assert_array_equal(np.sort(discarded), [2, 3])


@pytest.mark.parametrize(
    "index, message",
    [
        (
            pd.MultiIndex.from_tuples([("A", 0, "extra")]),
            "two-level index",
        ),
        (
            pd.MultiIndex.from_tuples([("A", "invalid")]),
            "identity must be an integer",
        ),
    ],
)
def test_no_diff_dist_rejects_invalid_index(index, message):
    transition_df = pd.DataFrame(index=index)

    with pytest.raises(TypeError, match=message):
        an.no_diff_dist(transition_df, ["A"])


def test_no_diff_dist_keeps_pair_with_unknown_donor():
    index = pd.MultiIndex.from_tuples([("D: X, A: B, dist: 1", 0)])
    transition_df = pd.DataFrame(
        {
            "abbreviation": ["PAIR"],
            "initial_state": ["S1_S0"],
            "final_state": ["S0_S1"],
            "photon": [False],
            "transition_type": [SimpleNamespace(mechanism="CUSTOM")],
        },
        index=index,
    )

    collapsed, discarded_by_position, discarded = an.no_diff_dist(
        transition_df, ["A", "B"]
    )

    pd.testing.assert_frame_equal(collapsed, transition_df)
    assert discarded_by_position == {}
    np.testing.assert_array_equal(discarded, np.array([], dtype=np.int64))


def test_get_absorbing_state_times(sim_tr_set_1f_bl):
    absorbing_state_times = an.get_absorbing_state_times(simulation=sim_tr_set_1f_bl)
    assert np.isnan(absorbing_state_times)


def test_get_absorbing_state_times_2(sim_tr_set_2f_diff):
    absorbing_state_times = an.get_absorbing_state_times(simulation=sim_tr_set_2f_diff)
    assert np.all(np.isnan(absorbing_state_times))


@pytest.mark.parametrize(
    "state_series, time_series",
    [(None, np.array([0.0])), (np.array([[0]]), None)],
)
def test_get_absorbing_state_times_requires_completed_simulation(
    state_series, time_series
):
    simulation = SimpleNamespace(
        state_series=state_series,
        time_series=time_series,
    )

    with pytest.raises(ValueError, match="completed simulation"):
        an.get_absorbing_state_times(simulation)


def test_get_absorbing_state_times_with_shared_destination():
    ground = tr.SingleState.S0.value
    bleached = tr.SingleState.B.value
    simulation = SimpleNamespace(
        transition_set=SimpleNamespace(
            absorbing_states={identity: np.array([bleached]) for identity in range(3)}
        ),
        state_series=np.array(
            [
                [ground, ground, bleached, bleached],
                [ground, bleached, bleached, bleached],
                [ground, ground, ground, ground],
            ],
            dtype=np.int8,
        ),
        time_series=np.array([0.0, 2.0, 5.0, 8.0]),
    )

    absorbing_state_times = an.get_absorbing_state_times(simulation)

    np.testing.assert_array_equal(absorbing_state_times, [2.0, 5.0, np.nan])


def test_get_absorbing_state_times_with_distinct_destinations():
    simulation = SimpleNamespace(
        transition_set=SimpleNamespace(
            absorbing_states={
                0: np.array([tr.SingleState.B.value, tr.SingleState.OFF.value])
            }
        ),
        state_series=np.array([[tr.SingleState.S0.value, tr.SingleState.B.value]]),
        time_series=np.array([0.0, 1.0]),
    )

    np.testing.assert_array_equal(an.get_absorbing_state_times(simulation), [1.0])


def test_get_delta_absorbing_state_times():
    absorbing_state_times = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )

    deltas = an.get_delta_absorbing_state_times(
        absorbing_state_times=absorbing_state_times
    )

    expected = np.array(
        [
            [1, 10],
            [1, 10],
            [1, 10],
        ]
    )
    assert np.array_equal(deltas, expected)
