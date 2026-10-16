import logging
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.stats as stats

from fluopy import prediction as pr
from fluopy import simulation as si
from fluopy import transitions as tr

# test_prediction_# includes testing of...
# ...predict_transition_occurrences()
# ...predict_transition_occurrences_abs()
# ...predict_state_occurrences()
# ...predict_lifetimes()
# ...infer_stats()


# test with 3 fluorophores
def test_prediction_1(tr_set_bl_et_3f):
    with pytest.raises(
        ValueError, match="prediction not available for more than 2 fluorophores"
    ):
        pr.Prediction(transition_set=tr_set_bl_et_3f)


@pytest.mark.parametrize(
    "start_at, message",
    [
        ((), "number of starting states"),
        ((0, 1), "number of starting states"),
        ((-1,), "not a valid combined state"),
        ((1000,), "not a valid combined state"),
        ((1.5,), "not a valid combined state"),
    ],
)
def test_prediction_rejects_invalid_start_at(tr_set_1f, start_at, message):
    with pytest.raises(ValueError, match=message):
        pr.Prediction(tr_set_1f, start_at=start_at)


def test_connected_non_absorbing_prediction_is_independent_of_start_at(
    tr_set_1f,
):
    frequencies_0 = pr.Prediction(
        transition_set=tr_set_1f, start_at=(0,)
    ).frequency_transitions
    frequencies_1 = pr.Prediction(
        transition_set=tr_set_1f, start_at=(1,)
    ).frequency_transitions

    np.testing.assert_array_equal(frequencies_0, frequencies_1)


def test_prediction_defaults_to_same_ground_state_as_simulation(tr_set_1f_bl):
    prediction = pr.Prediction(tr_set_1f_bl)
    simulation = si.Simulation(tr_set_1f_bl)

    simulation.run(size=1, seed=42)

    assert prediction.start_at == (0,)
    np.testing.assert_array_equal(simulation.state_series[:, 0], prediction.start_at)


@pytest.mark.parametrize(
    "transition_set_fixture",
    ["tr_set_1f", "tr_set_1f_bl", "tr_set_et_2f_diff"],
)
def test_sparse_prediction_matches_dense(request, transition_set_fixture):
    dense_transition_set = request.getfixturevalue(transition_set_fixture)
    sparse_transition_set = dense_transition_set.with_matrix_format("sparse")

    dense_prediction = pr.Prediction(dense_transition_set)
    sparse_prediction = pr.Prediction(sparse_transition_set)

    np.testing.assert_allclose(
        sparse_prediction.frequency_transitions,
        dense_prediction.frequency_transitions,
    )
    for fluorophore in dense_prediction.frequency_states:
        np.testing.assert_allclose(
            sparse_prediction.frequency_states[fluorophore],
            dense_prediction.frequency_states[fluorophore],
        )


def test_self_transition_affects_events_but_not_state_statistics(
    tr_set_self_cycle,
):
    prediction = pr.Prediction(tr_set_self_cycle)

    np.testing.assert_allclose(
        prediction.frequency_transitions, [9 / 11, 1 / 11, 1 / 11]
    )
    np.testing.assert_allclose(prediction.frequency_states["testfluo_1"], [0.5, 0.5])
    np.testing.assert_allclose(prediction.mean_transition_times, [0.1, 0.1, 1.0])
    np.testing.assert_allclose(prediction.mean_lifetimes["testfluo_1"], [1.0, 1.0])
    np.testing.assert_allclose(prediction.state_occupations["testfluo_1"], [0.5, 0.5])


def test_self_transition_only_prediction(tr_set_self_cycle):
    transition_set = tr_set_self_cycle.filter_by_identity([1, 2])

    prediction = pr.Prediction(transition_set)

    np.testing.assert_array_equal(prediction.frequency_transitions, [1.0])
    np.testing.assert_array_equal(prediction.frequency_states["testfluo_1"], [1.0])
    np.testing.assert_allclose(prediction.mean_transition_times, [1 / 9])
    np.testing.assert_array_equal(prediction.mean_lifetimes["testfluo_1"], [np.inf])
    np.testing.assert_array_equal(prediction.state_occupations["testfluo_1"], [1.0])


def test_sparse_self_transition_only_prediction(tr_set_self_cycle):
    transition_set = tr_set_self_cycle.filter_by_identity([1, 2])

    prediction = pr.Prediction(transition_set.with_matrix_format("sparse"))

    np.testing.assert_array_equal(prediction.frequency_transitions, [1.0])
    np.testing.assert_array_equal(prediction.mean_lifetimes["testfluo_1"], [np.inf])


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize("self_rates", [(1, 9), (9, 1), (1, 0)])
@pytest.mark.parametrize("initial_states", [(0, 1), (1, 0)])
def test_persistent_state_statistics_do_not_depend_on_self_event_rates(
    flu_sys_2xcy5, matrix_format, self_rates, initial_states
):
    transitions = [
        tr.Transition(
            tr.TransitionType("SELF0", tr.SingleState.S0, tr.SingleState.S0, False),
            rate=self_rates[0],
            fluorophore_ids=[0, 1],
        ),
        tr.Transition(
            tr.TransitionType("SELF1", tr.SingleState.S1, tr.SingleState.S1, False),
            rate=self_rates[1],
            fluorophore_ids=[0, 1],
        ),
    ]
    transition_set = tr.TransitionSet(
        {"testfluo_1": transitions},
        flu_sys_2xcy5,
        keep_zero_rates=True,
        matrix_format=matrix_format,
    )
    prediction = pr.Prediction(transition_set, start_at=initial_states)

    np.testing.assert_allclose(
        prediction.frequency_transitions, np.asarray(self_rates) / sum(self_rates)
    )
    np.testing.assert_allclose(prediction.frequency_states["testfluo_1"], [0.5, 0.5])
    np.testing.assert_array_equal(
        prediction.mean_lifetimes["testfluo_1"], [np.inf, np.inf]
    )
    np.testing.assert_allclose(prediction.state_occupations["testfluo_1"], [0.5, 0.5])


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize("emission_rate", [0, 1])
def test_prediction_handles_retained_zero_rate_transitions(
    flu_sys_cy5, matrix_format, emission_rate
):
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(
                    tr.TransitionType.EXCITATION, rate=0, fluorophore_ids=[0]
                ),
                tr.Transition(
                    tr.TransitionType.FLUORESCENT_EMISSION,
                    rate=emission_rate,
                    fluorophore_ids=[0],
                ),
            ]
        },
        flu_sys_cy5,
        keep_zero_rates=True,
        matrix_format=matrix_format,
    )

    prediction = pr.Prediction(transition_set)

    emission_mean = 1 if emission_rate else np.inf
    np.testing.assert_array_equal(
        prediction.mean_transition_times, [np.inf, emission_mean]
    )
    np.testing.assert_array_equal(
        prediction.mean_lifetimes["testfluo_1"], [np.inf, emission_mean]
    )


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_same_name_fluorophores_share_lifetime_predictions(
    flu_sys_2xcy5, matrix_format
):
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(transition_type, rate, [0, 1])
                for transition_type, rate in [
                    (tr.TransitionType.EXCITATION, 2),
                    (tr.TransitionType.FLUORESCENT_EMISSION, 3),
                ]
            ]
        },
        flu_sys_2xcy5,
        matrix_format=matrix_format,
    )

    prediction = pr.Prediction(transition_set)

    np.testing.assert_allclose(prediction.mean_transition_times, [0.5, 1 / 3])
    np.testing.assert_allclose(prediction.mean_lifetimes["testfluo_1"], [0.5, 1 / 3])
    np.testing.assert_allclose(prediction.state_occupations["testfluo_1"], [0.6, 0.4])


def test_distinct_names_can_have_different_kinetics(flu_sys_1xcy5_1xatto643):
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(tr.TransitionType.EXCITATION, 1, [0]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [0]),
            ],
            "testfluo_2": [
                tr.Transition(tr.TransitionType.EXCITATION, 9, [1]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [1]),
            ],
        },
        flu_sys_1xcy5_1xatto643,
    )

    prediction = pr.Prediction(transition_set)

    np.testing.assert_allclose(prediction.mean_lifetimes["testfluo_1"], [1, 1])
    np.testing.assert_allclose(prediction.mean_lifetimes["testfluo_2"], [1 / 9, 1])


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize("absorbing", [False, True])
def test_paired_state_frequencies_use_common_event_counts(
    flu_sys_1xcy5_1xatto643, matrix_format, absorbing
):
    paired_excitation = tr.TransitionType(
        "ACTIVATE",
        tr.PairedState.S0_S0,
        tr.PairedState("S0_S1", tr.SingleState.S0, tr.SingleState.S1),
        False,
        "CUSTOM",
    )
    paired_transitions = [
        tr.Transition(paired_excitation, rate=1, fluorophore_ids=[(0, 1)])
    ]
    if absorbing:
        paired_termination = tr.TransitionType(
            "TERMINATE",
            tr.PairedState.S0_S0,
            tr.PairedState("B_B", tr.SingleState.B, tr.SingleState.B),
            False,
            "CUSTOM",
        )
        paired_transitions.append(
            tr.Transition(paired_termination, rate=1, fluorophore_ids=[(0, 1)])
        )
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(
                    tr.TransitionType(
                        "SELF", tr.SingleState.S0, tr.SingleState.S0, False
                    ),
                    rate=9,
                    fluorophore_ids=[0],
                )
            ],
            "testfluo_2": [
                tr.Transition(
                    tr.TransitionType.FLUORESCENT_EMISSION,
                    rate=1,
                    fluorophore_ids=[1],
                )
            ],
            "D: testfluo_1, A: testfluo_2, dist: 2.0": paired_transitions,
        },
        flu_sys_1xcy5_1xatto643,
        matrix_format=matrix_format,
    )
    prediction = pr.Prediction(transition_set, start_at=(0, 0))

    assert prediction.absorbing_chain is absorbing
    acceptor_states = transition_set.single_states["testfluo_2"]
    expected_state_frequencies = np.where(
        acceptor_states == tr.SingleState.B.value, 0, 0.5
    )
    np.testing.assert_allclose(
        prediction.frequency_states["testfluo_2"], expected_state_frequencies
    )
    paired_frequency = 1 / 19
    expected_transition_frequencies = [1 - paired_frequency, 1, paired_frequency]
    if absorbing:
        expected_transition_frequencies.append(0)
    np.testing.assert_allclose(
        prediction.frequency_transitions, expected_transition_frequencies
    )


def test_prediction_uses_only_the_reachable_subchain(flu_sys_cy5):
    intermediate = tr.SingleState("INTERMEDIATE_CUSTOM", 10)
    terminal = tr.SingleState("TERMINAL_CUSTOM", 11)
    path_transition = tr.TransitionType("PATH", tr.SingleState.T1, intermediate, False)
    terminal_transition = tr.TransitionType("TERMINAL", intermediate, terminal, False)
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(
                    tr.TransitionType.EXCITATION,
                    rate=1,
                    fluorophore_ids=[0],
                ),
                tr.Transition(
                    tr.TransitionType.FLUORESCENT_EMISSION,
                    rate=1,
                    fluorophore_ids=[0],
                ),
                tr.Transition(path_transition, rate=1, fluorophore_ids=[0]),
                tr.Transition(terminal_transition, rate=1, fluorophore_ids=[0]),
            ]
        },
        flu_sys_cy5,
    )

    cycle_prediction = pr.Prediction(transition_set, start_at=(0,))
    terminal_prediction = pr.Prediction(transition_set, start_at=(intermediate.value,))

    assert not cycle_prediction.absorbing_chain
    np.testing.assert_array_equal(
        cycle_prediction.frequency_transitions, [0.5, 0.5, 0, 0]
    )
    assert terminal_prediction.absorbing_chain
    np.testing.assert_array_equal(
        terminal_prediction.frequency_transitions, [0, 0, 0, 0]
    )


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_prediction_rejects_reachable_terminal_and_closed_class(
    flu_sys_cy5, matrix_format
):
    source = tr.SingleState("SOURCE_CUSTOM", 10)
    branch = tr.SingleState("BRANCH_CUSTOM", 11)
    cycle_1 = tr.SingleState("CYCLE_1_CUSTOM", 12)
    cycle_2 = tr.SingleState("CYCLE_2_CUSTOM", 13)
    terminal = tr.SingleState("TERMINAL_CUSTOM", 14)
    transition_types = [
        tr.TransitionType("START", source, branch, False),
        tr.TransitionType("TO_CYCLE", branch, cycle_1, False),
        tr.TransitionType("TO_TERMINAL", branch, terminal, False),
        tr.TransitionType("CYCLE_FORWARD", cycle_1, cycle_2, False),
        tr.TransitionType("CYCLE_BACK", cycle_2, cycle_1, False),
    ]
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(transition_type, rate=1, fluorophore_ids=[0])
                for transition_type in transition_types
            ]
        },
        flu_sys_cy5,
        matrix_format=matrix_format,
    )

    with pytest.raises(ValueError, match="closed nonterminal class"):
        pr.Prediction(transition_set, start_at=(branch.value,))

    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = transition_set
    method = prediction.predict_transition_occurrences_absorbing
    if matrix_format == "sparse":
        method = prediction.predict_transition_occurrences_absorbing_sparse
    with pytest.raises(ValueError, match="closed nonterminal class"):
        method(start_at=(branch.value,))


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_absorbing_prediction_requires_reachable_terminal_state(
    tr_set_1f, matrix_format
):
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = tr_set_1f.with_matrix_format(matrix_format)
    method = prediction.predict_transition_occurrences_absorbing
    if matrix_format == "sparse":
        method = prediction.predict_transition_occurrences_absorbing_sparse

    with pytest.raises(ValueError, match="no terminal state is reachable"):
        method(start_at=(0,))


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_prediction_rejects_multiple_recurrent_classes(flu_sys_cy5, matrix_format):
    source = tr.SingleState("SOURCE_CUSTOM", 10)
    branch = tr.SingleState("BRANCH_CUSTOM", 11)
    recurrent_1 = tr.SingleState("RECURRENT_1_CUSTOM", 12)
    recurrent_2 = tr.SingleState("RECURRENT_2_CUSTOM", 13)
    transition_types = [
        tr.TransitionType("START", source, branch, False),
        tr.TransitionType("TO_FIRST", branch, recurrent_1, False),
        tr.TransitionType("TO_SECOND", branch, recurrent_2, False),
        tr.TransitionType("FIRST_SELF", recurrent_1, recurrent_1, False),
        tr.TransitionType("SECOND_SELF", recurrent_2, recurrent_2, False),
    ]
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(transition_type, rate=1, fluorophore_ids=[0])
                for transition_type in transition_types
            ]
        },
        flu_sys_cy5,
        matrix_format=matrix_format,
    )

    with pytest.raises(ValueError, match="unique stationary distribution"):
        pr.Prediction(transition_set, start_at=(branch.value,))


@pytest.mark.parametrize("solver_failure", [RuntimeError, np.nan])
def test_sparse_prediction_rejects_stationary_solver_failure(
    tr_set_1f, monkeypatch, solver_failure
):
    def fail_or_return_nan(*args):
        if solver_failure is RuntimeError:
            raise RuntimeError
        return np.full(args[1].shape, np.nan)

    monkeypatch.setattr(pr, "spsolve", fail_or_return_nan)

    with pytest.raises(ValueError, match="unique stationary distribution"):
        pr.Prediction(tr_set_1f.with_matrix_format("sparse"))


@pytest.mark.parametrize(
    "matrix_format, solver_failure",
    [
        ("dense", np.linalg.LinAlgError),
        ("dense", np.nan),
        ("sparse", RuntimeError),
        ("sparse", np.nan),
    ],
)
def test_absorbing_prediction_rejects_solver_failure(
    tr_set_1f_bl, monkeypatch, matrix_format, solver_failure
):
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = tr_set_1f_bl.with_matrix_format(matrix_format)

    def fail_or_return_nan(*args):
        if isinstance(solver_failure, type):
            raise solver_failure
        return np.full(args[1].shape, np.nan)

    method = prediction.predict_transition_occurrences_absorbing
    if matrix_format == "dense":
        monkeypatch.setattr(pr.np.linalg, "solve", fail_or_return_nan)
    else:
        monkeypatch.setattr(pr, "spsolve", fail_or_return_nan)
        method = prediction.predict_transition_occurrences_absorbing_sparse

    with pytest.raises(ValueError, match="transient subchain is singular"):
        method()


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize(
    "start_at, expected_frequencies, expected_occupations",
    [
        ((0,), [2 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0]),
        ((1,), [1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0]),
    ],
)
def test_absorbing_prediction_uses_start_at(
    flu_sys_cy5,
    matrix_format,
    start_at,
    expected_frequencies,
    expected_occupations,
):
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(tr.TransitionType.EXCITATION, 1, [0]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [0]),
                tr.Transition(
                    tr.TransitionType(
                        "BLEACH", tr.SingleState.S1, tr.SingleState.B, False
                    ),
                    1,
                    [0],
                ),
            ]
        },
        flu_sys_cy5,
        matrix_format=matrix_format,
    )
    prediction = pr.Prediction(transition_set, start_at=start_at)

    np.testing.assert_allclose(prediction.frequency_transitions, expected_frequencies)
    np.testing.assert_allclose(
        prediction.state_occupations["testfluo_1"], expected_occupations
    )


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
@pytest.mark.parametrize("reverse_incoming_order", [False, True])
@pytest.mark.parametrize("start_at", [None, (1,)])
def test_absorbing_prediction_does_not_count_arbitrary_incoming_transition(
    flu_sys_cy5, matrix_format, reverse_incoming_order, start_at
):
    bleach = tr.TransitionType("BLEACH", tr.SingleState.S1, tr.SingleState.B, False)
    incoming = [
        tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 1, [0]),
        tr.Transition(tr.TransitionType.SINGLET_QUENCHING, 3, [0]),
    ]
    if reverse_incoming_order:
        incoming.reverse()
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(tr.TransitionType.EXCITATION, 1, [0]),
                *incoming,
                tr.Transition(bleach, 1, [0]),
            ]
        },
        flu_sys_cy5,
        matrix_format=matrix_format,
    )

    prediction = pr.Prediction(transition_set, start_at=start_at)

    excitation_count = 5 if start_at is None else 4
    expected_counts = {
        tr.TransitionType.EXCITATION: excitation_count,
        tr.TransitionType.FLUORESCENT_EMISSION: 1,
        tr.TransitionType.SINGLET_QUENCHING: 3,
        bleach: 0,
    }
    expected = np.array(
        [
            expected_counts[kind]
            for kind in transition_set.transition_df["transition_type"]
        ]
    )
    np.testing.assert_allclose(
        prediction.frequency_transitions, expected / expected.sum()
    )
    np.testing.assert_allclose(
        prediction.state_occupations["testfluo_1"],
        np.array([excitation_count, 1, 0]) / (excitation_count + 1),
    )


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_absorbing_prediction_started_in_absorbing_state_is_zero(
    tr_set_1f_bl, matrix_format
):
    transition_set = tr_set_1f_bl.with_matrix_format(matrix_format)
    start_at = sorted(transition_set.terminal_state_combinations)[0]

    prediction = pr.Prediction(transition_set, start_at=start_at)

    np.testing.assert_array_equal(
        prediction.frequency_transitions, np.zeros(tr_set_1f_bl.transition_df.shape[0])
    )


def test_prediction_warns_for_multiple_terminal_state_combinations(
    tr_set_bl_et_2f_diff, monkeypatch, caplog
):
    combinations = [(4, 3), (5, 3)]
    monkeypatch.setattr(
        pr.Prediction,
        "_get_terminal_state_combinations",
        lambda self: combinations,
    )
    monkeypatch.setattr(
        pr.Prediction,
        "predict_transition_occurrences_absorbing",
        lambda self, start_at=None: np.zeros(
            self.transition_set.transition_df.shape[0]
        ),
    )
    monkeypatch.setattr(
        pr.Prediction,
        "predict_state_occurrences",
        lambda self: {
            fluorophore: np.zeros(states.size)
            for fluorophore, states in self.transition_set.single_states.items()
        },
    )

    with caplog.at_level(logging.WARNING):
        pr.Prediction(transition_set=tr_set_bl_et_2f_diff)

    assert (
        "multiple terminal state combinations are available; predicted transition "
        "frequencies depend on start_at."
    ) in caplog.text


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_predicted_frequencies_remain_zero_without_expected_visits(
    tr_set_1f, matrix_format
):
    transition_matrix = np.zeros_like(tr_set_1f.transition_matrix)
    if matrix_format == "sparse":
        transition_matrix = pr.sparse.csr_array(transition_matrix)
    transition_set = SimpleNamespace(
        transition_matrix=transition_matrix,
        transition_df=tr_set_1f.transition_df,
        combined_state_transitions_df=tr_set_1f.combined_state_transitions_df,
        single_states=tr_set_1f.single_states,
    )
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = transition_set

    method = prediction.predict_transition_occurrences
    if matrix_format == "sparse":
        method = prediction.predict_transition_occurrences_sparse
    frequency_transitions = method()
    prediction.frequency_transitions = frequency_transitions
    frequency_states = prediction.predict_state_occurrences()

    np.testing.assert_array_equal(frequency_transitions, np.zeros(7))
    np.testing.assert_array_equal(frequency_states["testfluo_1"], np.zeros(4))


@pytest.mark.parametrize(
    "terminal_states",
    [
        [(4, 6), (4, 7), (5, 6), (5, 7)],
        [],
    ],
)
def test_get_terminal_state_combinations(terminal_states):
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = SimpleNamespace(
        terminal_state_combinations=frozenset(terminal_states),
    )

    combinations = prediction._get_terminal_state_combinations()

    assert combinations == terminal_states


def test_get_multiple_terminal_states_for_one_fluorophore():
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.transition_set = SimpleNamespace(
        terminal_state_combinations=frozenset({(4,), (5,)}),
    )

    combinations = prediction._get_terminal_state_combinations()

    assert combinations == [(4,), (5,)]


# test with 2 different fluorophores, energy transfer, bleaching
def test_prediction_2(tr_set_bl_et_2f_diff, caplog):
    with caplog.at_level(logging.WARNING):
        prediction = pr.Prediction(transition_set=tr_set_bl_et_2f_diff)
        assert (
            "Only frequencies are available for systems with paired transitions; "
            "lifetimes and occupations are not available." in caplog.text
        )
        assert (
            "absorbing states have a lifetime of inf and a frequency / occupation of 0. Absorbing transitions have a frequency of 0."
            in caplog.text
        )
    caplog.clear()

    assert prediction.paired_transitions
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
            [4.9999685023e-01, 4.9999685028e-01, 2.5101160787e-07, 6.0484724788e-06, 0]
        ),
        "testfluo_2": np.array(
            [4.9999749819e-01, 4.9999749824e-01, 5.0035678866e-06, 0]
        ),
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    assert prediction.transition_time_distributions is None
    assert prediction.lifetime_distributions is None
    assert prediction.mean_transition_times is None
    assert prediction.mean_lifetimes is None
    assert prediction.state_occupations is None


# test with 2 same fluorophores, energy transfer, bleaching
def test_prediction_3(tr_set_bl_et_2f_same, caplog):
    with caplog.at_level(logging.WARNING):
        prediction = pr.Prediction(transition_set=tr_set_bl_et_2f_same)
        assert (
            "Only frequencies are available for systems with paired transitions; "
            "lifetimes and occupations are not available." in caplog.text
        )
        assert (
            "absorbing states have a lifetime of inf and a frequency / occupation of 0. Absorbing transitions have a frequency of 0."
            in caplog.text
        )
    caplog.clear()

    assert prediction.paired_transitions
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
            [4.999999e-01, 4.999999e-01, 6.773337e-09, 1.632129e-07, 0.000000e00]
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
def test_prediction_4(tr_set_et_2f_diff, caplog):
    with caplog.at_level(logging.WARNING):
        prediction = pr.Prediction(transition_set=tr_set_et_2f_diff)
        assert (
            "Only frequencies are available for systems with paired transitions; "
            "lifetimes and occupations are not available." in caplog.text
        )
        caplog.clear()

    assert prediction.paired_transitions
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
            [4.9999728494e-01, 4.9999728494e-01, 2.1637091256e-07, 5.2137569291e-06]
        ),
        "testfluo_2": np.array([4.9999965356e-01, 4.9999965356e-01, 6.9287945081e-07]),
    }
    for fluorophore, freq in prediction.frequency_states.items():
        np.testing.assert_allclose(freq, exp_freq_states[fluorophore], rtol=1e-5)
    assert prediction.transition_time_distributions is None
    assert prediction.lifetime_distributions is None
    assert prediction.mean_transition_times is None
    assert prediction.mean_lifetimes is None
    assert prediction.state_occupations is None


# test with 2 different fluorophores, no energy transfer, no bleaching
def test_prediction_5(tr_set_2f_diff):
    prediction = pr.Prediction(transition_set=tr_set_2f_diff)
    assert not prediction.paired_transitions
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
    for _, distr_col in prediction.lifetime_distributions.items():
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
def test_prediction_6(tr_set_1f_bl, caplog):
    with caplog.at_level(logging.WARNING):
        prediction = pr.Prediction(transition_set=tr_set_1f_bl)
        assert (
            "absorbing states have a lifetime of inf and a frequency / occupation of 0. Absorbing transitions have a frequency of 0."
            in caplog.text
        )
        caplog.clear()

    assert not prediction.paired_transitions
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
    for _, distr_col in prediction.lifetime_distributions.items():
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
def test_prediction_7(tr_set_1f):
    prediction = pr.Prediction(transition_set=tr_set_1f)
    assert not prediction.paired_transitions
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
    for _, distr_col in prediction.lifetime_distributions.items():
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


def test_infer_stats_rejects_unavailable_lifetimes():
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.lifetime_distributions = None

    with pytest.raises(ValueError, match="unavailable for paired transitions"):
        prediction.infer_stats()


@pytest.mark.parametrize(
    "method, attribute, ylabel",
    [
        ("plot_frequency_transitions", "frequency_transitions", "Prob. occurrence"),
        ("plot_mean_transition_times", "mean_transition_times", r"$\tau$ (s)"),
    ],
)
def test_prediction_transition_bar_plots(pred_tr_set_1f, method, attribute, ylabel):
    expected = getattr(pred_tr_set_1f, attribute)

    ax = getattr(pred_tr_set_1f, method)(yscale="linear")

    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], expected)
    assert ax.get_yscale() == "linear"
    assert ax.get_ylabel() == ylabel
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["testfluo_1"]
    assert len(ax.get_xticklabels()) == expected.size


@pytest.mark.parametrize(
    "method, attribute, ylabel",
    [
        ("plot_frequency_states", "frequency_states", "Prob. occurrence"),
        ("plot_mean_lifetimes", "mean_lifetimes", r"$\tau$ (s)"),
        ("plot_state_occupations", "state_occupations", "Prob. occupation"),
    ],
)
def test_prediction_state_bar_plots(pred_tr_set_1f, method, attribute, ylabel):
    values_by_fluorophore = getattr(pred_tr_set_1f, attribute)
    expected = np.concatenate(list(values_by_fluorophore.values()))

    ax = getattr(pred_tr_set_1f, method)(yscale="linear")

    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], expected)
    assert ax.get_yscale() == "linear"
    assert ax.get_ylabel() == ylabel
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["testfluo_1"]
    assert len(ax.get_xticklabels()) == expected.size


def test_prediction_distribution_plots(pred_tr_set_1f):
    fluorophore = "testfluo_1"
    state_identity = int(pred_tr_set_1f.transition_set.single_states[fluorophore][0])
    state_index = 0

    lifetime_ax = pred_tr_set_1f.plot_lifetime_distributions(
        fluorophore=fluorophore, state_identity=state_identity
    )
    lifetime_x = lifetime_ax.lines[0].get_xdata()
    lifetime_distribution = pred_tr_set_1f.lifetime_distributions[fluorophore][
        state_index
    ]
    assert lifetime_x.size == 1000
    assert lifetime_x[-1] == pytest.approx(
        pred_tr_set_1f.mean_lifetimes[fluorophore][state_index] * 10
    )
    np.testing.assert_allclose(
        lifetime_ax.lines[0].get_ydata(), lifetime_distribution.pdf(lifetime_x)
    )
    assert lifetime_ax.get_xlabel() == "lifetime [s]"
    assert lifetime_ax.get_ylabel() == "PD"

    transition_x = np.array([0.0, 1e-9, 2e-9])
    transition_ax = pred_tr_set_1f.plot_transition_time_distributions(
        fluorophore=fluorophore, transition_id=0, x=transition_x
    )
    transition_distribution = pred_tr_set_1f.transition_time_distributions[0]
    np.testing.assert_array_equal(transition_ax.lines[0].get_xdata(), transition_x)
    np.testing.assert_allclose(
        transition_ax.lines[0].get_ydata(), transition_distribution.pdf(transition_x)
    )
    assert transition_ax.get_xlabel() == "time to transition [s]"
    assert transition_ax.get_ylabel() == "PD"

    default_transition_ax = pred_tr_set_1f.plot_transition_time_distributions(
        fluorophore=fluorophore, transition_id=0
    )
    default_transition_x = default_transition_ax.lines[0].get_xdata()
    assert default_transition_x.size == 1000
    assert default_transition_x[-1] == pytest.approx(
        pred_tr_set_1f.mean_transition_times[0] * 10
    )


def test_prediction_rejects_constant_lifetime_plot(pred_tr_set_1f_bl):
    fluorophore = "testfluo_1"
    absorbing_state = int(
        pred_tr_set_1f_bl.transition_set.single_states[fluorophore][-1]
    )

    with pytest.raises(ValueError, match="lifetimes are all equal to inf"):
        pred_tr_set_1f_bl.plot_lifetime_distributions(
            fluorophore=fluorophore, state_identity=absorbing_state
        )


@pytest.mark.parametrize(
    "method, arguments, message",
    [
        ("plot_mean_transition_times", {}, "mean_transition_times"),
        ("plot_mean_lifetimes", {}, "mean_lifetimes"),
        ("plot_state_occupations", {}, "state_occupations"),
        (
            "plot_lifetime_distributions",
            {"fluorophore": "testfluo_1", "state_identity": 0},
            "lifetime_distributions",
        ),
        (
            "plot_transition_time_distributions",
            {"fluorophore": "testfluo_1", "transition_id": 0},
            "transition_time_distributions",
        ),
    ],
)
def test_prediction_plots_reject_energy_transfer(method, arguments, message):
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.paired_transitions = True

    with pytest.raises(ValueError, match=message):
        getattr(prediction, method)(**arguments)


@pytest.mark.parametrize(
    "method, attribute, arguments, message",
    [
        (
            "plot_mean_transition_times",
            "mean_transition_times",
            {},
            "mean transition times are unavailable",
        ),
        (
            "plot_mean_lifetimes",
            "mean_lifetimes",
            {},
            "mean lifetimes are unavailable",
        ),
        (
            "plot_state_occupations",
            "state_occupations",
            {},
            "state occupations are unavailable",
        ),
        (
            "plot_lifetime_distributions",
            "lifetime_distributions",
            {"fluorophore": "testfluo_1", "state_identity": 0},
            "lifetime distributions are unavailable",
        ),
        (
            "plot_transition_time_distributions",
            "transition_time_distributions",
            {"fluorophore": "testfluo_1", "transition_id": 0},
            "transition-time distributions are unavailable",
        ),
    ],
)
def test_prediction_plots_reject_missing_statistics(
    method, attribute, arguments, message
):
    prediction = pr.Prediction.__new__(pr.Prediction)
    prediction.paired_transitions = False
    setattr(prediction, attribute, None)
    if method == "plot_lifetime_distributions":
        prediction.mean_lifetimes = {"testfluo_1": np.array([1.0])}
    elif method == "plot_transition_time_distributions":
        prediction.mean_transition_times = np.array([1.0])

    with pytest.raises(ValueError, match=message):
        getattr(prediction, method)(**arguments)
