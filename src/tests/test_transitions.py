import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import copy
import warnings
import numpy as np
import pandas as pd
import transitions as tr


def test_singlestate():
    assert tr.SingleState.S0.value == 0
    assert tr.SingleState.S1.value == 1
    assert tr.SingleState.S2.value == 2
    assert tr.SingleState.T1.value == 3
    assert tr.SingleState.T2.value == 4
    assert tr.SingleState.Cis.value == 5
    assert tr.SingleState.OFF.value == 6
    assert tr.SingleState.B.value == 7


def test_pairedstate():
    assert tr.PairedState.S1_S0.value == [tr.SingleState.S1, tr.SingleState.S0]
    assert tr.PairedState.S1_S0.single_state_values == (1, 0)
    assert tr.PairedState.S1_S0.acceptor == tr.SingleState.S0
    assert tr.PairedState.S1_S0.donor == tr.SingleState.S1


def test_transitiontype():
    assert tr.TransitionType.EXCITATION.abbreviation == 'EXC'
    assert tr.TransitionType.EXCITATION.initial_state == tr.SingleState.S0
    assert tr.TransitionType.EXCITATION.final_state == tr.SingleState.S1
    assert tr.TransitionType.EXCITATION.photon is False


@pytest.mark.parametrize('transition_object,expected',
                         [[[tr.TransitionType.EXCITATION, 5],
                           [tr.TransitionType.EXCITATION, 5, tr.TransitionType.EXCITATION.abbreviation,
                            tr.TransitionType.EXCITATION.initial_state, tr.TransitionType.EXCITATION.final_state,
                            tr.TransitionType.EXCITATION.photon, None]],
                          [[tr.TransitionType.HOMO_FRET, 6.1, 2.7509],
                           [tr.TransitionType.HOMO_FRET, 6.1, tr.TransitionType.HOMO_FRET.abbreviation + '(2.8)',
                            tr.TransitionType.HOMO_FRET.initial_state, tr.TransitionType.HOMO_FRET.final_state,
                            tr.TransitionType.HOMO_FRET.photon, 2.751]]],
                         indirect=['transition_object'])
def test_transition(transition_object, expected):
    assert transition_object.transition_type == expected[0]
    assert transition_object.rate == expected[1]
    assert transition_object.abbreviation == expected[2]
    assert transition_object.initial_state == expected[3]
    assert transition_object.final_state == expected[4]
    assert transition_object.photon == expected[5]
    assert transition_object.id is None
    if expected[6] is None:
        assert transition_object.distance is None
        assert not transition_object.energy_transfer
    else:
        assert transition_object.distance == expected[6]
        assert transition_object.energy_transfer
        with pytest.raises(AttributeError):
            tr.Transition(transition_object.transition_type, transition_object.rate)


@pytest.mark.parametrize('transitionlist,expected',
                         [[[[tr.TransitionType.EXCITATION, 5], [tr.TransitionType.HOMO_FRET, 6.1, 2.75]],
                           np.array([0, 1])],
                          [[[tr.TransitionType.EXCITATION, 5], [tr.TransitionType.ISOMERIZATION, 2]],
                           np.array([0, 1, 5])]],
                         indirect=['transitionlist'])
def test_get_single_states(transitionlist, expected):
    np.testing.assert_array_equal(tr.get_single_states(transitions=transitionlist), expected)


def test_transitionset(transition_set_object):
    new_transitions = transition_set_object.transitions
    old_transitions = copy.deepcopy(transition_set_object.transitions)
    for i, transition in enumerate(transition_set_object.transitions):
        assert transition.id == i
        assert transition.rate != 0
        if transition.distance == 7:
            transition.distance = 6
    np.testing.assert_array_equal(transition_set_object.single_states, np.array([0, 1]))
    assert transition_set_object.combined_state_transitions_df is None
    assert transition_set_object.transition_matrix is None
    assert transition_set_object.row_sums is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr.TransitionSet(transitions=old_transitions, fluorophore_system=transition_set_object.fluorophore_system)
    with pytest.warns(UserWarning):
        tr.TransitionSet(transitions=new_transitions, fluorophore_system=transition_set_object.fluorophore_system)


@pytest.mark.parametrize('remove_list,expected',
                         [[[tr.TransitionType.EXCITATION.abbreviation, tr.TransitionType.HOMO_FRET.abbreviation],
                           [tr.TransitionType.FLUORESCENT_EMISSION.abbreviation,
                            tr.TransitionType.INTERNAL_CONVERSION_S.abbreviation,
                            tr.TransitionType.INTERSYSTEM_CROSSING_ST.abbreviation]],
                          [[tr.TransitionType.HOMO_FRET.abbreviation + '(7.0)'],
                           [tr.TransitionType.EXCITATION.abbreviation,
                            tr.TransitionType.FLUORESCENT_EMISSION.abbreviation,
                            tr.TransitionType.INTERNAL_CONVERSION_S.abbreviation,
                            tr.TransitionType.INTERSYSTEM_CROSSING_ST.abbreviation,
                            tr.TransitionType.HOMO_FRET.abbreviation + '(9.9)']]])
def test_filter_by_abbreviation(transition_set_object, remove_list, expected):
    transition_set = transition_set_object.filter_by_abbreviation(remove_list)
    for transition in transition_set.transitions:
        assert transition.abbreviation in expected


@pytest.mark.parametrize('change_dict,expected',
                         [[{'FLU': 2}, [5, 2, 1, 9, 9]],
                          [{'FLU': 0}, [5, 1, 9, 9]],
                          [{'HFRET(9.9)': 2.2}, [5, 4, 1, 9, 2.2]]])
def test_adjust_rates(transition_set_object, change_dict, expected):
    transition_set = transition_set_object.adjust_rates(change_dict)
    for transition, expected_rate in zip(transition_set.transitions, expected):
        assert transition.rate == expected_rate


@pytest.mark.parametrize('parameters,expected',
                         [[[[0, 1, 2], 2], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]],
                          [[[0, 1, 2], 1], [(0,), (1,), (2,)]]])
def test_get_state_combinations(parameters, expected):
    state_combinations = tr.get_state_combinations(*parameters)
    assert state_combinations == expected


def test_get_combined_state_transitions():
    state_combinations = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    combined_state_transitions = tr.get_combined_state_transitions(state_combinations)
    assert combined_state_transitions == [((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 0), (1, 0)),
                                           ((0, 0), (1, 1)), ((0, 0), (1, 2)), ((0, 0), (2, 0)), ((0, 0), (2, 1)),
                                           ((0, 0), (2, 2)), ((0, 1), (0, 0)), ((0, 1), (0, 1)), ((0, 1), (0, 2)),
                                           ((0, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 1), (1, 2)), ((0, 1), (2, 0)),
                                           ((0, 1), (2, 1)), ((0, 1), (2, 2)), ((0, 2), (0, 0)), ((0, 2), (0, 1)),
                                           ((0, 2), (0, 2)), ((0, 2), (1, 0)), ((0, 2), (1, 1)), ((0, 2), (1, 2)),
                                           ((0, 2), (2, 0)), ((0, 2), (2, 1)), ((0, 2), (2, 2)), ((1, 0), (0, 0)),
                                           ((1, 0), (0, 1)), ((1, 0), (0, 2)), ((1, 0), (1, 0)), ((1, 0), (1, 1)),
                                           ((1, 0), (1, 2)), ((1, 0), (2, 0)), ((1, 0), (2, 1)), ((1, 0), (2, 2)),
                                           ((1, 1), (0, 0)), ((1, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (1, 0)),
                                           ((1, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 1), (2, 0)), ((1, 1), (2, 1)),
                                           ((1, 1), (2, 2)), ((1, 2), (0, 0)), ((1, 2), (0, 1)), ((1, 2), (0, 2)),
                                           ((1, 2), (1, 0)), ((1, 2), (1, 1)), ((1, 2), (1, 2)), ((1, 2), (2, 0)),
                                           ((1, 2), (2, 1)), ((1, 2), (2, 2)), ((2, 0), (0, 0)), ((2, 0), (0, 1)),
                                           ((2, 0), (0, 2)), ((2, 0), (1, 0)), ((2, 0), (1, 1)), ((2, 0), (1, 2)),
                                           ((2, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 0), (2, 2)), ((2, 1), (0, 0)),
                                           ((2, 1), (0, 1)), ((2, 1), (0, 2)), ((2, 1), (1, 0)), ((2, 1), (1, 1)),
                                           ((2, 1), (1, 2)), ((2, 1), (2, 0)), ((2, 1), (2, 1)), ((2, 1), (2, 2)),
                                           ((2, 2), (0, 0)), ((2, 2), (0, 1)), ((2, 2), (0, 2)), ((2, 2), (1, 0)),
                                           ((2, 2), (1, 1)), ((2, 2), (1, 2)), ((2, 2), (2, 0)), ((2, 2), (2, 1)),
                                           ((2, 2), (2, 2))]


@pytest.mark.parametrize('transition_object,expected',
                         [[[tr.TransitionType.EXCITATION, 8], [[(0, 0), (0, 1), 'EXC', None, 8, False],
                                                               [(0, 1), (1, 1), 'EXC', None, 8, False]]],
                          [[tr.TransitionType.OXIDATION, 7], []]],
                         indirect=['transition_object'])
def test_rate_assignment_standard(transition_object, expected):
    combined_state_transitions = [((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 1), (1, 1)),
                                  ((0, 0), (1, 1)), ((0, 3), (2, 3))]
    transition_rate_list = tr.rate_assignment_standard(transition_object, [], combined_state_transitions)
    assert transition_rate_list == expected


@pytest.mark.parametrize('transition_object,distance_lookup,expected',
                         [[[tr.TransitionType.HOMO_FRET, 8, 4], {(0, 1): 5, (1, 0): 5, (0, 2): 4, (2, 0): 4, (1, 2): 7,
                                                                 (2, 1): 7},
                          [[(0, 0, 1), (1, 0, 0), 'HFRET(4.0)', None, 8, False]]],
                          [[tr.TransitionType.HOMO_FRET, 8, 4], {(0, 1): 5, (1, 0): 5, (0, 2): 4, (1, 2): 7, (2, 1): 7},
                           'KeyError'],
                          [[tr.TransitionType.HOMO_FRET, 8, 2], {(0, 1): 5, (1, 0): 5, (0, 2): 4, (2, 0): 4, (1, 2): 7,
                                                                 (2, 1): 7},
                          []]],
                         indirect=['transition_object'])
def test_rate_assignment_energy_transfer(transition_object, distance_lookup, expected):
    combined_state_transitions = [((0, 0, 0), (0, 0, 0)), ((0, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 0, 0)),
                                  ((2, 0, 1), (1, 0, 0))]
    if expected == 'KeyError':
        with pytest.raises(KeyError):
            tr.rate_assignment_energy_transfer(transition_object, [], combined_state_transitions, distance_lookup)
    else:
        transition_rate_list = tr.rate_assignment_energy_transfer(transition_object, [], combined_state_transitions,
                                                                  distance_lookup)
        assert transition_rate_list == expected


def test_construct_transition_rate_list():
    transitions = [tr.Transition(tr.TransitionType.EXCITATION, 8), tr.Transition(tr.TransitionType.HOMO_FRET, 7, 4)]
    combined_state_transitions = [((0, 0, 0), (0, 0, 0)), ((0, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 0, 0)),
                                  ((2, 0, 1), (1, 0, 0)), ((0, 2, 0), (0, 2, 1))]
    distance_lookup = {(0, 1): 5, (1, 0): 5, (0, 2): 4, (2, 0): 4, (1, 2): 7, (2, 1): 7}
    transition_rate_list = tr.construct_transition_rate_list(transitions, combined_state_transitions, distance_lookup)
    assert transition_rate_list == [[(0, 2, 0), (0, 2, 1), 'EXC', None, 8, False],
                                    [(0, 0, 1), (1, 0, 0), 'HFRET(4.0)', None, 7, False]]


def test_construct_transition_matrix():
    transition_rate_list = [[(0, 2, 0), (0, 2, 1), 'EXC', None, 8, False],
                            [(0, 0, 1), (1, 0, 0), 'HFRET(4.0)', None, 7, False],
                            [(1, 0, 0), (2, 0, 0), 'ISC_ST', None, 5, False],
                            [(2, 0, 0), (0, 0, 0), 'ISC_TS', None, 1, False],
                            [(0, 2, 1), (0, 2, 0), 'FLU', None, 10, True],
                            [(0, 2, 1), (0, 2, 2), 'ISC_ST', None, 8, False]]
    combined_state_transitions_df = pd.DataFrame(transition_rate_list, columns=['initial_state', 'final_state',
                                                                                'abbreviation', 'transition_id', 'rate',
                                                                                'photon'])
    combined_state_transitions_df.index.name = 'id'
    transition_matrix, row_sums = tr.construct_transition_matrix(combined_state_transitions_df)
    expected_tm = np.array([[0, 0, 0, 0, 10/18, 8/18],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])
    expected_row_sums = np.array([18, 5, 1, 0, 8, 0])
    np.testing.assert_allclose(transition_matrix, expected_tm)
    np.testing.assert_allclose(row_sums, expected_row_sums)


def test_finalize(transition_set_object):
    transition_set_object.finalize()
    assert transition_set_object.combined_state_transitions_df is not None
    assert transition_set_object.transition_matrix is not None
    assert transition_set_object.row_sums is not None


@pytest.mark.parametrize('parameters,expected',
                         [[['', '', None, None], 'AttributeError'],
                          [['', '', [], []], 'AttributeError'],
                          [[tr.TransitionType.HOMO_FRET, {(0, 1): 5, (1, 0): 5, (0, 2): 4, (2, 0): 4, (1, 2): 7,
                                                          (2, 1): 7}, [3, 0, 6], None],
                           [tr.Transition(tr.TransitionType.HOMO_FRET, 6, 4),
                            tr.Transition(tr.TransitionType.HOMO_FRET, 3, 5),
                            tr.Transition(tr.TransitionType.HOMO_FRET, 0, 7)]],
                           [[tr.TransitionType.HOMO_FRET, {(0, 1): 5, (1, 0): 5, (0, 2): 4, (2, 0): 4, (1, 2): 7,
                                                          (2, 1): 7}, None,
                            {'emission_rate': 5, 'spectral_overlap_integral': 2, 'dipole_orientation_factor': 0.4}],
                           [tr.Transition(tr.TransitionType.HOMO_FRET, 8.5791015625e-14, 4),
                            tr.Transition(tr.TransitionType.HOMO_FRET, 2.24896e-14, 5),
                            tr.Transition(tr.TransitionType.HOMO_FRET, 2.98685071696317e-15, 7)]]])
def test_get_energy_transfer_transitions(parameters, expected):
    if expected == 'AttributeError':
        with pytest.raises(AttributeError):
            tr.get_energy_transfer_transitions(*parameters)
    else:
        energy_transfer_transitions = tr.get_energy_transfer_transitions(*parameters)
        assert energy_transfer_transitions == expected


@pytest.mark.parametrize('fluorophore_system_object,parameters,expected',
                         [[[['Atto488', [0, 0]], ['Cy5', [0, 1]]], [2, 600, False], 'ValueError'],
                          [[['Cy5', [0, 0]], ['Cy5', [0, 1]], ['Cy5', [1, 1]]], [2, 600, False], []],
                          [[['Cy5', [0, 0]], ['Cy5', [0, 1]], ['Cy5', [1, 1]]], [2, 600, True], []]],
                         indirect=['fluorophore_system_object'])
def test_load_transitions(fluorophore_system_object, parameters, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            tr.load_transitions(fluorophore_system_object, *parameters)
    else:
        transitions = tr.load_transitions(fluorophore_system_object, *parameters)
        index = 11
        expected = [tr.Transition(tr.TransitionType.EXCITATION, 1946561.1940894993),
                    tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 2.7e+08),
                    tr.Transition(tr.TransitionType.INTERSYSTEM_CROSSING_ST, 8.3e5),
                    tr.Transition(tr.TransitionType.INTERSYSTEM_CROSSING_TS, 5e5),
                    tr.Transition(tr.TransitionType.ISOMERIZATION, 2e7),
                    tr.Transition(tr.TransitionType.BACKISOMERIZATION, 102695.97797787128),
                    tr.Transition(tr.TransitionType.INTERNAL_CONVERSION_S, 709169999.9999999),
                    tr.Transition(tr.TransitionType.REDUCTION, 9.6e6),
                    tr.Transition(tr.TransitionType.OXIDATION, 0.2),
                    tr.Transition(tr.TransitionType.HOMO_FRET, 245101500000000.0, 1.0),
                    tr.Transition(tr.TransitionType.HOMO_FRET, 30665462018995.47, 1.414)]
        if parameters[2]:
            expected.insert(9, tr.Transition(tr.TransitionType.PHOTOBLEACHING_1, 1.0))
            index = 12
        assert transitions[:index] == expected
