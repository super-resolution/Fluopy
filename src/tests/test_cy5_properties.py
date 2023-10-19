import pytest
import src.transitions as tr


@pytest.mark.parametrize('parameters',
                         [[2, 600, False, False, None, False],
                          [2, 600, False, True, {(0, 1): 1, (1, 0): 1}, False],
                          [2, 600, False, True, {(0, 1): 1, (1, 0): 1}, True]])
def test_derive_transitions(cy5_object, parameters):

    transitions = cy5_object.derive_transitions(*parameters)
    print(transitions)
    expected = [tr.Transition(tr.TransitionType.EXCITATION, 1946561.1940894993),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 2.7e+08),
                tr.Transition(tr.TransitionType.INTERSYSTEM_CROSSING_ST, 8.3e5),
                tr.Transition(tr.TransitionType.INTERSYSTEM_CROSSING_TS, 5e3),
                tr.Transition(tr.TransitionType.ISOMERIZATION, 2e7),
                tr.Transition(tr.TransitionType.BACKISOMERIZATION, 102695.97797787128),
                tr.Transition(tr.TransitionType.INTERNAL_CONVERSION_S, 709169999.9999999),]
    if parameters[3]:
        expected_2 = [tr.Transition(tr.TransitionType.HOMO_FRET, 245101500000000.0, 1.0),
                      tr.Transition(tr.TransitionType.CIS_FRET_1, 474390000000000.0, 1.0),
                      # tr.Transition(tr.TransitionType.CIS_FRET_2, 245101500000000.0, 1.0),
                      tr.Transition(tr.TransitionType.S_T_ANNIHILATION, 142317000000000.0, 1.0),
                      tr.Transition(tr.TransitionType.OFF_FRET, 15813000000000.0, 1.0)]
        expected = expected + expected_2
    if parameters[5]:
        expected_3 = [tr.Transition(tr.TransitionType.ET_CYCLE_T, 438344.0494535317),
                      tr.Transition(tr.TransitionType.ET_CYCLE_S, 4383440.494535317),
                      tr.Transition(tr.TransitionType.REDUCTION_T, 438.3440494535317),
                      tr.Transition(tr.TransitionType.REDUCTION_S, 4383.440494535317),
                      tr.Transition(tr.TransitionType.OXIDATION, 0.02)]
        expected[7:7] = expected_3

    assert transitions == expected
