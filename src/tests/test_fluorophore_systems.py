import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import fluorophore_systems as fc
import initialize as ini
import processing as pr
import gillespie_algorithm as ga
import emitting_transitions as et


def test_init_fluorophoresystem():
    system = fc.FluorophoreSystem(2, 1, ("S0", "S1", "T1", "R", "B"), {"k_S0_S1": [0.4, "excitation"],
                                                                       "k_S1_T1": [5.8, "intersystem crossing"],
                                                                       "k_T1_S0": [1e-2, "relaxation"],
                                                                       "k_T1_R": [13, "reduction"],
                                                                       "k_R_S0": [0.3, "oxidation"]},
                                  induction_rate=4)
    assert system.number == 2
    assert system.distances == 1
    assert system.states == ("S0", "S1", "T1", "R", "B")
    assert system.rates == {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                            "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"], "k_R_S0": [0.3, "oxidation"]}
    assert system.state_names == ["S0_S0", "S0_S1", "S0_T1", "S0_R", "S0_B", "S1_S0", "S1_S1", "S1_T1", "S1_R", "S1_B",
                                  "T1_S0", "T1_S1", "T1_T1", "T1_R", "T1_B", "R_S0", "R_S1", "R_T1", "R_R", "R_B",
                                  "B_S0", "B_S1", "B_T1", "B_R", "B_B"]
    for joined_state, expected in zip(system.Joined_States, ini.state_pairs(system.number, system.states)):
        assert joined_state.name == expected.name
        assert joined_state.value == expected.value
    assert system.transitions == ini.transition_pairs(system.Joined_States)
    goal_assigned_rate_dict, goal_rate_name_dict = ini.transition_rate_dict(system.rates, system.transitions)
    goal_assigned_rate_dict, goal_rate_name_dict = ini.induction(goal_assigned_rate_dict, goal_rate_name_dict,
                                                                 system.transitions, 4, system.states)
    assert system.assigned_rate_dict == goal_assigned_rate_dict
    assert system.rate_name_dict == goal_rate_name_dict
    assert np.array_equal(system.initial_row_vector, ini.initial_row_vector(system.transitions))
    _, goal_transition_matrix, goal_row_sums = ini.transition_matrices(system.assigned_rate_dict, system.transitions)
    assert np.array_equal(system.transition_matrix, goal_transition_matrix)
    assert np.array_equal(system.row_sums, goal_row_sums)
    assert not system.time_series
    assert not system.time_step_series
    assert not system.state_series
    assert not system.transition_series
    assert not system.duplices
    assert not system.unique_series
    assert not system.unique_states
    assert not system.unique_joined_states
    assert not system.unique_names
    assert not system.unique_transitions
    assert not system.unique_series_converted
    assert not system.occupation_time_total
    assert not system.occupation_time_mean


def test_simulate_fluorophoresystem():
    system = fc.FluorophoreSystem(2, 1, ("S0", "S1", "T1", "R", "B"), {"k_S0_S1": [0.4, "excitation"],
                                                                       "k_S1_T1": [5.8, "intersystem crossing"],
                                                                       "k_T1_S0": [1e-2, "relaxation"],
                                                                       "k_T1_R": [13, "reduction"],
                                                                       "k_R_S0": [0.3, "oxidation"]},
                                  induction_rate=4)
    system.simulate(100, 100, "cy")
    goal_time_series, goal_time_step_series, goal_state_series, goal_transition_series = ga.direct_method_py(
        system.row_sums, system.initial_row_vector, system.transition_matrix, system.rate_name_dict, 100, 100)
    assert np.array_equal(system.time_series, goal_time_series)
    assert np.array_equal(system.time_step_series, goal_time_step_series)
    assert np.array_equal(system.state_series, goal_state_series)
    # "cy" version generates goal_transition_series only of type None


def test_process_fluorophoresystem():
    system = fc.FluorophoreSystem(2, 1, ("S0", "S1", "T1", "R", "B"), {"k_S0_S1": [0.4, "excitation"],
                                                                       "k_S1_T1": [5.8, "intersystem crossing"],
                                                                       "k_T1_S0": [1e-2, "relaxation"],
                                                                       "k_T1_R": [13, "reduction"],
                                                                       "k_R_S0": [0.3, "oxidation"]},
                                  induction_rate=4)
    system.simulate(100, 100, "py")
    system.process()
    goal_duplices = pr.identify_duplices(system.state_names)
    goal_unique_series, goal_unique_states, goal_unique_joined_states, goal_unique_names = \
        pr.uniques(goal_duplices, system.state_series, system.Joined_States)
    goal_unique_transitions = ini.transition_pairs(goal_unique_joined_states)
    goal_unique_series_converted = pr.convert_unique_states(goal_unique_series, goal_unique_states)
    goal_occupation_time_total, goal_occupation_time_mean = \
        pr.occupation_t(system.time_step_series, goal_unique_series, goal_unique_states, system.absorbing_states)

    assert system.duplices == goal_duplices
    assert np.array_equal(system.unique_series, goal_unique_series)
    assert np.array_equal(system.unique_states, goal_unique_states)
    assert system.unique_joined_states == goal_unique_joined_states
    assert system.unique_names == goal_unique_names
    assert system.unique_transitions == goal_unique_transitions
    assert np.array_equal(system.unique_series_converted, goal_unique_series_converted)
    assert np.array_equal(system.occupation_time_total, goal_occupation_time_total)
    assert np.array_equal(system.occupation_time_mean, goal_occupation_time_mean)


def test_init_jablonskimodel():
    system = fc.JablonskiModel(2, 1, {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                                      "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"],
                                      "k_R_S0": [0.3, "oxidation"]},
                               induction_rate=4)
    assert system.number == 2
    assert system.distances == 1
    assert system.states == ("S0", "S1", "T1", "R", "B")
    assert system.rates == {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                            "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"],
                            "k_R_S0": [0.3, "oxidation"]}
    assert system.state_names == ["S0_S0", "S0_S1", "S0_T1", "S0_R", "S0_B", "S1_S0", "S1_S1", "S1_T1", "S1_R", "S1_B",
                                  "T1_S0", "T1_S1", "T1_T1", "T1_R", "T1_B", "R_S0", "R_S1", "R_T1", "R_R", "R_B",
                                  "B_S0", "B_S1", "B_T1", "B_R", "B_B"]
    assert system.assigned_rate_dict == {'S0_S0__S0_S1': 0.4, 'S0_S0__S1_S0': 0.4, 'S0_S1__S1_S1': 0.4,
                                         'S0_T1__S1_T1': 0.4, 'S0_R__S1_R': 0.4, 'S0_B__S1_B': 0.4,
                                         'S1_S0__S1_S1': 0.4, 'T1_S0__T1_S1': 0.4, 'R_S0__R_S1': 0.4,
                                         'B_S0__B_S1': 0.4, 'S0_S1__S0_T1': 5.8, 'S1_S0__T1_S0': 5.8,
                                         'S1_S1__S1_T1': 5.8, 'S1_S1__T1_S1': 5.8, 'S1_T1__T1_T1': 5.8,
                                         'S1_R__T1_R': 5.8, 'S1_B__T1_B': 5.8, 'T1_S1__T1_T1': 5.8,
                                         'R_S1__R_T1': 5.8, 'B_S1__B_T1': 5.8, 'S0_T1__S0_S0': 0.01,
                                         'S1_T1__S1_S0': 0.01, 'T1_S0__S0_S0': 0.01, 'T1_S1__S0_S1': 0.01,
                                         'T1_T1__S0_T1': 0.01, 'T1_T1__T1_S0': 0.01, 'T1_R__S0_R': 0.01,
                                         'T1_B__S0_B': 0.01, 'R_T1__R_S0': 0.01, 'B_T1__B_S0': 0.01,
                                         'S0_T1__S0_R': 13, 'S1_T1__S1_R': 13, 'T1_S0__R_S0': 13,
                                         'T1_S1__R_S1': 13, 'T1_T1__T1_R': 13, 'T1_T1__R_T1': 13,
                                         'T1_R__R_R': 13, 'T1_B__R_B': 13, 'R_T1__R_R': 13,
                                         'B_T1__B_R': 13, 'S0_R__S0_S0': 0.3, 'S1_R__S1_S0': 0.3,
                                         'T1_R__T1_S0': 0.3, 'R_S0__S0_S0': 0.3, 'R_S1__S0_S1': 0.3,
                                         'R_T1__S0_T1': 0.3, 'R_R__S0_R': 0.3, 'R_R__R_S0': 0.3,
                                         'R_B__S0_B': 0.3, 'B_R__B_S0': 0.3, 'S1_R__S0_S0': 4,
                                         'R_S1__S0_S0': 4}
    assert system.rate_name_dict == {(0, 1): 'excitation', (0, 5): 'excitation', (1, 6): 'excitation',
                                     (2, 7): 'excitation', (3, 8): 'excitation', (4, 9): 'excitation',
                                     (5, 6): 'excitation', (10, 11): 'excitation', (15, 16): 'excitation',
                                     (20, 21): 'excitation', (1, 2): 'intersystem crossing',
                                     (5, 10): 'intersystem crossing', (6, 7): 'intersystem crossing',
                                     (6, 11): 'intersystem crossing', (7, 12): 'intersystem crossing',
                                     (8, 13): 'intersystem crossing', (9, 14): 'intersystem crossing',
                                     (11, 12): 'intersystem crossing', (16, 17): 'intersystem crossing',
                                     (21, 22): 'intersystem crossing', (2, 0): 'relaxation', (7, 5): 'relaxation',
                                     (10, 0): 'relaxation', (11, 1): 'relaxation', (12, 2): 'relaxation',
                                     (12, 10): 'relaxation', (13, 3): 'relaxation', (14, 4): 'relaxation',
                                     (17, 15): 'relaxation', (22, 20): 'relaxation', (2, 3): 'reduction',
                                     (7, 8): 'reduction', (10, 15): 'reduction', (11, 16): 'reduction',
                                     (12, 13): 'reduction', (12, 17): 'reduction', (13, 18): 'reduction',
                                     (14, 19): 'reduction', (17, 18): 'reduction', (22, 23): 'reduction',
                                     (3, 0): 'oxidation', (8, 5): 'oxidation', (13, 10): 'oxidation',
                                     (15, 0): 'oxidation', (16, 1): 'oxidation', (17, 2): 'oxidation',
                                     (18, 3): 'oxidation', (18, 15): 'oxidation', (19, 4): 'oxidation',
                                     (23, 20): 'oxidation', (8, 0): 'induction', (16, 0): 'induction'}
    assert np.array_equal(system.row_sums, np.array([0.8, 6.2, 13.41, 0.7, 0.4, 6.2, 11.6, 18.81, 10.1,
                                                     5.8, 13.41, 18.81, 26.02, 13.31, 13.01, 0.7, 10.1,
                                                     13.31, 0.6, 0.3, 0.4, 5.8, 13.01, 0.3, 0.]))
    assert not system.time_series
    assert not system.time_step_series
    assert not system.state_series
    assert not system.duplices
    assert not system.unique_series
    assert not system.unique_states
    assert not system.unique_joined_states
    assert not system.unique_names
    assert not system.unique_transitions
    assert not system.unique_series_converted
    assert not system.occupation_time_total
    assert not system.occupation_time_mean
    assert not system.emitting_transitions
    assert not system.emitting_mask
    assert not system.detected_emission_mask
    assert not system.pandas_series
    assert not system.on_periods
    assert not system.off_periods
    assert not system.autocorrelation


def test_emitters_jablonskimodel():
    system_1 = fc.JablonskiModel(2, 1, {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                                        "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"],
                                        "k_R_S0": [0.3, "oxidation"], "k_S1_S0": [10, "emission"]},
                                 induction_rate=4)
    system_1.simulate()
    system_1.emitters(photon_collection=0.7, resample="5ms", unit="s", threshold=0, memory=0, use_unique=True)
    system_2 = fc.JablonskiModel(2, 1, {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                                        "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"],
                                        "k_R_S0": [0.3, "oxidation"], "k_S1_S0": [10, "emission"]},
                                 induction_rate=4)
    system_2.simulate()
    system_2.emitters(photon_collection=0.7, resample="5ms", unit="s", threshold=0, memory=0, use_unique=False)

    assert np.array_equal(system_1.emitting_mask, system_2.emitting_mask)  # check if using uniques doesn't influence
    # the outcome

    goal_emitting_transitions, goal_emitting_transitions_indices = \
        et.identify_emitting_transitions(system_1.unique_transitions, system_1.states)

    assert system_1.emitting_transitions == goal_emitting_transitions
    assert system_1.emitting_transitions_indices == goal_emitting_transitions_indices

    goal_emitting_mask = et.emitter_mask(system_1.unique_series_converted, goal_emitting_transitions_indices)
    assert np.array_equal(system_1.emitting_mask, goal_emitting_mask)
    goal_detected_emission_mask = et.detected_emissions(goal_emitting_mask, 0.7, 100)
    assert np.array_equal(system_1.detected_emission_mask, goal_detected_emission_mask)
    goal_pandas_series = et.pandas_event_time_series(system_1.time_series[goal_detected_emission_mask], "s", "5ms")
    assert system_1.pandas_series.equals(goal_pandas_series)
    goal_on_periods, goal_off_periods, _, _ = et.blink_statistics(goal_pandas_series, 0, 0, False)
    assert np.array_equal(system_1.on_periods, goal_on_periods)
    assert np.array_equal(system_1.off_periods, goal_off_periods)


def test_fcs_jablonskimodel():
    system = fc.JablonskiModel(2, 1, {"k_S0_S1": [0.4, "excitation"], "k_S1_T1": [5.8, "intersystem crossing"],
                                      "k_T1_S0": [1e-2, "relaxation"], "k_T1_R": [13, "reduction"],
                                      "k_R_S0": [0.3, "oxidation"], "k_S1_S0": [10, "emission"]},
                               induction_rate=4)
    system.simulate()
    system.emitters(photon_collection=0.7, resample="5ms", unit="s", threshold=0, memory=0, use_unique=True)
    system.fcs()
    goal_autocorrelation = pr.autocorrelate(system.pandas_series, normalize=True, log=True, m=2, deltat=5e-3)
    assert np.array_equal(system.autocorrelation[0], goal_autocorrelation[0])
    assert np.array_equal(system.autocorrelation[1], goal_autocorrelation[1])


def test_init_cy5():
    system = fc.Cy5(2, 1, {"k_tS0_tS1": [0.3, "excitation"], "k_tS1_tS0": [5.8, "emission"]}, induction_rate=0.4)
    assert system.number == 2
    assert system.distances == 1
    assert system.states == ("tS0", "tS1", "tT1", "R", "B", "cS0", "cS1", "cT1")
    assert system.rates == {"k_tS0_tS1": [0.3, "excitation"], "k_tS1_tS0": [5.8, "emission"]}


def test_init_onoffmodel():
    system = fc.OnOffModel(2, 1, {"k_ON_OFF": [5, "turn off"], "k_OFF_ON": [2, "turn on"],
                                  "k_ON_B": [0.01, "bleaching"]},
                           induction_rate=1)
    assert system.number == 2
    assert system.distances == 1
    assert system.states == ("ON", "OFF", "B")
    assert system.rates == {"k_ON_OFF": [5, "turn off"], "k_OFF_ON": [2, "turn on"], "k_ON_B": [0.01, "bleaching"]}
    assert system.state_names == ["ON_ON", "ON_OFF", "ON_B", "OFF_ON", "OFF_OFF", "OFF_B", "B_ON", "B_OFF", "B_B"]
    assert system.assigned_rate_dict == {'ON_ON__ON_OFF': 5, 'ON_ON__OFF_ON': 5, 'ON_OFF__OFF_OFF': 5,
                                         'ON_B__OFF_B': 5, 'OFF_ON__OFF_OFF': 5, 'B_ON__B_OFF': 5,
                                         'ON_OFF__ON_ON': 3, 'OFF_ON__ON_ON': 3, 'OFF_OFF__ON_OFF': 2,
                                         'OFF_OFF__OFF_ON': 2, 'OFF_B__ON_B': 2, 'B_OFF__B_ON': 2,
                                         'ON_ON__ON_B': 0.01, 'ON_ON__B_ON': 0.01, 'ON_OFF__B_OFF': 0.01,
                                         'ON_B__B_B': 0.01, 'OFF_ON__OFF_B': 0.01, 'B_ON__B_B': 0.01}
    assert system.rate_name_dict == {(0, 1): 'turn off', (0, 3): 'turn off', (1, 4): 'turn off', (2, 5): 'turn off',
                                     (3, 4): 'turn off', (6, 7): 'turn off', (1, 0): 'turn on', (3, 0): 'turn on',
                                     (4, 1): 'turn on', (4, 3): 'turn on', (5, 2): 'turn on', (7, 6): 'turn on',
                                     (0, 2): 'bleaching', (0, 6): 'bleaching', (1, 7): 'bleaching', (2, 8): 'bleaching',
                                     (3, 5): 'bleaching', (6, 8): 'bleaching'}
    assert np.array_equal(system.row_sums, np.array([10.02, 8.01, 5.01, 8.01, 4., 2., 5.01, 2., 0.]))
    assert not system.time_series
    assert not system.time_step_series
    assert not system.state_series
    assert not system.duplices
    assert not system.unique_series
    assert not system.unique_states
    assert not system.unique_joined_states
    assert not system.unique_names
    assert not system.unique_transitions
    assert not system.unique_series_converted
    assert not system.occupation_time_total
    assert not system.occupation_time_mean
    assert not system.on_counts
    assert not system.emissions
    assert not system.emission_time_series


def test_emitters_onoffmodel():
    system = fc.OnOffModel(2, 1, {"k_ON_OFF": [5, "turn off"], "k_OFF_ON": [2, "turn on"],
                                  "k_ON_B": [0.01, "bleaching"]},
                           induction_rate=1)
    system.simulate()
    system.emitters(s0s1_rate=10, s1s0_rate=3, resample=2, seed=100)
    goal_on_counts = et.on_states(system.state_names)
    assert np.array_equal(system.on_counts, goal_on_counts)
    goal_emissions, goal_emission_time_series = et.emission_count(10, 3, goal_on_counts, system.state_series,
                                                                  system.time_step_series, 2, 100)
    assert np.array_equal(system.emissions, goal_emissions)
    assert np.array_equal(system.emission_time_series, goal_emission_time_series)
