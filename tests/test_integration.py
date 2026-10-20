import numpy as np
import pandas as pd
import pytest

from fluopy import analysis as an
from fluopy import emissions as em
from fluopy import prediction as pr
from fluopy import simulation as si
from fluopy import transitions as tr


@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_filtered_dye_model_through_prediction_simulation_and_emissions(
    tr_set_1f, matrix_format
):
    original = tr_set_1f.transition_df.copy(deep=True)
    removed_ids = original.index.get_level_values(1)[
        ~original["abbreviation"].isin(["EXC", "FLU"])
    ]
    transition_set = tr_set_1f.remove_transitions(removed_ids)
    transition_set = transition_set.adjust_rates({0: 2, 1: 4})
    transition_set = transition_set.with_matrix_format(matrix_format).finalize()

    prediction = pr.Prediction(transition_set)
    simulation = si.Simulation(transition_set)
    simulation.run(size=100, seed=42)
    analysis = an.Analysis(simulation)
    emissions = em.Emissions(frame_time="1s", seed=43)
    emissions.extract(simulation)

    np.testing.assert_array_equal(simulation.state_series[0], [0, 1] * 50 + [0])
    np.testing.assert_array_equal(analysis.frequency_transitions, [0.5, 0.5])
    np.testing.assert_allclose(
        analysis.frequency_transitions, prediction.frequency_transitions
    )
    np.testing.assert_allclose(prediction.mean_lifetimes["testfluo_1"], [0.5, 0.25])
    np.testing.assert_allclose(
        prediction.state_occupations["testfluo_1"], [2 / 3, 1 / 3]
    )
    np.testing.assert_array_equal(
        emissions.event_time_points["all"], simulation.time_series[2::2]
    )
    frame_ends = emissions.event_time_series.index.to_numpy()
    expected_counts, _ = np.histogram(
        simulation.time_series[2::2], bins=np.insert(frame_ends, 0, 0)
    )
    np.testing.assert_array_equal(emissions.event_time_series["all"], expected_counts)
    assert expected_counts.sum() > 0
    pd.testing.assert_frame_equal(tr_set_1f.transition_df, original)


@pytest.mark.parametrize("second_pet_rate", [0, 3])
def test_distance_specific_model_through_analysis_plots(
    flu_sys_2xcy5_1xatto643, second_pet_rate
):
    pet = tr.TransitionType(
        "FRET", tr.PairedState.S1_S0, tr.PairedState.S0_S1, False, "PET"
    )
    transition_set = tr.TransitionSet(
        {
            "testfluo_1": [
                tr.Transition(tr.TransitionType.EXCITATION, 2, [0, 1]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 4, [0, 1]),
            ],
            "D: testfluo_1, A: testfluo_2, dist: 2.0": [
                tr.Transition(tr.TransitionType.FRET, 1, [(0, 2)]),
                tr.Transition(pet, 2, [(0, 2)]),
            ],
            "D: testfluo_1, A: testfluo_2, dist: 1.0": [
                tr.Transition(pet, second_pet_rate, [(1, 2)]),
                tr.Transition(tr.TransitionType.FRET, 3, [(1, 2)]),
            ],
            "testfluo_2": [
                tr.Transition(tr.TransitionType.EXCITATION, 2, [2]),
                tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 4, [2]),
            ],
        },
        flu_sys_2xcy5_1xatto643,
        keep_zero_rates=True,
    )
    original = transition_set.transition_df.copy(deep=True)
    simulation = si.Simulation(transition_set)
    simulation.run(size=1000, seed=42)
    analysis = an.Analysis(simulation)

    frequencies = analysis.frequency_transitions
    expected_frequencies = [
        frequencies[0],
        frequencies[1],
        frequencies[2] + frequencies[5],
        frequencies[3] + frequencies[4],
        frequencies[6],
        frequencies[7],
    ]
    distributions = analysis.transition_time_distributions
    fret_times = np.concatenate([distributions[2], distributions[5]])
    pet_times = np.concatenate([distributions[3], distributions[4]])
    assert distributions[2].size > 0
    assert distributions[5].size > 0
    if second_pet_rate == 0:
        assert distributions[4].size == 0
    else:
        assert distributions[4].size > 0
    expected_means = [
        analysis.mean_transition_times[0],
        analysis.mean_transition_times[1],
        fret_times.mean(),
        pet_times.mean(),
        analysis.mean_transition_times[6],
        analysis.mean_transition_times[7],
    ]

    frequency_ax = analysis.plot_frequency_transitions(diff_dist=False, yscale="linear")
    mean_ax = analysis.plot_mean_transition_times(diff_dist=False, yscale="linear")

    np.testing.assert_allclose(
        [patch.get_height() for patch in frequency_ax.patches], expected_frequencies
    )
    np.testing.assert_allclose(
        [patch.get_height() for patch in mean_ax.patches], expected_means
    )
    assert [text.get_text() for text in frequency_ax.get_legend().get_texts()] == [
        "testfluo_1",
        "D: testfluo_1, A: testfluo_2",
        "testfluo_2",
    ]
    pd.testing.assert_frame_equal(transition_set.transition_df, original)
