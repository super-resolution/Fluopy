import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict
from fluopy import transitions as tr


def test_singlestate():
    assert tr.SingleState.S0.value == 0
    assert tr.SingleState.S1.value == 1
    assert tr.SingleState.S2.value == 2
    assert tr.SingleState.T1.value == 3
    assert tr.SingleState.T2.value == 4
    assert tr.SingleState.B.value == 5
    assert tr.SingleState.cis.value == 6
    assert tr.SingleState.OFF.value == 7
    assert tr.SingleState.OFF2.value == 8
    assert tr.SingleState.R.value == 9
    assert len(tr.SingleState) == 10


def test_pairedstate():
    assert tr.PairedState.S1_S0.value == [tr.SingleState.S1, tr.SingleState.S0]
    assert tr.PairedState.S1_S0.single_state_values == (1, 0)
    assert tr.PairedState.S1_S0.acceptor == tr.SingleState.S0
    assert tr.PairedState.S1_S0.donor == tr.SingleState.S1


def test_transitiontype():
    assert tr.TransitionType.EXCITATION.abbreviation == "EXC"
    assert (
        tr.TransitionType.EXCITATION.value.abbreviation
        == tr.TransitionType.EXCITATION.abbreviation
    )
    assert tr.TransitionType.EXCITATION.initial_state == tr.SingleState.S0
    assert (
        tr.TransitionType.EXCITATION.value.initial_state
        == tr.TransitionType.EXCITATION.initial_state
    )
    assert tr.TransitionType.EXCITATION.final_state == tr.SingleState.S1
    assert (
        tr.TransitionType.EXCITATION.value.final_state
        == tr.TransitionType.EXCITATION.final_state
    )
    assert tr.TransitionType.EXCITATION.photon is False
    assert (
        tr.TransitionType.EXCITATION.value.photon == tr.TransitionType.EXCITATION.photon
    )
    for transition_type in tr.TransitionType:
        assert len(transition_type.value.__dict__) == 4


@pytest.mark.parametrize(
    "transition_type, fluorophore_ids, expected",
    [
        [tr.TransitionType.EXCITATION, [1, 2], ""],
        [tr.TransitionType.EXCITATION, [(1, 2)], "ValueError1"],
        [tr.TransitionType.FRET, [(1, 2)], ""],
        [tr.TransitionType.FRET, [1, 2], "ValueError2"],
    ],
)
def test_transition(transition_type, fluorophore_ids, expected):
    if expected == "ValueError1":
        with pytest.raises(
            ValueError,
            match="EXC is not an energy transfer, fluorophore_ids has to be a list of "
            "ints.",
        ):
            transition = tr.Transition(
                transition_type=transition_type, rate=1, fluorophore_ids=fluorophore_ids
            )
    elif expected == "ValueError2":
        with pytest.raises(
            ValueError,
            match="FRET is energy transfer, fluorophore_ids have to be tuples of "
            "fluorophore pairs.",
        ):
            transition = tr.Transition(
                transition_type=transition_type, rate=1, fluorophore_ids=fluorophore_ids
            )
    else:
        transition = tr.Transition(
            transition_type=transition_type, rate=1, fluorophore_ids=fluorophore_ids
        )
        assert transition.identity is None
        assert transition.transition_type == transition_type
        assert transition.abbreviation == transition_type.abbreviation
        assert transition.initial_state == transition_type.initial_state
        assert transition.final_state == transition_type.final_state
        assert transition.rate == 1
        assert transition.photon == transition_type.photon
        assert transition.fluorophore_ids == fluorophore_ids


class TestTransitionSet:

    def test_init_empty(self, flu_sys_cy5):
        with pytest.raises(TypeError):
            tr.TransitionSet()

    def test_init(self, flu_sys_cy5):
        transition_1 = tr.Transition(
            tr.TransitionType.EXCITATION, rate=1e6, fluorophore_ids=[0]
        )
        transition_2 = tr.Transition(
            tr.TransitionType.FLUORESCENT_EMISSION, rate=1e9, fluorophore_ids=[0]
        )
        transitions = {
            "testfluo_1": [transition_1, transition_2],
        }
        fluorophore_system = flu_sys_cy5
        transition_set = tr.TransitionSet(
            transitions=transitions,
            fluorophore_system=fluorophore_system
        )
        assert transition_set.transitions == transitions
        assert transition_set.fluorophore_system == fluorophore_system
        assert isinstance(transition_set.combined_state_transitions_df, pd.DataFrame)
        assert len(transition_set.row_sums) == 2
        assert list(transition_set.single_states.keys()) == ['testfluo_1']
        assert isinstance(transition_set.transition_df, pd.DataFrame)
        assert len(transition_set.transition_df) == 2
        assert transition_set.transition_matrix.shape == (2, 2)

        transition_set = tr.TransitionSet(
            transitions=transitions,
            fluorophore_system=fluorophore_system
        )
        transition_set.finalize()

        assert transition_set.transitions == transitions
        assert transition_set.fluorophore_system == fluorophore_system
        assert isinstance(transition_set.combined_state_transitions_df, pd.DataFrame)
        assert len(transition_set.row_sums) == 2
        assert list(transition_set.single_states.keys()) == ['testfluo_1']
        assert isinstance(transition_set.transition_df, pd.DataFrame)
        assert len(transition_set.transition_df) == 2
        assert transition_set.transition_matrix.shape == (2, 2)

    @pytest.mark.parametrize(
        "transition_type, fluorophore_ids, fluo_comb, expected",
        [
            [tr.TransitionType.FRET, [(0, 1)], "testfluo_1-testfluo_1", "ValueError1"],
            [
                tr.TransitionType.FRET,
                [(2, 1)],
                "D: testfluo_1, A: testfluo_1, dist: 1.0",
                "ValueError2",
            ],
            [
                tr.TransitionType.FRET,
                [(1, 2)],
                "D: testfluo_1, A: testfluo_1, dist: 1.0",
                "ValueError3",
            ],
            [
                tr.TransitionType.FRET,
                [(1, 0)],
                "D: testfluo_1, A: testfluo_1, dist: 2.0",
                "ValueError4",
            ],
            [tr.TransitionType.EXCITATION, [1], "testfluo_2", "ValueError5"],
        ],
    )
    def test_transition_set_errors(
        self, transition_type, fluorophore_ids, fluo_comb, expected, request
    ):
        transitions = {
            fluo_comb: [
                tr.Transition(
                    transition_type=transition_type, rate=1, fluorophore_ids=fluorophore_ids
                )
            ]
        }
        fluorophore_system = request.getfixturevalue("flu_sys_2xcy5_1xatto643")
        if expected == "ValueError1":
            with pytest.raises(
                ValueError,
                match="energy transfers have to be defined in transitions with the "
                "key 'D: {name of donor}, A: {name of acceptor}, dist: "
                "{distance between them}'",
            ):
                transition_set = tr.TransitionSet(
                    transitions=transitions, fluorophore_system=fluorophore_system
                )
        elif expected == "ValueError2":
            with pytest.raises(
                ValueError,
                match="testfluo_1 indicated to be at identity 2, testfluo_2 found.",
            ):
                transition_set = tr.TransitionSet(
                    transitions=transitions, fluorophore_system=fluorophore_system
                )
        elif expected == "ValueError3":
            # looks similar to ValueError2 but tests different Error
            with pytest.raises(
                ValueError,
                match="testfluo_1 indicated to be at identity 2, testfluo_2 found.",
            ):
                transition_set = tr.TransitionSet(
                    transitions=transitions, fluorophore_system=fluorophore_system
                )
        elif expected == "ValueError4":
            with pytest.raises(ValueError, match="2.0 nm indicated, 1.0 nm found."):
                transition_set = tr.TransitionSet(
                    transitions=transitions, fluorophore_system=fluorophore_system
                )
        elif expected == "ValueError5":
            with pytest.raises(
                ValueError,
                match="testfluo_2 indicated to be at identity 1, testfluo_1 found.",
            ):
                transition_set = tr.TransitionSet(
                    transitions=transitions, fluorophore_system=fluorophore_system
                )


    # get_single_states is tested indirectly within test_transition_set
    def test_transition_set(self, request):
        zero_rate_transition = tr.Transition(
            transition_type=tr.TransitionType.INTERNAL_CONVERSION_S,
            rate=0,
            fluorophore_ids=[0, 1],
        )
        transitions = {
            "testfluo_1": [
                tr.Transition(
                    transition_type=tr.TransitionType.EXCITATION,
                    rate=1,
                    fluorophore_ids=[0, 1],
                ),
                tr.Transition(
                    transition_type=tr.TransitionType.FLUORESCENT_EMISSION,
                    rate=1e-1,
                    fluorophore_ids=[0, 1],
                ),
                zero_rate_transition,
                tr.Transition(
                    transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST,
                    rate=1,
                    fluorophore_ids=[0, 1],
                ),
            ],
            "D: testfluo_1, A: testfluo_1, dist: 1.0": [
                tr.Transition(
                    transition_type=tr.TransitionType.FRET,
                    rate=1,
                    fluorophore_ids=[(0, 1), (1, 0)],
                )
            ],
            "D: testfluo_1, A: testfluo_2, dist: 2.0": [
                tr.Transition(
                    transition_type=tr.TransitionType.FRET, rate=1, fluorophore_ids=[(0, 2)]
                )
            ],
            "testfluo_2": [
                tr.Transition(
                    transition_type=tr.TransitionType.EXCITATION,
                    rate=1,
                    fluorophore_ids=[2],
                ),
                tr.Transition(
                    transition_type=tr.TransitionType.FLUORESCENT_EMISSION,
                    rate=1e-1,
                    fluorophore_ids=[2],
                ),
            ],
        }
        fluorophore_system = request.getfixturevalue("flu_sys_2xcy5_1xatto643")
        transition_set = tr.TransitionSet(
            transitions=transitions, fluorophore_system=fluorophore_system
        )
        i = 0
        for transition_collection in transition_set.transitions.values():
            for transition in transition_collection:
                assert transition.identity == i
                i += 1
                assert transition.abbreviation != "IC"
        assert transition_set.fluorophore_system == fluorophore_system
        assert transition_set.transitions == transitions
        assert list(transition_set.transition_df.columns) == [
            "transition_type",
            "abbreviation",
            "initial_state",
            "final_state",
            "rate",
            "photon",
            "fluorophore_ids",
            "absorbing",
        ]
        assert (
            transition_set.transition_df.loc[
                transition_set.transition_df["absorbing"], "abbreviation"
            ]
        ).tolist() == ["ISC_ST"]
        test_single_states = {
            "testfluo_1": np.array([0, 1, 3]),
            "testfluo_2": np.array([0, 1]),
        }
        for key in transition_set.single_states:
            assert key in test_single_states
            assert np.array_equal(
                transition_set.single_states[key], test_single_states[key]
            )

        assert isinstance(transition_set.combined_state_transitions_df, pd.DataFrame)
        assert len(transition_set.row_sums) == 61
        assert list(transition_set.single_states.keys()) == ['testfluo_1', 'testfluo_2']
        assert isinstance(transition_set.transition_df, pd.DataFrame)
        assert len(transition_set.transition_df) == 7
        assert transition_set.transition_matrix.shape == (61, 61)


    def test_transition_set_filter_by_identity(self, tr_set_bl_et_3f):
        assert (
            tr_set_bl_et_3f.transition_df.index.get_level_values(1).tolist()
            == (np.arange(0, 22, 1, dtype=int)).tolist()
        )
        assert (
            "D: testfluo_1, A: testfluo_2, dist: 2.0"
            in tr_set_bl_et_3f.transition_df.index.get_level_values(0).tolist()
        )
        tr_set_bl_et_filtered = tr_set_bl_et_3f.filter_by_identity(remove_list=[4, 10])
        assert (
            tr_set_bl_et_filtered.transition_df.index.get_level_values(1).tolist()
            == (np.arange(0, 20, 1, dtype=int)).tolist()
        )
        assert (
            "D: testfluo_1, A: testfluo_2, dist: 2.0"
            not in tr_set_bl_et_filtered.transition_df.index.get_level_values(0).tolist()
        )


    def test_transition_set_adjust_rates(self, tr_set_bl_et_3f):
        assert tr_set_bl_et_3f.transition_df["rate"].tolist()[:6] == [
            5815700.439305622,
            270000000.0,
            830000.0,
            5000.0,
            20000000.0,
            109542.37650972935,
        ]
        tr_set_bl_et_adjusted = tr_set_bl_et_3f.adjust_rates(change_dict={4: 1})
        assert tr_set_bl_et_adjusted.transition_df["rate"].tolist()[:6] == [
            5815700.439305622,
            270000000.0,
            830000.0,
            5000.0,
            1.0,
            109542.37650972935,
        ]


    def test_transition_set_remove_absorbing_states(self, tr_set_bl_et_3f):
        assert tr_set_bl_et_3f.transition_df["absorbing"].any()
        tr_set_et = tr_set_bl_et_3f.remove_absorbing_states()
        assert not tr_set_et.transition_df["absorbing"].any()


    def test_transition_set_remove_energy_transfers(self, tr_set_bl_et_3f):
        assert any(
            "dist" in s
            for s in tr_set_bl_et_3f.transition_df.index.get_level_values(0).tolist()
        )
        tr_set_bl = tr_set_bl_et_3f.remove_energy_transfers()
        assert not any(
            "dist" in s for s in tr_set_bl.transition_df.index.get_level_values(0).tolist()
        )


@pytest.mark.parametrize(
    "single_states, dirnames, expected",
    [
        [
            {"testfluo_1": [0, 1, 2]},
            ["flu_obj_cy5_1", "flu_obj_cy5_2"],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        ],
        [
            {"testfluo_1": [0, 1, 2], "testfluo_2": [0, 1, 2]},
            ["flu_obj_cy5_1", "flu_obj_atto643"],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        ],
        [{"testfluo_1": [0, 1, 2]}, ["flu_obj_cy5_1"], [(0,), (1,), (2,)]],
    ],
)
def test_get_state_combinations(single_states, dirnames, request, expected):
    fluorophores = [request.getfixturevalue(dirname) for dirname in dirnames]

    state_combinations = tr.get_state_combinations(
        single_states=single_states, fluorophores=fluorophores
    )
    assert state_combinations == expected


def test_get_combined_state_transitions():
    combined_state_transitions = tr.get_combined_state_transitions(
        state_combinations=[(0, 0), (0, 1), (1, 2)]
    )
    assert combined_state_transitions == [
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 2)),
        ((0, 1), (0, 0)),
        ((0, 1), (0, 1)),
        ((0, 1), (1, 2)),
        ((1, 2), (0, 0)),
        ((1, 2), (0, 1)),
        ((1, 2), (1, 2)),
    ]


def test_rate_assignment_standard():
    combined_state_transitions = [
        ((0, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (1, 1, 0)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (1, 0, 0)),
        ((0, 4, 5), (1, 4, 5)),
    ]
    transition = pd.Series(
        asdict(
            tr.Transition(
                transition_type=tr.TransitionType.EXCITATION,
                rate=1,
                fluorophore_ids=[0, 1],
            )
        )
    )
    transition_rate_list = tr.rate_assignment_standard(
        transition=transition,
        transition_id=0,
        transition_rate_list=[],
        combined_state_transitions=combined_state_transitions,
    )
    expected = [
        [(0, 0, 0), (0, 1, 0), [1], "EXC", 0, 1, False],
        [(0, 1, 0), (1, 1, 0), [0], "EXC", 0, 1, False],
        [(0, 4, 5), (1, 4, 5), [0], "EXC", 0, 1, False],
    ]
    assert transition_rate_list == expected


def test_rate_assignment_energy_transfer():
    combined_state_transitions = [
        ((0, 0, 0), (1, 1, 1)),
        ((0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (1, 0, 1)),
        ((0, 1, 5), (1, 0, 5)),
    ]
    transition = pd.Series(
        asdict(
            tr.Transition(
                transition_type=tr.TransitionType.FRET,
                rate=1,
                fluorophore_ids=[(0, 1), (1, 0)],
            )
        )
    )
    transition_rate_list = tr.rate_assignment_energy_transfer(
        transition=transition,
        transition_id=0,
        transition_rate_list=[],
        combined_state_transitions=combined_state_transitions,
    )
    expected = [
        [(0, 1, 0), (1, 0, 0), [1, 0], "FRET", 0, 1, False],
        [(1, 0, 0), (0, 1, 0), [0, 1], "FRET", 0, 1, False],
        [(0, 1, 5), (1, 0, 5), [1, 0], "FRET", 0, 1, False],
    ]
    assert transition_rate_list == expected


def test_construct_transition_rate_list():
    transition_1 = pd.Series(
        asdict(
            tr.Transition(
                transition_type=tr.TransitionType.EXCITATION,
                rate=1,
                fluorophore_ids=[0, 1],
            )
        )
    )
    transition_2 = pd.Series(
        asdict(
            tr.Transition(
                transition_type=tr.TransitionType.FRET,
                rate=1,
                fluorophore_ids=[(0, 1), (1, 0)],
            )
        )
    )
    transition_df = pd.concat([transition_1, transition_2], axis=1).transpose()
    transition_df.index = pd.MultiIndex.from_tuples(
        [("testfluo_1", 0), ("D: testfluo_1, A: testfluo_1, dist: 1.0", 1)]
    )
    combined_state_transitions = [
        ((0, 0, 0), (1, 1, 1)),
        ((0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (1, 0, 1)),
        ((0, 1, 5), (1, 0, 5)),
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (1, 1, 0)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 4, 5), (1, 4, 5)),
    ]
    transition_rate_list = tr.construct_transition_rate_list(
        transition_df=transition_df,
        combined_state_transitions=combined_state_transitions,
    )
    expected = [
        [(0, 1, 0), (1, 1, 0), [0], "EXC", 0, 1, False],
        [(0, 0, 0), (0, 1, 0), [1], "EXC", 0, 1, False],
        [(0, 4, 5), (1, 4, 5), [0], "EXC", 0, 1, False],
        [(0, 1, 0), (1, 0, 0), [1, 0], "FRET", 1, 1, False],
        [(1, 0, 0), (0, 1, 0), [0, 1], "FRET", 1, 1, False],
        [(0, 1, 5), (1, 0, 5), [1, 0], "FRET", 1, 1, False],
    ]
    assert transition_rate_list == expected


def test_construct_transition_matrix():
    combined_state_transitions_df = pd.DataFrame(
        {
            "initial_state": [(0, 0), (0, 1), (0, 1)],
            "final_state": [(0, 1), (1, 0), (0, 0)],
            "abbreviation": ["EXC", "FRET", "FLU"],
            "transition_id": [0, 1, 2],
            "rate": [1, 1e4, 2],
            "photon": [False, False, True],
        }
    )
    transition_matrix, row_sums = tr.construct_transition_matrix(
        combined_state_transitions_df=combined_state_transitions_df
    )
    expected_transition_matrix = np.array(
        [
            [0.0000e00, 9.9980e-01, 1.9996e-04],
            [0.0000e00, 0.0000e00, 0.0000e00],
            [1.0000e00, 0.0000e00, 0.0000e00],
        ]
    )

    expected_row_sums = np.array([10002, 0, 1])
    np.testing.assert_allclose(transition_matrix, expected_transition_matrix, rtol=1e-6)
    np.testing.assert_allclose(row_sums, expected_row_sums, rtol=1e-5)


def test_transition_set_finalize(tr_set_bl_et_3f):
    assert tr_set_bl_et_3f.combined_state_transitions_df.columns.tolist() == [
        "initial_state",
        "final_state",
        "fluorophore_ids",
        "abbreviation",
        "transition_id",
        "rate",
        "photon",
    ]
    assert tr_set_bl_et_3f.combined_state_transitions_df.shape == (516, 7)
    assert tr_set_bl_et_3f.transition_matrix.shape == (516, 516)
    assert tr_set_bl_et_3f.row_sums.shape == (516,)


@pytest.mark.parametrize(
    "dirnames, distance, overwrite, exclude, include, expected",
    [
        [["flu_obj_cy5_1", "flu_obj_cy5_1"], 1, None, None, None, 0],
        [["flu_obj_cy5_1", "flu_obj_atto643"], 1, None, None, None, 1],
        [["flu_obj_cy5_1", "flu_obj_cy5_1"], 1, {"s0": [0.1, 1]}, None, None, 2],
        [["flu_obj_cy5_1", "flu_obj_cy5_1"], 1, None, ["s0"], None, 3],
        [
            ["flu_obj_cy5_1", "flu_obj_cy5_1"],
            1,
            None,
            None,
            {"t1": [(tr.TransitionType.S_T_ANNI_BLEACH, 0.5)]},
            4,
        ],
    ],
)
def test_derive_energy_transfer_transitions(
    dirnames, request, distance, overwrite, exclude, include, expected
):
    donor_data = request.getfixturevalue(dirnames[0]).constants
    acceptor_data = request.getfixturevalue(dirnames[1]).constants
    transitions = tr.derive_energy_transfer_transitions(
        donor_data=donor_data,
        acceptor_data=acceptor_data,
        fluorophore_ids=[(1, 2)],
        dipole_orientation_factor=2 / 3,
        distance=distance,
        refractive_index=1,
        overwrite=overwrite,
        exclude=exclude,
        include=include,
    )
    if expected == 0:
        assert len(transitions) == 3
        assert transitions[0].rate == 248130286338181.22
    elif expected == 1:
        assert len(transitions) == 1
    elif expected == 2:
        assert transitions[0].rate == 0.1 * 248130286338181.22
    elif expected == 3:
        assert len(transitions) == 2
    elif expected == 4:
        assert len(transitions) == 4
        assert_total_rate = 0
        for transition in transitions:
            if transition.initial_state == tr.PairedState.S1_T1:
                assert_total_rate += transition.rate
                if transition.abbreviation == "STA_B":
                    assert_rate = transition.rate
        assert assert_rate == 0.5 * assert_total_rate


@pytest.mark.parametrize(
    "irradiance, bleaching, dstorm, summarize",
    [[0, False, False, False], [1, False, False, True], [1, True, True, False]],
)
def test_derive_transitions(irradiance, bleaching, dstorm, summarize, request):
    fluorophore_data = request.getfixturevalue("flu_obj_cy5_1").constants
    transitions = tr.derive_transitions(
        summarize=summarize,
        fluorophore_data=fluorophore_data,
        fluorophore_ids=[1],
        irradiance=irradiance,
        wavelength=640,
        bleaching=bleaching,
        dstorm=dstorm,
    )
    dstorm_checker = [
        "PET_TS",
        "PET_SS",
        "PET_SO",
        "PET_TO",
        "TE",
        "PU",
        "PET_TR",
        "OXI",
    ]
    if not dstorm:
        for transition in transitions:
            assert transition.abbreviation not in dstorm_checker
    else:
        for transition in transitions:
            if transition.abbreviation in dstorm_checker:
                dstorm_checker.remove(transition.abbreviation)
        assert len(dstorm_checker) == 0
    if not bleaching:
        for transition in transitions:
            assert transition.abbreviation not in ["BLE"]
    else:
        assert any(transition.abbreviation == "BLE" for transition in transitions)
    if irradiance == 0:
        for transition in transitions:
            if transition.abbreviation == "EXC":
                assert transition.rate == 0
    else:
        for transition in transitions:
            if transition.abbreviation == "EXC":
                assert transition.rate != 0
    summarize_checker = ["S1S0SUM", "cisS0SUM", "T1S0SUM"]
    if summarize:
        for transition in transitions:
            if transition.abbreviation in summarize_checker:
                summarize_checker.remove(transition.abbreviation)
        assert len(summarize_checker) == 0
    else:
        for transition in transitions:
            assert transition.abbreviation not in summarize_checker


def test_interpolate_data():
    data = pd.DataFrame(
        {"Wavelengths": [617, 619, 620, 621], "y": [0.5, 0.6, 0.7, 0.6]}
    )
    interpolated = tr.interpolate_data(
        minimum_wavelength=610, maximum_wavelength=621, data=data
    )
    np.testing.assert_array_equal(
        interpolated,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.475, 0.6, 0.7, 0.6]),
    )
