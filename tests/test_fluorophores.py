import logging

import numpy as np
import pytest

from fluopy import emissions as em
from fluopy import fluorophores as fl
from fluopy import transitions as tr
from fluopy.fluo_data import FluorophoreData, Spectrum, testfluo_1, testfluo_2


@pytest.mark.parametrize(
    "name, position, exp_identity, exp_name, exp_position, exp_constants",
    [
        ["testfluo_1", [0, 0], None, "testfluo_1", np.array([0, 0]), testfluo_1],
        [
            "testfluo_2",
            [1.5, 0.34],
            None,
            "testfluo_2",
            np.array([1.5, 0.34]),
            testfluo_2,
        ],
        ["aa", [0, -5], None, "aa", np.array([0, -5]), None],
    ],
)
def test_fluorophore(
    name, position, exp_identity, exp_name, exp_position, exp_constants, caplog
):
    if name == "aa":
        with caplog.at_level(logging.WARNING):
            fluorophore = fl.Fluorophore(name=name, position=position)
            assert (
                "There is no FluorophoreData for Fluorophore aa in fluopy.fluo_data. Parameters have to be defined manually."
                in caplog.text
            )
        caplog.clear()
    else:
        fluorophore = fl.Fluorophore(name=name, position=position)
    assert fluorophore.identity == exp_identity
    assert fluorophore.name == exp_name
    np.testing.assert_array_equal(fluorophore.position, exp_position)
    if exp_constants is not None:
        assert isinstance(fluorophore.constants, FluorophoreData)
    else:
        assert fluorophore.constants is None


def test_fluorophore_uses_identity_equality():
    first = fl.Fluorophore("testfluo_1", [0, 0])
    second = fl.Fluorophore("testfluo_1", [0, 0])

    assert first == first
    assert first != second


def test_fluorophore_copies_position():
    position = np.array([1.0, 2.0])
    fluorophore = fl.Fluorophore("testfluo_1", position)

    position[0] = 10

    np.testing.assert_array_equal(
        fluorophore.position,
        [1.0, 2.0],
    )


def test_fluorophore_position_is_read_only():
    fluorophore = fl.Fluorophore("testfluo_1", [1, 2])

    with pytest.raises(ValueError, match="read-only"):
        fluorophore.position[0] = 10


@pytest.mark.parametrize(
    "positions, expected",
    [
        [
            [[1, 1], [2, 1], [1, 2]],
            {
                (0, 1): 1.0,
                (0, 2): 1.0,
                (1, 0): 1.0,
                (1, 2): 1.414,
                (2, 0): 1.0,
                (2, 1): 1.414,
            },
        ],
        [[[0, 0]], {}],
        [[[0, 0], [0, 0]], {(0, 1): 0.0, (1, 0): 0.0}],
        [[[-1, 0], [0, 0]], {(0, 1): 1.0, (1, 0): 1.0}],
    ],
)
def test_get_distances(positions, expected):
    assert fl.get_distances(positions=positions) == expected


@pytest.mark.parametrize(
    "dirnames, exp_distances, exp_count, multi_type",
    [
        [["flu_obj_cy5_1"], {}, 1, False],
        [["flu_obj_cy5_1", "flu_obj_cy5_2"], {(0, 1): 1, (1, 0): 1}, 2, False],
        [["flu_obj_cy5_1", "flu_obj_cy5_1"], "ValueError1", None, False],
        [["flu_obj_atto643", "flu_obj_cy5_1"], {(0, 1): 2, (1, 0): 2}, 2, True],
        [
            ["flu_obj_unknown", "flu_obj_cy5_1", "flu_obj_cy5_2"],
            {
                (0, 1): 3.0,
                (0, 2): 2.0,
                (1, 0): 3.0,
                (1, 2): 1.0,
                (2, 0): 2.0,
                (2, 1): 1.0,
            },
            3,
            True,
        ],
    ],
)
def test_fluorophore_system(
    dirnames, request, exp_distances, exp_count, multi_type, caplog
):
    if "flu_obj_unknown" in dirnames:
        with caplog.at_level(logging.WARNING):
            fluorophores = [request.getfixturevalue(dirname) for dirname in dirnames]
            assert (
                "There is no FluorophoreData for Fluorophore aa in fluopy.fluo_data. Parameters have to be defined manually."
                in caplog.text
            )
        caplog.clear()
    else:
        fluorophores = [request.getfixturevalue(dirname) for dirname in dirnames]
    if exp_distances == "ValueError1":
        with pytest.raises(
            ValueError,
            match="at least two fluorophores are indistinguishable at the 0.001 nm "
            "distance resolution. Also check for duplicates.",
        ):
            fluorophore_system = fl.FluorophoreSystem(fluorophores=fluorophores)
    else:
        fluorophore_system = fl.FluorophoreSystem(fluorophores=fluorophores)
        for i, (fluorophore_sys, fluorophore) in enumerate(
            zip(fluorophore_system.fluorophores, fluorophores)
        ):
            assert fluorophore_sys.identity == i
            assert fluorophore_sys == fluorophore
        assert fluorophore_system.distances == exp_distances
        assert fluorophore_system.count == exp_count
        assert fluorophore_system.multi_type == multi_type


def test_fluorophore_system_copies_input_sequence():
    fluorophores = [
        fl.Fluorophore("testfluo_1", [0, 0]),
        fl.Fluorophore("testfluo_1", [1, 0]),
    ]
    system = fl.FluorophoreSystem(fluorophores)

    fluorophores.append(fl.Fluorophore("testfluo_1", [2, 0]))

    assert isinstance(system.fluorophores, tuple)
    assert len(system.fluorophores) == 2
    assert system.count == 2


def test_fluorophore_system_rejects_positions_below_distance_resolution():
    fluorophores = [
        fl.Fluorophore("testfluo_1", [0, 0]),
        fl.Fluorophore("testfluo_1", [0.0004, 0]),
    ]

    with pytest.raises(
        ValueError,
        match="indistinguishable at the 0.001 nm distance resolution",
    ):
        fl.FluorophoreSystem(fluorophores)


def test_fluorophore_system_requires_fluorophores():
    with pytest.raises(
        ValueError,
        match="a fluorophore system must contain at least one fluorophore.",
    ):
        fl.FluorophoreSystem([])


def test_fluorophore_system_requires_matching_dimensions():
    fluorophores = [
        fl.Fluorophore("testfluo_1", [0, 0]),
        fl.Fluorophore("testfluo_1", [1, 0, 0]),
    ]

    with pytest.raises(
        ValueError,
        match="all fluorophore positions must have the same dimension.",
    ):
        fl.FluorophoreSystem(fluorophores)


def test_same_name_requires_same_fluorophore_data():
    data_1 = FluorophoreData()
    data_2 = FluorophoreData()

    fluorophores = [
        fl.Fluorophore("custom", [0, 0], data_1),
        fl.Fluorophore("custom", [1, 0], data_2),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "fluorophores with the same name must share the same "
            "FluorophoreData object: custom"
        ),
    ):
        fl.FluorophoreSystem(fluorophores)


def test_same_name_accepts_shared_fluorophore_data():
    data = FluorophoreData()

    fluorophores = [
        fl.Fluorophore("custom", [0, 0], data),
        fl.Fluorophore("custom", [1, 0], data),
    ]

    system = fl.FluorophoreSystem(fluorophores)

    assert not system.multi_type


# all other load_transitions parameters are tested in derive_transitions
@pytest.mark.parametrize("energy_transfer", [[True], [False]])
@pytest.mark.parametrize(
    "dirname, expected_true, expected_false, expected_warnings, "
    "energy_transfer_parameters",
    [
        ["flu_sys_unk", [], [], True, None],
        ["flu_sys_unk_cy5", ["testfluo_1"], ["testfluo_1"], True, None],
        [
            "flu_sys_2xcy5_1xatto643",
            [
                "testfluo_1",
                "D: testfluo_1, A: testfluo_1, dist: 1.0",
                "D: testfluo_1, A: testfluo_2, dist: 2.0",
                "D: testfluo_1, A: testfluo_2, dist: 1.0",
                "testfluo_2",
                "D: testfluo_2, A: testfluo_1, dist: 2.0",
                "D: testfluo_2, A: testfluo_1, dist: 1.0",
            ],
            ["testfluo_1", "testfluo_2"],
            False,
            {"exclude": ["t1", "s0"]},
        ],
    ],
)
def test_fluorophore_system_load_transitions(
    dirname,
    request,
    expected_true,
    expected_false,
    expected_warnings,
    energy_transfer,
    energy_transfer_parameters,
    caplog,
):
    if expected_warnings:
        with caplog.at_level(logging.WARNING):
            fluorophore_system = request.getfixturevalue(dirname)
            assert (
                "There is no FluorophoreData for Fluorophore aa in fluopy.fluo_data. Parameters have to be defined manually."
                in caplog.text
            )
        caplog.clear()
    else:
        fluorophore_system = request.getfixturevalue(dirname)
    if expected_warnings:
        with caplog.at_level(logging.WARNING):
            transitions = fluorophore_system.load_transitions(
                energy_transfer=energy_transfer
            )
            assert (
                "load_transitions() not available for this kind of fluorophore: aa."
                in caplog.text
            )
        caplog.clear()
    elif energy_transfer_parameters is not None:
        with caplog.at_level(logging.WARNING):
            transitions = fluorophore_system.load_transitions(
                energy_transfer=energy_transfer,
                energy_transfer_parameters=energy_transfer_parameters,
            )
            assert (
                "'overwrite', 'exclude' or 'include' in energy_transfer_parameters will affect all types of fluorophores."
                in caplog.text
            )
        caplog.clear()
    else:
        transitions = fluorophore_system.load_transitions(
            energy_transfer=energy_transfer
        )
    if energy_transfer:
        assert list(transitions) == expected_true
    else:
        assert list(transitions) == expected_false


def test_load_transitions_warns_once_per_unknown_name(caplog):
    fluorophores = [
        fl.Fluorophore("unknown", [0, 0]),
        fl.Fluorophore("unknown", [1, 0]),
    ]
    system = fl.FluorophoreSystem(fluorophores)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        transitions = system.load_transitions()

    assert transitions == {}
    assert (
        caplog.text.count(
            "load_transitions() not available for this kind of fluorophore: unknown."
        )
        == 1
    )


def test_load_transitions_does_not_mutate_energy_transfer_parameters(
    flu_sys_cy5,
):
    parameters = {
        "refractive_index": 1.4,
        "exclude": ["t1"],
    }
    original = {
        "refractive_index": 1.4,
        "exclude": ["t1"],
    }

    flu_sys_cy5.load_transitions(
        energy_transfer_parameters=parameters,
        dstorm=False,
    )

    assert parameters == original


def test_load_transitions_does_not_mutate_dstorm_parameters(
    flu_sys_cy5,
):
    parameters = {
        "concentration": 100,
    }

    flu_sys_cy5.load_transitions(
        energy_transfer=False,
        dstorm=True,
        dstorm_parameters=parameters,
    )

    assert parameters == {
        "concentration": 100,
    }


@pytest.mark.parametrize(
    "position_1, position_2, expected",
    [
        [[0, 0], [1, 0], np.array([0.5, 0.866025])],
        [[1, 0], [0, 0], np.array([0.5, -0.866025])],
    ],
)
def test_triangle_third_position(position_1, position_2, expected):
    np.testing.assert_allclose(
        fl.triangle_third_position(position_1=position_1, position_2=position_2),
        expected,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "distance, count, shape, expected",
    [
        [5, 4, None, np.array([[0, 0], [5, 0], [0, 5], [5, 5]])],
        [5, 3, "triangle", np.array([[0, 0], [5, 0], [2.5, 4.3301]])],
        [5, 3, "square", np.array([[0, 0], [5, 0], [0, 5]])],
        [5, 3, "elipse", "ValueError1"],
        [5, 2, None, np.array([[0, 0], [5, 0]])],
        [5, 1, None, np.array([[0, 0]])],
        [5, 5, None, "Warning1"],
    ],
)
def test_get_positions_from_distance(distance, count, shape, expected, caplog):
    if isinstance(expected, str):
        if expected == "ValueError1":
            with pytest.raises(
                ValueError,
                match="shape elipse not known. Can either be 'triangle' or 'square'.",
            ):
                fl.get_positions_from_distance(
                    distance=distance, count=count, shape=shape
                )
        elif expected == "Warning1":
            with caplog.at_level(logging.WARNING):
                fl.get_positions_from_distance(
                    distance=distance, count=count, shape=shape
                )
                assert "If count is above 4" in caplog.text
            caplog.clear()
    else:
        np.testing.assert_allclose(
            fl.get_positions_from_distance(distance=distance, count=count, shape=shape),
            expected,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    "name, distance, count, expected",
    [
        ["testfluo_1", 5, 3, [[0, 0], [5, 0], [2.5, 4.3301]]],
        ["aa", 1, 2, [[0, 0], [1, 0]]],
    ],
)
def test_construct_fluorophores(name, distance, count, expected, caplog):
    expected = np.asarray(expected)
    if name == "aa":
        with caplog.at_level(logging.WARNING):
            fluorophores = fl.construct_fluorophores(name, distance, count)
            assert (
                "There is no FluorophoreData for Fluorophore aa in fluopy.fluo_data. Parameters have to be defined manually."
                in caplog.text
            )
        caplog.clear()
    else:
        fluorophores = fl.construct_fluorophores(name, distance, count)
    assert len(fluorophores) == count
    for fluorophore, position in zip(fluorophores, expected):
        assert fluorophore.name == name
        np.testing.assert_allclose(fluorophore.position, position, rtol=1e-5)


def test_custom_fluorophore_automatic_transitions_and_bandpass():
    fluorophore_data = FluorophoreData(
        QUANTUM_YIELD=0.5,
        FLUORESCENCE_LIFETIME=2e-9,
        emission_spectrum=Spectrum(
            wavelengths=[500, 510, 520],
            values=[0, 1, 0],
        ),
        absorption_spectra={
            "s0": Spectrum(
                wavelengths=[500, 510, 520],
                values=[1000, 2000, 1000],
            )
        },
    )
    fluorophore = fl.Fluorophore(
        name="custom",
        position=[0, 0],
        constants=fluorophore_data,
    )
    fluorophore_system = fl.FluorophoreSystem(fluorophores=[fluorophore])

    transitions = fluorophore_system.load_transitions(
        wavelength=510,
        irradiance=1,
        energy_transfer=False,
        dstorm=False,
    )

    assert list(transitions) == ["custom"]

    excitation = next(
        transition
        for transition in transitions["custom"]
        if transition.transition_type is tr.TransitionType.EXCITATION
    )
    emission = next(
        transition
        for transition in transitions["custom"]
        if transition.transition_type is tr.TransitionType.FLUORESCENT_EMISSION
    )

    assert excitation.rate > 0
    assert emission.rate > 0

    transition_set = tr.TransitionSet(
        transitions=transitions,
        fluorophore_system=fluorophore_system,
    )
    emitting_transition_ids = em.get_emitting_transition_ids(
        bandpass=(505, 515),
        transition_set=transition_set,
    )

    assert emitting_transition_ids
    assert all(
        probability == pytest.approx(0.75)
        for probability in emitting_transition_ids.values()
    )


def test_custom_fluorophores_automatic_energy_transfer():
    donor_data = FluorophoreData(
        QUANTUM_YIELD=0.5,
        FLUORESCENCE_LIFETIME=2e-9,
        emission_spectrum=Spectrum(
            wavelengths=[500, 510, 520],
            values=[0, 1, 0],
        ),
        absorption_spectra={
            "s0": Spectrum(
                wavelengths=[505, 515],
                values=[1000, 2000],
            )
        },
    )
    acceptor_data = FluorophoreData(
        QUANTUM_YIELD=0.6,
        FLUORESCENCE_LIFETIME=3e-9,
        emission_spectrum=Spectrum(
            wavelengths=[505, 515, 525],
            values=[0, 1, 0],
        ),
        absorption_spectra={
            "s0": Spectrum(
                wavelengths=[500, 510, 520],
                values=[500, 2000, 500],
            )
        },
    )

    donor = fl.Fluorophore(
        name="custom_donor",
        position=[0, 0],
        constants=donor_data,
    )
    acceptor = fl.Fluorophore(
        name="custom_acceptor",
        position=[5, 0],
        constants=acceptor_data,
    )
    fluorophore_system = fl.FluorophoreSystem(fluorophores=[donor, acceptor])

    transitions = fluorophore_system.load_transitions(
        wavelength=510,
        irradiance=1,
        energy_transfer=True,
        dstorm=False,
    )

    forward_key = "D: custom_donor, A: custom_acceptor, dist: 5.0"
    reverse_key = "D: custom_acceptor, A: custom_donor, dist: 5.0"

    assert forward_key in transitions
    assert reverse_key in transitions

    forward_fret = next(
        transition
        for transition in transitions[forward_key]
        if transition.transition_type is tr.TransitionType.FRET
    )
    reverse_fret = next(
        transition
        for transition in transitions[reverse_key]
        if transition.transition_type is tr.TransitionType.FRET
    )

    assert forward_fret.rate > 0
    assert reverse_fret.rate > 0

    transition_set = tr.TransitionSet(
        transitions=transitions,
        fluorophore_system=fluorophore_system,
    )

    assert not transition_set.transition_df.empty
