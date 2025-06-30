import pytest
import numpy as np
from fluopy import fluorophores as fl
from fluopy.fluo_data import TestFluo_1, TestFluo_2


@pytest.mark.parametrize(
    "name, position, exp_identity, exp_name, exp_position, exp_constants",
    [
        ["testfluo_1", [0, 0], None, "testfluo_1", np.array([0, 0]), TestFluo_1],
        [
            "testfluo_2",
            [1.5, 0.34],
            None,
            "testfluo_2",
            np.array([1.5, 0.34]),
            TestFluo_2,
        ],
        ["aa", [0, -5], None, "aa", np.array([0, -5]), None],
    ],
)
def test_fluorophore(
    name, position, exp_identity, exp_name, exp_position, exp_constants
):
    if name == "aa":
        with pytest.warns(
            UserWarning,
            match="Fluorophore aa not known. Parameters have to be defined manually.",
        ):
            fluorophore = fl.Fluorophore(name=name, position=position)
    else:
        fluorophore = fl.Fluorophore(name=name, position=position)
    assert fluorophore.identity == exp_identity
    assert fluorophore.name == exp_name
    np.testing.assert_array_equal(fluorophore.position, exp_position)
    if exp_constants is not None:
        assert isinstance(fluorophore.constants, exp_constants)
    else:
        assert fluorophore.constants is None


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
def test_fluorophore_system(dirnames, request, exp_distances, exp_count, multi_type):
    if "flu_obj_unknown" in dirnames:
        with pytest.warns(
            UserWarning,
            match="Fluorophore aa not known. Parameters have to be defined manually.",
        ):
            fluorophores = [request.getfixturevalue(dirname) for dirname in dirnames]
    else:
        fluorophores = [request.getfixturevalue(dirname) for dirname in dirnames]
    if exp_distances == "ValueError1":
        with pytest.raises(
            ValueError,
            match="at least two fluorophores share the same position. Also "
            "check for duplicates.",
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
):
    if expected_warnings:
        with pytest.warns(
            UserWarning,
            match="Fluorophore aa not known. Parameters have to be defined manually.",
        ):
            fluorophore_system = request.getfixturevalue(dirname)
    else:
        fluorophore_system = request.getfixturevalue(dirname)
    if expected_warnings:
        with pytest.warns(
            UserWarning,
            match="load_transitions\\(\\) not available for this kind of fluorophore: "
            "aa.",
        ):
            transitions = fluorophore_system.load_transitions(
                energy_transfer=energy_transfer
            )
    elif energy_transfer_parameters is not None:
        with pytest.warns(
            UserWarning,
            match="'overwrite', 'exclude' or 'include' in energy_transfer_parameters "
            "will effect all types of fluorophores.",
        ):
            transitions = fluorophore_system.load_transitions(
                energy_transfer=energy_transfer,
                energy_transfer_parameters=energy_transfer_parameters,
            )
    else:
        transitions = fluorophore_system.load_transitions(
            energy_transfer=energy_transfer
        )
    if energy_transfer:
        assert list(transitions) == expected_true
    else:
        assert list(transitions) == expected_false


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
def test_get_positions_from_distance(distance, count, shape, expected):
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
            with pytest.warns(
                UserWarning, match="If count is above 4"
            ):
                fl.get_positions_from_distance(
                    distance=distance, count=count, shape=shape
                )
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
def test_construct_fluorophores(name, distance, count, expected):
    expected = np.asarray(expected)
    if name == "aa":
        with pytest.warns(
            UserWarning,
            match="Fluorophore aa not known. Parameters have to be defined manually.",
        ):
            fluorophores = fl.construct_fluorophores(name, distance, count)
    else:
        fluorophores = fl.construct_fluorophores(name, distance, count)
    assert len(fluorophores) == count
    for fluorophore, position in zip(fluorophores, expected):
        assert fluorophore.name == name
        np.testing.assert_allclose(fluorophore.position, position, rtol=1e-5)
