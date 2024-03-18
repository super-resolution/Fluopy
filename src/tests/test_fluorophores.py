import pytest
import numpy as np
import src.fluorophores as fl
from src.fluo_data import Cy5, Atto643


@pytest.mark.parametrize(
    "name, position, exp_identity, exp_name, exp_position, exp_constants",
    [
        ["cy5", [0, 0], None, "cy5", np.array([0, 0]), Cy5],
        ["atto643", [1.5, 0.34], None, "atto643", np.array([1.5, 0.34]), Atto643],
        ["aa", [0, -5], None, "aa", np.array([0, -5]), None],
    ],
)
@pytest.mark.filterwarnings("ignore:Fluorophore:UserWarning")
def test_fluorophore(
    name, position, exp_identity, exp_name, exp_position, exp_constants
):
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
    "dirnames, exp_distances, exp_count",
    [
        [["flu_obj_cy5_1"], {}, 1],
        [["flu_obj_cy5_1", "flu_obj_cy5_2"], {(0, 1): 1, (1, 0): 1}, 2],
        [["flu_obj_cy5_1", "flu_obj_cy5_1"], "ValueError1", None],
        [["flu_obj_atto643", "flu_obj_cy5_1"], {(0, 1): 2, (1, 0): 2}, 2],
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
        ],
    ],
)
@pytest.mark.filterwarnings("ignore:Fluorophore:UserWarning")
def test_fluorophore_system(dirnames, request, exp_distances, exp_count):
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


# all other load_transitions parameters are tested in derive_transitions
@pytest.mark.parametrize("energy_transfer", [[True], [False]])
@pytest.mark.parametrize(
    "dirname, expected_true, expected_false",
    [
        ["flu_sys_unk", [], []],
        ["flu_sys_unk_cy5", ["cy5"], ["cy5"]],
        [
            "flu_sys_2xcy5_1xatto643",
            [
                "cy5",
                "D: cy5, A: cy5, dist: 1.0",
                "D: cy5, A: atto643, dist: 2.0",
                "D: cy5, A: atto643, dist: 1.0",
                "atto643",
                "D: atto643, A: cy5, dist: 2.0",
                "D: atto643, A: cy5, dist: 1.0",
            ],
            ["cy5", "atto643"],
        ],
    ],
)
@pytest.mark.filterwarnings("ignore:Fluorophore:UserWarning")
@pytest.mark.filterwarnings("ignore:load_transitions():UserWarning")
def test_fluorophore_system_load_transitions(
    dirname, request, expected_true, expected_false, energy_transfer
):
    fluorophore_system = request.getfixturevalue(dirname)
    transitions = fluorophore_system.load_transitions(energy_transfer=energy_transfer)
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
        [5, 0, None, "ValueError2"],
        [5, 5, None, "ValueError2"],
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
        elif expected == "ValueError2":
            with pytest.raises(
                ValueError, match="count has to be one of 1, 2, 3 or 4."
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
    [["cy5", 5, 3, [[0, 0], [5, 0], [2.5, 4.3301]]], ["aa", 1, 2, [[0, 0], [1, 0]]]],
)
@pytest.mark.filterwarnings("ignore:Fluorophore:UserWarning")
def test_construct_fluorophores(name, distance, count, expected):
    expected = np.asarray(expected)
    fluorophores = fl.construct_fluorophores(name, distance, count)
    assert len(fluorophores) == count
    for fluorophore, position in zip(fluorophores, expected):
        assert fluorophore.name == name
        np.testing.assert_allclose(fluorophore.position, position, rtol=1e-5)
