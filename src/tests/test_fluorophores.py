import pytest
import numpy as np
import src.fluorophores as fl
import src.fluorophore_collection.cy5 as cy5


@pytest.mark.filterwarnings("ignore:Fluorophore:UserWarning")
@pytest.mark.parametrize('fluorophore_object,expected',
                         [[['atto488', [3.5, -4]], [None, 'atto488', np.array([3.5, -4]), None]],
                          [['Cy5', [1, 1], 'test'], [None, 'Cy5', np.array([1, 1]), cy5.Cy5('test')]]],
                          indirect=['fluorophore_object'])
def test_fluorophore(fluorophore_object, expected):
    if expected[0] is None:
        assert fluorophore_object.id is None
    else:
        assert fluorophore_object.id == expected[0]
    assert fluorophore_object.name == expected[1]
    np.testing.assert_array_equal(fluorophore_object.position, expected[2])
    if expected[3] is None:
        assert fluorophore_object.constants is None
    else:
        assert fluorophore_object.constants == expected[3]


@pytest.mark.parametrize('positions,expected',
                         [[[[1, 1], [2, 1], [1, 2]], {(0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 2): 1.414, (2, 0): 1,
                                                      (2, 1): 1.414}],
                          [[[0, 1], [1, 1]], {(0, 1): 1, (1, 0): 1}],
                          [[[1, 1]], {}]])
def test_get_distances(positions, expected):
    assert fl.get_distances(positions) == pytest.approx(expected)


@pytest.mark.parametrize('fluorophore_system_object,expected',
                         [[[['Cy5', [0, 0]], ['Cy5', [0, 1]]], [{(0, 1): 1, (1, 0): 1}, 2]]],
                         indirect=['fluorophore_system_object'])
def test_fluorophoresystem(fluorophore_system_object, expected):
    for i, fluorophore in enumerate(fluorophore_system_object.fluorophores):
        assert fluorophore.id == i
    assert fluorophore_system_object.distances == expected[0]
    assert fluorophore_system_object.count == expected[1]


@pytest.mark.parametrize('position_1,position_2,expected',
                         [[[0, 0], [1, 0], np.array([0.5, 0.866025])],
                          [[1, 0], [0, 0], np.array([0.5, -0.866025])]])
def test_triangle_third_position(position_1, position_2, expected):
    np.testing.assert_allclose(fl.triangle_third_position(position_1, position_2), expected, rtol=1e-5)


@pytest.mark.parametrize('distance,count,expected',
                         [[5, 4, np.array([[0, 0], [5, 0], [0, 5], [5, 5]])],
                          [5, 3, np.array([[0, 0], [5, 0], [2.5, 4.330127]])],
                          [5, 2, np.array([[0, 0], [5, 0]])],
                          [5, 1, np.array([[0, 0]])],
                          [5, 0, 'AttributeError'],
                          [5, 5, 'AttributeError']])
def test_get_positions_from_distance(distance, count, expected):
    if isinstance(expected, str):
        if expected == 'AttributeError':
            with pytest.raises(AttributeError):
                fl.get_positions_from_distance(distance, count)
    else:
        np.testing.assert_allclose(fl.get_positions_from_distance(distance, count), expected)


@pytest.mark.parametrize('name,distance,count,expected',
                         [['cy5', 5, 3, [[0, 0], [5, 0], [2.5, 4.3301]]]])
def test_construct_fluorophores(name, distance, count, expected):
    expected = np.asarray(expected)
    fluorophores = fl.construct_fluorophores(name, distance, count)
    assert len(fluorophores) == count
    for fluorophore, position in zip(fluorophores, expected):
        assert fluorophore.name == name
        np.testing.assert_allclose(fluorophore.position, position, rtol=1e-5)
