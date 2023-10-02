import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import gc
import numpy as np
import simulation as si


@pytest.mark.parametrize('expected',
                         ['ValueError', ''])
def test_simulation(transition_set_object, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            si.Simulation(transition_set_object)
    else:
        transition_set_object.finalize()
        simulation = si.Simulation(transition_set_object)
        assert simulation.transitions is not None
        assert simulation.time_series is None
        assert simulation.transition_series is None
        assert simulation.state_series is None


def test_direct_method_steps(transition_set_object):
    transition_set_object.finalize()
    size = 20
    time_series, transition_series = si.direct_method_steps(transition_matrix=transition_set_object.transition_matrix,
                                                            row_sums=transition_set_object.row_sums, start_index=0,
                                                            size=size, seed=3, use_memmap=None)
    assert time_series[0] == 0
    assert time_series.size == size + 1
    assert transition_series.size == size
    time_series_memmap, transition_series_memmap = \
        si.direct_method_steps(transition_matrix=transition_set_object.transition_matrix,
                               row_sums=transition_set_object.row_sums,
                               start_index=0, size=size, seed=3, use_memmap='')
    np.testing.assert_array_equal(time_series, time_series_memmap)
    np.testing.assert_array_equal(transition_series, transition_series_memmap)
    assert len(time_series_memmap.base) == time_series.size * 64 / 8
    assert len(transition_series_memmap.base) == transition_series.size * 32 / 8
    del time_series_memmap
    del transition_series_memmap
    gc.collect()
    os.remove('time_series')
    os.remove('transition_series')


def test_direct_method_steps_bleach(transition_set_object_bleach):
    transition_set_object_bleach.finalize()
    size = 100
    time_series, transition_series = \
        si.direct_method_steps(transition_matrix=transition_set_object_bleach.transition_matrix,
                               row_sums=transition_set_object_bleach.row_sums, start_index=0,
                               size=size, seed=3, use_memmap=None)
    assert time_series[0] == 0
    assert time_series.size != size + 1
    assert transition_series.size != size
    assert time_series.size == transition_series.size + 1
    time_series_memmap, transition_series_memmap = \
        si.direct_method_steps(transition_matrix=transition_set_object_bleach.transition_matrix,
                               row_sums=transition_set_object_bleach.row_sums,
                               start_index=0, size=size, seed=3, use_memmap='')
    np.testing.assert_array_equal(time_series, time_series_memmap)
    np.testing.assert_array_equal(transition_series, transition_series_memmap)
    assert len(time_series_memmap.base) == time_series.size * 64 / 8
    assert len(transition_series_memmap.base) == transition_series.size * 32 / 8
    del time_series_memmap
    del transition_series_memmap
    gc.collect()
    os.remove('time_series')
    os.remove('transition_series')


def test_direct_method_time(transition_set_object):
    transition_set_object.finalize()
    end_time = 10
    time_series, transition_series = si.direct_method_time(transition_matrix=transition_set_object.transition_matrix,
                                                           row_sums=transition_set_object.row_sums, start_index=0,
                                                           size=10, end_time=end_time, seed=3, use_memmap=None)
    assert time_series[0] == 0
    assert time_series.size == transition_series.size + 2
    assert time_series[-1] == end_time
    time_series_memmap, transition_series_memmap = \
        si.direct_method_time(transition_matrix=transition_set_object.transition_matrix,
                              row_sums=transition_set_object.row_sums,
                              start_index=0, size=10, end_time=end_time, seed=3, use_memmap='')
    np.testing.assert_array_equal(time_series, time_series_memmap)
    np.testing.assert_array_equal(transition_series, transition_series_memmap)
    assert len(time_series_memmap.base) == time_series.size * 64 / 8
    assert len(transition_series_memmap.base) == transition_series.size * 32 / 8
    del time_series_memmap
    del transition_series_memmap
    gc.collect()
    os.remove('time_series')
    os.remove('transition_series')


def test_direct_method_time_bleach(transition_set_object_bleach):
    transition_set_object_bleach.finalize()
    end_time = 10
    time_series, transition_series = si.direct_method_time(transition_matrix=transition_set_object_bleach.transition_matrix,
                                                           row_sums=transition_set_object_bleach.row_sums, start_index=0,
                                                           size=10, end_time=end_time, seed=3, use_memmap=None)
    assert time_series[0] == 0
    assert time_series.size == transition_series.size + 2
    assert time_series[-1] == end_time
    assert transition_set_object_bleach.combined_state_transitions_df['final_state'][transition_series[-1]] == (7, 7)
    time_series_memmap, transition_series_memmap = \
        si.direct_method_time(transition_matrix=transition_set_object_bleach.transition_matrix,
                              row_sums=transition_set_object_bleach.row_sums,
                              start_index=0, size=10, end_time=end_time, seed=3, use_memmap='')
    np.testing.assert_array_equal(time_series, time_series_memmap)
    np.testing.assert_array_equal(transition_series, transition_series_memmap)
    assert len(time_series_memmap.base) == time_series.size * 64 / 8
    assert len(transition_series_memmap.base) == transition_series.size * 32 / 8
    del time_series_memmap
    del transition_series_memmap
    gc.collect()
    os.remove('time_series')
    os.remove('transition_series')


@pytest.mark.parametrize('start_at,memmap,expected',
                         [[None, None, [None, None]],
                          [(1, 1, 0), None, ['ValueError', None]],
                          [(1, 1, 0, 1), '', [np.array([1, 1, 0, 1]), '']]])
def test_run(start_at, memmap, expected, transition_set_object):
    transition_set_object.finalize()
    simulation = si.Simulation(transition_set_object)
    if expected[0] == 'ValueError':
        with pytest.raises(ValueError):
            simulation.run(start_at=start_at, size=10, end_time=None, seed=3, use_memmap=memmap)
    else:
        simulation.run(start_at=start_at, size=10, end_time=None, seed=3, use_memmap=memmap)
        if expected[0] is None:
            assert not np.any(simulation.state_series[:, 0])
        else:
            np.testing.assert_array_equal(simulation.state_series[:, 0], expected[0])
        if expected[1] is None:
            assert simulation.memmap_path is None
        else:
            assert simulation.memmap_path == expected[1]
            assert len(simulation.state_series.base) == simulation.state_series.shape[1] * 8 / 8 * \
                   simulation.state_series.shape[0]  # 8 bit per number (* 8), result in byte (/ 8)
        assert simulation.state_series.shape[1] == simulation.transition_series.size + 1


def test_delete_memmaps(transition_set_object):
    transition_set_object.finalize()
    simulation = si.Simulation(transition_set_object)
    simulation.run(start_at=None, size=10, end_time=None, seed=3, use_memmap='')
    assert os.path.isfile(os.path.join('', 'transition_series'))
    assert hasattr(simulation, 'transition_series')
    simulation.delete_memmaps()
    assert not hasattr(simulation, 'transition_series')
    assert not os.path.isfile(os.path.join('', 'transition_series'))
