from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fluopy import routines as routines_module
from fluopy.routines import (
    emission_post_processing,
    fingerprint_analysis,
    get_bleaching_times,
    get_delta_bleaching_times,
    truncate_fingerprints,
)


def test_emission_post_processing(em_large):
    emission = deepcopy(em_large)
    emission_post_processing(emis=emission, seed=1)
    assert not emission.event_time_series.equals(em_large.event_time_series)


def test_get_bleaching_times(sim_tr_set_1f_bl):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_1f_bl)
    assert np.isnan(bleaching_times)


def test_get_bleaching_times_2(sim_tr_set_2f_diff):
    bleaching_times = get_bleaching_times(simulation=sim_tr_set_2f_diff)
    assert np.all(np.isnan(bleaching_times))


def test_get_delta_bleaching_times():
    bleaching_times = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    deltas = get_delta_bleaching_times(bleaching_times=bleaching_times)
    deltas_expected = np.array(
        [
            [1, 10],
            [1, 10],
            [1, 10],
        ]
    )
    assert np.array_equal(deltas, deltas_expected)


def test_fingerprint_analysis_with_single_run_batch(monkeypatch):
    index = np.round(np.linspace(0, 300, 300001), decimals=12)
    event_time_series = pd.Series(np.zeros(index.size, dtype=np.int32), index=index)
    event_time_series.iloc[100] = 1

    class FakeSimulation:
        def __init__(self, transition_set):
            pass

        def run(self, **kwargs):
            pass

    class FakeEmissions:
        def __init__(self, **kwargs):
            self.event_time_points = np.array([0.1])
            self.event_time_series = event_time_series.copy()

        def extract(self, simulation):
            pass

    saved_batches = []
    saved_bleaching_times = []

    def capture_parquet(frame, path):
        saved_batches.append((frame, path))

    def capture_numpy(path, values):
        saved_bleaching_times.append((path, values))

    monkeypatch.setattr(routines_module.si, "Simulation", FakeSimulation)
    monkeypatch.setattr(routines_module.em, "Emissions", FakeEmissions)
    monkeypatch.setattr(
        routines_module, "get_bleaching_times", lambda simulation: np.array([300.0])
    )
    monkeypatch.setattr(
        routines_module, "emission_post_processing", lambda emis, seed: None
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", capture_parquet)
    monkeypatch.setattr(routines_module.np, "save", capture_numpy)
    transition_set = SimpleNamespace(fluorophore_system=SimpleNamespace(count=1))

    fingerprint, bleaching_times, delta_times = fingerprint_analysis(
        transition_set=transition_set,
        batch_size=1,
        batches=1,
        filepath="output",
        filename="test",
        seed=1,
    )

    assert len(saved_batches) == 1
    saved_batch, saved_path = saved_batches[0]
    assert isinstance(saved_batch, pd.DataFrame)
    assert saved_batch.columns.to_list() == [0]
    assert saved_path == Path("output/single_runs_test_batch_0.parquet")
    assert fingerprint.iloc[-1] == 1
    np.testing.assert_array_equal(bleaching_times, [[300.0]])
    np.testing.assert_array_equal(delta_times[0][0], [0.1])
    assert saved_bleaching_times[0][0] == Path("output/bleaching_times_test.npy")
    np.testing.assert_array_equal(saved_bleaching_times[0][1], bleaching_times)


def test_truncate_fingerprints():
    fingerprint = pd.Series([1, 2, 3, 4, 5])
    new_fingerprint = truncate_fingerprints(fingerprint=fingerprint, low=1, high=4)
    fingerprint_expected = pd.Series([0, 0.5, 1], index=[1, 2, 3])
    assert new_fingerprint.equals(fingerprint_expected)
