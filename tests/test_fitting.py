"""
Unit tests for fitting.py
"""

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from fluopy.distributions import ExponentialMixtureModel
from fluopy.fitting import (
    convert_dicts,
    load_from_array,
    log_likelihood_hist_v1,
    log_likelihood_hist_v2,
    prepare_constraints,
    prepare_exp_mixture_parameters,
    prepare_pfa_parameters,
    save_as_array,
)


class TestLogLikelihoodHistV1:

    def test_init(self):
        model = ExponentialMixtureModel
        params = {
            "pis": [0.2],
            "lambdas": [0.1, 1],
        }
        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        results = log_likelihood_hist_v1(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            truncation_low=10,
            truncation_up=90,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.5051141015797)

        bin_edges = (0.0, 10.0, 50.0, 100.0)
        counts = (5.0, 10.0, 3.0)
        results = log_likelihood_hist_v1(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            truncation_low=10,
            truncation_up=90,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.5051141015797)

        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        results = log_likelihood_hist_v1(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            truncation_low=0,
            truncation_up=90,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.50511410157997)

    def test_large_counts_not_observed_increases_nll(self):
        model = ExponentialMixtureModel
        params = {
            "pis": [0.2],
            "lambdas": [0.1, 1],
        }
        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        ll0 = log_likelihood_hist_v1(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            truncation_low=10,
            truncation_up=90,
            counts_not_observed=0,
        )
        ll100 = log_likelihood_hist_v1(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            truncation_low=10,
            truncation_up=90,
            counts_not_observed=100,
        )
        assert ll0 < ll100


class TestLogLikelihoodHistV2:

    def test_init(self):
        model = ExponentialMixtureModel
        params = {
            "pis": [0.2],
            "lambdas": [0.1, 1],
        }
        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        results = log_likelihood_hist_v2(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.5051141015797)

        bin_edges = (0.0, 10.0, 50.0, 100.0)
        counts = (5.0, 10.0, 3.0)
        results = log_likelihood_hist_v2(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.5051141015797)

        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        results = log_likelihood_hist_v2(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            counts_not_observed=0,
        )
        assert results == pytest.approx(46.50511410157997)

    def test_large_counts_not_observed_increases_nll(self):
        model = ExponentialMixtureModel
        params = {
            "pis": [0.2],
            "lambdas": [0.1, 1],
        }
        bin_edges = np.array([0.0, 10.0, 50.0, 100.0])
        counts = np.array([5.0, 10.0, 3.0])
        ll0 = log_likelihood_hist_v2(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            counts_not_observed=0,
        )
        ll100 = log_likelihood_hist_v2(
            model=model,
            params=params,
            counts=counts,
            bin_edges=bin_edges,
            counts_not_observed=100,
        )
        assert ll0 < ll100


class TestPrepareConstraints:

    def test_returns_linear_constraint_and_bounds(self):
        n = 3
        lc, bounds = prepare_constraints(n=n, z=-1)
        assert isinstance(lc, LinearConstraint)
        assert isinstance(bounds, Bounds)

        # Bounds should cover n*3 parameters when z=-1.
        assert len(bounds.lb) == n * 3
        assert len(bounds.ub) == n * 3

    def test_z_not_minus_one_bounds_shape(self):
        """Bounds should cover 5 + (n-1)*3 parameters when z != -1."""
        n = 3
        z = 0
        lc, bounds = prepare_constraints(n=n, z=z)
        expected = 5 + (n - 1) * 3
        assert len(bounds.lb) == expected
        assert len(bounds.ub) == expected


class TestPreparePfaParameters:

    def test_z_minus_one_two_datasets(self):
        """z=-1: each dataset gets [p, 1-p, lam_b, lam_nb]."""
        params = [0.3, 1.0, 0.5, 0.6, 2.0, 0.8]
        result = prepare_pfa_parameters(z=-1, n=2, params=params)
        assert set(result.keys()) == {0, 1}
        # dataset 0: p=0.3, 1-p=0.7, lam_b=1.0, lam_nb=0.5
        assert result[0] == pytest.approx([0.3, 0.7, 1.0, 0.5])
        # dataset 1: p=0.6, 1-p=0.4, lam_b=2.0, lam_nb=0.8
        assert result[1] == pytest.approx([0.6, 0.4, 2.0, 0.8])

    def test_z_minus_one_three_datasets(self):
        """z=-1 with 3 datasets returns 3 keys."""
        params = [0.2, 1.0, 0.4, 0.5, 1.5, 0.3, 0.8, 2.0, 0.6]
        result = prepare_pfa_parameters(z=-1, n=3, params=params)
        assert len(result) == 3

    def test_z_zero(self):
        """z=0: first dataset gets 3-component params."""
        # params: [uz, vz, lam_bz, lam_nbz1, lam_nbz2, p1, lam_b1, lam_nb1]
        params = [0.4, 0.5, 2.0, 1.0, 0.5, 0.6, 1.5, 0.3]
        result = prepare_pfa_parameters(z=0, n=2, params=params)
        assert len(result) == 2

        uz, vz = 0.4, 0.5
        pz1 = uz  # 0.4
        pz2 = (1 - uz) * vz  # 0.3
        pz3 = (1 - uz) * (1 - vz)  # 0.3
        assert result[0] == pytest.approx([pz1, pz2, pz3, 2.0, 1.0, 0.5])


class TestPrepareExpMixtureParameters:

    def test_z_minus_one(self):
        """Each entry should have 'pis' (length 1) and 'lambdas' (length 2)."""
        params = [0.3, 1.0, 0.5, 0.6, 2.0, 0.8]  # dataset 0  # dataset 1
        result = prepare_exp_mixture_parameters(z=-1, n=2, params=params)
        for key in result:
            assert "pis" in result[key]
            assert "lambdas" in result[key]
            assert len(result[key]["pis"]) == 1
            assert len(result[key]["lambdas"]) == 2

        assert result[0]["pis"] == pytest.approx([0.3])
        assert result[0]["lambdas"] == pytest.approx([1.0, 0.5])
        assert result[1]["pis"] == pytest.approx([0.6])
        assert result[1]["lambdas"] == pytest.approx([2.0, 0.8])

    def test_z_equal_zero_three_component(self):
        """Dataset z should have pis length 2 and lambdas length 3."""
        params = [0.4, 0.5, 2.0, 1.0, 0.5, 0.6, 1.5, 0.3]
        result = prepare_exp_mixture_parameters(z=0, n=2, params=params)
        assert len(result[0]["pis"]) == 2
        assert len(result[0]["lambdas"]) == 3


class TestSaveLoadArray:

    def test_round_trip(self, tmp_path):
        parameter_dict = {0: [0.3, 1.0, 0.5], 1: [0.6, 2.0, 0.8]}
        filepath = str(tmp_path / "params.npy")

        save_as_array(parameter_dict=parameter_dict, filepath=filepath)
        loaded = load_from_array(filepath)

        for key in parameter_dict:
            assert loaded[key] == pytest.approx(parameter_dict[key])

    def test_single_key(self, tmp_path):
        parameter_dict = {0: [0.1, 0.9, 2.5, 0.4]}
        filepath = str(tmp_path / "single.npy")

        save_as_array(parameter_dict, filepath)
        loaded = load_from_array(filepath)

        for key in parameter_dict:
            assert loaded[key] == pytest.approx(parameter_dict[key])


class TestConvertDicts:

    def test_length_4_entry(self):
        """A length-4 entry should produce pis=[p0], lambdas=[lam2, lam3]."""
        pfa_dict = {0: [0.3, 0.7, 1.5, 0.5]}
        result = convert_dicts(pfa_dict)
        assert result[0]["pis"] == pytest.approx([0.3])
        assert result[0]["lambdas"] == pytest.approx([1.5, 0.5])

    def test_length_6_entry(self):
        """A length-6 entry should produce pis=[p0,p1], lambdas=[p3,p4,p5]."""
        pfa_dict = {0: [0.4, 0.3, 0.3, 2.0, 1.0, 0.5]}
        result = convert_dicts(pfa_dict)
        assert result[0]["pis"] == pytest.approx([0.4, 0.3])
        assert result[0]["lambdas"] == pytest.approx([2.0, 1.0, 0.5])

    def test_mixed_entries(self):
        """Mixed-length entries both converted correctly."""
        pfa_dict = {
            0: [0.4, 0.3, 0.3, 2.0, 1.0, 0.5],
            1: [0.6, 0.4, 1.5, 0.3],
        }
        result = convert_dicts(pfa_dict)
        assert len(result[0]["pis"]) == 2
        assert len(result[1]["pis"]) == 1

    def test_empty_dict(self):
        assert convert_dicts({}) == {}
