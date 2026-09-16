"""
Unit tests for fitting.py
"""

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult

import fluopy.fitting as fitting
from fluopy.distributions import ExponentialMixtureModel
from fluopy.fitting import (
    convert_dicts,
    load_from_array,
    log_likelihood_hist_marginal_v2,
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
        bin_edges = np.array([10.0, 20.0, 50.0, 90.0])
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
        assert results == pytest.approx(71.82550273177682)

        bin_edges = (10.0, 20.0, 50.0, 90.0)
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
        assert results == pytest.approx(71.82550273177682)

        bin_edges = np.array([0.0, 10.0, 50.0, 90.0])
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
        assert results == pytest.approx(46.54028819370892)

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

    def test_full_support_with_no_unobserved_counts(self):
        result = log_likelihood_hist_v2(
            model=ExponentialMixtureModel,
            params={"pis": [], "lambdas": [1]},
            counts=[1],
            bin_edges=[0, np.inf],
            counts_not_observed=0,
        )

        assert result == pytest.approx(0)


def test_log_likelihood_hist_marginal_v2_with_small_truncation():
    truncation_up = 0.001

    def uniform_pdf_part(*, call, x, i, normalize):
        return np.full_like(x, 1 / truncation_up, dtype=np.float64)

    result = log_likelihood_hist_marginal_v2(
        model=ExponentialMixtureModel,
        params={"pis": [], "lambdas": [1000]},
        counts=[1],
        bin_edges=[0, truncation_up],
        counts_not_observed=0,
        truncation_low=0,
        truncation_up=truncation_up,
        pfa_pdf_part=uniform_pdf_part,
        pdf_part_index=0,
    )

    assert result == pytest.approx(1, rel=1e-4)


@pytest.mark.parametrize(
    "likelihood, extra_arguments",
    [
        (
            fitting.log_likelihood_hist_v1,
            {"truncation_low": 0, "truncation_up": 1},
        ),
        (
            fitting.log_likelihood_hist_marginal_v1,
            {
                "truncation_low": 0,
                "truncation_up": 1,
                "pfa_cdf_part": lambda x, i, normalize: 1.0,
                "cdf_part_index": 0,
            },
        ),
        (fitting.log_likelihood_hist_v2, {}),
        (
            fitting.log_likelihood_hist_marginal_v2,
            {
                "truncation_low": 0,
                "truncation_up": 1,
                "pfa_pdf_part": lambda call, x, i, normalize: 1.0,
                "pdf_part_index": 0,
            },
        ),
    ],
)
def test_public_likelihood_rejects_mismatched_histogram(
    likelihood,
    extra_arguments,
):
    with pytest.raises(ValueError, match="exactly one more value"):
        likelihood(
            model=ExponentialMixtureModel,
            params={"pis": [], "lambdas": [1]},
            counts=[1],
            bin_edges=[0, 1, 2],
            counts_not_observed=0,
            **extra_arguments,
        )


@pytest.mark.parametrize(
    "fitter",
    [fitting.fit_multiple_mixture_v1, fitting.fit_multiple_mixture_v2],
)
def test_fitter_penalizes_ill_conditioned_hypoexponential_rates(monkeypatch, fitter):
    candidate = np.array(
        [
            0.8,
            0.95165,
            0.5,
            0.7,
            0.95055,
            0.4,
            0.6,
            0.94945,
            0.3,
            0.5,
            0.94835,
            0.2,
        ]
    )

    def differential_evolution(objective, **kwargs):
        assert objective(candidate) == np.inf
        return OptimizeResult(x=candidate, fun=0.0, success=True)

    monkeypatch.setattr(fitting, "differential_evolution", differential_evolution)
    datasets = [np.array([1]) for _ in range(4)]

    result = fitter(
        datasets=datasets,
        bin_edges=[0, 1],
        pfa_bin_edges=[0, 1],
        pfa_counts=[1],
    )

    assert result.fun == 0.0


@pytest.mark.parametrize(
    "fitter, marginal_likelihood_name",
    [
        (
            fitting.fit_multiple_mixture_v1,
            "log_likelihood_hist_marginal_v1",
        ),
        (
            fitting.fit_multiple_mixture_v2,
            "log_likelihood_hist_marginal_v2",
        ),
    ],
)
def test_fitter_uses_custom_truncation_up(
    monkeypatch,
    fitter,
    marginal_likelihood_name,
):
    domains = []
    marginal_truncation_limits = []
    original_pfa_model = fitting.dist.Photoswitching_fingerprint_model
    original_marginal_likelihood = getattr(fitting, marginal_likelihood_name)

    def pfa_model(*, params, domain):
        domains.append(domain)
        return original_pfa_model(params=params, domain=domain)

    def marginal_likelihood(**kwargs):
        marginal_truncation_limits.append(kwargs["truncation_up"])
        return original_marginal_likelihood(**kwargs)

    candidate = np.array([0.8, 1.2, 0.4, 0.7, 0.8, 0.2])

    def differential_evolution(objective, **kwargs):
        objective_value = objective(candidate)
        assert np.isfinite(objective_value)
        return OptimizeResult(x=candidate, fun=objective_value, success=True)

    monkeypatch.setattr(fitting.dist, "Photoswitching_fingerprint_model", pfa_model)
    monkeypatch.setattr(fitting, marginal_likelihood_name, marginal_likelihood)
    monkeypatch.setattr(fitting, "differential_evolution", differential_evolution)

    fitter(
        datasets=[np.array([1]), np.array([1])],
        bin_edges=[0, 1],
        truncation_up=17,
    )

    assert domains == [(0, 17)]
    assert marginal_truncation_limits == [17]


def test_fitter_rejects_nonfinite_best_objective(monkeypatch):
    def differential_evolution(objective, **kwargs):
        return OptimizeResult(fun=np.inf, success=False)

    monkeypatch.setattr(fitting, "differential_evolution", differential_evolution)

    with pytest.raises(RuntimeError, match="did not find parameters"):
        fitting.fit_multiple_mixture_v1(
            datasets=[np.array([1])],
            bin_edges=[0, 1],
        )


@pytest.mark.parametrize(
    "fitter",
    [fitting.fit_multiple_mixture_v1, fitting.fit_multiple_mixture_v2],
)
def test_fitter_rejects_z_below_minus_one(fitter):
    with pytest.raises(ValueError, match="z must be -1 or between 0"):
        fitter(
            datasets=[np.array([1]), np.array([1])],
            bin_edges=[0, 1],
            z=-2,
        )


@pytest.mark.parametrize(
    "arguments, error",
    [
        (
            {"datasets": [np.array([1])], "bin_edges": [0, 1, 2]},
            "exactly one more value",
        ),
        (
            {"datasets": [np.array([-1])], "bin_edges": [0, 1]},
            "counts must be finite and nonnegative",
        ),
        (
            {"datasets": [np.array([0])], "bin_edges": [0, 1], "norm": True},
            "Cannot normalize an empty histogram",
        ),
        (
            {
                "datasets": [np.array([1])],
                "bin_edges": [0, 1],
                "pfa_counts": [1],
            },
            "pfa_bin_edges and pfa_counts must be provided together",
        ),
        (
            {
                "datasets": [np.array([1])],
                "bin_edges": [0, 1],
                "counts_not_observed": [],
            },
            "counts_not_observed must have one value per dataset",
        ),
    ],
)
@pytest.mark.parametrize(
    "fitter",
    [fitting.fit_multiple_mixture_v1, fitting.fit_multiple_mixture_v2],
)
def test_fitter_validates_histogram_inputs(fitter, arguments, error):
    with pytest.raises(ValueError, match=error):
        fitter(**arguments)


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
