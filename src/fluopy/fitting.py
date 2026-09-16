"""
Tools for fitting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    OptimizeResult,
    differential_evolution,
)

from . import distributions as dist

__all__: list[str] = []


def _prepare_histogram_inputs(
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    counts_not_observed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    counts_array = np.asarray(counts, dtype=np.float64)
    bin_edges_array = np.asarray(bin_edges, dtype=np.float64)

    if counts_array.ndim != 1 or bin_edges_array.ndim != 1:
        raise ValueError("counts and bin_edges must be one-dimensional.")
    if bin_edges_array.size != counts_array.size + 1:
        raise ValueError("bin_edges must contain exactly one more value than counts.")
    if not np.all(np.isfinite(counts_array)) or np.any(counts_array < 0):
        raise ValueError("counts must be finite and nonnegative.")
    if np.any(np.isnan(bin_edges_array)) or np.any(np.diff(bin_edges_array) <= 0):
        raise ValueError(
            "bin_edges must be strictly increasing and cannot contain NaN."
        )
    if not np.isfinite(counts_not_observed) or counts_not_observed < 0:
        raise ValueError("counts_not_observed must be finite and nonnegative.")

    return counts_array, bin_edges_array


def _validate_fitter_histogram_inputs(
    datasets: Sequence[npt.ArrayLike],
    bin_edges: npt.ArrayLike,
    counts_not_observed: Sequence[int],
    pfa_bin_edges: npt.ArrayLike | None,
    pfa_counts: npt.ArrayLike | None,
    pfa_counts_not_observed: int,
    norm: bool,
) -> None:
    if not datasets:
        raise ValueError("datasets must contain at least one histogram.")
    if len(counts_not_observed) != len(datasets):
        raise ValueError("counts_not_observed must have one value per dataset.")

    for data, count_not_observed in zip(datasets, counts_not_observed):
        data_array, _ = _prepare_histogram_inputs(
            data,
            bin_edges,
            count_not_observed,
        )
        if norm and data_array.sum() == 0:
            raise ValueError("Cannot normalize an empty histogram.")

    if (pfa_bin_edges is None) != (pfa_counts is None):
        raise ValueError("pfa_bin_edges and pfa_counts must be provided together.")
    if not np.isfinite(pfa_counts_not_observed) or pfa_counts_not_observed < 0:
        raise ValueError("pfa_counts_not_observed must be finite and nonnegative.")
    if pfa_counts is not None and pfa_bin_edges is not None:
        pfa_counts_array, _ = _prepare_histogram_inputs(
            pfa_counts,
            pfa_bin_edges,
            pfa_counts_not_observed,
        )
        if norm and pfa_counts_array.sum() == 0:
            raise ValueError("Cannot normalize an empty PFA histogram.")


def negative_log_likelihood_hist_observation_window(
    model: Callable[..., Any],
    params: Iterable[Any],
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a distribution specified by its CDF and parameters.
    The observation window is defined explicitly by truncation_low and truncation_up.
    Bin probabilities are conditional on an event falling within this window, while
    separate likelihood terms account for observing or not observing an event. In the
    bin-complement formulation, the histogram bins instead define the observable range.

    Parameters
    ----------
    model
        A custom distribution (class) defined in distributions.py.
    params
        Parameters of the distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.
    counts_not_observed
        Number of events not observed due to truncation.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    counts_array, bin_edges_array = _prepare_histogram_inputs(
        counts,
        bin_edges,
        counts_not_observed,
    )

    a = bin_edges_array[:-1]
    b = bin_edges_array[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = model(params, domain=(truncation_low, truncation_up)).cdf(b) - model(
        params, domain=(truncation_low, truncation_up)
    ).cdf(a)
    probs = np.clip(probs, a_min=1e-14, a_max=1)
    log_likelihood_bin = np.sum(counts_array * np.log(probs))

    prob_event = model(params, domain=(0, np.inf)).cdf(truncation_up) - model(
        params, domain=(0, np.inf)
    ).cdf(truncation_low)
    # probability of observing an event
    # within the truncation range (given the distribution is non-truncated)
    prob_event = np.clip(prob_event, a_min=1e-14, a_max=1 - 1e-14)
    log_likelihood_no_observation = np.log1p(-prob_event) * counts_not_observed
    log_likelihood_observation = np.sum(counts_array) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )
    negative_log_likelihood = -log_likelihood

    return float(negative_log_likelihood)


def negative_log_likelihood_hist_marginal_observation_window(
    model: Callable[..., Any],
    params: Iterable[Any],
    pfa_cdf_part: Callable[..., Any],
    cdf_part_index: int,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a marginal distribution specified by its CDF and
    parameters. The random observation window is bounded above by truncation_up. Bin
    probabilities are conditional on observing an event, while separate likelihood
    terms use the model's observation probability. In the marginal bin-complement
    formulation, the probability of no event is instead the complement of the
    probabilities covered by the histogram bins.

    Parameters
    ----------
    model
        A custom marginal distribution (class) defined in distributions.py.
    params
        Parameters of the distribution.
    pfa_cdf_part
        CDF part of the PFA distribution to be used in the marginal distribution.
    cdf_part_index
        Index of the CDF part of the PFA distribution to be used in the marginal
        distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.
    counts_not_observed
        Number of events not observed due to truncation.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    if truncation_low != 0:
        raise ValueError("Marginal distribution only defined for truncation_low = 0.")
    counts_array, bin_edges_array = _prepare_histogram_inputs(
        counts,
        bin_edges,
        counts_not_observed,
    )

    a = bin_edges_array[:-1]
    b = bin_edges_array[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    current_model = model(
        params=params,
        pfa_cdf_part=pfa_cdf_part,
        cdf_part_index=cdf_part_index,
        truncation_up=truncation_up,
    )
    probs = current_model.cdf(b) - current_model.cdf(a)
    probs = np.clip(probs, a_min=1e-14, a_max=1)
    log_likelihood_bin = np.sum(counts_array * np.log(probs))

    prob_event = current_model.observation_probability
    # probability of observing an event
    # within the truncation range (given the distribution is non-truncated)
    prob_event = np.clip(prob_event, a_min=1e-14, a_max=1 - 1e-14)
    log_likelihood_no_observation = np.log1p(-prob_event) * counts_not_observed
    log_likelihood_observation = np.sum(counts_array) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )
    negative_log_likelihood = -log_likelihood

    return float(negative_log_likelihood)


def negative_log_likelihood_hist_bin_complement(
    model: Callable[..., Any],
    params: Iterable[Any],
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a distribution specified by its CDF and parameters.
    The histogram bins define the observable range, and the probability of not
    observing an event is one minus the sum of their probabilities. In the
    observation-window formulation, this probability is instead calculated from
    explicit lower and upper observation limits.

    Parameters
    ----------
    model
        A custom distribution (class) defined in distributions.py.
    params
        Parameters of the distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    counts_not_observed
        Number of events not observed due to truncation.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    counts_array, bin_edges_array = _prepare_histogram_inputs(
        counts,
        bin_edges,
        counts_not_observed,
    )

    a = bin_edges_array[:-1]
    b = bin_edges_array[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = model(params, domain=(0, np.inf)).cdf(b) - model(
        params, domain=(0, np.inf)
    ).cdf(a)

    p_no_event = np.clip(1 - np.sum(probs), a_min=1e-14, a_max=1)
    probs = np.clip(probs, a_min=1e-14, a_max=1)

    log_likelihood_bin = np.sum(counts_array * np.log(probs))

    log_likelihood_no_observation = np.log(p_no_event) * counts_not_observed
    log_likelihood = log_likelihood_bin + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return float(negative_log_likelihood)


def negative_log_likelihood_hist_marginal_bin_complement(
    model: Callable[..., Any],
    params: Iterable[Any],
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    counts_not_observed: int,
    truncation_low: float,
    truncation_up: float,
    pfa_pdf_part: Callable[..., Any],
    pdf_part_index: int,
) -> float:
    """
    Negative log-likelihood of a marginal distribution of a sample X from model, where
    the upper truncation is a random variable Y ~ fixed truncation - T, and T is a
    random variable following a part of PFA distribution. The histogram bins define
    the observable range, and the probability of not observing an event is one minus
    the sum of their marginal probabilities. In the marginal observation-window
    formulation, this probability is instead supplied by the marginal model.

    Parameters
    ----------
    model
        A custom marginal distribution (class) defined in distributions.py.
    params
        Parameters of the distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    counts_not_observed
        Number of events not observed due to truncation.
    truncation_low
        Fixed lower truncation.
    truncation_up
        Fixed upper truncation.
    pfa_pdf_part
        PDF part of the PFA distribution to be used in the marginal distribution.
    pdf_part_index
        Index of the PDF part of the PFA distribution to be used in the marginal
        distribution.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    if truncation_low != 0:
        raise ValueError("Marginal distribution only defined for truncation_low = 0.")
    counts_array, bin_edges_array = _prepare_histogram_inputs(
        counts,
        bin_edges,
        counts_not_observed,
    )

    x_grid = dist._marginal_integration_grid(truncation_up)
    weights = np.asarray(
        pfa_pdf_part(
            call=None,
            x=x_grid,
            i=pdf_part_index,
            normalize=True,
        ),
        dtype=np.float64,
    )
    a = bin_edges_array[:-1, None]
    b = bin_edges_array[1:, None]
    true_limits = (truncation_up - x_grid)[None, :]
    eff_limit = np.minimum(b, true_limits)
    valid = (a < true_limits).astype(float)
    qk = model(params, domain=(0, np.inf)).cdf(eff_limit) - model(
        params, domain=(0, np.inf)
    ).cdf(a)
    qk *= valid
    probs = simpson(qk * weights[None, :], x=x_grid, axis=1)

    p_no_event = np.clip(1 - np.sum(probs), a_min=1e-14, a_max=1)
    probs = np.clip(probs, a_min=1e-14, a_max=1)

    log_likelihood_bin = np.sum(counts_array * np.log(probs))
    log_likelihood_no_observation = np.log(p_no_event) * counts_not_observed
    log_likelihood = log_likelihood_bin + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return float(negative_log_likelihood)


def fit_multiple_mixture_v1(
    datasets: list[npt.NDArray[np.int64]],
    bin_edges: npt.ArrayLike,
    z: int = -1,
    constr: bool = True,
    norm: bool = False,
    counts_not_observed: list[int] | None = None,
    pfa_bin_edges: npt.ArrayLike | None = None,
    pfa_counts: npt.ArrayLike | None = None,
    pfa_counts_not_observed: int | None = None,
    truncation_up: float = 300,
    **diff_ev: Any,
) -> OptimizeResult:
    """
    Fit multiple datasets with exponential mixture models using maximum likelihood
    estimation. One dataset can be fitted with a three-component mixture model, while
    the others are fitted with two-component mixture models.
    If pfa_bin_edges and pfa_counts are provided, the PFA distribution is also fitted,
    sharing parameters with the mixture models.

    Uses observation-window negative log-likelihoods. The observation limits determine
    the probability of observing an event independently of the histogram bin coverage.

    Parameters
    ----------
    datasets
        List of datasets to be fitted. Each dataset should be a 1D array-like of
        histogram counts.
    bin_edges
        Edges of the histogram bins. Should be the same for all datasets.
    z
        Index of the dataset to be fitted with a three-component mixture model. If -1,
        all datasets are fitted with two-component mixture models.
    constr
        Whether to apply constraints on the parameters.
    norm
        Whether to normalize the negative log-likelihood by the number of observed
        events in each dataset.
    counts_not_observed
        List of number of events not observed due to truncation for each dataset.
        If None, assumed to be 0 for all datasets.
    pfa_bin_edges
        Edges of the histogram bins for the PFA data. If None, no PFA data is fitted.
    pfa_counts
        Counts of the histogram for the PFA data. If None, no PFA data is fitted.
    pfa_counts_not_observed
        Number of events not observed due to truncation for the PFA data. If None,
        assumed to be 0.
    diff_ev
        Additional arguments to be passed to scipy.optimize.differential_evolution.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a OptimizeResult object.
    """
    bin_edges_array = np.asarray(bin_edges, dtype=np.float64)

    if counts_not_observed is None:
        counts_not_observed = [0 for _ in datasets]
    elif norm:
        raise ValueError(
            "Normalization to num observed events not possible if general "
            "log-likelihood of observation/no observation terms are included."
        )
    if pfa_counts_not_observed is None:
        pfa_counts_not_observed = 0
    _validate_fitter_histogram_inputs(
        datasets,
        bin_edges_array,
        counts_not_observed,
        pfa_bin_edges,
        pfa_counts,
        pfa_counts_not_observed,
        norm,
    )

    if z != -1 and not 0 <= z < len(datasets) - 1:
        raise ValueError(
            "z must be -1 or between 0 and number of datasets - 1."
            " The last dataset is assumed to always be a mixture of two exponentials."
        )

    def global_objective(params: npt.NDArray[np.float64]) -> float:
        total_negative_log_likelihood = 0.0
        pfa_params = prepare_pfa_parameters(z=z, n=len(datasets), params=params)
        exp_mixture_params = convert_dicts(pfa_params)
        for i, data in enumerate(datasets):
            use: Callable[..., float]
            use_model: Callable[..., Any]
            use_parameters: dict[str, Any]
            if i != 0:
                pfa_cdf_part = dist.Photoswitching_fingerprint_model(
                    params=pfa_params,
                    domain=(0, truncation_up),
                ).cdf_part
                cdf_part_index = i - 1
                use_model = dist.ExponentialMixtureMarginalModel
                use = negative_log_likelihood_hist_marginal_observation_window
                use_parameters = {
                    "pfa_cdf_part": pfa_cdf_part,
                    "cdf_part_index": cdf_part_index,
                }
            else:
                use_model = dist.ExponentialMixtureModel
                use = negative_log_likelihood_hist_observation_window
                use_parameters = {}

            negative_log_likelihood = use(
                model=use_model,
                params=exp_mixture_params[i],
                counts=data,
                bin_edges=bin_edges_array,
                truncation_low=0,
                truncation_up=truncation_up,
                counts_not_observed=counts_not_observed[i],
                **use_parameters,
            )
            if norm:
                negative_log_likelihood /= data.sum()  # data is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        if pfa_bin_edges is not None and pfa_counts is not None:
            pfa_bin_edges_array = np.asarray(pfa_bin_edges, dtype=np.float64)
            pfa_counts_array = np.asarray(pfa_counts, dtype=np.float64)
            negative_log_likelihood = negative_log_likelihood_hist_observation_window(
                model=dist.Photoswitching_fingerprint_model,
                params=pfa_params,
                counts=pfa_counts_array,
                bin_edges=pfa_bin_edges_array,
                truncation_low=0,
                truncation_up=truncation_up,
                counts_not_observed=pfa_counts_not_observed,
            )
            if norm:
                negative_log_likelihood /= (
                    pfa_counts_array.sum()
                )  # pfa_counts is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        return total_negative_log_likelihood

    def objective_for_optimizer(params: npt.NDArray[np.float64]) -> float:
        try:
            return global_objective(params)
        except dist.IllConditionedHypoexponentialError:
            return np.inf

    linear_constraint, bounds = prepare_constraints(len(datasets), z)

    if not constr:
        linear_constraint = ()
    result = differential_evolution(
        objective_for_optimizer,
        bounds=bounds,
        constraints=linear_constraint,
        **diff_ev,
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(
            "Optimization did not find parameters for which the hypoexponential "
            "distribution can be evaluated reliably."
        )
    return result


def fit_multiple_mixture_v2(
    datasets: list[npt.NDArray[np.int64]],
    bin_edges: npt.ArrayLike,
    z: int = -1,
    constr: bool = True,
    norm: bool = False,
    counts_not_observed: list[int] | None = None,
    pfa_bin_edges: npt.ArrayLike | None = None,
    pfa_counts: npt.ArrayLike | None = None,
    pfa_counts_not_observed: int | None = None,
    truncation_up: float = 300,
    **diff_ev: Any,
) -> OptimizeResult:
    """
    Fit multiple datasets with exponential mixture models using maximum likelihood
    estimation. One dataset can be fitted with a three-component mixture model, while
    the others are fitted with two-component mixture models.
    If pfa_bin_edges and pfa_counts are provided, the PFA distribution is also fitted,
    sharing parameters with the mixture models.

    Uses bin-complement negative log-likelihoods. The histogram bins define the
    observable range, and their probability complement represents unobserved events.

    Parameters
    ----------
    datasets
        List of datasets to be fitted. Each dataset should be a 1D array-like of
        histogram counts.
    bin_edges
        Edges of the histogram bins. Should be the same for all datasets.
    z
        Index of the dataset to be fitted with a three-component mixture model. If -1,
        all datasets are fitted with two-component mixture models.
    constr
        Whether to apply constraints on the parameters.
    norm
        Whether to normalize the negative log-likelihood by the number of observed
        events in each dataset.
    counts_not_observed
        List of number of events not observed due to truncation for each dataset.
        If None, assumed to be 0 for all datasets.
    pfa_bin_edges
        Edges of the histogram bins for the PFA data. If None, no PFA data is fitted.
    pfa_counts
        Counts of the histogram for the PFA data. If None, no PFA data is fitted.
    pfa_counts_not_observed
        Number of events not observed due to truncation for the PFA data. If None,
        assumed to be 0.
    diff_ev
        Additional arguments to be passed to scipy.optimize.differential_evolution.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a OptimizeResult object.
    """
    bin_edges_array = np.asarray(bin_edges, dtype=np.float64)

    if counts_not_observed is None:
        counts_not_observed = [0 for _ in datasets]
    elif norm:
        raise ValueError(
            "Normalization to num observed events not possible if general "
            "log-likelihood of observation/no observation terms are included."
        )
    if pfa_counts_not_observed is None:
        pfa_counts_not_observed = 0
    _validate_fitter_histogram_inputs(
        datasets,
        bin_edges_array,
        counts_not_observed,
        pfa_bin_edges,
        pfa_counts,
        pfa_counts_not_observed,
        norm,
    )

    if z != -1 and not 0 <= z < len(datasets) - 1:
        raise ValueError(
            "z must be -1 or between 0 and number of datasets - 1."
            " The last dataset is assumed to always be a mixture of two exponentials."
        )

    def global_objective(params: npt.NDArray[np.float64]) -> float:
        total_negative_log_likelihood = 0.0
        pfa_params = prepare_pfa_parameters(z=z, n=len(datasets), params=params)
        exp_mixture_params = convert_dicts(pfa_params)
        for i, data in enumerate(datasets):
            use: Callable[..., float]
            use_parameters: dict[str, Any]
            if i != 0:
                pfa_pdf_part = dist.Photoswitching_fingerprint_model(
                    params=pfa_params,
                    domain=(0, truncation_up),
                ).pdf_part
                pdf_part_index = i - 1
                use = negative_log_likelihood_hist_marginal_bin_complement
                use_parameters = {
                    "truncation_low": 0,
                    "truncation_up": truncation_up,
                    "pfa_pdf_part": pfa_pdf_part,
                    "pdf_part_index": pdf_part_index,
                }
            else:
                use = negative_log_likelihood_hist_bin_complement
                use_parameters = {}

            negative_log_likelihood = use(
                model=dist.ExponentialMixtureModel,
                params=exp_mixture_params[i],
                counts=data,
                bin_edges=bin_edges_array,
                counts_not_observed=counts_not_observed[i],
                **use_parameters,
            )
            if norm:
                negative_log_likelihood /= data.sum()  # data is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        if pfa_bin_edges is not None and pfa_counts is not None:
            pfa_bin_edges_array = np.asarray(pfa_bin_edges, dtype=np.float64)
            pfa_counts_array = np.asarray(pfa_counts, dtype=np.float64)
            negative_log_likelihood = negative_log_likelihood_hist_bin_complement(
                model=dist.Photoswitching_fingerprint_model,
                params=pfa_params,
                counts=pfa_counts_array,
                bin_edges=pfa_bin_edges_array,
                counts_not_observed=pfa_counts_not_observed,
            )
            if norm:
                negative_log_likelihood /= (
                    pfa_counts_array.sum()
                )  # pfa_counts is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        return total_negative_log_likelihood

    def objective_for_optimizer(params: npt.NDArray[np.float64]) -> float:
        try:
            return global_objective(params)
        except dist.IllConditionedHypoexponentialError:
            return np.inf

    linear_constraint, bounds = prepare_constraints(len(datasets), z)

    if not constr:
        linear_constraint = ()
    result = differential_evolution(
        objective_for_optimizer,
        bounds=bounds,
        constraints=linear_constraint,
        **diff_ev,
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(
            "Optimization did not find parameters for which the hypoexponential "
            "distribution can be evaluated reliably."
        )
    return result


def prepare_constraints(n: int, z: int) -> tuple[LinearConstraint, Bounds]:
    """
    Prepare constraints for the optimization problem.
    lam_b[i] > lam_b[i+1], lam_b[i] > lam_nb[i], p[i] > pi[i+1],
    lam_nb[i] > lam_nb[i+1]. For lam_b and lam_nb, a minimum difference of 1e-3 is
    enforced to ensure inequality.

    Parameters
    ----------
    n
        Number of datasets.
    z
        Index of the dataset to be fitted with a three-component mixture model. If -1,
        all datasets are fitted with two-component mixture models.

    Returns
    -------
    linear_constraint : LinearConstraint
        Linear constraints for the optimization problem.
    bounds : Bounds
        Bounds for the optimization problem.
    """
    linear_constraint = None
    bounds = None
    A = []
    lb = []
    ub = []
    if z == -1:
        # parameter structure: [p1, lam_b1, lam_nb1, p2, ...]
        param_count = n * 3
        # lam_b[i] > lam_b[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            row[i * 3 + 1] = 1
            row[i * 3 + 4] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        # lam_b[i] > lam_nb[i]
        for i in range(n):
            row = [0] * param_count
            row[i * 3 + 1] = 1
            row[i * 3 + 2] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        # p[i] > pi[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            row[i * 3] = 1
            row[i * 3 + 3] = -1
            A.append(row)
            lb.append(0)
            ub.append(np.inf)
        # lam_nb[i] > lam_nb[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            row[i * 3 + 2] = 1
            row[i * 3 + 5] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        linear_constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds([0, 1e-9, 1e-9] * n, [1, 5, 5] * n)
    else:
        # parameter structure: [pz, vz, lam_bz, lam_nbz1, lam_nbz2, px, lam_bx, lam_nbx, ...],
        # where x is the smallest index not equal to z
        param_count = 5 + (n - 1) * 3
        # lam_b[i] > lam_b[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            if i == z:
                row[2] = 1
                row[i * 3 + 1 + 5] = -1
            elif i == z - 1:
                row[i * 3 + 1 + 5] = 1
                row[2] = -1
            else:
                row[(i - (i > z)) * 3 + 1 + 5] = 1
                row[(i - (i > z)) * 3 + 4 + 5] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        # lam_b[i] > lam_nb[i]
        for i in range(n):
            row = [0] * param_count
            if i == z:
                row[2] = 1
                row[3] = -1
                row_2 = [0] * param_count
                row_2[2] = 1
                row_2[4] = -1
                A.append(row_2)
                lb.append(1e-3)
                ub.append(np.inf)
            else:
                row[(i - (i > z)) * 3 + 1 + 5] = 1
                row[(i - (i > z)) * 3 + 2 + 5] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        # p[i] > pi[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            if i == z:
                row[0] = 1
                row[i * 3 + 5] = -1
            elif i == z - 1:
                row[i * 3 + 5] = 1
                row[0] = -1
            else:
                row[(i - (i > z)) * 3 + 5] = 1
                row[(i - (i > z)) * 3 + 3 + 5] = -1
            A.append(row)
            lb.append(0)
            ub.append(1)
        # lam_nb[i] > lam_nb[i+1]
        for i in range(n - 1):
            row = [0] * param_count
            if i == z:
                row[3] = 1
                row[i * 3 + 2 + 5] = -1
                row_2 = [0] * param_count
                row_2[4] = 1
                row_2[i * 3 + 2 + 5] = -1
                A.append(row_2)
                lb.append(1e-3)
                ub.append(np.inf)
            elif i == z - 1:
                row[i * 3 + 2 + 5] = 1
                row[3] = -1
                row_2 = [0] * param_count
                row_2[i * 3 + 2 + 5] = 1
                row_2[4] = -1
                A.append(row_2)
                lb.append(1e-3)
                ub.append(np.inf)
            else:
                row[(i - (i > z)) * 3 + 2 + 5] = 1
                row[(i - (i > z)) * 3 + 5 + 5] = -1
            A.append(row)
            lb.append(1e-3)
            ub.append(np.inf)
        linear_constraint = LinearConstraint(A=A, lb=lb, ub=ub)
        bounds = Bounds(
            lb=[0, 0, 1e-9, 1e-9, 1e-9] + [0, 1e-9, 1e-9] * (n - 1),
            ub=[1, 1, 5, 5, 5] + [1, 5, 5] * (n - 1),
        )

    return linear_constraint, bounds


def prepare_pfa_parameters(
    z: int,
    n: int,
    params: Sequence[float] | npt.NDArray[np.float64],
) -> dict[int, list[float]]:
    """
    Prepare parameters for the PFA distribution.

    Parameters
    ----------
    z
        Index of the dataset to be fitted with a three-component mixture model. If -1,
        all datasets are fitted with two-component mixture models.
    n
        Number of datasets.
    params
        List of parameters from the optimization.

    Returns
    -------
    parameters : dict
        Dictionary of parameters for the PFA distribution.
    """
    parameters: dict[int, list[float]] = {}
    if z != -1:
        uz = params[0]
        vz = params[1]
        pz1 = uz
        pz2 = (1 - uz) * vz
        pz3 = (1 - uz) * (1 - vz)
        for i in range(n):
            if i == z:
                parameters[i] = [pz1, pz2, pz3, params[2], params[3], params[4]]
            else:
                parameters[i] = [
                    params[(i - (i > z)) * 3 + 5],
                    1 - params[(i - (i > z)) * 3 + 5],
                    params[(i - (i > z)) * 3 + 1 + 5],
                    params[(i - (i > z)) * 3 + 2 + 5],
                ]
    else:
        for i in range(n):
            parameters[i] = [
                params[i * 3],
                1 - params[i * 3],
                params[i * 3 + 1],
                params[i * 3 + 2],
            ]
    return parameters


def prepare_exp_mixture_parameters(
    z: int,
    n: int,
    params: Sequence[float] | npt.NDArray[np.float64],
) -> dict[int, dict[str, list[float]]]:
    """
    Prepare parameters for the exponential mixture model.

    Parameters
    ----------
    z
        Index of the dataset to be fitted with a three-component mixture model. If -1,
        all datasets are fitted with two-component mixture models.
    n
        Number of datasets.
    params
        List of parameters from the optimization.

    Returns
    -------
    parameters : dict
        Dictionary of parameters for the exponential mixture model.
    """
    return convert_dicts(
        prepare_pfa_parameters(
            z=z,
            n=n,
            params=params,
        )
    )


def save_as_array(
    parameter_dict: Mapping[int, Sequence[float]],
    filepath: str | PathLike[str],
) -> None:
    """
    Save parameters as a numpy array. The dictionary keys are repeated for each
    parameter value, and the values are flattened into a single array.

    Parameters
    ----------
    parameter_dict
        Dictionary of parameters to be saved.
    filepath
        Path to the file where the parameters will be saved.
    """
    indices: list[int] = []
    parameters: list[float] = []
    for key, value in parameter_dict.items():
        indices += [key] * len(value)
        parameters += value

    save_array = np.array([indices, parameters])
    np.save(file=filepath, arr=save_array)


def load_from_array(filepath: str | PathLike[str]) -> dict[int, list[float]]:
    """
    Load parameters from a numpy array saved in a file. The array is expected to
    have two rows: the first row contains the dictionary keys, and the second row
    contains the corresponding parameter values.

    Parameters
    ----------
    filepath
        Path to the file from which the parameters will be loaded.

    Returns
    -------
    parameter_dict : dict
        Dictionary of parameters loaded from the file.
    """
    parameter_array = np.load(filepath)
    parameter_df = pd.DataFrame(parameter_array.T, columns=["key", "value"])
    parameter_df["key"] = parameter_df["key"].astype(int)
    parameter_dict: dict[int, list[float]] = {}
    for key, values in parameter_df.groupby("key")["value"]:
        if not isinstance(key, (int, np.integer)):
            raise TypeError("saved parameter keys must be integers.")
        parameter_dict[int(key)] = [float(value) for value in values]
    return parameter_dict


def convert_dicts(
    pfa_dict: Mapping[int, Sequence[float]],
) -> dict[int, dict[str, list[float]]]:
    """
    Convert a dictionary of PFA parameters to a dictionary of exponential mixture
    parameters.

    Parameters
    ----------
    pfa_dict
        Parameters of the underlying exponential mixture distributions. Dict indexed
        by (0, 1, ..., n-1). The indices denote the number of photobleaching events
        that have occurred. Each entry is a list of length 4, one of the entries
        can be of length 6. The first half of the list contains the pis, the second
        half the lambdas.
    """
    exp_mixture_dict: dict[int, dict[str, list[float]]] = {}
    for key, value in pfa_dict.items():
        if len(value) == 6:
            pis = [value[0], value[1]]
            lambdas = [value[3], value[4], value[5]]
        elif len(value) == 4:
            pis = [value[0]]
            lambdas = [value[2], value[3]]
        else:
            raise ValueError("PFA parameter values must have length 4 or 6.")
        exp_mixture_dict[key] = {"pis": pis, "lambdas": lambdas}
    return exp_mixture_dict
