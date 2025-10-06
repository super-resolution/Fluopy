"""
Tools for fitting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, differential_evolution

from . import distributions as dist

if TYPE_CHECKING:
    pass


__all__: list[str] = []


def log_likelihood_hist_v1(
    model,
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a distribution specified by its CDF and parameters. The
    distribution is truncated between truncation_low and truncation_up, meaning the
    probability of observing an event outside this range is 0. This is why here, to
    include the log-likelihood term of not observing an event due to truncation, we also
    add a term for the probability of observing an event within the truncation range.

    Parameters
    ----------
    model
        A custom distribution defined in distributions.py.
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
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = model(params, domain=(truncation_low, truncation_up)).cdf(b) - model(
        params, domain=(truncation_low, truncation_up)
    ).cdf(a)
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    prob_event = model(params, domain=(0, np.inf)).cdf(truncation_up) - model(
        params, domain=(0, np.inf)
    ).cdf(truncation_low)
    # probability of observing an event
    # within the truncation range (given the distribution is non-truncated)
    prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
    log_likelihood_no_observation = np.log1p(-prob_event) * counts_not_observed
    log_likelihood_observation = np.sum(counts) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def log_likelihood_hist_marginal_v1(
    model,
    params: Iterable,
    pfa_cdf_part: Callable,
    cdf_part_index: int,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a marginal distribution specified by its CDF and parameters.
    The marginal distribution support is (0, truncation_up) and has no option to be
    defined on (0, inf), however the marginal distribution has an attribute returning
    the probability of observing an event within (0, truncation_up) assuming the support
    was (0, inf).

    Parameters
    ----------
    model
        A custom marginal distribution defined in distributions.py.
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
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = model(params, pfa_cdf_part, cdf_part_index, truncation_up).cdf(b) - model(
        params, pfa_cdf_part, cdf_part_index, truncation_up
    ).cdf(a)
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    prob_event = model(params, pfa_cdf_part, cdf_part_index, truncation_up).P_obs
    # probability of observing an event
    # within the truncation range (given the distribution is non-truncated)
    prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
    log_likelihood_no_observation = np.log1p(-prob_event) * counts_not_observed
    log_likelihood_observation = np.sum(counts) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def log_likelihood_hist_v2(
    model,
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    counts_not_observed: int,
) -> float:
    """
    Negative log-likelihood of a distribution specified by its CDF and parameters. The
    distribution support is (0, inf), meaning the sum of probabilities of the given bins
    may not be 1. This is why here, to include the log-likelihood term of not observing an
    event due to truncation, we can do 1 - sum(probabilities_bins).

    Parameters
    ----------
    model
        A custom distribution defined in distributions.py.
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
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = model(params, domain=(0, np.inf)).cdf(b) - model(
        params, domain=(0, np.inf)
    ).cdf(a)
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    log_likelihood_no_observation = np.log1p(-np.sum(probs)) * counts_not_observed
    log_likelihood = log_likelihood_bin + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def log_likelihood_hist_marginal_v2(
    model,
    params,
    counts,
    bin_edges,
    counts_not_observed,
    truncation_low,
    truncation_up,
    pfa_pdf_part: Callable,
    pdf_part_index: int,
) -> float:
    """
    Negative log-likelihood of a marginal distribution of a sample X from model, where
    the upper truncation is a random variable Y ~ fixed truncation - T, and T is a random
    variable following a part of PFA distribution.
    The distribution support is (0, inf), meaning the sum of probabilities of the given bins
    may not be 1. This is why here, to include the log-likelihood term of not observing an
    event due to truncation, we can do 1 - sum(probabilities_bins).
    """
    if truncation_low != 0:
        raise ValueError("Marginal distribution only defined for truncation_low = 0.")
    x_grid = np.logspace(np.log10(0.01), np.log10(truncation_up), 200)
    x_grid = np.insert(x_grid, 0, 0)
    weights = pfa_pdf_part(
        call=None,
        x=x_grid,
        i=pdf_part_index,
        normalize=True,
    )
    a = bin_edges[:-1][:, None]
    b = bin_edges[1:][:, None]
    true_limits = (truncation_up - x_grid)[None, :]
    eff_limit = np.minimum(b, true_limits)
    valid = (a < true_limits).astype(float)
    qk = model(params, domain=(0, np.inf)).cdf(eff_limit) - model(
        params, domain=(0, np.inf)
    ).cdf(a)
    qk *= valid
    probs = np.trapezoid(qk * weights[None, :], x=x_grid, axis=1)
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    p_no_event = 1 - np.sum(probs)
    p_no_event = np.clip(p_no_event, 1e-14, None)  # avoid log(0)

    log_likelihood_no_observation = np.log(p_no_event) * counts_not_observed
    log_likelihood = log_likelihood_bin + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def fit_multiple_mixture_v1(
    datasets,
    bin_edges,
    z=-1,
    constr=True,
    norm=False,
    counts_not_observed=None,
    pfa_bin_edges=None,
    pfa_counts=None,
    pfa_counts_not_observed=None,
    **diff_ev,
):
    if counts_not_observed is None:
        counts_not_observed = [0 for _ in datasets]
    else:
        if norm:
            raise ValueError(
                "Normalization to num observed events not possible if general "
                "log-likelihood of observation/no observation terms are included."
            )
    if pfa_counts_not_observed is None:
        pfa_counts_not_observed = 0

    if z != -1 and z < len(datasets) - 1:
        add = 5
    elif z != -1 and z >= len(datasets) - 1:
        raise ValueError(
            "z must be -1 or between 0 and number of datasets - 1."
            " The last dataset is assumed to always be a mixture of two exponentials."
        )
    else:
        add = 0

    def global_objective(params):
        total_negative_log_likelihood = 0
        counter = 0
        pfa_params = prepare_pfa_parameters(z=z, n=len(datasets), params=params)
        for i, data in enumerate(datasets):
            if i != 0:
                pfa_cdf_part = dist.Photoswitching_fingerprint_model(
                    params=pfa_params,
                    domain=(0, 300),
                ).cdf_part
                cdf_part_index = i - 1
                use_model = dist.ExponentialMixtureMarginalModel
                use = log_likelihood_hist_marginal_v1
                use_parameters = [pfa_cdf_part, cdf_part_index]
            else:
                use_model = dist.ExponentialMixtureModel
                use = log_likelihood_hist_v1
                use_parameters = []
            if i == z:
                parameters = {
                    "pis": [params[0], (1 - params[0]) * params[1]],
                    "lambdas": [params[2], params[3], params[4]],
                }
            else:
                parameters = {
                    "pis": [params[add + counter * 3]],
                    "lambdas": [
                        params[add + counter * 3 + 1],
                        params[add + counter * 3 + 2],
                    ],
                }
                counter += 1

            negative_log_likelihood = use(
                use_model,
                parameters,
                *use_parameters,
                data,
                bin_edges,
                0,
                300,
                counts_not_observed[i],
            )
            if norm:
                negative_log_likelihood /= data.sum()  # data is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        if pfa_bin_edges is not None and pfa_counts is not None:
            negative_log_likelihood = log_likelihood_hist_v1(
                dist.Photoswitching_fingerprint_model,
                pfa_params,
                pfa_counts,
                pfa_bin_edges,
                0,
                300,
                counts_not_observed=pfa_counts_not_observed,
            )
            if norm:
                negative_log_likelihood /= (
                    pfa_counts.sum()
                )  # pfa_counts is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        return total_negative_log_likelihood

    linear_constraint, bounds = prepare_constraints(len(datasets), z)

    if not constr:
        linear_constraint = ()
    result = differential_evolution(
        global_objective, bounds=bounds, constraints=linear_constraint, **diff_ev
    )
    return result


def fit_multiple_mixture_v2(
    datasets,
    bin_edges,
    z=-1,
    constr=True,
    norm=False,
    counts_not_observed=None,
    pfa_bin_edges=None,
    pfa_counts=None,
    pfa_counts_not_observed=None,
    **diff_ev,
):
    if counts_not_observed is None:
        counts_not_observed = [0 for _ in datasets]
    else:
        if norm:
            raise ValueError(
                "Normalization to num observed events not possible if general "
                "log-likelihood of observation/no observation terms are included."
            )
    if pfa_counts_not_observed is None:
        pfa_counts_not_observed = 0

    if z != -1 and z < len(datasets) - 1:
        add = 5
    elif z != -1 and z >= len(datasets) - 1:
        raise ValueError(
            "z must be -1 or between 0 and number of datasets - 1."
            " The last dataset is assumed to always be a mixture of two exponentials."
        )
    else:
        add = 0

    def global_objective(params):
        total_negative_log_likelihood = 0
        counter = 0
        pfa_params = prepare_pfa_parameters(z=z, n=len(datasets), params=params)
        for i, data in enumerate(datasets):
            if i != 0:
                pfa_pdf_part = dist.Photoswitching_fingerprint_model(
                    params=pfa_params,
                    domain=(0, 300),
                ).pdf_part
                pdf_part_index = i - 1
                use = log_likelihood_hist_marginal_v2
                use_parameters = [0, 300, pfa_pdf_part, pdf_part_index]
            else:
                use = log_likelihood_hist_v2
                use_parameters = []
            if i == z:
                parameters = {
                    "pis": [params[0], (1 - params[0]) * params[1]],
                    "lambdas": [params[2], params[3], params[4]],
                }
            else:
                parameters = {
                    "pis": [params[add + counter * 3]],
                    "lambdas": [
                        params[add + counter * 3 + 1],
                        params[add + counter * 3 + 2],
                    ],
                }
                counter += 1

            negative_log_likelihood = use(
                dist.ExponentialMixtureModel,
                parameters,
                data,
                bin_edges,
                counts_not_observed[i],
                *use_parameters,
            )
            if norm:
                negative_log_likelihood /= data.sum()  # data is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        if pfa_bin_edges is not None and pfa_counts is not None:
            negative_log_likelihood = log_likelihood_hist_v2(
                dist.Photoswitching_fingerprint_model,
                pfa_params,
                pfa_counts,
                pfa_bin_edges,
                counts_not_observed=pfa_counts_not_observed,
            )
            if norm:
                negative_log_likelihood /= (
                    pfa_counts.sum()
                )  # pfa_counts is histogrammed
            total_negative_log_likelihood += negative_log_likelihood
        return total_negative_log_likelihood

    linear_constraint, bounds = prepare_constraints(len(datasets), z)

    if not constr:
        linear_constraint = ()
    result = differential_evolution(
        global_objective, bounds=bounds, constraints=linear_constraint, **diff_ev
    )
    return result


def prepare_constraints(n, z):
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
        linear_constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(
            [0, 0, 1e-9, 1e-9, 1e-9] + [0, 1e-9, 1e-9] * (n - 1),
            [1, 1, 5, 5, 5] + [1, 5, 5] * (n - 1),
        )

    return linear_constraint, bounds


def prepare_pfa_parameters(z, n, params):
    parameters = {}
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


def prepare_exp_mixture_parameters(z, n, params):
    parameters = {}
    if z != -1:
        uz = params[0]
        vz = params[1]
        pz1 = uz
        pz2 = (1 - uz) * vz
        for i in range(n):
            if i == z:
                parameters[i] = {
                    "pis": [pz1, pz2],
                    "lambdas": [params[2], params[3], params[4]],
                }
            else:
                parameters[i] = {
                    "pis": [params[(i - (i > z)) * 3 + 5]],
                    "lambdas": [
                        params[(i - (i > z)) * 3 + 1 + 5],
                        params[(i - (i > z)) * 3 + 2 + 5],
                    ],
                }
    else:
        for i in range(n):
            parameters[i] = {
                "pis": [params[i * 3]],
                "lambdas": [params[i * 3 + 1], params[i * 3 + 2]],
            }
    return parameters


def save_as_array(parameter_dict, filepath):
    indices = []
    parameters = []
    for key, value in parameter_dict.items():
        indices += [key] * len(value)
        parameters += value

    save_array = np.array([indices, parameters])
    np.save(filepath, save_array)


def load_from_array(filepath):
    parameter_array = np.load(filepath)
    parameter_df = pd.DataFrame(parameter_array.T, columns=["key", "value"])
    parameter_df["key"] = parameter_df["key"].astype(int)
    parameter_dict = parameter_df.groupby("key")["value"].apply(list).to_dict()
    return parameter_dict


def convert_dicts(pfa_dict):
    exp_mixture_dict = {}
    for key, value in pfa_dict.items():
        if len(value) == 6:
            pis = [value[0], value[1]]
            lambdas = [value[3], value[4], value[5]]
        elif len(value) == 4:
            pis = [value[0]]
            lambdas = [value[2], value[3]]
        exp_mixture_dict[key] = {"pis": pis, "lambdas": lambdas}
    return exp_mixture_dict
