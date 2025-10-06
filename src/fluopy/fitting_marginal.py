"""
Tools for fitting marginal.

Module fitting marginal: Here we define each log likelihood bins unconditionally
(i.e., the fitting distribution is non-truncated), and include 1-sum(bin_probs) for
the probability of observing no event. This is the case for all log-likelihoods that
are involved to keep the same basis in order to sum them for differential evolution.
This is needed, because this is the way we define the log-likelihoods of two- or three-
exponential mixtures that describe the n != 0 dataset. Those datasets have no constant
truncation limit, so we need the marginal distribution over the varying truncation
limit. The truncation limit is the distribution of the PFA using the n - 1 fluorophores
for the nth dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, differential_evolution

from . import distributions as dist

# from .fitting import pfa_log_likelihood

if TYPE_CHECKING:
    pass


def trapezoid_weights(truncation_up, n):
    t = np.linspace(0.0, truncation_up, n)
    dt = t[1] - t[0]
    weights = np.full_like(t, dt)
    return t, weights


# def two_expon_mixture_marginal_pdf(
#     x: float | npt.ArrayLike,
#     pi: float,
#     lambda1: float,
#     lambda2: float,
#     pfa_cdf: callable,
#     truncation_up: float = 300,
# ):
#     """
#     PDF of a mixture of two exponential distribution marginalized over truncation
#     limits distributed according to pfa_pdf.
#     """
#     pdf_two_expon = dist.two_expon_mixture_pdf(x, pi, lambda1, lambda2)
#     survival_pfa = pfa_cdf(truncation_up - x)
#     pdf_unnormalized = pdf_two_expon * survival_pfa

#     x_vals = np.linspace(0, truncation_up, 300001)
#     pdf_two_expon_grid = dist.two_expon_mixture_pdf(x_vals, pi, lambda1, lambda2)
#     pdf_unnormalized_grid = pdf_two_expon_grid * pfa_cdf(truncation_up - x_vals)
#     P_obs = np.trapezoid(pdf_unnormalized_grid, x_vals)

#     return pdf_unnormalized / P_obs


# def two_expon_mixture_marginal_cdf(
#     x: float | npt.ArrayLike,
#     pi: float,
#     lambda1: float,
#     lambda2: float,
#     pfa_cdf: callable,
# ):
#     """
#     CDF of a mixture of two exponential distribution marginalized over truncation
#     limits distributed according to pfa_pdf.
#     """
#     pdf_vals = two_expon_mixture_marginal_pdf(x, pi, lambda1, lambda2, pfa_cdf)
#     cdf_vals = cumulative_trapezoid(pdf_vals, x, initial=0)
#     return cdf_vals


# def three_expon_mixture_marginal_pdf(
#     x: float | npt.ArrayLike,
#     pi1: float,
#     pi2: float,
#     lambda1: float,
#     lambda2: float,
#     lambda3: float,
#     pfa_cdf: callable,
# ):
#     """
#     PDF of a mixture of two exponential distribution marginalized over truncation
#     limits distributed according to pfa_pdf.
#     """
#     pdf_three_expon = dist.three_expon_mixture_pdf(x, pi1, pi2, lambda1, lambda2, lambda3)
#     survival_pfa = pfa_cdf(300 - x)
#     pdf_unnormalized = pdf_three_expon * survival_pfa
#     P_obs, _ = quad(lambda y: dist.three_expon_mixture_pdf(y, pi1, pi2, lambda1, lambda2, lambda3) * pfa_cdf(300 - y), 0, np.inf, epsabs=1e-9)
#     if np.isnan(P_obs):
#         P_obs = 1
#     return pdf_unnormalized / P_obs


# def three_expon_mixture_marginal_cdf(
#     x: float | npt.ArrayLike,
#     pi1: float,
#     pi2: float,
#     lambda1: float,
#     lambda2: float,
#     lambda3: float,
#     pfa_cdf: callable,
# ):
#     """
#     CDF of a mixture of two exponential distribution marginalized over truncation
#     limits distributed according to pfa_pdf.
#     """
#     pdf_vals = three_expon_mixture_marginal_pdf(x, pi1, pi2, lambda1, lambda2, lambda3, pfa_cdf)
#     cdf_vals = cumulative_trapezoid(pdf_vals, x, initial=0)

#     return cdf_vals/cdf_vals[-1]


def mixture_log_likelihood_hist_marginal(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    pfa_pdf_part: callable,
    pdf_part_index: int,
    rel_events_not_observed: float,
) -> float:
    """
    Negative log-likelihood of a mixture of two exponential distributions (probability
    within bins).

    Parameters
    ----------
    params
        Parameters of the mixture distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.

    Returns
    -------
    float
        Negative log-likelihood to be minimized.
    """
    pi, lambda1, lambda2 = params

    if truncation_low != 0:
        raise NotImplementedError("Only truncation_low = 0 is implemented.")
    if not (0 <= pi <= 1) or lambda1 <= 0 or lambda2 <= 0:
        return 1e20

    t_grid, w = trapezoid_weights(truncation_up, 1000)
    w *= pfa_pdf_part(
        t_grid,
        pdf_part_index,
        normalize=True,
    )

    # bin probabilities
    a = bin_edges[:-1][:, None]
    b = bin_edges[1:][:, None]
    true_limits = (truncation_up - t_grid)[None, :]
    eff_limit = np.minimum(b, true_limits)

    # bin is valid if lower edge is smaller than true limit
    valid = (a < true_limits).astype(float)

    # the probability that an event falls in bin [a, b] and is observed equals:
    qk = (
        dist.two_expon_mixture_cdf(eff_limit, pi, lambda1, lambda2)
        - dist.two_expon_mixture_cdf(a, pi, lambda1, lambda2)
    ) * valid

    # integrate over grid with weights
    probs = qk @ w
    # probability of observing no event
    P0 = 1 - np.sum(probs)

    log_likelihood_bin = np.sum(counts * np.log(np.clip(probs, 1e-14, None)))
    factor = rel_events_not_observed / (1 - rel_events_not_observed)
    log_likelihood_no_observation = (
        np.log(np.clip(P0, 1e-14, None)) * factor * np.sum(counts)
    )

    negative_log_likelihood = -(log_likelihood_bin + log_likelihood_no_observation)

    return negative_log_likelihood


def mixture_log_likelihood_hist_three_exp_marginal(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    pfa_pdf_part: callable,
    pdf_part_index: int,
    rel_events_not_observed: float,
) -> float:
    """
    Negative log-likelihood of a mixture of two exponential distributions (probability
    within bins).

    Parameters
    ----------
    params
        Parameters of the mixture distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.

    Returns
    -------
    float
        Negative log-likelihood to be minimized.
    """
    u, v, lambda1, lambda2, lambda3 = params
    if truncation_low != 0:
        raise NotImplementedError("Only truncation_low = 0 is implemented.")
    pi1 = u
    pi2 = (1 - u) * v
    pi3 = (1 - u) * (1 - v)
    if (
        not (0 <= pi1 <= 1)
        or not (0 <= pi2 <= 1)
        or not (0 <= pi3 <= 1)
        or lambda1 <= 0
        or lambda2 <= 0
        or lambda3 <= 0
    ):
        return 1e20

    t_grid, w = trapezoid_weights(truncation_up, 1000)

    w *= pfa_pdf_part(t_grid, pdf_part_index, normalize=True)

    # bin probabilities
    a = bin_edges[:-1][:, None]
    b = bin_edges[1:][:, None]
    true_limits = (truncation_up - t_grid)[None, :]
    eff_limit = np.minimum(b, true_limits)

    # bin is valid if lower edge is smaller than true limit
    valid = (a < true_limits).astype(float)

    # the probability that an event falls in bin [a, b] and is observed equals:
    qk = (
        dist.three_expon_mixture_cdf(eff_limit, pi1, pi2, lambda1, lambda2, lambda3)
        - dist.three_expon_mixture_cdf(a, pi1, pi2, lambda1, lambda2, lambda3)
    ) * valid

    # integrate over grid with weights
    probs = qk @ w
    # probability of observing no event
    P0 = 1 - np.sum(probs)

    log_likelihood_bin = np.sum(counts * np.log(np.clip(probs, 1e-14, None)))
    factor = rel_events_not_observed / (1 - rel_events_not_observed)
    log_likelihood_no_observation = (
        np.log(np.clip(P0, 1e-14, None)) * factor * np.sum(counts)
    )

    negative_log_likelihood = -(log_likelihood_bin + log_likelihood_no_observation)

    return negative_log_likelihood


def mixture_log_likelihood_hist(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    rel_events_not_observed: float,
) -> float:
    """
    Negative log-likelihood of a mixture of two exponential distributions (probability
    within bins).

    Parameters
    ----------
    params
        Parameters of the mixture distribution.
    counts
        Counts of the histogram.
    bin_edges
        Edges of the histogram bins.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.

    Returns
    -------
    float
        Negative log-likelihood to be minimized.
    """
    pi, lambda1, lambda2 = params

    if not (0 <= pi <= 1) or lambda1 <= 0 or lambda2 <= 0:
        return 1e20

    a = bin_edges[:-1]
    b = bin_edges[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    # e^-lam a - e^-lam b is the same as 1 - e^-lam a - (1 - e^-lam b)
    probs = dist.two_expon_mixture_cdf(
        b, pi, lambda1, lambda2, domain=(0, np.inf)
    ) - dist.two_expon_mixture_cdf(a, pi, lambda1, lambda2, domain=(0, np.inf))
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))
    factor = rel_events_not_observed / (1 - rel_events_not_observed)
    log_likelihood_no_observation = factor * np.sum(counts) * np.log(1 - np.sum(probs))

    log_likelihood = log_likelihood_bin + log_likelihood_no_observation

    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def mixture_log_likelihood_hist_three_exp(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    rel_events_not_observed: float,
) -> float:
    """
    Negative log-likelihood of a mixture of three exponential distributions (probability
    within bins).
    """
    u, v, lambda1, lambda2, lambda3 = params
    pi1 = u
    pi2 = (1 - u) * v
    pi3 = (1 - u) * (1 - v)
    if (
        not (0 <= pi1 <= 1)
        or not (0 <= pi2 <= 1)
        or not (0 <= pi3 <= 1)
        or lambda1 <= 0
        or lambda2 <= 0
        or lambda3 <= 0
    ):
        return 1e20
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # calculate the probability of observing an event between a and b if the distribution
    # is truncated between trunc_low and trunc_up
    probs = dist.three_expon_mixture_cdf(
        b, pi1, pi2, lambda1, lambda2, lambda3, domain=(0, np.inf)
    ) - dist.three_expon_mixture_cdf(
        a, pi1, pi2, lambda1, lambda2, lambda3, domain=(0, np.inf)
    )
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))
    factor = rel_events_not_observed / (1 - rel_events_not_observed)
    log_likelihood_no_observation = factor * np.sum(counts) * np.log(1 - np.sum(probs))

    log_likelihood = log_likelihood_bin + log_likelihood_no_observation

    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def pfa_log_likelihood(
    params: dict,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    rel_events_not_observed: float,
) -> float:
    a = bin_edges[:-1]
    b = bin_edges[1:]
    probs = dist.Photoswitching_fingerprint_model(params, domain=(0, np.inf)).cdf(
        b
    ) - dist.Photoswitching_fingerprint_model(params, domain=(0, np.inf)).cdf(a)
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))
    factor = rel_events_not_observed / (1 - rel_events_not_observed)
    log_likelihood_no_observation = factor * np.sum(counts) * np.log(1 - np.sum(probs))
    log_likelihood = log_likelihood_bin + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def fit_multiple_mixture(
    datasets,
    bin_edges,
    z=-1,
    constr=True,
    norm=False,
    rel_events_not_observed=None,
    pfa_bin_edges=None,
    pfa_counts=None,
    pfa_not_observed=None,
    **diff_ev,
):
    if rel_events_not_observed is None:
        rel_events_not_observed = [0 for _ in datasets]
    else:
        if norm:
            raise ValueError(
                "Normalization to num observed events not possible if general "
                "log-likelihood of observation/no observation terms are included."
            )
    if pfa_not_observed is None:
        pfa_not_observed = 0

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
        for i, data in enumerate(datasets):
            if i != 0:
                pfa_pdf_part = dist.Photoswitching_fingerprint_model(
                    params=prepare_pfa_parameters(z=z, n=i, params=params),
                    domain=(0, 300),
                ).pdf_part
                use_1 = mixture_log_likelihood_hist_marginal
                use_2 = mixture_log_likelihood_hist_three_exp_marginal
                pdf_part_index = i - 1
                use_parameters = [
                    0,
                    300,
                    pfa_pdf_part,
                    pdf_part_index,
                ]
            else:
                use_1 = mixture_log_likelihood_hist
                use_2 = mixture_log_likelihood_hist_three_exp
                use_parameters = []
            if i == z:
                parameters = params[0:5]
                negative_log_likelihood = use_2(
                    parameters,
                    data,
                    bin_edges,
                    *use_parameters,
                    rel_events_not_observed=rel_events_not_observed[i],
                )

            else:
                parameters = params[add + counter * 3 : (add + 3) + counter * 3]
                negative_log_likelihood = use_1(
                    parameters,
                    data,
                    bin_edges,
                    *use_parameters,
                    rel_events_not_observed=rel_events_not_observed[i],
                )

                counter += 1
            if norm:
                negative_log_likelihood /= data.sum()  # data is histogrammed
            total_negative_log_likelihood += negative_log_likelihood

        if pfa_bin_edges is not None and pfa_counts is not None:
            parameters = prepare_pfa_parameters(z, len(datasets), params)
            negative_log_likelihood = pfa_log_likelihood(
                parameters,
                pfa_counts,
                pfa_bin_edges,
                rel_events_not_observed=pfa_not_observed,
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
                parameters[i] = [pz1, pz2, params[2], params[3], params[4]]
            else:
                parameters[i] = [
                    params[(i - (i > z)) * 3 + 5],
                    params[(i - (i > z)) * 3 + 1 + 5],
                    params[(i - (i > z)) * 3 + 2 + 5],
                ]
    else:
        for i in range(n):
            parameters[i] = [
                params[i * 3],
                params[i * 3 + 1],
                params[i * 3 + 2],
            ]
    return parameters
