"""
Module fitting
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, differential_evolution, minimize
from scipy.stats import expon
from sklearn.metrics import r2_score, root_mean_squared_error

from . import distributions as dist

if TYPE_CHECKING:
    pass


def mixture_log_likelihood(
    params: Iterable,
    data: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    number_no_events: int = 0,
) -> float:
    """
    Negative log-likelihood of a mixture of two exponential distributions (pdf).

    Parameters
    ----------
    params
        Parameters of the mixture distribution.
    data
        Sample.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.
    number_no_events
        Number of runs without events.

    Returns
    -------
    float
        Negative log-likelihood to be minimized.
    """
    pi, lambda1, lambda2 = params

    if not (0 <= pi <= 1) or lambda1 <= 0 or lambda2 <= 0:
        return 1e20

    norm1 = (1 - np.exp(-lambda1 * truncation_up)) - (  # adjust for truncation
        1 - np.exp(-lambda1 * truncation_low)
    )
    norm2 = (1 - np.exp(-lambda2 * truncation_up)) - (
        1 - np.exp(-lambda2 * truncation_low)
    )

    exp1 = pi * expon.pdf(data, scale=1 / lambda1) / norm1
    exp2 = (1 - pi) * expon.pdf(data, scale=1 / lambda2) / norm2
    pdf = exp1 + exp2
    safe_pdf = np.clip(pdf, 1e-14, None)
    log_likelihood_observation = np.log(safe_pdf)
    if number_no_events == 0:
        log_likelihood_no_observation = 0
    else:
        prob_event = pi * norm1 + (1 - pi) * norm2  # probability of observing an event
        # within the truncation range (given the distribution is non-truncated)
        prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
        log_likelihood_no_observation = np.log1p(-prob_event) * number_no_events

    log_likelihood = np.sum(log_likelihood_observation) + log_likelihood_no_observation
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def mixture_log_likelihood_hist(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    rel_events_not_observed: int,
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
    probs = (
        pi * (np.exp(-lambda1 * a) - np.exp(-lambda1 * b))
        + (1 - pi) * (np.exp(-lambda2 * a) - np.exp(-lambda2 * b))
    ) / (
        pi * (np.exp(-lambda1 * truncation_low) - np.exp(-lambda1 * truncation_up))
        + (1 - pi)
        * (np.exp(-lambda2 * truncation_low) - np.exp(-lambda2 * truncation_up))
    )
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    if rel_events_not_observed == 0:
        log_likelihood_no_observation = 0
        log_likelihood_observation = 0
    else:
        prob_event = pi * (
            np.exp(-lambda1 * truncation_low) - np.exp(-lambda1 * truncation_up)
        ) + (1 - pi) * (
            np.exp(-lambda2 * truncation_low) - np.exp(-lambda2 * truncation_up)
        )  # probability of observing an event
        # within the truncation range (given the distribution is non-truncated)
        prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
        factor = rel_events_not_observed / (1 - rel_events_not_observed)
        log_likelihood_no_observation = np.log1p(-prob_event) * np.sum(counts) * factor
        log_likelihood_observation = np.sum(counts) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )

    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def mixture_log_likelihood_hist_three_exp(
    params: Iterable,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    rel_events_not_observed: int,
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
    probs = (
        pi1 * (np.exp(-lambda1 * a) - np.exp(-lambda1 * b))
        + pi2 * (np.exp(-lambda2 * a) - np.exp(-lambda2 * b))
        + pi3 * (np.exp(-lambda3 * a) - np.exp(-lambda3 * b))
    ) / (
        pi1 * (np.exp(-lambda1 * truncation_low) - np.exp(-lambda1 * truncation_up))
        + pi2 * (np.exp(-lambda2 * truncation_low) - np.exp(-lambda2 * truncation_up))
        + pi3 * (np.exp(-lambda3 * truncation_low) - np.exp(-lambda3 * truncation_up))
    )
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))

    if rel_events_not_observed == 0:
        log_likelihood_no_observation = 0
        log_likelihood_observation = 0
    else:
        prob_event = (
            pi1 * (np.exp(-lambda1 * truncation_low) - np.exp(-lambda1 * truncation_up))
            + pi2
            * (np.exp(-lambda2 * truncation_low) - np.exp(-lambda2 * truncation_up))
            + pi3
            * (np.exp(-lambda3 * truncation_low) - np.exp(-lambda3 * truncation_up))
        )  # probability of observing an event
        # within the truncation range (given the distribution is non-truncated)
        prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
        factor = rel_events_not_observed / (1 - rel_events_not_observed)
        log_likelihood_no_observation = np.log1p(-prob_event) * np.sum(counts) * factor
        log_likelihood_observation = np.sum(counts) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )

    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def pfa_log_likelihood(
    params: dict,
    counts: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    truncation_low: float,
    truncation_up: float,
    rel_events_not_observed: int,
) -> float:
    a = bin_edges[:-1]
    b = bin_edges[1:]
    probs = dist.Photoswitching_fingerprint_model(
        params, domain=(truncation_low, truncation_up)
    ).cdf(b) - dist.Photoswitching_fingerprint_model(
        params, domain=(truncation_low, truncation_up)
    ).cdf(
        a
    )
    probs = np.clip(probs, 1e-14, None)  # avoid log(0)
    log_likelihood_bin = np.sum(counts * np.log(probs))
    if rel_events_not_observed == 0:
        log_likelihood_no_observation = 0
        log_likelihood_observation = 0
    else:
        prob_event = dist.Photoswitching_fingerprint_model(
            params, domain=(truncation_low, truncation_up)
        ).cdf(truncation_up) - dist.Photoswitching_fingerprint_model(
            params, domain=(truncation_low, truncation_up)
        ).cdf(
            truncation_low
        )  # probability of observing an event
        # within the truncation range (given the distribution is non-truncated)
        prob_event = np.minimum(prob_event, 1 - 1e-14)  # avoid log(0)
        factor = rel_events_not_observed / (1 - rel_events_not_observed)
        log_likelihood_no_observation = np.log1p(-prob_event) * np.sum(counts) * factor
        log_likelihood_observation = np.sum(counts) * np.log(prob_event)
    log_likelihood = (
        log_likelihood_bin + log_likelihood_no_observation + log_likelihood_observation
    )
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood


def estimate_mixture_parameters(
    data: npt.ArrayLike,
    initial_guess: npt.ArrayLike,
    bounds: Iterable,
    truncation_low: float = 0,
    truncation_up: float = 300,
    number_no_events: int = 0,
    method: str = "L-BFGS-B",
) -> tuple[float, float, float]:
    """
    Estimate the parameters of a mixture of two exponential distributions (pdf).

    Parameters
    ----------
    data
        Sample or histogram data. If a histogram is provided, it should be a list
        containing two elements: the counts and the bin edges.
    initial_guess
        Initial guess for the optimization. For parameters, see return values.
    bounds
        Bounds for the optimization. For parameters, see return values.
    truncation_low
        Lower truncation of the distribution.
    truncation_up
        Upper truncation of the distribution.
    number_no_events
        Number of runs without events. Set to 0 if all runs produced events or if no
        influence of log-likelihood of no events is desired.
    method
        Optimization method.

    Returns
    -------
    pi : float
        Weight of the first exponential distribution.
    lambda1 : float
        Rate of the first exponential distribution.
    lambda2 : float
        Rate of the second exponential distribution.
    """
    if isinstance(data, np.ndarray):
        optimization_function = mixture_log_likelihood
        args = (data, truncation_low, truncation_up, number_no_events)
    elif isinstance(data, list) and len(data) == 2:
        optimization_function = mixture_log_likelihood_hist
        args = (data[0], data[1], truncation_low, truncation_up, number_no_events)
    result = minimize(
        optimization_function,
        initial_guess,
        args=args,
        bounds=bounds,
        method=method,
    )
    pi, lambda1, lambda2 = result.x

    return pi, lambda1, lambda2


def goodness_of_fit(
    fits: npt.NDArray[np.float64], data: npt.NDArray[np.float64], p: int
) -> tuple[float, float, float, float]:
    """
    Calculate the goodness of fit for the given fits and data.

    Parameters
    ----------
    fits
        The fitted values from the model.
    data
        The observed data values.
    p
        The number of parameters in the model.

    Returns
    -------
    ss_res : float
        The sum of squared residuals.
    rmse : float
        The root mean squared error.
    r2 : float
        The coefficient of determination (R^2).
    adj_r2 : float
        The adjusted R^2, which accounts for the number of predictors in the model.
    """
    r2 = r2_score(data, fits)
    residuals = data - fits
    ss_res = np.sum(residuals**2)
    rmse = root_mean_squared_error(data, fits)
    adj_r2 = 1 - (1 - r2) * (data.size - 1) / (data.size - p - 1)

    return ss_res, rmse, r2, adj_r2


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
    if pfa_not_observed is None:
        pfa_not_observed = 0

    if z != -1 and z < len(datasets):
        add = 5
    else:
        add = 0

    def global_objective(params):
        total_negative_log_likelihood = 0
        counter = 0
        for i, data in enumerate(datasets):
            if i == z:
                parameters = params[0:5]
                negative_log_likelihood = mixture_log_likelihood_hist_three_exp(
                    parameters,
                    data,
                    bin_edges,
                    0,
                    300,
                    rel_events_not_observed=rel_events_not_observed[i],
                )
            else:
                parameters = params[add + counter * 3 : (add + 3) + counter * 3]
                negative_log_likelihood = mixture_log_likelihood_hist(
                    parameters,
                    data,
                    bin_edges,
                    0,
                    300,
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
                0,
                300,
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
