"""
Module fitting
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, differential_evolution, minimize
from scipy.stats import expon

from . import distributions as dist

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult


def ps_fingerprint_cdf_fit_1f(
    x: npt.ArrayLike, y: npt.ArrayLike, **diff_ev: Any
) -> OptimizeResult:
    """
    Fit a 1f distribution to the data using the differential evolution algorithm.

    Parameters
    ----------
    x
        The x values of the data.
    y
        The y values of the data.
    diff_ev
        Additional parameters for the differential evolution algorithm.

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    def objective_function(params):
        domain = (x[0], x[-1])
        y_pred = dist.ps_fingerprint_cdf_1f(x, *params, domain=domain)
        error = np.sum((y - y_pred) ** 2)

        return error

    linear_constraint = LinearConstraint(
        A=[[1, -1, 0]],  # lam1_1 > lam1_2
        lb=[1e-3],
        ub=[np.inf],
    )
    bounds = Bounds(
        [1e-9, 1e-9, 0],
        [500, 500, 1],
    )
    result = differential_evolution(
        objective_function, bounds=bounds, constraints=linear_constraint, **diff_ev
    )
    return result


def ps_fingerprint_cdf_fit_2f(
    x: npt.ArrayLike, y: npt.ArrayLike, **diff_ev: Any
) -> OptimizeResult:
    """
    Fit a 2f distribution to the data using the differential evolution algorithm.

    Parameters
    ----------
    x
        The x values of the data.
    y
        The y values of the data.
    diff_ev
        Additional parameters for the differential evolution algorithm.

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """

    def objective_function(params):
        domain = (x[0], x[-1])
        y_pred = dist.ps_fingerprint_cdf_2f(x, *params, domain=domain)
        error = np.sum((y - y_pred) ** 2)

        return error

    linear_constraint = LinearConstraint(
        A=[
            [1, -1, 0, 0, 0, 0],  # lam1_1 > lam2_1
            [1, 0, -1, 0, 0, 0],  # lam1_1 > lam1_2
            [0, 1, 0, -1, 0, 0],  # lam2_1 > lam2_2
            [0, 0, 1, -1, 0, 0],  # lam1_2 > lam2_2
            [0, 0, 0, 0, 1, -1],  # pi1 > pi2
        ],
        lb=[1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        ub=[np.inf, np.inf, np.inf, np.inf, 1],
    )
    bounds = Bounds([1e-9, 1e-9, 1e-9, 1e-9, 0, 0], [500, 500, 500, 500, 1, 1])
    result = differential_evolution(
        objective_function, bounds=bounds, constraints=linear_constraint, **diff_ev
    )
    return result


def ps_fingerprint_cdf_fit_3f(
    x: npt.ArrayLike, y: npt.ArrayLike, **diff_ev: Any
) -> OptimizeResult:
    """
    Fit a 3f distribution to the data using the differential evolution algorithm.

    Parameters
    ----------
    x
        The x values of the data.
    y
        The y values of the data.
    diff_ev
        Additional parameters for the differential evolution algorithm.

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """

    def objective_function(params):
        domain = (x[0], x[-1])
        y_pred = dist.ps_fingerprint_cdf_3f(x, *params, domain=domain)
        error = np.sum((y - y_pred) ** 2)

        return error

    linear_constraint = LinearConstraint(
        A=[
            [1, -1, 0, 0, 0, 0, 0, 0, 0],  # lam1_1 > lam2_1
            [0, 1, -1, 0, 0, 0, 0, 0, 0],  # lam2_1 > lam3_1
            [1, 0, 0, -1, 0, 0, 0, 0, 0],  # lam1_1 > lam1_2
            [0, 1, 0, 0, -1, 0, 0, 0, 0],  # lam2_1 > lam2_2
            [0, 0, 1, 0, 0, -1, 0, 0, 0],  # lam3_1 > lam3_2
            [0, 0, 0, 1, -1, 0, 0, 0, 0],  # lam1_2 > lam2_2
            [0, 0, 0, 0, 1, -1, 0, 0, 0],  # lam2_2 > lam3_2
            [0, 0, 0, 0, 0, 0, 1, -1, 0],  # pi1 > pi2
            [0, 0, 0, 0, 0, 0, 0, 1, -1],  # pi2 > pi3
        ],
        lb=[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        ub=[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1],
    )
    bounds = Bounds(
        [1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 0, 0],
        [500, 500, 500, 500, 500, 500, 500, 1, 1],
    )
    result = differential_evolution(
        objective_function, bounds=bounds, constraints=linear_constraint, **diff_ev
    )
    return result


def ps_fingerprint_cdf_fit_4f(
    x: npt.ArrayLike, y: npt.ArrayLike, **diff_ev: Any
) -> OptimizeResult:
    """
    Fit a 4f distribution to the data using the differential evolution algorithm.

    Parameters
    ----------
    x
        The x values of the data.
    y
        The y values of the data.
    diff_ev
        Additional parameters for the differential evolution algorithm.

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """

    def objective_function(params):
        domain = (x[0], x[-1])
        y_pred = dist.ps_fingerprint_cdf_4f(x, *params, domain=domain)
        error = np.sum((y - y_pred) ** 2)

        return error

    linear_constraint = LinearConstraint(
        A=[
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lam1_1 > lam2_1
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lam2_1 > lam3_1
            [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # lam3_1 > lam4_1
            [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # lam1_1 > lam1_2
            [0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # lam2_1 > lam2_2
            [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # lam3_1 > lam3_2
            [0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0],  # lam4_1 > lam4_2
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # lam1_2 > lam2_2
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],  # lam2_2 > lam3_2
            [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # lam3_2 > lam4_2
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],  # pi1 > pi2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # pi2 > pi3
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # pi3 > pi4
        ],
        lb=[
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
        ],
        ub=[
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            1,
            1,
            1,
        ],
    )
    bounds = Bounds(
        [1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 0, 0, 0, 0],
        [500, 500, 500, 500, 500, 500, 500, 500, 1, 1, 1, 1],
    )
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        constraints=linear_constraint,
        **diff_ev,
    )
    return result


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
        return np.inf

    norm1 = (1 - np.exp(-lambda1 * truncation_up)) - (  # adjust for truncation
        1 - np.exp(-lambda1 * truncation_low)
    )
    norm2 = (1 - np.exp(-lambda2 * truncation_up)) - (
        1 - np.exp(-lambda2 * truncation_low)
    )

    exp1 = pi * expon.pdf(data, scale=1 / lambda1) / norm1
    exp2 = (1 - pi) * expon.pdf(data, scale=1 / lambda2) / norm2

    log_likelihood_observation = np.log(exp1 + exp2)
    if number_no_events == 0:
        log_likelihood_no_observation = 0
    else:
        prob_event = pi * norm1 + (1 - pi) * norm2
        prob_event = np.minimum(prob_event, 1 - 1e-14)
        log_likelihood_no_observation = np.log1p(-prob_event) * number_no_events

    log_likelihood = np.sum(log_likelihood_observation) + log_likelihood_no_observation
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
        Sample.
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
    result = minimize(
        mixture_log_likelihood,
        initial_guess,
        args=(data, truncation_low, truncation_up, number_no_events),
        bounds=bounds,
        method=method,
    )
    pi, lambda1, lambda2 = result.x

    return pi, lambda1, lambda2
