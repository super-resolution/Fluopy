"""
Module distributions
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import expon

if TYPE_CHECKING:
    from fluopy.fluopy_types import RandomGeneratorSeed


def hypoexponential_distribution_cdf(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    CDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the hypoexponential distribution.
    """
    cdf = 1
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        cdf -= np.exp(-arg * x) * np.prod(other_args) / np.prod(-arg + other_args)

    return cdf


def hypoexponential_distribution_pdf(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf += np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)

    return pdf


def hypoexponential_distribution_pdf_1st_order_derivative(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    First order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    pdf_1st_order_derivative : float | npt.NDArray[np.float64]
        First order derivative of the PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf_1st_order_derivative = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf_1st_order_derivative += (
            -arg * np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)
        )

    return pdf_1st_order_derivative


def hypoexponential_distribution_pdf_2nd_order_derivative(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    Second order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    pdf_2nd_order_derivative : float | npt.NDArray[np.float64]
        Second order derivative of the PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf_2nd_order_derivative = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf_2nd_order_derivative += (
            arg**2 * np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)
        )

    return pdf_2nd_order_derivative


class Photoswitching_fingerprint_model:
    """
    Model to describe photoswitching fingerprints produced by n fluorophores where there
    is bias in time (e.g., increased probability of ON in the beginning).
    """

    def __init__(
        self,
        params: dict,
        weights: float | npt.ArrayLike = None,
        domain: tuple[float, float] = (0, np.inf),
    ) -> None:
        """
        Parameters
        ----------
        params
            Parameters of the underlying exponential distributions.
        weights
            Weights of each fluorophore (1D).
        domain
            Domain of the model. Default is (0, np.inf).
        """
        self.params = params
        if weights is None:
            weights = np.ones(len(params)) / len(params)
        self.weights = np.asarray(weights)
        self.domain = domain
        lengths = {k: len(params[k]) for k in params}
        max_length = max(lengths.values())
        self.z = (
            -1
            if list(lengths.values()).count(max_length) > 1
            else max(lengths, key=lengths.get)
        )

    def pdf(
        self, x: float | npt.ArrayLike, order: int = 0
    ) -> float | npt.NDArray[np.float64]:
        """
        PDF

        Parameters
        ----------
        x
            Sample.
        order : int
            Order of the derivative of the PDF to be calculated.

        Returns
        -------
        pdf : float | npt.NDArray[np.float64]
            Model output.
        """
        if order == 0:
            call = hypoexponential_distribution_pdf
        elif order == 1:
            caller = inspect.stack()[1].function
            if caller != "dpdf":
                raise ValueError("Call dpdf instead of setting order=1.")
            call = hypoexponential_distribution_pdf_1st_order_derivative
        elif order == 2:
            caller = inspect.stack()[1].function
            if caller != "ddpdf":
                raise ValueError("Call ddpdf instead of setting order=2.")
            call = hypoexponential_distribution_pdf_2nd_order_derivative
        else:
            raise ValueError("Order has to be 0, 1, or 2.")

        n = len(self.params)
        pdf = 0
        for i in range(n):
            lambdas, pis = photoswitching_fingerprint_prepare(
                self.params,
                i + 1,
                self.z,
            )
            pdf_part = 0
            for lambda_combo, pi_combo in zip(lambdas, pis):
                pi_set = np.prod(pi_combo)
                pdf_part += pi_set * call(
                    x,
                    *lambda_combo,
                )
            pdf += self.weights[i] * pdf_part

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(self.domain[-1], extra=True)
            F_0 = self.cdf(self.domain[0], extra=True)
            pdf = pdf / (F_1 - F_0)  # true for pdf, dpdf, ddpdf

        return pdf

    def cdf(
        self, x: float | npt.ArrayLike, extra: bool = False
    ) -> float | npt.NDArray[np.float64]:
        """
        CDF

        Parameters
        ----------
        x
            Sample.
        extra
            ...

        Returns
        -------
        cdf : float | npt.NDArray[np.float64]
            Model output.
        """
        n = len(self.params)
        cdf = 0
        for i in range(n):
            lambdas, pis = photoswitching_fingerprint_prepare(
                self.params,
                i + 1,
                self.z,
            )
            cdf_part = 0
            for lambda_combo, pi_combo in zip(lambdas, pis):
                pi_set = np.prod(pi_combo)
                cdf_part += pi_set * hypoexponential_distribution_cdf(
                    x,
                    *lambda_combo,
                )
            cdf += self.weights[i] * cdf_part
        if extra:
            return cdf

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(self.domain[-1], extra=True)
            F_0 = self.cdf(self.domain[0], extra=True)
            cdf = (cdf - F_0) / (F_1 - F_0)

        return cdf

    def dpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        first derivate of PDF
        """
        dpdf = self.pdf(x, order=1)

        return dpdf

    def ddpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        second derivate of PDF
        """
        ddpdf = self.pdf(x, order=2)

        return ddpdf

    def logp(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Logarithm of the PDF
        """
        logp = np.log(self.pdf(x))  # lower-level implementation of log does not provide
        # much numerical stability since sum of e^x terms has to be calculated first.

        return logp

    def quantile_function(self) -> None:
        """
        Quantile function
        """
        raise ValueError(
            "Quantile function has no closed form. Inverse CDF has to be "
            "calculated numerically."
        )


def photoswitching_fingerprint_prepare(
    params: dict,
    n: int,
    z: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Get combinations of lambdas and pis for the photoswitching fingerprint model.
    See model derivation for details.

    Parameters
    ----------
    params
        Parameters of the underlying exponential distributions.
    n
        Number of fluorophores.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    tuple[list[list[float]], list[list[float]]]
        Combinations of lambdas and pis for the photoswitching fingerprint model.
    """
    valid_combinations = generate_combinations(n, z)
    lambdas = map_to_lambdas(valid_combinations, params, z)
    pis = get_pis(valid_combinations, params, z)
    return lambdas, pis


def generate_combinations(n: int, z: int):
    """
    Generate all valid convolutions of exponential distributions.

    Parameters
    ----------
    n
        Number of fluorophores.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    valid_combos : 2-D np.ndarray
        All valid combinations of exponential distributions. Array of shape (m, n) where
        m is the number of valid combinations.
        For each m, the n columns represent the n fluorophores. Each entry can be 0
        (biased exponential distribution), 1 (non-biased exponential distribution of
        two-component mixture), 2 (first non-biased exponential distribution of
        three-component mixture), or 3 (second non-biased exponential distribution of
        three-component mixture).
    """
    arrays = []
    for i in range(n):
        if i == z:
            arrays.append([0, 2, 3])  # b, nb_1 and nb_2
        else:
            arrays.append([0, 1])  # b and nb_0
    combos = np.array(list(product(*arrays)), dtype=int)
    if combos.shape[1] == 1:
        valid_combos = combos
    zeros = combos == 0
    curr_zeros = zeros[:, 1:]
    prev_zeros = zeros[:, :-1]
    mask = np.all((~curr_zeros) | prev_zeros, axis=1)
    valid_combos = combos[mask]

    return valid_combos


def map_to_lambdas(
    combos: npt.NDArray[np.int_], params: dict, z: int
) -> npt.NDArray[np.float64]:
    """
    Map combinations to lambdas.

    Parameters
    ----------
    combos
        All valid combinations of exponential distributions.
    params
        Parameters of the underlying exponential distributions.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    result : 2-D np.ndarray
        Mapped lambdas. Array of shape (m, n) where m is the number of valid combinations
        and n the number of fluorophores.
    """
    result = np.empty_like(combos, dtype=float)
    for idx in range(combos.shape[1]):
        col = combos[:, idx]
        if idx == z:
            mapping = {0: params[idx][3], 2: params[idx][4], 3: params[idx][5]}
        else:
            mapping = {0: params[idx][2], 1: params[idx][3]}

        mapper = np.vectorize(mapping.get)
        result[:, idx] = mapper(col)
    return result


def get_pis(
    combos: npt.NDArray[np.int_], params: dict, z: int
) -> npt.NDArray[np.float64]:
    """
    Get pis for each combination.

    Parameters
    ----------
    combos
        All valid combinations of exponential distributions.
    params
        Parameters of the underlying exponential distributions.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    pis : 2-D np.ndarray
        Mapped pis. Array of shape (m, n) where m is the number of valid combinations
        and n the number of fluorophores.
    """
    pis = np.ones_like(combos, dtype=float)
    if z != -1:
        normalize = params[z][1] + params[z][2]
    else:
        normalize = 1
    for idx in range(combos.shape[1]):
        col_all = combos[:, idx]
        col_filt = combos[1:, idx]
        mapping = {
            0: params[idx][0],
            1: params[idx][1],
            2: params[idx][1],
            3: params[idx][2],
        }
        mapper = np.vectorize(mapping.get, otypes=[float])
        if idx == combos.shape[1] - 1:
            pis[:, idx] = mapper(col_all)
        else:
            zeros = np.where(col_filt == 0)[0]
            pis[1:, :][zeros, idx] = mapper(col_filt[zeros])

            ones = np.where(col_filt == 1)[0]
            if idx == 0:
                pis[1:, :][ones, idx] = mapper(col_filt[ones])
            else:
                mask = combos[1:, :][ones, idx - 1] == 0
                pis[1:, :][ones[mask], idx] = mapper(col_filt[ones[mask]])

            twos_threes = np.isin(col_filt, [2, 3])
            twos_threes = np.where(twos_threes)[0]
            if idx == 0:
                pis[1:, :][twos_threes, idx] = mapper(col_filt[twos_threes])
            else:
                mask = combos[1:, :][twos_threes, idx - 1] == 0
                pis[1:, :][twos_threes[mask], idx] = mapper(col_filt[twos_threes[mask]])
                pis[1:, :][twos_threes[~mask], idx] = (
                    mapper(col_filt[twos_threes[~mask]]) / normalize
                )
    return pis


def two_expon_mixture_cdf(
    x: float | npt.ArrayLike,
    p: float,
    lambda1: float,
    lambda2: float,
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of a mixture of two exponential distributions.
    Automatic truncation is applied if x contains more than two values.

    Parameters
    ----------
    x
        Sample.
    lambda1
        Rate of the first exponential distribution.
    lambda2
        Rate of the second exponential distribution.
    p
        Weight of the first exponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of two exponential mixture distribution.
    """
    cdf = p * expon.cdf(x, scale=1 / lambda1) + (1 - p) * expon.cdf(
        x, scale=1 / lambda2
    )
    if cdf.size > 2:
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    return cdf


def two_expon_mixture_pdf(
    x: float | npt.ArrayLike, p: float, lambda1: float, lambda2: float
) -> float | npt.NDArray[np.float64]:
    """
    Probability density function of a mixture of two exponential distributions.
    Automatic truncation is applied if x contains more than two values.

    Parameters
    ----------
    x
        Sample.
    lambda1
        Rate of the first exponential distribution.
    lambda2
        Rate of the second exponential distribution.
    p
        Weight of the first exponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        PDF of two exponential mixture distribution.
    """
    pdf = p * expon.pdf(x, scale=1 / lambda1) + (1 - p) * expon.pdf(
        x, scale=1 / lambda2
    )
    cdf = p * expon.cdf(x, scale=1 / lambda1) + (1 - p) * expon.cdf(
        x, scale=1 / lambda2
    )
    if pdf.size > 2:
        pdf = pdf / (cdf[-1] - cdf[0])

    return pdf


def three_expon_mixture_cdf(
    x: float | npt.ArrayLike,
    p1: float,
    p2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of a mixture of three exponential distributions.
    Automatic truncation is applied if x contains more than two values.

    Parameters
    ----------
    x
        Sample.
    lambda1
        Rate of the first exponential distribution.
    lambda2
        Rate of the second exponential distribution.
    lambda3
        Rate of the third exponential distribution.
    p1
        Weight of the first exponential distribution.
    p2
        Weight of the second exponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of three exponential mixture distribution.
    """
    cdf = (
        p1 * expon.cdf(x, scale=1 / lambda1)
        + p2 * expon.cdf(x, scale=1 / lambda2)
        + (1 - p1 - p2) * expon.cdf(x, scale=1 / lambda3)
    )
    if cdf.size > 2:
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    return cdf


def three_expon_mixture_pdf(
    x: float | npt.ArrayLike,
    p1: float,
    p2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> float | npt.NDArray[np.float64]:
    """
    Probability density function of a mixture of three exponential distributions.
    Automatic truncation is applied if x contains more than two values.

    Parameters
    ----------
    x
        Sample.
    lambda1
        Rate of the first exponential distribution.
    lambda2
        Rate of the second exponential distribution.
    lambda3
        Rate of the third exponential distribution.
    p1
        Weight of the first exponential distribution.
    p2
        Weight of the second exponential distribution.

    Returns
    -------
    float | npt.NDArray[np.float64]
        PDF of three exponential mixture distribution.
    """
    pdf = (
        p1 * expon.pdf(x, scale=1 / lambda1)
        + p2 * expon.pdf(x, scale=1 / lambda2)
        + (1 - p1 - p2) * expon.pdf(x, scale=1 / lambda3)
    )
    cdf = (
        p1 * expon.cdf(x, scale=1 / lambda1)
        + p2 * expon.cdf(x, scale=1 / lambda2)
        + (1 - p1 - p2) * expon.cdf(x, scale=1 / lambda3)
    )
    if pdf.size > 2:
        pdf = pdf / (cdf[-1] - cdf[0])

    return pdf


def rejection_sampling(
    pdf: Callable[Any, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    batch: int,
    size: int,
    parameters: Iterable[Any],
    seed: RandomGeneratorSeed,
):
    """
    Technique to sample from a distribution with a known PDF.
    Adapted from https://cosmiccoding.com.au/tutorials/rejection_sampling/.
    Needed if the inverse function of the CDF is not analytically computable.

    Parameters
    ----------
    pdf
        Probability density function of distribution of interest.
    x_min
        Sample space lower bound.
    x_max
        Sample space upper bound.
    y_min
        Probability density lower bound.
    y_max
        Probability density upper bound.
    batch
        Number of possible samples tested simultaneously.
    size
        Number of samples to be generated.
    parameters
        Parameters to be passed to pdf in corresponding order.
    seed
        A seed to initialize the BitGenerator.

    Returns
    -------
    samples : np.ndarray
        Generated samples.
    """
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < size:
        x = rng.uniform(low=x_min, high=x_max, size=batch)
        y = rng.uniform(low=y_min, high=y_max, size=batch)
        samples += x[y < pdf(*parameters, x)].tolist()
    samples = np.array(samples)

    return samples[:size]
