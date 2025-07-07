"""
Module distributions
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import i1
from scipy.stats import expon, rv_discrete

# type definition for random number generator seed
RandomGeneratorSeed = (
    None
    | int
    | Sequence[int]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
)


def high_gain_amplification_noise_distribution(
    x_min: int = 1, x_max: int = 100, v: float = 1, gain: float = 100
) -> rv_discrete:
    """
    The high gain amplification noise distribution as proposed in
    https://doi.org/10.1117/12.2004621 with the adjustment of not considering 0 as a
    possible variable's value. The support is limited to a maximum x of around
    125000 * gain / v.
    Applies if gain is added to poisson distributed photon counts. This is the case, if
    the interarrival time is exponentially distributed or can be approximated with an
    exponential distribution. Resembles a high gain approximation, indicating a better
    fit for higher gains. Indeed, low gains of 1 to 10 should be avoided.

    Parameters
    ----------
    x_min
        Minimum support value.
    x_max
        Maximum support value.
    v
        The mean of the non-amplified (nearly) poissonian photon count distribution.
        Has to include 0 counts.
    gain
        The gain applied to the photon counts.

    Returns
    -------
    distribution : scipy.stats._distn_infrastructure.rv_sample
        High gain amplification noise distribution.
    """
    # the value z of iv cannot be larger than ~714:
    if x_max > 120000 * gain / v:
        raise ValueError("x_max is too large (> 120,000 * gain / v).")
    x = np.arange(start=x_min, stop=x_max)

    x = x.astype(float)

    probabilities = (
        1
        / x
        * np.exp(-(x / gain + v))
        * np.sqrt(v * x / gain)
        * i1(2 * np.sqrt(v * x / gain))
    )
    probabilities = probabilities / np.sum(probabilities)
    distribution = rv_discrete(name="high_gain_distr", values=(x, probabilities))

    return distribution


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
        lambdas: float | npt.ArrayLike,
        pis_orig: float | npt.ArrayLike,
        domain: tuple[float, float] = (0, np.inf),
    ) -> None:
        """
        Parameters
        ----------
        lambdas
            Rates of the underlying exponential distributions (2D).
        pis_orig
            Weights of the underlying exponential distributions (1D).
        domain
            Domain of the model. Default is (0, np.inf).
        """
        self.lambdas = np.asarray(lambdas)
        pis_orig = np.asarray(pis_orig)
        self.pis_orig = np.array([pis_orig, 1 - pis_orig])
        self.domain = domain

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

        n = self.lambdas.shape[1]
        pdf = 0
        for i in range(n):
            valid_combinations, pis = photoswitching_fingerprint_prepare(
                i + 1, self.pis_orig
            )
            pdf_part = 0
            for j in range(i + 2):
                pi_set = np.prod(pis[j])
                pdf_part += pi_set * call(
                    x,
                    *self.lambdas[
                        valid_combinations[j], np.arange(valid_combinations[j].shape[0])
                    ],
                )
            pdf += 1 / n * pdf_part

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
        n = self.lambdas.shape[1]
        cdf = 0
        for i in range(n):
            valid_combinations, pis = photoswitching_fingerprint_prepare(
                i + 1, self.pis_orig
            )
            cdf_part = 0
            for j in range(i + 2):
                pi_set = np.prod(pis[j])
                cdf_part += pi_set * hypoexponential_distribution_cdf(
                    x,
                    *self.lambdas[
                        valid_combinations[j], np.arange(valid_combinations[j].shape[0])
                    ],
                )
            cdf += 1 / n * cdf_part
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
    n: int, pis: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Get indices for valid lambda combinations for the photoswitching fingerprint model.
    Modifies the weights of the underlying exponential distributions by converting
    probabilities to 1, if their counterpart is not valid.

    Parameters
    ----------
    n
        Number of fluorophores.
    pis
        Weights of the underlying exponential distributions.

    Returns
    -------
    valid_combinations : 2-D array_like
        Valid combinations of lambdas.
    pis : 2-D array_like
        Modified weights of the underlying exponential distributions.
    """
    combinations = product([0, 1], repeat=n)
    valid_combinations = [
        comb
        for comb in combinations
        if all(not (comb[h] == 0 and comb[h - 1] == 1) for h in range(1, n))
    ]
    valid_combinations = np.array(valid_combinations, dtype=int)
    pis = pis[valid_combinations, np.arange(valid_combinations.shape[1])]
    for n, comb in enumerate(valid_combinations):
        ones = np.where(comb == 1)[0]
        if ones.size > 1:
            pis[n, ones[1:]] = 1

    return valid_combinations, pis


def ps_fingerprint_cdf_1f(
    x: float | npt.ArrayLike,
    lam1_1: float,
    lam1_2: float,
    pi1: float,
    domain: tuple[float, float],
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of the photoswitching fingerprint model with one
    fluorophore.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lam1_1
        Rate of the first (biased) exponential distribution.
    lam1_2
        Rate of the second (non-biased) exponential distribution.
    pi1
        Weight of the first exponential distribution.
    domain
        Domain of the model.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1], [lam1_2]],
        pis_orig=[pi1],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_2f(
    x: float | npt.ArrayLike,
    lam1_1: float,
    lam2_1: float,
    lam1_2: float,
    lam2_2: float,
    pi1: float,
    pi2: float,
    domain,
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of the photoswitching fingerprint model with two
    fluorophores.

    Parameters
    ----------
    x
        Sample.
    lam1_1
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam1_2
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    pi1
        Weight of the first exponential distribution of the first fluorophore.
    pi2
        Weight of the first exponential distribution of the second fluorophore.
    domain
        Domain of the model.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1], [lam1_2, lam2_2]],
        pis_orig=[pi1, pi2],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_3f(
    x: float | npt.ArrayLike,
    lam1_1: float,
    lam2_1: float,
    lam3_1: float,
    lam1_2: float,
    lam2_2: float,
    lam3_2: float,
    pi1: float,
    pi2: float,
    pi3: float,
    domain: tuple[float, float],
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of the photoswitching fingerprint model with three
    fluorophores.

    Parameters
    ----------
    x
        Sample.
    lam1_1
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam3_1
        Rate of the first (biased) exponential distribution of the third fluorophore.
    lam1_2
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    lam3_2
        Rate of the second (non-biased) exponential distribution of the third fluorophore.
    pi1
        Weight of the first exponential distribution of the first fluorophore.
    pi2
        Weight of the first exponential distribution of the second fluorophore.
    pi3
        Weight of the first exponential distribution of the third fluorophore.
    domain
        Domain of the model.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1, lam3_1], [lam1_2, lam2_2, lam3_2]],
        pis_orig=[pi1, pi2, pi3],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_4f(
    x: float | npt.ArrayLike,
    lam1_1: float,
    lam2_1: float,
    lam3_1: float,
    lam4_1: float,
    lam1_2: float,
    lam2_2: float,
    lam3_2: float,
    lam4_2: float,
    pi1: float,
    pi2: float,
    pi3: float,
    pi4: float,
    domain: tuple[float, float],
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of the photoswitching fingerprint model with four
    fluorophores.

    Parameters
    ----------
    x
        Sample.
    lam1_1
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam3_1
        Rate of the first (biased) exponential distribution of the third fluorophore.
    lam4_1
        Rate of the first (biased) exponential distribution of the fourth fluorophore.
    lam1_2
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    lam3_2
        Rate of the second (non-biased) exponential distribution of the third fluorophore.
    lam4_2
        Rate of the second (non-biased) exponential distribution of the fourth fluorophore.
    pi1
        Weight of the first exponential distribution of the first fluorophore.
    pi2
        Weight of the first exponential distribution of the second fluorophore.
    pi3
        Weight of the first exponential distribution of the third fluorophore.
    pi4
        Weight of the first exponential distribution of the fourth fluorophore.
    domain
        Domain of the model.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1, lam3_1, lam4_1], [lam1_2, lam2_2, lam3_2, lam4_2]],
        pis_orig=[pi1, pi2, pi3, pi4],
        domain=domain,
    ).cdf(x)

    return cdf


def two_expon_mixture_cdf(
    x: float | npt.ArrayLike, lambda1: float, lambda2: float, p: float
) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function of a mixture of two exponential distributions.

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
    x: float | npt.ArrayLike, lambda1: float, lambda2: float, p: float
) -> float | npt.NDArray[np.float64]:
    """
    Probability density function of a mixture of two exponential distributions.

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
