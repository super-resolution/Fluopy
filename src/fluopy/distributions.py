"""
Module distributions
"""

import inspect
from itertools import product

import numpy as np
from scipy.special import i1
from scipy.stats import expon, rv_discrete


def high_gain_amplification_noise_distribution(x_min=1, x_max=100, v=1, gain=100):
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
    x_min : int
        Minimum support value.
    x_max : int
        Maximum support value.
    v : float
        The mean of the non-amplified (nearly) poissonian photon count distribution.
        Has to include 0 counts.
    gain : float
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


def hypoexponential_distribution_cdf(x, *args):
    """
    CDF of the hypoexponential distribution.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of the hypoexponential distribution.
    """
    cdf = 1
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        cdf -= np.exp(-arg * x) * np.prod(other_args) / np.prod(-arg + other_args)

    return cdf


def hypoexponential_distribution_pdf(x, *args):
    """
    PDF of the hypoexponential distribution.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    pdf : float, 1-D array_like
        PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf += np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)

    return pdf


def hypoexponential_distribution_pdf_1st_order_derivative(x, *args):
    """
    First order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    pdf_1st_order_derivative : float, 1-D array_like
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


def hypoexponential_distribution_pdf_2nd_order_derivative(x, *args):
    """
    Second order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution.

    Returns
    -------
    pdf_2nd_order_derivative : float, 1-D array_like
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

    def __init__(self, lambdas, pis_orig, domain=(0, np.inf)):
        """
        Parameters
        ----------
        lambdas : float, 2-D array_like
            Rates of the underlying exponential distributions.
        pis_orig : float, 1-D array_like
            Weights of the underlying exponential distributions.
        domain : tuple
            Domain of the model. Default is (0, np.inf).
        """
        self.lambdas = np.asarray(lambdas)
        pis_orig = np.asarray(pis_orig)
        self.pis_orig = np.array([pis_orig, 1 - pis_orig])
        self.domain = domain

    def pdf(self, x, order=0):
        """
        PDF

        Parameters
        ----------
        x : float, 1-D array_like
            Sample.
        order : int
            Order of the derivative of the PDF to be calculated.

        Returns
        -------
        pdf : float, 1-D array_like
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

    def cdf(self, x, extra=False):
        """
        CDF

        Parameters
        ----------
        x : float, 1-D array_like
            Sample.

        Returns
        -------
        cdf : float, 1-D array_like
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

    def dpdf(self, x):
        """
        first derivate of PDF
        """
        dpdf = self.pdf(x, order=1)

        return dpdf

    def ddpdf(self, x):
        """
        second derivate of PDF
        """
        ddpdf = self.pdf(x, order=2)

        return ddpdf

    def logp(self, x):
        """
        Logarithm of the PDF
        """
        logp = np.log(self.pdf(x))  # lower-level implementation of log does not provide
        # much numerical stability since sum of e^x terms has to be calculated first.

        return logp

    def quantile_function():
        """
        Quantile function
        """
        raise ValueError(
            "Quantile function has no closed form. Inverse CDF has to be "
            "calculated numerically."
        )


def photoswitching_fingerprint_prepare(n, pis):
    """
    Get indices for valid lambda combinations for the photoswitching fingerprint model.
    Modifies the weights of the underlying exponential distributions by converting
    probabilities to 1, if their counterpart is not valid.

    Parameters
    ----------
    n : int
        Number of fluorophores.
    pis : 2-D array_like
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


def ps_fingerprint_cdf_1f(x, lam1_1, lam1_2, pi1, domain):
    """
    Cumulative distribution function of the photoswitching fingerprint model with one
    fluorophore.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lam1_1 : float
        Rate of the first (biased) exponential distribution.
    lam1_2 : float
        Rate of the second (non-biased) exponential distribution.
    pi1 : float
        Weight of the first exponential distribution.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1], [lam1_2]],
        pis_orig=[pi1],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_2f(x, lam1_1, lam2_1, lam1_2, lam2_2, pi1, pi2, domain):
    """
    Cumulative distribution function of the photoswitching fingerprint model with two
    fluorophores.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lam1_1 : float
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1 : float
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam1_2 : float
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2 : float
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    pi1 : float
        Weight of the first exponential distribution of the first fluorophore.
    pi2 : float
        Weight of the first exponential distribution of the second fluorophore.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1], [lam1_2, lam2_2]],
        pis_orig=[pi1, pi2],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_3f(
    x, lam1_1, lam2_1, lam3_1, lam1_2, lam2_2, lam3_2, pi1, pi2, pi3, domain
):
    """
    Cumulative distribution function of the photoswitching fingerprint model with three
    fluorophores.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lam1_1 : float
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1 : float
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam3_1 : float
        Rate of the first (biased) exponential distribution of the third fluorophore.
    lam1_2 : float
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2 : float
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    lam3_2 : float
        Rate of the second (non-biased) exponential distribution of the third fluorophore.
    pi1 : float
        Weight of the first exponential distribution of the first fluorophore.
    pi2 : float
        Weight of the first exponential distribution of the second fluorophore.
    pi3 : float
        Weight of the first exponential distribution of the third fluorophore.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1, lam3_1], [lam1_2, lam2_2, lam3_2]],
        pis_orig=[pi1, pi2, pi3],
        domain=domain,
    ).cdf(x)

    return cdf


def ps_fingerprint_cdf_4f(
    x,
    lam1_1,
    lam2_1,
    lam3_1,
    lam4_1,
    lam1_2,
    lam2_2,
    lam3_2,
    lam4_2,
    pi1,
    pi2,
    pi3,
    pi4,
    domain,
):
    """
    Cumulative distribution function of the photoswitching fingerprint model with four
    fluorophores.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lam1_1 : float
        Rate of the first (biased) exponential distribution of the first fluorophore.
    lam2_1 : float
        Rate of the first (biased) exponential distribution of the second fluorophore.
    lam3_1 : float
        Rate of the first (biased) exponential distribution of the third fluorophore.
    lam4_1 : float
        Rate of the first (biased) exponential distribution of the fourth fluorophore.
    lam1_2 : float
        Rate of the second (non-biased) exponential distribution of the first fluorophore.
    lam2_2 : float
        Rate of the second (non-biased) exponential distribution of the second fluorophore.
    lam3_2 : float
        Rate of the second (non-biased) exponential distribution of the third fluorophore.
    lam4_2 : float
        Rate of the second (non-biased) exponential distribution of the fourth fluorophore.
    pi1 : float
        Weight of the first exponential distribution of the first fluorophore.
    pi2 : float
        Weight of the first exponential distribution of the second fluorophore.
    pi3 : float
        Weight of the first exponential distribution of the third fluorophore.
    pi4 : float
        Weight of the first exponential distribution of the fourth fluorophore.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of the photoswitching fingerprint model.
    """
    cdf = Photoswitching_fingerprint_model(
        lambdas=[[lam1_1, lam2_1, lam3_1, lam4_1], [lam1_2, lam2_2, lam3_2, lam4_2]],
        pis_orig=[pi1, pi2, pi3, pi4],
        domain=domain,
    ).cdf(x)

    return cdf


def two_expon_mixture_cdf(x, lambda1, lambda2, p):
    """
    Cumulative distribution function of a mixture of two exponential distributions.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lambda1 : float
        Rate of the first exponential distribution.
    lambda2 : float
        Rate of the second exponential distribution.
    p : float
        Weight of the first exponential distribution.

    Returns
    -------
    cdf : float, 1-D array_like
        CDF of two exponential mixture distribution.
    """
    cdf = p * expon.cdf(x, scale=1 / lambda1) + (1 - p) * expon.cdf(
        x, scale=1 / lambda2
    )
    if cdf.size > 2:
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    return cdf


def two_expon_mixture_pdf(x, lambda1, lambda2, p):
    """
    Probability density function of a mixture of two exponential distributions.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lambda1 : float
        Rate of the first exponential distribution.
    lambda2 : float
        Rate of the second exponential distribution.
    p : float
        Weight of the first exponential distribution.

    Returns
    -------
    pdf : float, 1-D array_like
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


def rejection_sampling(pdf, x_min, x_max, y_min, y_max, batch, size, parameters, seed):
    """
    Technique to sample from a distribution with a known PDF.
    Adapted from https://cosmiccoding.com.au/tutorials/rejection_sampling/.
    Needed if the inverse function of the CDF is not analytically computable.

    Parameters
    ----------
    pdf : callable
        Probability density function of distribution of interest.
    x_min : float
        Sample space lower bound.
    x_max : float
        Sample space upper bound.
    y_min : float
        Probability density lower bound.
    y_max : float
        Probability density upper bound.
    batch : int
        Number of possible samples tested simultaneously.
    size : int
        Number of samples to be generated.
    parameters : list
        Parameters to be passed to pdf in corresponding order.
    seed : None, int, BitGenerator, Generator
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
