"""
Module distributions
"""
import numpy as np
from scipy.special import i1
from scipy.stats import rv_discrete


def high_gain_amplification_noise_distribution(x_min=1, x_max=100, v=1, gain=100):
    """
    The high gain amplification noise distribution as proposed in https://doi.org/10.1117/12.2004621 with the
    adjustment of not considering 0 as a possible variable's value. The support is limited to a maximum x of around
    125000 * gain / v.
    Applies if gain is added to poisson distributed photon counts. This is the case, if the interarrival time is
    exponentially distributed or can be approximated with an exponential distribution.
    Resembles a high gain approximation, indicating a better fit for higher gains. Indeed, low gains of 1 to 10 should
    be avoided.

    Parameters
    ----------
    x_min : int
        Minimum support value.
    x_max : int
        Maximum support value.
    v : float
        The mean of the non-amplified (nearly) poissonian photon count distribution. Has to include 0 counts.
    gain : float
        The gain applied to the photon counts.

    Returns
    -------
    distribution : scipy.stats._distn_infrastructure.rv_sample
        High gain amplification noise distribution.
    """
    # the value z of iv cannot be larger than ~714:
    if x_max > 120000 * gain / v:
        raise ValueError('x_max is too large (> 120,000 * gain / v).')
    x = np.arange(start=x_min, stop=x_max)

    x = x.astype(float)

    probabilities = 1 / x * np.exp(-(x/gain + v)) * np.sqrt(v * x/gain) * i1(2*np.sqrt(v * x/gain))
    probabilities = probabilities / np.sum(probabilities)
    distribution = rv_discrete(name='high_gain_distr', values=(x, probabilities))

    return distribution


def hypoexponential_distribution_two_parameters_pdf(a, b, x):
    """
    Probability density function of two-parameter hypoexponential distribution.

    Parameters
    ----------
    a : float
        Rate of one of the underlying exponential distributions.
    b : float
        Rate of the other of the underlying exponential distributions.
    x : float, 1-D array_like
        Sample.

    Returns
    -------
    pdf : float, np.ndarray
        Probability densities.
    """
    x = np.asarray(x)
    pdf = a*b / (a-b) * (np.exp(-b * x) - np.exp(-a * x))

    return pdf


def hypoexponential_distribution_two_parameters_cdf(a, b, x):
    """
    Cumulative distribution function of two-parameter hypoexponential distribution.

    Parameters
    ----------
    a : float
        Rate of one of the underlying exponential distributions.
    b : float
        Rate of the other of the underlying exponential distributions.
    x : float, 1-D array_like
        Sample.

    Returns
    -------
    cdf : float, np.ndarray
        Probabilities of samples less than or equal to x.
    """
    x = np.asarray(x)
    cdf = 1 - b / (b - a) * np.exp(-a * x) + a / (b - a) * np.exp(-b * x)
    # note that this is the cdf and not the pdf, so the /(r2 - r1) makes sense

    return cdf


def hypoexponential_distribution_three_parameters_pdf(a, b, c, x):
    """
    Probability density function of three-parameter hypoexponential distribution.

    Parameters
    ----------
    a : float
        Rate of the first of the underlying exponential distributions.
    b : float
        Rate of the second of the underlying exponential distributions.
    c : float
        Rate of the third of the underlying exponential distributions.
    x : float, 1-D array_like
        Sample.

    Returns
    -------
    pdf : float, np.ndarray
        Probability densities.
    """
    x = np.asarray(x)
    z = a * b * c
    pdf = np.exp(-c * x) * z / ((a - c) * (b - c)) + np.exp(-a * x) * z / ((-a + b) * (-a + c)) + \
        np.exp(-b * x) * z / ((a - b) * (-b + c))

    return pdf


def hypoexponential_distribution_three_parameters_cdf(a, b, c, x):
    """
    Cumulative distribution function of three-parameter hypoexponential distribution.

    Parameters
    ----------
    a : float
        Rate of the first of the underlying exponential distributions.
    b : float
        Rate of the second of the underlying exponential distributions.
    c : float
        Rate of the third of the underlying exponential distributions.
    x : float, 1-D array_like
        Sample.

    Returns
    -------
    cdf : float, np.ndarray
        Probabilities of samples less than or equal to x.
    """
    x = np.asarray(x)
    cdf = 1 - np.exp(-c * x) * a * b / ((a - c) * (b - c)) - np.exp(-a * x) * b * c / ((-a + b) * (-a + c)) - \
        np.exp(-b * x) * a * c / ((a - b) * (-b + c))

    return cdf


# The inverse function of the cdf is not computable (not analytically, at least). Hence, the strategy will be to not
# use inverse transform sampling, but rejection sampling


def rejection_sampling(pdf, x_min, x_max, y_min, y_max, batch, size, parameters, seed):
    """
    Technique to sample from a distribution with a known PDF.
    Adapted from https://cosmiccoding.com.au/tutorials/rejection_sampling/.

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
