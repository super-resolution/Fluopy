import numpy as np

from mpmath import nsum, inf, factorial
from scipy.special import gamma, i1
from scipy.stats import rv_discrete


def high_gain_amplification_noise_distribution(x_min=1, x_max=100, v=1, gain=100):
    """
    The high gain amplification noise distribution as proposed in https://doi.org/10.1117/12.2004621 with the
    adjustment of not considering 0 as a possible variable's value. The support is limited to a maximum x of
    127499 * gain / v.
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
    x = np.arange(x_min, x_max)

    x = x.astype(float)

    probabilities = 1 / x * np.exp(-(x/gain + v)) * np.sqrt(v * x/gain) * i1(2*np.sqrt(v * x/gain))
    probabilities = probabilities / np.sum(probabilities)
    distribution = rv_discrete(name='high_gain_distr', values=(x, probabilities))

    return distribution


def calculate_lambda(r1, r2):
    lam = 1 / (1/r1 + 1/r2)

    return lam


def hypoexponential_distribution_two_parameters_pdf(a, b, x):

    pdf = a*b / (a-b) * (np.exp(-b * x) - np.exp(-a * x))

    return pdf


def hypoexponential_distribution_two_parameters_cdf(a, b, x):

    cdf = 1 - b / (b - a) * np.exp(-a * x) + a / (b - a) * np.exp(-b * x)
    # note that this is the cdf and not the pdf, so the /(r2 - r1) makes sense

    return cdf


def hypoexponential_distribution_three_parameters_pdf(a, b, c, x):

    z = a * b * c
    pdf = np.exp(-c * x) * z /((a-c) * (b-c)) + np.exp(-a * x) * z /((-a+b) * (-a+c)) + np.exp(-b * x) * z /((a-b) * (-b+c))

    return pdf


def hypoexponential_distribution_three_parameters_cdf(a, b, c, x):

    cdf = 1 - np.exp(-c * x) * a * b /((a-c) * (b-c)) - np.exp(-a * x) * b * c /((-a+b) * (-a+c)) - np.exp(-b * x) * a * c /((a-b) * (-b+c))

    return cdf


# The inverse function of the cdf is not computable (not analytically, at least). Hence, the strategy will be to not
# use inverse transform sampling, but rejection sampling


def rejection_sampling(pdf, x_min, x_max, y_min, y_max, batch, size, parameters):

    samples = []
    while len(samples) < size:
        x = np.random.uniform(x_min, x_max, size=batch)
        y = np.random.uniform(y_min, y_max, size=batch)
        samples += x[y < pdf(*parameters, x)].tolist()

    return samples[:size]


def phi(b, c, w, z):
    f = b**2/c
    result = float(nsum(lambda k, l: f * w**k * z**l / (factorial(l)*factorial(k)), [0, inf], [0, inf]))

    return result


def c_x(t, a, b, x):
    part1 = (a * b)**x
    part2 = t**(2*x)/gamma(1+2*x)
    part3 = phi(x, 1+2*x, -t*a, -t*b)
    func = part1 * part2 * part3

    return func


def counts_hypo_interarrival_times_pmf(t, a, b, x):
    pmf = c_x(t, a, b, x) - c_x(t, a, b, x+1)

    return pmf
