"""
Module distributions
"""

import numpy as np
from scipy.special import i1
from scipy.stats import rv_discrete
from scipy.stats import expon
from scipy.optimize import minimize
from itertools import product


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
        cdf -= np.exp(-arg * x) * np.prod(other_args) / np.prod(- arg + other_args)

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
        pdf += np.exp(-arg * x) * np.prod(all_args) / np.prod(- arg + other_args)

    return pdf


def photoswitching_fingerprint_prepare(n, pis):
    """
    Get indices for valid lambda combinations for the photoswitching fingerprint model.
    Modifies the weights of the underlying exponential distributions by converting
    probabilities to 1, if their counterpart is not valid.

    Parameters
    ----------
    valid_combinations : 2-D array_like
        Valid combinations of lambdas.
    pis : 2-D array_like
        Modified weights of the underlying exponential distributions.
    """
    combinations = product([0, 1], repeat=n)
    valid_combinations = [
    comb for comb in combinations if all(
            not (comb[h] == 0 and comb[h-1] == 1) for h in range(1, n)
        )
    ]
    valid_combinations = np.array(valid_combinations, dtype=int)
    pis = pis[valid_combinations, np.arange(valid_combinations.shape[1])]
    for n, comb in enumerate(valid_combinations):
        ones = np.where(comb == 1)[0]
        if ones.size > 1:
            pis[n, ones[1:]] = 1

    return valid_combinations, pis


def photoswitching_fingerprint_cdf(x, lambdas, pis_orig):
    """
    Model to describe photoswitching fingerprints produced by n fluorophores where there
    is bias in time (e.g., increased probability of ON in the beginning). The model is 
    adjusted to account for truncations.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lambdas : float, 2-D array_like
        Rates of the underlying exponential distributions.
    pis : float, 1-D array_like
        Weights of the underlying exponential distributions.
    """
    lambdas = np.asarray(lambdas)
    pis_orig = np.asarray(pis_orig)
    pis_orig = np.array([pis_orig, 1-pis_orig])
    n = lambdas.shape[1]
    cdf = 0
    for i in range(n):
        valid_combinations, pis = photoswitching_fingerprint_prepare(i+1, pis_orig)
        cdf_part = 0
        for j in range(i+2):
            pi_set = np.prod(pis[j])
            cdf_part += pi_set * hypoexponential_distribution_cdf(
                x, 
                *lambdas[
                    valid_combinations[j], np.arange(valid_combinations[j].shape[0])
                ]
            )
        cdf += 1/n * cdf_part
    
    # truncation
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    return cdf


def fit_1_fluorophore(x, lam1_1, lam1_2, pi1):
    cdf = photoswitching_fingerprint_cdf(
        x, lambdas=[[lam1_1], [lam1_2]], pis_orig=[pi1]
    )
    return cdf


def fit_2_fluorophore(x, lam1_1, lam2_1, lam1_2, lam2_2, pi1, pi2):
    cdf = photoswitching_fingerprint_cdf(
        x, lambdas=[[lam1_1, lam2_1], [lam1_2, lam2_2]], pis_orig=[pi1, pi2]
    )
    return cdf


def fit_3_fluorophore(x, lam1_1, lam2_1, lam3_1, lam1_2, lam2_2, lam3_2, pi1, pi2, pi3):
    cdf = photoswitching_fingerprint_cdf(
        x,
        lambdas=[[lam1_1, lam2_1, lam3_1], [lam1_2, lam2_2, lam3_2]],
        pis_orig=[pi1, pi2, pi3],
    )
    return cdf


def fit_4_fluorophore(x, lam1_1, lam2_1, lam3_1, lam4_1, lam1_2, lam2_2, lam3_2, lam4_2, 
                      pi1, pi2, pi3, pi4):
    cdf = photoswitching_fingerprint_cdf(
        x, 
        lambdas=[[lam1_1, lam2_1, lam3_1, lam4_1], [lam1_2, lam2_2, lam3_2, lam4_2]], 
        pis_orig=[pi1, pi2, pi3, pi4])
    return cdf


def photoswitching_fingerprint_pdf(x, lambdas, pis):
    """
    PDF of the photoswitching fingerprint model.

    Parameters
    ----------
    x : float, 1-D array_like
        Sample.
    lambdas : float, 2-D array_like
        Rates of the underlying exponential distributions.
    pis : float, 1-D array_like
        Weights of the underlying exponential distributions.
    """
    lambdas = np.asarray(lambdas)
    pis = np.asarray(pis)
    n = lambdas.shape[1]
    pdf = 0
    for i in range(n):
        valid_combinations, pis = photoswitching_fingerprint_prepare(i+1, pis)
        pdf_part = 0
        for j in range(i+2):
            pi_set = np.prod(pis[j])
            pdf_part += pi_set * hypoexponential_distribution_pdf(
                x, 
                *lambdas[
                    valid_combinations[j], np.arange(valid_combinations[j].shape[0])
                ]
            )
        pdf += 1/n * pdf_part
    # truncation
    pdf = pdf / np.trapz(pdf, x)  # could also be normalized using cdf[-1] - cdf[0]

    return pdf


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
    model : float, 1-D array_like
        Model output.
    """
    total_cdf = (p * expon.cdf(x, scale=1/lambda1) + 
                (1-p) * expon.cdf(x, scale=1/lambda2))
    model = (total_cdf - total_cdf[0]) / (total_cdf[-1] - total_cdf[0])

    return model


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
    model : float, 1-D array_like
        Model output.
    """
    total_pdf = (p * expon.pdf(x, scale=1/lambda1) + 
                (1-p) * expon.pdf(x, scale=1/lambda2))
    total_cdf = (p * expon.cdf(x, scale=1/lambda1) + 
            (1-p) * expon.cdf(x, scale=1/lambda2))
    model = total_pdf / (total_cdf[-1] - total_cdf[0])

    return model


def mixture_log_likelihood(params, data, truncation_low, truncation_up, 
                           number_no_events=0):
    """
    Negative log-likelihood of a mixture of two exponential distributions (pdf).

    Parameters
    ----------
    params : list
        Parameters of the mixture distribution.
    data : np.ndarray
        Sample.
    truncation_low : float
        Lower truncation of the distribution.
    truncation_up : float
        Upper truncation of the distribution.
    number_no_events : int
        Number of runs without events.

    Returns
    -------
    negative_log_likelihood : float
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
        prob_event = np.minimum(prob_event, 1-1e-14)
        log_likelihood_no_observation = np.log1p(- prob_event) * number_no_events

    log_likelihood = (np.sum(log_likelihood_observation) +
                    log_likelihood_no_observation)
    negative_log_likelihood = - log_likelihood

    return negative_log_likelihood


def estimate_mixture_parameters(
    data, 
    initial_guess, 
    bounds, 
    truncation_low=0, 
    truncation_up=300, 
    number_no_events=0,
    method="L-BFGS-B",
):
    """
    Estimate the parameters of a mixture of two exponential distributions (pdf).

    Parameters
    ----------
    data : np.ndarray
        Sample.
    initial_guess : list
        Initial guess for the optimization. For parameters, see return values.
    bounds : list
        Bounds for the optimization. For parameters, see return values.
    truncation_low : float
        Lower truncation of the distribution.
    truncation_up : float
        Upper truncation of the distribution.
    number_no_events : int
        Number of runs without events. Set to 0 if all runs produced events or if no
        influence of log-likelihood of no events is desired. 
    method : str
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
