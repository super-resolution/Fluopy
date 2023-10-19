import pytest
import src.distributions as dist
import numpy as np


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 130000, 1, 1], 'ValueError'],
                          [[1, 125940, 1, 20], 32.1022]])
def test_high_gain_amplification_noise_distribution(parameters, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            dist.high_gain_amplification_noise_distribution(*parameters)
    else:
        distribution = dist.high_gain_amplification_noise_distribution(*parameters)
        np.testing.assert_allclose(distribution.mean(), expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 2, 1], 0.46509],
                          [[1, 2, [1, 2, 3]], np.array([0.46509, 0.23404, 0.09462])]])
def test_hypoexponential_distribution_two_parameters_pdf(parameters, expected):
    pdf = dist.hypoexponential_distribution_two_parameters_pdf(*parameters)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 2, 1], 0.3996],
                          [[1, 2, [1, 2, 3]], np.array([0.3996, 0.7476, 0.9029])]])
def test_hypoexponential_distribution_two_parameters_cdf(parameters, expected):
    cdf = dist.hypoexponential_distribution_two_parameters_cdf(*parameters)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 2, 3, 1], 0.44099],
                          [[1, 2, 3, [1, 2, 3]], np.array([0.440988, 0.303548, 0.134859])]])
def test_hypoexponential_distribution_three_parameters_pdf(parameters, expected):
    pdf = dist.hypoexponential_distribution_three_parameters_pdf(*parameters)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 2, 3, 1], 0.25258],
                          [[1, 2, 3, [1, 2, 3]], np.array([0.25258, 0.64646, 0.85795])]])
def test_hypoexponential_distribution_three_parameters_cdf(parameters, expected):
    cdf = dist.hypoexponential_distribution_three_parameters_cdf(*parameters)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[dist.hypoexponential_distribution_two_parameters_pdf,
                           1, 10, 0, 1, 2, 10, [1, 2], 1], np.array([1.5644, 1.0524, 3.1889, 1.6937, 4.2642, 2.4269,
                                                                     1.4138, 2.0629, 1.1785, 2.2404])]])
def test_rejection_sampling(parameters, expected):
    samples = dist.rejection_sampling(*parameters)
    np.testing.assert_allclose(samples, expected, rtol=1e-4)



