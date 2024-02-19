import pytest
import src.distributions as dist
import numpy as np


@pytest.mark.parametrize('x_min, x_max, v, gain, expected',
                         [[1, 130000, 1, 1, 'ValueError'],
                          [1, 125940, 1, 20, 32.1022]])
def test_high_gain_amplification_noise_distribution(x_min, x_max, v, gain, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            dist.high_gain_amplification_noise_distribution(x_min=x_min, x_max=x_max, v=v, gain=gain)
    else:
        distribution = dist.high_gain_amplification_noise_distribution(x_min=x_min, x_max=x_max, v=v, gain=gain)
        np.testing.assert_allclose(distribution.mean(), expected, rtol=1e-4)


@pytest.mark.parametrize('a, b, x, expected',
                         [[1, 2, 1, 0.46509],
                          [1, 2, [1, 2, 3], np.array([0.46509, 0.23404, 0.09462])]])
def test_hypoexponential_distribution_two_parameters_pdf(a, b, x, expected):
    pdf = dist.hypoexponential_distribution_two_parameters_pdf(a=a, b=b, x=x)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize('a, b, x, expected',
                         [[1, 2, 1, 0.3996],
                          [1, 2, [1, 2, 3], np.array([0.3996, 0.7476, 0.9029])]])
def test_hypoexponential_distribution_two_parameters_cdf(a, b, x, expected):
    cdf = dist.hypoexponential_distribution_two_parameters_cdf(a=a, b=b, x=x)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


@pytest.mark.parametrize('a, b, c, x, expected',
                         [[1, 2, 3, 1, 0.44099],
                          [1, 2, 3, [1, 2, 3], np.array([0.440988, 0.303548, 0.134859])]])
def test_hypoexponential_distribution_three_parameters_pdf(a, b, c, x, expected):
    pdf = dist.hypoexponential_distribution_three_parameters_pdf(a=a, b=b, c=c, x=x)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize('a, b, c, x, expected',
                         [[1, 2, 3, 1, 0.25258],
                          [1, 2, 3, [1, 2, 3], np.array([0.25258, 0.64646, 0.85795])]])
def test_hypoexponential_distribution_three_parameters_cdf(a, b, c, x, expected):
    cdf = dist.hypoexponential_distribution_three_parameters_cdf(a=a, b=b, c=c, x=x)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


def test_rejection_sampling():
    samples = dist.rejection_sampling(pdf=dist.hypoexponential_distribution_two_parameters_pdf,
                                      x_min=1, x_max=10, y_min=0, y_max=1, batch=2, size=10, parameters=[1, 2],
                                      seed=1)
    expected = np.array([1.5644, 1.0524, 3.1889, 1.6937, 4.2642, 2.4269, 1.4138, 2.0629, 1.1785, 2.2404])
    np.testing.assert_allclose(samples, expected, rtol=1e-4)
