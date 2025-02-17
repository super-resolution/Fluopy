import pytest
import numpy as np
import src.distributions as dist


@pytest.mark.parametrize(
    "x_min, x_max, v, gain, expected",
    [[1, 130000, 1, 1, "ValueError"], [1, 125940, 1, 20, 32.1022]],
)
def test_high_gain_amplification_noise_distribution(x_min, x_max, v, gain, expected):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            dist.high_gain_amplification_noise_distribution(
                x_min=x_min, x_max=x_max, v=v, gain=gain
            )
    else:
        distribution = dist.high_gain_amplification_noise_distribution(
            x_min=x_min, x_max=x_max, v=v, gain=gain
        )
        np.testing.assert_allclose(distribution.mean(), expected, rtol=1e-4)


@pytest.mark.parametrize(
    "x, args, expected",
    [[1, [1], 0.6321], [1, [1, 0.9], 0.2452], [1, [1, 0.99, 1.01], 0.08029]],
)
def test_hypoexponential_distribution_cdf(x, args, expected):
    cdf = dist.hypoexponential_distribution_cdf(x, *args)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "x, args, expected",
    [[1, [1], 0.36788], [1, [1, 0.9], 0.34821], [1, [1, 0.99, 1.01], 0.18392]],
)
def test_hypoexponential_distribution_pdf(x, args, expected):
    pdf = dist.hypoexponential_distribution_pdf(x, *args)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


def test_photoswitching_fingerprint_prepare():
    valid_combinations, pis = dist.photoswitching_fingerprint_prepare(
        n=3, pis=np.array([[1, 0.7, 0.5], [0, 0.3, 0.5]])
    )
    expected_combinations = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    expected_pis = np.array([[1, 0.7, 0.5], [1, 0.7, 0.5], [1, 0.3, 1], [0, 1, 1]])
    np.testing.assert_array_equal(valid_combinations, expected_combinations)
    np.testing.assert_array_equal(pis, expected_pis)


def test_PFM_cdf():
    cdf = dist.Photoswitching_fingerprint_model(
        lambdas=[[1, 0.7, 0.5], [0.7, 0.5, 0.3]], pis_orig=[1, 0.7, 0.5]
    ).cdf(x=2)
    expected = 0.4838
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


def test_PFM_pdf():
    pdf = dist.Photoswitching_fingerprint_model(
        lambdas=[[1, 0.7, 0.5], [0.7, 0.5, 0.3]], pis_orig=[1, 0.7, 0.5]
    ).pdf(x=2)
    expected = 0.17106
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


def test_two_expon_mixture_cdf():
    cdf = dist.two_expon_mixture_cdf(x=2, lambda1=1, lambda2=0.7, p=0.3)
    expected = 0.7868
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


def test_two_expon_mixture_pdf():
    pdf = dist.two_expon_mixture_pdf(x=2, lambda1=1, lambda2=0.7, p=0.3)
    expected = 0.16143
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)
