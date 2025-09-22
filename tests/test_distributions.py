import numpy as np
import pytest

from fluopy import distributions as dist


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
    lambdas, pis = dist.photoswitching_fingerprint_prepare(
        params={0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]},
        n=3,
        z=-1,
    )
    np.testing.assert_array_equal(
        lambdas, [[1, 0.7, 0.5], [1, 0.7, 0.3], [1, 0.5, 0.3], [0.7, 0.5, 0.3]]
    )
    np.testing.assert_array_equal(
        pis, [[1, 1, 0.5], [1, 0.7, 0.5], [1, 0.3, 0.5], [0, 1, 0.5]]
    )


def test_PFM_cdf():
    cdf = dist.Photoswitching_fingerprint_model(
        params={0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]},
    ).cdf(x=2)
    expected = 0.487748
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


def test_PFM_pdf():
    pdf = dist.Photoswitching_fingerprint_model(
        params={0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]},
    ).pdf(x=2)
    expected = 0.174595
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


def test_two_expon_mixture_cdf():
    cdf = dist.two_expon_mixture_cdf(x=2, lambda1=1, lambda2=0.7, p=0.3)
    expected = 0.7868
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


def test_two_expon_mixture_pdf():
    pdf = dist.two_expon_mixture_pdf(x=2, lambda1=1, lambda2=0.7, p=0.3)
    expected = 0.16143
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)
