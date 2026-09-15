from collections.abc import Callable

import numpy as np
import pytest

from fluopy import (
    ExponentialMixtureMarginalModel,
    ExponentialMixtureModel,
    IllConditionedHypoexponentialError,
    Photoswitching_fingerprint_model,
    hypoexponential_distribution_cdf,
    hypoexponential_distribution_pdf,
    hypoexponential_distribution_pdf_1st_order_derivative,
    hypoexponential_distribution_pdf_2nd_order_derivative,
)
from fluopy.distributions import (
    generate_combinations,
    get_pis,
    map_to_lambdas,
    photoswitching_fingerprint_prepare,
)


@pytest.mark.parametrize(
    "x, args, expected",
    [
        [1, [1], 0.6321],
        [[1, 2], [1], [0.6321, 0.864665]],
        [1, [1, 0.9], 0.2452],
        [1, [1, 0.99, 1.01], 0.08029],
    ],
)
def test_hypoexponential_distribution_cdf(x, args, expected):
    cdf = hypoexponential_distribution_cdf(x, *args)
    np.testing.assert_allclose(cdf, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "x, args, expected",
    [
        [1, [1], 0.36788],
        [[1, 2], [1], [0.36788, 0.135335]],
        [1, [1, 0.9], 0.34821],
        [1, [1, 0.99, 1.01], 0.18392],
    ],
)
def test_hypoexponential_distribution_pdf(x, args, expected):
    pdf = hypoexponential_distribution_pdf(x, *args)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "x, args, expected",
    [
        [1, [1], -0.367879],
        [[1, 2], [1], [-0.367879, -0.135335]],
        [1, [1, 0.9], 0.017701],
        [1, [1, 0.99, 1.01], 0.18392],
    ],
)
def test_hypoexponential_distribution_pdf_1st_order_derivative(x, args, expected):
    pdf = hypoexponential_distribution_pdf_1st_order_derivative(x, *args)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "x, args, expected",
    [
        [1, [1], 0.367879],
        [[1, 2], [1], [0.367879, 0.135335]],
        [1, [1, 0.9], -0.347022],
        [1, [1, 0.99, 1.01], -0.183914],
    ],
)
def test_hypoexponential_distribution_pdf_2nd_order_derivative(x, args, expected):
    pdf = hypoexponential_distribution_pdf_2nd_order_derivative(x, *args)
    np.testing.assert_allclose(pdf, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "call",
    [
        hypoexponential_distribution_cdf,
        hypoexponential_distribution_pdf,
        hypoexponential_distribution_pdf_1st_order_derivative,
        hypoexponential_distribution_pdf_2nd_order_derivative,
    ],
)
def test_hypoexponential_distribution_rejects_ill_conditioned_rates(call):
    rates = np.array([0.9497, 0.9499, 0.9501, 0.9503])
    np.testing.assert_array_equal(np.round(rates, decimals=2), [0.95] * 4)

    with pytest.raises(IllConditionedHypoexponentialError, match="Rates are too close"):
        call(1, *rates)


@pytest.mark.parametrize(
    "call, expected",
    [
        (hypoexponential_distribution_cdf, 0.01607397677271227),
        (hypoexponential_distribution_pdf, 0.05249902090014569),
        (
            hypoexponential_distribution_pdf_1st_order_derivative,
            0.1076231323351729,
        ),
        (
            hypoexponential_distribution_pdf_2nd_order_derivative,
            0.06313078403110652,
        ),
    ],
)
def test_hypoexponential_distribution_accepts_close_stable_rates(call, expected):
    rates = np.array([0.9451, 0.9484, 0.9516, 0.9549])
    np.testing.assert_array_equal(np.round(rates, decimals=2), [0.95] * 4)

    result = call(1, *rates)

    assert result == pytest.approx(expected, rel=1e-7)


def test_fingerprint_skips_zero_weight_ill_conditioned_combinations():
    model = Photoswitching_fingerprint_model(
        params={
            0: [1, 0, 3, 0.9497],
            1: [1, 0, 2.5, 0.9499],
            2: [1, 0, 2, 0.9501],
            3: [0.5, 0.5, 1.5, 0.9503],
        }
    )
    expected = 0.5 * hypoexponential_distribution_pdf(1, 3, 2.5, 2, 1.5)
    expected += 0.5 * hypoexponential_distribution_pdf(1, 3, 2.5, 2, 0.9503)

    result = model.pdf_part(call=None, x=1, i=3)

    assert result == pytest.approx(expected)


class TestPhotoswitchingFingerprintModel:

    pfm = Photoswitching_fingerprint_model(
        params={0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]},
    )

    def test_init(self):
        assert self.pfm.params == {
            0: [1, 0, 1, 0.7],
            1: [0.7, 0.3, 0.7, 0.5],
            2: [0.5, 0.5, 0.5, 0.3],
        }

    def test_PFM_cdf(self):
        cdf = self.pfm.cdf(x=2)
        expected = 0.487748
        np.testing.assert_allclose(cdf, expected, rtol=1e-4)

    def test_PFM_pdf(self):
        pdf = self.pfm.pdf(x=2)
        expected = 0.174595
        np.testing.assert_allclose(pdf, expected, rtol=1e-4)


class TestExponentialMixtureModel:
    model = ExponentialMixtureModel(
        params={"lambdas": [1, 10], "pis": [0.2]},
    )

    def test_init(self):
        assert self.model.params == {"lambdas": [1, 10], "pis": [0.2]}
        assert self.model.domain == (0, np.inf)

    def test_PFM_cdf(self):
        cdf = self.model.cdf(x=2)
        expected = 0.972933
        np.testing.assert_allclose(cdf, expected, rtol=1e-4)

    def test_PFM_pdf(self):
        pdf = self.model.pdf(x=2)
        expected = 0.027067
        np.testing.assert_allclose(pdf, expected, rtol=1e-4)


class TestExponentialMixtureMarginalModel:
    model = ExponentialMixtureMarginalModel(
        params={"lambdas": [1, 3, 9], "pis": [0.2, 0.5]},
        pfa_cdf_part=lambda x, i, normalize: 0.5,
        cdf_part_index=1,
        truncation_up=0.5,
    )

    def test_init(self):
        assert self.model.params == {"lambdas": [1, 3, 9], "pis": [0.2, 0.5]}
        assert isinstance(self.model.pfa_cdf_part, Callable)
        assert self.model.cdf_part_index == 1
        assert self.model.truncation_up == 0.5

    def test_PFM_cdf(self):
        cdf = self.model.cdf(x=(0.1, 2))
        expected = [0.427669, 1]
        np.testing.assert_allclose(cdf, expected, rtol=1e-4)

    def test_PFM_pdf(self):
        pdf = self.model.pdf(x=(0.1, 2))
        expected = [3.128881, 0]
        np.testing.assert_allclose(pdf, expected, rtol=1e-4)


def test_generate_combinations():
    valid_combinations = generate_combinations(n=1, z=-1)
    np.testing.assert_array_equal(valid_combinations, [[0], [1]])

    valid_combinations = generate_combinations(n=3, z=-1)
    np.testing.assert_array_equal(
        valid_combinations, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]
    )

    valid_combinations = generate_combinations(n=3, z=0)
    np.testing.assert_array_equal(
        valid_combinations, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [2, 1, 1], [3, 1, 1]]
    )


def test_generate_combinations_with_middle_three_component():
    valid_combinations = generate_combinations(n=3, z=1)

    np.testing.assert_array_equal(
        valid_combinations,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 2, 1],
            [0, 3, 1],
            [1, 2, 1],
            [1, 3, 1],
        ],
    )


def test_map_to_lambdas():
    valid_combinations = generate_combinations(n=3, z=-1)
    params = {0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]}
    lambdas = map_to_lambdas(combos=valid_combinations, params=params, z=-1)
    np.testing.assert_array_equal(
        lambdas, [[1, 0.7, 0.5], [1, 0.7, 0.3], [1, 0.5, 0.3], [0.7, 0.5, 0.3]]
    )


def test_get_pis():
    valid_combinations = generate_combinations(n=3, z=-1)
    params = {0: [1, 0, 1, 0.7], 1: [0.7, 0.3, 0.7, 0.5], 2: [0.5, 0.5, 0.5, 0.3]}
    pis = get_pis(combos=valid_combinations, params=params, z=-1)
    np.testing.assert_array_equal(
        pis, [[1, 1, 0.5], [1, 0.7, 0.5], [1, 0.3, 0.5], [0, 1, 0.5]]
    )


def test_get_pis_three_component_stage_fully_biased():
    params = {
        0: [1, 0, 3, 1.5],
        1: [1, 0, 0, 2, 1, 0.5],
        2: [0.5, 0.5, 0.4, 0.2],
    }
    combos = generate_combinations(n=3, z=1)

    with np.errstate(divide="raise", invalid="raise"):
        pis = get_pis(combos=combos, params=params, z=1)

    weights = np.prod(pis, axis=1)
    np.testing.assert_array_equal(weights, [0.5, 0.5, 0, 0, 0, 0])


def test_photoswitching_fingerprint_prepare():
    lambdas, pis = photoswitching_fingerprint_prepare(
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
