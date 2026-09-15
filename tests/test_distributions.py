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
def test_hypoexponential_distribution_is_zero_below_support(call):
    assert call(-1, 1) == 0


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

    def test_PFM_pdf_derivatives(self):
        x = np.array([0.5, 1, 2])

        dpdf = self.pfm.dpdf(x)
        ddpdf = self.pfm.ddpdf(x)

        np.testing.assert_allclose(
            dpdf,
            [-0.09251863, -0.07558411, -0.05525348],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            ddpdf,
            [0.04209551, 0.02729518, 0.01577208],
            rtol=1e-6,
        )

    def test_PFM_respects_domain_and_preserves_underlying_parts(self):
        model = Photoswitching_fingerprint_model(
            params={0: [1, 0, 1, 0.7]},
            domain=(1, 2),
        )
        x = np.array([0, 1, 1.5, 2, 3])

        np.testing.assert_array_equal(model.pdf(x)[[0, 4]], [0, 0])
        np.testing.assert_array_equal(model.cdf(x)[[0, 1, 3, 4]], [0, 0, 1, 1])
        assert model.pdf_part(None, x=3, i=0, normalize=False) > 0
        assert model.pdf_part(None, x=3, i=0, normalize=True) == 0
        assert model.cdf_part(x=3, i=0, normalize=False) < 1
        assert model.cdf_part(x=3, i=0, normalize=True) == 1


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

    def test_domain_and_underlying_cdf(self):
        model = ExponentialMixtureModel(
            params={"lambdas": [1, 2], "pis": [0.5]},
            domain=(1, 2),
        )
        x = np.array([0, 1, 1.5, 2, 3])

        np.testing.assert_array_equal(model.pdf(x)[[0, 4]], [0, 0])
        np.testing.assert_array_equal(model.cdf(x)[[0, 1, 3, 4]], [0, 0, 1, 1])
        assert 0 < model.cdf(3, extra=True) < 1


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

    def test_grid_scales_with_small_truncation(self):
        model = ExponentialMixtureMarginalModel(
            params={"lambdas": [1], "pis": []},
            pfa_cdf_part=lambda x, i, normalize: 0.5,
            cdf_part_index=0,
            truncation_up=0.001,
        )

        assert model.x_grid[0] == 0
        assert model.x_grid[-1] == 0.001
        assert np.all(np.diff(model.x_grid) > 0)

    def test_sharp_density_observation_probability(self):
        model = ExponentialMixtureMarginalModel(
            params={"lambdas": [10000], "pis": []},
            pfa_cdf_part=lambda x, i, normalize: 1.0,
            cdf_part_index=0,
            truncation_up=1,
        )

        assert model.P_obs == pytest.approx(1, rel=2e-3)
        assert np.all(np.diff(model.cdf_grid) >= 0)
        assert model.cdf_grid[-1] == 1


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
