"""
Random variable distributions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_simpson, simpson
from scipy.stats import expon

__all__: list[str] = [
    "IllConditionedHypoexponentialError",
    "hypoexponential_distribution_cdf",
    "hypoexponential_distribution_pdf",
    "hypoexponential_distribution_pdf_1st_order_derivative",
    "hypoexponential_distribution_pdf_2nd_order_derivative",
    "Photoswitching_fingerprint_model",
    "ExponentialMixtureModel",
    "ExponentialMixtureMarginalModel",
]


DistributionValue = float | npt.NDArray[np.float64]


def _restrict_pdf_to_domain(
    x: npt.ArrayLike,
    pdf: DistributionValue,
    domain: tuple[float, float],
) -> DistributionValue:
    values = np.asarray(x)
    inside = (values >= domain[0]) & (values <= domain[1])
    result = np.where(inside, pdf, 0.0)

    return float(result) if result.ndim == 0 else result


def _restrict_cdf_to_domain(
    x: npt.ArrayLike,
    cdf: DistributionValue,
    domain: tuple[float, float],
) -> DistributionValue:
    values = np.asarray(x)
    result = np.where(values <= domain[0], 0.0, cdf)
    result = np.where(values >= domain[1], 1.0, result)

    return float(result) if result.ndim == 0 else result


def _marginal_integration_grid(
    truncation_up: float,
    points_per_side: int = 100,
) -> npt.NDArray[np.float64]:
    if not np.isfinite(truncation_up) or truncation_up <= 0:
        raise ValueError("truncation_up must be finite and greater than zero.")

    smallest_fraction = np.sqrt(np.finfo(np.float64).eps)
    linear_fraction_start = 1 / points_per_side
    geometric_point_count = points_per_side // 2
    geometric_fractions = np.geomspace(
        smallest_fraction,
        linear_fraction_start,
        geometric_point_count,
        endpoint=False,
    )
    linear_fractions = np.linspace(
        linear_fraction_start,
        0.5,
        points_per_side - geometric_point_count,
    )
    edge_fractions = np.concatenate((geometric_fractions, linear_fractions))

    return np.unique(
        np.concatenate(
            (
                [0.0],
                truncation_up * edge_fractions,
                truncation_up * (1 - edge_fractions),
                [truncation_up],
            )
        )
    )


class IllConditionedHypoexponentialError(ValueError):
    """Raised when hypoexponential partial fractions cannot be evaluated reliably."""


class HypoexponentialCall(Protocol):
    def __call__(self, x: npt.ArrayLike, *args: int | float) -> DistributionValue: ...


def _partial_fraction_is_ill_conditioned(
    coefficients: npt.NDArray[np.float64], scale: float
) -> bool:
    eps = np.finfo(np.float64).eps
    estimated_roundoff = eps * np.sum(np.abs(coefficients))

    return bool(
        not np.isfinite(estimated_roundoff) or estimated_roundoff > np.sqrt(eps) * scale
    )


def _prepare_hypoexponential_rates(
    args: Sequence[int | float], order: int | None
) -> npt.NDArray[np.float64]:
    rates = np.asarray(args, dtype=np.float64)
    if rates.size == 0:
        raise ValueError("At least one rate is required.")
    if not np.all(np.isfinite(rates)):
        raise ValueError("Rates must be finite.")
    if np.any(rates <= 0):
        raise ValueError("Rates must be positive.")
    if np.unique(rates).size != rates.size:
        raise IllConditionedHypoexponentialError("Rates must be distinct.")

    differences = rates[None, :] - rates[:, None]
    np.fill_diagonal(differences, 1.0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        pdf_coefficients = np.prod(rates) / np.prod(differences, axis=1)

    if order is None:
        coefficients = pdf_coefficients / rates
        scale = 1.0
    else:
        coefficients = pdf_coefficients * (-rates) ** order
        scale = float(np.max(rates) ** (order + 1))

    if _partial_fraction_is_ill_conditioned(coefficients, scale):
        raise IllConditionedHypoexponentialError(
            "Rates are too close for stable evaluation of the hypoexponential "
            "distribution."
        )

    return rates


def hypoexponential_distribution_cdf(
    x: npt.ArrayLike, *args: int | float
) -> float | npt.NDArray[np.float64]:
    """
    CDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the hypoexponential distribution.
    """
    rates = _prepare_hypoexponential_rates(args, order=None)
    values = np.asarray(x)
    evaluation_values = np.maximum(values, 0.0)
    cdf = 1
    for arg in rates:
        other_args = rates[rates != arg]
        cdf -= (
            np.exp(-arg * evaluation_values)
            * np.prod(other_args)
            / np.prod(-arg + other_args)
        )

    return _restrict_cdf_to_domain(values, cdf, (0, np.inf))


def hypoexponential_distribution_pdf(
    x: npt.ArrayLike, *args: int | float
) -> float | npt.NDArray[np.float64]:
    """
    PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        PDF of the hypoexponential distribution.
    """
    rates = _prepare_hypoexponential_rates(args, order=0)
    values = np.asarray(x)
    evaluation_values = np.maximum(values, 0.0)
    pdf = 0
    for arg in rates:
        other_args = rates[rates != arg]
        pdf += (
            np.exp(-arg * evaluation_values)
            * np.prod(rates)
            / np.prod(-arg + other_args)
        )

    return _restrict_pdf_to_domain(values, pdf, (0, np.inf))


def hypoexponential_distribution_pdf_1st_order_derivative(
    x: npt.ArrayLike, *args: int | float
) -> float | npt.NDArray[np.float64]:
    """
    First order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        First order derivative of the PDF of the hypoexponential distribution.
    """
    rates = _prepare_hypoexponential_rates(args, order=1)
    values = np.asarray(x)
    evaluation_values = np.maximum(values, 0.0)
    pdf_1st_order_derivative = 0
    for arg in rates:
        other_args = rates[rates != arg]
        pdf_1st_order_derivative += (
            -arg
            * np.exp(-arg * evaluation_values)
            * np.prod(rates)
            / np.prod(-arg + other_args)
        )

    return _restrict_pdf_to_domain(values, pdf_1st_order_derivative, (0, np.inf))


def hypoexponential_distribution_pdf_2nd_order_derivative(
    x: npt.ArrayLike, *args: int | float
) -> float | npt.NDArray[np.float64]:
    """
    Second order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        Second order derivative of the PDF of the hypoexponential distribution.
    """
    rates = _prepare_hypoexponential_rates(args, order=2)
    values = np.asarray(x)
    evaluation_values = np.maximum(values, 0.0)
    pdf_2nd_order_derivative = 0
    for arg in rates:
        other_args = rates[rates != arg]
        pdf_2nd_order_derivative += (
            arg**2
            * np.exp(-arg * evaluation_values)
            * np.prod(rates)
            / np.prod(-arg + other_args)
        )

    return _restrict_pdf_to_domain(values, pdf_2nd_order_derivative, (0, np.inf))


class Photoswitching_fingerprint_model:
    """
    Model to describe photoswitching fingerprints produced by n fluorophores where there
    is bias in time (e.g., increased probability of ON in the beginning).
    """

    def __init__(
        self,
        params: Mapping[int, Sequence[int | float]],
        weights: float | npt.ArrayLike | None = None,
        domain: tuple[float, float] = (0, np.inf),
    ) -> None:
        """
        Parameters
        ----------
        params
            Parameters of the underlying exponential mixture distributions. Dict indexed
            by (0, 1, ..., n-1). The indices denote the number of photobleaching events
            that have occurred. Each entry is a list of length 4, one of the entries
            can be of length 6. The first half of the list contains the pis, the second
            half the lambdas.
        weights
            Weights of each fluorophore (1D). Defaults to equal weights.
        domain
            Domain of the model. Default is (0, np.inf). If domain is not (0, inf),
            the PDF and CDF are normalized to the domain.
        """
        self.params = params
        if weights is None:
            weights = np.ones(len(params)) / len(params)
        self.weights = np.asarray(weights)
        self.domain = domain
        self.z = next((k for k, v in params.items() if len(v) == 6), -1)

    def pdf_part(
        self,
        call: HypoexponentialCall | None,
        x: float | npt.ArrayLike,
        i: int,
        normalize: bool = False,
    ) -> float | npt.NDArray[np.float64]:
        """
        PDF for the arrival times (not delta arrival times) after the i-th fluorophore
        has photobleached.

        Parameters
        ----------
        call
            Function to calculate the PDF (or its derivative) of the hypoexponential
            distribution. If None, hypoexponential_distribution_pdf is used.
        x
            Sample.
        i
            Index of the fluorophore that has just photobleached (0 means no
            photobleaching so far).
        normalize
            Whether to normalize the PDF part to the domain. This is needed since if
            PDF parts are summarized into the full PDF, the full PDF is normalized. If
            PDF parts are used for other purposes (e.g., marginal distribution), they
            have to be normalized individually.

        Returns
        -------
        float | npt.NDArray[np.float64]
            PDF for the arrival times
        """
        if call is None:
            call = hypoexponential_distribution_pdf
        lambdas, pis = photoswitching_fingerprint_prepare(
            params=self.params,
            n=i + 1,
            z=self.z,
        )
        pdf_part: DistributionValue = 0.0
        for lambda_combo, pi_combo in zip(lambdas, pis):
            pi_set = np.prod(pi_combo)
            if pi_set == 0:
                continue
            pdf_part += pi_set * call(
                x,
                *lambda_combo,
            )

        if not normalize:
            return pdf_part

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf_part(x=self.domain[-1], i=i, normalize=False)
            F_0 = self.cdf_part(x=self.domain[0], i=i, normalize=False)
            pdf_part = pdf_part / (F_1 - F_0)

        return _restrict_pdf_to_domain(x, pdf_part, self.domain)

    def _evaluate_pdf(
        self,
        x: float | npt.ArrayLike,
        call: HypoexponentialCall,
    ) -> float | npt.NDArray[np.float64]:
        """
        Evaluate the PDF (or its derivative) of the photoswitching fingerprint model.

        Parameters
        ----------
        x
            Sample.
        call
            Function to calculate the PDF (or its derivative) of the hypoexponential
            distribution.

        Returns
        -------
        pdf
        """
        n = len(self.params)
        pdf: DistributionValue = 0.0
        for i in range(n):
            pdf_part = self.pdf_part(call=call, x=x, i=i, normalize=False)
            pdf += self.weights[i] * pdf_part

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            pdf = pdf / (F_1 - F_0)

        return _restrict_pdf_to_domain(x, pdf, self.domain)

    def pdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        PDF

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            PDF
        """
        return self._evaluate_pdf(x=x, call=hypoexponential_distribution_pdf)

    def cdf_part(
        self,
        x: float | npt.ArrayLike,
        i: int,
        normalize: bool = False,
    ) -> float | npt.NDArray[np.float64]:
        """
        CDF for the arrival times (not delta arrival times) after the i-th fluorophore
        has photobleached.

        Parameters
        ----------
        x
            Sample.
        i
            Index of the fluorophore that has just photobleached (0 means no
            photobleaching so far).
        normalize
            Whether to normalize the CDF part to the domain. This is needed since if
            CDF parts are summarized into the full CDF, the full CDF is normalized. If
            CDF parts are used for other purposes (e.g., marginal distribution), they
            have to be normalized individually.

        Returns
        -------
        float | npt.NDArray[np.float64]
            CDF for the arrival times
        """
        lambdas, pis = photoswitching_fingerprint_prepare(
            params=self.params,
            n=i + 1,
            z=self.z,
        )
        cdf_part: DistributionValue = 0.0
        for lambda_combo, pi_combo in zip(lambdas, pis):
            pi_set = np.prod(pi_combo)
            if pi_set == 0:
                continue
            cdf_part += pi_set * hypoexponential_distribution_cdf(
                x,
                *lambda_combo,
            )
        if not normalize:
            return cdf_part

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf_part(x=self.domain[-1], i=i, normalize=False)
            F_0 = self.cdf_part(x=self.domain[0], i=i, normalize=False)
            cdf_part = (cdf_part - F_0) / (F_1 - F_0)

        return _restrict_cdf_to_domain(x, cdf_part, self.domain)

    def cdf(
        self,
        x: float | npt.ArrayLike,
        extra: bool = False,
    ) -> float | npt.NDArray[np.float64]:
        """
        CDF

        Parameters
        ----------
        x
            Sample.
        extra
            If True, the CDF is not normalized to the domain. Needed for normalization
            of PDF and CDF.

        Returns
        -------
        float | npt.NDArray[np.float64]
            CDF
        """
        n = len(self.params)
        cdf: DistributionValue = 0.0
        for i in range(n):
            cdf_part = self.cdf_part(x=x, i=i, normalize=False)
            cdf += self.weights[i] * cdf_part
        if extra:
            return cdf

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            cdf = (cdf - F_0) / (F_1 - F_0)

        return _restrict_cdf_to_domain(x, cdf, self.domain)

    def dpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        First derivative of PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            First derivative of PDF
        """
        return self._evaluate_pdf(
            x=x,
            call=hypoexponential_distribution_pdf_1st_order_derivative,
        )

    def ddpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Second derivative of PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Second derivative of PDF
        """
        return self._evaluate_pdf(
            x=x,
            call=hypoexponential_distribution_pdf_2nd_order_derivative,
        )

    def logp(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Logarithm of the PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Logarithm of the PDF
        """
        logp = np.log(self.pdf(x))  # lower-level implementation of log does not provide
        # much numerical stability since sum of e^x terms has to be calculated first.

        return logp

    def quantile_function(self) -> None:
        """
        Quantile function.

        Returns
        -------
        None
        """
        raise ValueError(
            "Quantile function has no closed form. Inverse CDF has to be "
            "calculated numerically."
        )


def photoswitching_fingerprint_prepare(
    params: Mapping[int, Sequence[int | float]],
    n: int,
    z: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Get combinations of lambdas and pis for the photoswitching fingerprint model.
    Needed for the PDF and CDF parts. See model derivation for details.

    Parameters
    ----------
    params
        Parameters of the underlying exponential distributions.
    n
        Number of fluorophores needed to be considered. For the i-th CDF/PDF part,
        n = i + 1.
    z
        Index of the delta arrival time group that uses a mixture of three exponential
        distributions. -1 if none uses three exponential distributions.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Combinations of lambdas and pis for the photoswitching fingerprint model.
    """
    valid_combinations = generate_combinations(n=n, z=z)
    lambdas = map_to_lambdas(combos=valid_combinations, params=params, z=z)
    pis = get_pis(combos=valid_combinations, params=params, z=z)

    return lambdas, pis


def generate_combinations(n: int, z: int) -> npt.NDArray[np.int64]:
    """
    Generate combinations for all valid convolutions of exponential distributions.

    Parameters
    ----------
    n
        Number of fluorophores needed to be considered. For the i-th CDF/PDF part,
        n = i + 1.
    z
        Index of the delta arrival time group that uses a mixture of three exponential
        distributions. -1 if none uses three exponential distributions.

    Returns
    -------
    npt.NDArray[np.int64]
        All valid combinations of exponential distributions. Array of shape (m, n) where
        m is the number of valid combinations.
        For each m, the n columns represent the n delta arrival time groups needed to be
        considered.
        Each entry can be 0 (biased exponential distribution), 1 (non-biased exponential
        distribution of two-component mixture), 2 (first non-biased exponential
        distribution of three-component mixture), or 3 (second non-biased exponential
        distribution of three-component mixture).
    """
    combinations = []
    has_three_component = 0 <= z < n

    for first_nonbiased in range(n, -1, -1):
        combination = np.zeros(n, dtype=np.int64)
        combination[first_nonbiased:] = 1

        if has_three_component and first_nonbiased <= z:
            for component in (2, 3):
                current = combination.copy()
                current[z] = component
                combinations.append(current)
        else:
            combinations.append(combination)

    valid_combinations = np.array(combinations, dtype=np.int64)
    return valid_combinations


def map_to_lambdas(
    combos: npt.NDArray[np.int_], params: Mapping[int, Sequence[int | float]], z: int
) -> npt.NDArray[np.float64]:
    """
    Map combinations to lambdas.

    Parameters
    ----------
    combos
        All valid combinations of exponential distributions.
    params
        Parameters of the underlying exponential distributions.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    npt.NDArray[np.float64]
        Mapped lambdas. Array of shape (m, n) where m is the number of valid
        combinations and n the number of delta arrival time groups needed to be
        considered.
    """
    result = np.empty_like(combos, dtype=float)
    for idx in range(combos.shape[1]):
        col = combos[:, idx]
        if idx == z:
            mapping = {0: params[idx][3], 2: params[idx][4], 3: params[idx][5]}
        else:
            mapping = {0: params[idx][2], 1: params[idx][3]}
        result[:, idx] = np.array([mapping.get(x) for x in col])
    return result


def get_pis(
    combos: npt.NDArray[np.int_], params: Mapping[int, Sequence[int | float]], z: int
) -> npt.NDArray[np.float64]:
    """
    Get pis for each combination.

    Parameters
    ----------
    combos
        All valid combinations of exponential distributions.
    params
        Parameters of the underlying exponential distributions.
    z
        Index of the fluorophore that uses three exponential distributions.

    Returns
    -------
    npt.NDArray[np.float64]
        Mapped pis. Array of shape (m, n) where m is the number of valid combinations
        and n the number of delta arrival time groups needed to be considered.
    """
    pis = np.ones_like(combos, dtype=float)
    if z != -1:
        normalize = params[z][1] + params[z][2]
    else:
        normalize = 1
    for idx in range(combos.shape[1]):
        col_all = combos[:, idx]
        col_filt = combos[1:, idx]
        mapping = {
            0: params[idx][0],
            1: params[idx][1],
            2: params[idx][1],
            3: params[idx][2],
        }
        mapper = np.vectorize(pyfunc=mapping.get, otypes=[float])
        if idx == combos.shape[1] - 1:
            pis[:, idx] = mapper(col_all)
        else:
            zeros = np.where(col_filt == 0)[0]
            pis[1:, :][zeros, idx] = mapper(col_filt[zeros])

            ones = np.where(col_filt == 1)[0]
            if idx == 0:
                pis[1:, :][ones, idx] = mapper(col_filt[ones])
            else:
                mask = combos[1:, :][ones, idx - 1] == 0
                pis[1:, :][ones[mask], idx] = mapper(col_filt[ones[mask]])

            twos_threes_mask = np.isin(col_filt, [2, 3])
            twos_threes = np.where(twos_threes_mask)[0]
            if idx == 0:
                pis[1:, :][twos_threes, idx] = mapper(col_filt[twos_threes])
            else:
                mask = combos[1:, :][twos_threes, idx - 1] == 0
                pis[1:, :][twos_threes[mask], idx] = mapper(col_filt[twos_threes[mask]])
                if normalize == 0:
                    pis[1:, :][twos_threes[~mask], idx] = 0.0
                else:
                    pis[1:, :][twos_threes[~mask], idx] = (
                        mapper(col_filt[twos_threes[~mask]]) / normalize
                    )
    return pis


class ExponentialMixtureModel:
    """
    Model to describe a mixture of exponential distributions.
    """

    def __init__(
        self,
        params: dict[str, Sequence[int | float]],
        domain: tuple[float, float] = (0, np.inf),
    ) -> None:
        """
        Parameters
        ----------
        params
            Parameters of the underlying exponential distributions. Should contain
            keys "lambdas" (1D array-like of length n) and "pis" (1D array-like of
            length n-1). The last pi will be inferred as 1 - sum(pis).
        domain
            Domain of the model. Default is (0, np.inf).
        """
        self.params = params
        self.domain = domain

    def pdf(
        self,
        x: float | npt.ArrayLike,
    ) -> float | npt.NDArray[np.float64]:
        """
        Probability density function of a mixture of exponential distributions.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            PDF of the mixture of exponential distributions.
        """
        pdf: DistributionValue = 0.0
        for i, lam in enumerate(self.params["lambdas"]):
            if i == len(self.params["lambdas"]) - 1:
                p = 1 - np.sum(self.params["pis"])
            else:
                p = self.params["pis"][i]
            pdf += p * expon.pdf(x, scale=1 / lam)

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            pdf = pdf / (F_1 - F_0)

        return _restrict_pdf_to_domain(x, pdf, self.domain)

    def cdf(
        self,
        x: float | npt.ArrayLike,
        extra: bool = False,
    ) -> float | npt.NDArray[np.float64]:
        """
        Cumulative distribution function of a mixture of exponential distributions.

        Parameters
        ----------
        x
            Sample.
        extra
            ...

        Returns
        -------
        float | npt.NDArray[np.float64]
            CDF of the mixture of exponential distributions.
        """
        cdf: DistributionValue = 0.0
        for i, lam in enumerate(self.params["lambdas"]):
            if i == len(self.params["lambdas"]) - 1:
                p = 1 - np.sum(self.params["pis"])
            else:
                p = self.params["pis"][i]

            cdf += p * expon.cdf(x, scale=1 / lam)

        if extra:
            return cdf

        if self.domain != (0, np.inf):
            F_1: DistributionValue
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            cdf = (cdf - F_0) / (F_1 - F_0)

        return _restrict_cdf_to_domain(x, cdf, self.domain)


class ExponentialMixtureMarginalModel:
    """
    Model to describe the marginal distribution of a sample X from a mixture of
    exponential distributions, where the upper truncation is a random variable
    Y ~ fixed truncation - T, and T is a random variable following a part of PFA
    distribution.
    """

    def __init__(
        self,
        params: dict[str, Sequence[int | float]],
        pfa_cdf_part: Callable[[npt.ArrayLike, int, bool], DistributionValue],
        cdf_part_index: int,
        truncation_up: float,
    ) -> None:
        """
        Parameters
        ----------
        params
            Parameters of the underlying exponential distributions. Should contain
            keys "lambdas" (1D array-like of length n) and "pis" (1D array-like of
            length n-1). The last pi will be inferred as 1 - sum(pis).
        pfa_cdf_part
            Function to calculate the CDF part of the PFA distribution.
        cdf_part_index
            Index of the CDF part of the PFA distribution to be used.
        truncation_up
            Fixed upper truncation.
        """
        self.params = params
        self.pfa_cdf_part = pfa_cdf_part
        self.cdf_part_index = cdf_part_index
        self.truncation_up = truncation_up

        x_grid = _marginal_integration_grid(truncation_up)
        pdf_grid = ExponentialMixtureModel(params=params, domain=(0, np.inf)).pdf(
            x_grid
        ) * pfa_cdf_part(truncation_up - x_grid, cdf_part_index, True)
        # pfa_cdf_part because we want the distribution of T of the (n-1)th fluorophore,
        # not of all n-x fluorophores
        # CDF(truncation_up - x) because Pr(actual_truncation >= x) = Pr(truncation_up - T >= x) = Pr(T <= truncation_up - x) = CDF(truncation_up - x)
        # i.e., pfa_cdf_part does not describe the actual truncation of the two_expon_mixture, but truncation_up - T does.
        # if it described the actual truncation, we would multiply with (1 - CDF(x)) (Survival function)
        P_obs = simpson(y=pdf_grid, x=x_grid)
        if not np.isfinite(P_obs) or P_obs <= 0 or P_obs > 1 + 1e-6:
            raise ValueError("The calculated observation probability is invalid.")
        pdf_grid /= P_obs
        # normalization such that integral over domain is 1, i.e., pdf_grid describes the distribution
        # given that an event is observed
        cdf_grid = cumulative_simpson(pdf_grid, x=x_grid, initial=0)
        cdf_grid = np.maximum.accumulate(cdf_grid)
        cdf_grid /= cdf_grid[-1]

        self.pdf_grid = pdf_grid
        self.cdf_grid = cdf_grid
        self.x_grid = x_grid
        self.P_obs = P_obs

    def pdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Probability distribution function

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            PDF
        """
        values = np.asarray(x, dtype=np.float64)
        pdf = np.interp(values, xp=self.x_grid, fp=self.pdf_grid, left=0.0, right=0.0)

        return cast(DistributionValue, pdf)

    def cdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Cumulative distribution function

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        float | npt.NDArray[np.float64]
            CDF
        """
        values = np.asarray(x, dtype=np.float64)
        cdf = np.interp(values, xp=self.x_grid, fp=self.cdf_grid, left=0.0, right=1.0)

        return cast(DistributionValue, cdf)
