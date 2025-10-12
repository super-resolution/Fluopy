"""
Random variable distributions.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import expon

if TYPE_CHECKING:
    pass


__all__: list[str] = []


def hypoexponential_distribution_cdf(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    CDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        CDF of the hypoexponential distribution.
    """
    cdf = 1
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        cdf -= np.exp(-arg * x) * np.prod(other_args) / np.prod(-arg + other_args)

    return cdf


def hypoexponential_distribution_pdf(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    float | npt.NDArray[np.float64]
        PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf += np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)

    return pdf


def hypoexponential_distribution_pdf_1st_order_derivative(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    First order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    pdf_1st_order_derivative : float | npt.NDArray[np.float64]
        First order derivative of the PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf_1st_order_derivative = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf_1st_order_derivative += (
            -arg * np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)
        )

    return pdf_1st_order_derivative


def hypoexponential_distribution_pdf_2nd_order_derivative(
    x: float | npt.ArrayLike, *args: Any
) -> float | npt.NDArray[np.float64]:
    """
    Second order derivative of the PDF of the hypoexponential distribution.

    Parameters
    ----------
    x
        Sample.
    args : float, 1-D array_like
        Parameters (lambdas) of the hypoexponential distribution. Must be distinct and
        positive.

    Returns
    -------
    pdf_2nd_order_derivative : float | npt.NDArray[np.float64]
        Second order derivative of the PDF of the hypoexponential distribution.
    """
    all_args = np.array(args)
    pdf_2nd_order_derivative = 0
    for arg in args:
        other_args = np.array([other for other in args if other != arg])
        pdf_2nd_order_derivative += (
            arg**2 * np.exp(-arg * x) * np.prod(all_args) / np.prod(-arg + other_args)
        )

    return pdf_2nd_order_derivative


class Photoswitching_fingerprint_model:
    """
    Model to describe photoswitching fingerprints produced by n fluorophores where there
    is bias in time (e.g., increased probability of ON in the beginning).
    """

    def __init__(
        self,
        params: dict,
        weights: float | npt.ArrayLike = None,
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
        call: Callable | None,
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
        """
        if call is None:
            call = hypoexponential_distribution_pdf
        lambdas, pis = photoswitching_fingerprint_prepare(
            params=self.params,
            n=i + 1,
            z=self.z,
        )
        pdf_part = 0
        for lambda_combo, pi_combo in zip(lambdas, pis):
            pi_set = np.prod(pi_combo)
            pdf_part += pi_set * call(
                x,
                *lambda_combo,
            )

        if not normalize:
            return pdf_part

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf_part(x=self.domain[-1], i=i, normalize=False)
            F_0 = self.cdf_part(x=self.domain[0], i=i, normalize=False)
            pdf_part = pdf_part / (F_1 - F_0)

        return pdf_part

    def pdf(
        self, x: float | npt.ArrayLike, order: int = 0
    ) -> float | npt.NDArray[np.float64]:
        """
        PDF

        Parameters
        ----------
        x
            Sample.
        order : int
            Order of the derivative of the PDF to be calculated.

        Returns
        -------
        pdf : float | npt.NDArray[np.float64]
            Model output.
        """
        if order == 0:
            call = hypoexponential_distribution_pdf
        elif order == 1:
            caller = inspect.stack()[1].function
            if caller != "dpdf":
                raise ValueError("Call dpdf instead of pdf with setting order=1.")
            call = hypoexponential_distribution_pdf_1st_order_derivative
        elif order == 2:
            caller = inspect.stack()[1].function
            if caller != "ddpdf":
                raise ValueError("Call ddpdf instead of pdf with setting order=2.")
            call = hypoexponential_distribution_pdf_2nd_order_derivative
        else:
            raise ValueError("Order has to be 0, 1, or 2.")

        n = len(self.params)
        pdf = 0
        for i in range(n):
            pdf_part = self.pdf_part(call=call, x=x, i=i, normalize=False)
            pdf += self.weights[i] * pdf_part

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            pdf = pdf / (F_1 - F_0)  # true for pdf, dpdf, ddpdf

        return pdf

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
        """
        lambdas, pis = photoswitching_fingerprint_prepare(
            params=self.params,
            n=i + 1,
            z=self.z,
        )
        cdf_part = 0
        for lambda_combo, pi_combo in zip(lambdas, pis):
            pi_set = np.prod(pi_combo)
            cdf_part += pi_set * hypoexponential_distribution_cdf(
                x,
                *lambda_combo,
            )
        if not normalize:
            return cdf_part

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf_part(x=self.domain[-1], i=i, normalize=False)
            F_0 = self.cdf_part(x=self.domain[0], i=i, normalize=False)
            cdf_part = (cdf_part - F_0) / (F_1 - F_0)

        return cdf_part

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
        cdf : float | npt.NDArray[np.float64]
            Model output.
        """
        n = len(self.params)
        cdf = 0
        for i in range(n):
            cdf_part = self.cdf_part(x=x, i=i, normalize=False)
            cdf += self.weights[i] * cdf_part
        if extra:
            return cdf

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            cdf = (cdf - F_0) / (F_1 - F_0)

        return cdf

    def dpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        First derivative of PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        dpdf : float | npt.NDArray[np.float64]
            Model output.
        """
        dpdf = self.pdf(x, order=1)

        return dpdf

    def ddpdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Second derivative of PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        ddpdf : float | npt.NDArray[np.float64]
            Model output.
        """
        ddpdf = self.pdf(x, order=2)

        return ddpdf

    def logp(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Logarithm of the PDF.

        Parameters
        ----------
        x
            Sample.

        Returns
        -------
        logp : float | npt.NDArray[np.float64]
            Model output.
        """
        logp = np.log(self.pdf(x))  # lower-level implementation of log does not provide
        # much numerical stability since sum of e^x terms has to be calculated first.

        return logp

    def quantile_function(self) -> None:
        """
        Quantile function.
        """
        raise ValueError(
            "Quantile function has no closed form. Inverse CDF has to be "
            "calculated numerically."
        )


def photoswitching_fingerprint_prepare(
    params: dict,
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
    print(valid_combinations)
    lambdas = map_to_lambdas(combos=valid_combinations, params=params, z=z)
    pis = get_pis(combos=valid_combinations, params=params, z=z)

    return lambdas, pis


def generate_combinations(n: int, z: int) -> npt.NDArray[np.int64]:
    """
    Generate all valid convolutions of exponential distributions.

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
    valid_combos : 2-D np.ndarray
        All valid combinations of exponential distributions. Array of shape (m, n) where
        m is the number of valid combinations.
        For each m, the n columns represent the n delta arrival time groups needed to be
        considered.
        Each entry can be 0 (biased exponential distribution), 1 (non-biased exponential
        distribution of two-component mixture), 2 (first non-biased exponential
        distribution of three-component mixture), or 3 (second non-biased exponential
        distribution of three-component mixture).
    """
    arrays = []
    for i in range(n):
        if i == z:
            arrays.append([0, 2, 3])  # b, nb_1 and nb_2
        else:
            arrays.append([0, 1])  # b and nb_0
    combos = np.array(list(product(*arrays)), dtype=int)
    if combos.shape[1] == 1:
        valid_combos = combos
    zeros = combos == 0
    curr_zeros = zeros[:, 1:]
    prev_zeros = zeros[:, :-1]
    mask = np.all((~curr_zeros) | prev_zeros, axis=1)
    valid_combos = combos[mask]

    return valid_combos


def map_to_lambdas(
    combos: npt.NDArray[np.int_], params: dict, z: int
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
    result : 2-D np.ndarray
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
    combos: npt.NDArray[np.int_], params: dict, z: int
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
    pis : 2-D np.ndarray
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

            twos_threes = np.isin(col_filt, [2, 3])
            twos_threes = np.where(twos_threes)[0]
            if idx == 0:
                pis[1:, :][twos_threes, idx] = mapper(col_filt[twos_threes])
            else:
                mask = combos[1:, :][twos_threes, idx - 1] == 0
                pis[1:, :][twos_threes[mask], idx] = mapper(col_filt[twos_threes[mask]])
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
        params: dict,
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
        pdf = 0
        for i, lam in enumerate(self.params["lambdas"]):
            if i == len(self.params["lambdas"]) - 1:
                p = 1 - np.sum(self.params["pis"])
            else:
                p = self.params["pis"][i]
            pdf += p * expon.pdf(x, scale=1 / lam)

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            pdf = pdf / (F_1 - F_0)

        return pdf

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
        cdf = 0
        for i, lam in enumerate(self.params["lambdas"]):
            if i == len(self.params["lambdas"]) - 1:
                p = 1 - np.sum(self.params["pis"])
            else:
                p = self.params["pis"][i]

            cdf += p * expon.cdf(x, scale=1 / lam)

        if extra:
            return cdf

        if self.domain != (0, np.inf):
            if self.domain[-1] == np.inf:
                F_1 = 1
            else:
                F_1 = self.cdf(x=self.domain[-1], extra=True)
            F_0 = self.cdf(x=self.domain[0], extra=True)
            cdf = (cdf - F_0) / (F_1 - F_0)

        return cdf


class ExponentialMixtureMarginalModel:
    """
    Model to describe the marginal distribution of a sample X from a mixture of
    exponential distributions, where the upper truncation is a random variable
    Y ~ fixed truncation - T, and T is a random variable following a part of PFA
    distribution.
    """

    def __init__(
        self,
        params: dict,
        pfa_cdf_part: Callable,
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
        x_grid = np.logspace(np.log10(0.01), np.log10(truncation_up), 200)
        x_grid = np.insert(arr=x_grid, obj=0, values=0.0)
        pdf_grid = ExponentialMixtureModel(params=params, domain=(0, np.inf)).pdf(
            x_grid
        ) * pfa_cdf_part(x=truncation_up - x_grid, i=cdf_part_index, normalize=True)
        # pfa_cdf_part because we want the distribution of T of the (n-1)th fluorophore,
        # not of all n-x fluorophores
        # CDF(truncation_up - x) because Pr(actual_truncation >= x) = Pr(truncation_up - T >= x) = Pr(T <= truncation_up - x) = CDF(truncation_up - x)
        # i.e., pfa_cdf_part does not describe the actual truncation of the two_expon_mixture, but truncation_up - T does.
        # if it described the actual truncation, we would multiply with (1 - CDF(x)) (Survival function)
        P_obs = np.trapezoid(y=pdf_grid, x=x_grid)
        pdf_grid /= P_obs
        # normalization such that integral over domain is 1, i.e., pdf_grid describes the distribution
        # given that an event is observed
        cdf_grid = cumulative_trapezoid(pdf_grid, x=x_grid, initial=0)

        self.pdf_grid = pdf_grid
        self.cdf_grid = cdf_grid
        self.x_grid = x_grid
        self.P_obs = P_obs

    def pdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        pdf = np.interp(x, xp=self.x_grid, fp=self.pdf_grid, left=0.0, right=0.0)

        return pdf

    def cdf(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        cdf = np.interp(x, xp=self.x_grid, fp=self.cdf_grid, left=0.0, right=1.0)

        return cdf
