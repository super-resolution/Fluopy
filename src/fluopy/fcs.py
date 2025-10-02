"""
Fluorescence correlation spectroscopy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt
import multipletau as mp
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import figure as fi

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.emissions import Emissions
    from fluopy.fluopy_types import RandomGeneratorSeed


__all__: list[str] = ["FCS"]

logger = logging.getLogger(__name__)


class FCS:
    """
    Container of FCS-assocciated attributes and methods.

    Attributes
    ----------
    emissions : fluopy.emissions.Emissions
        Container for emission-associated attributes.
    autocorrelation : npt.NDArray[np.foat64]
        Autocorrelation values.
    tau : npt.NDArray[np.foat64]
        Time differences (i.e., τ, lag times).
    """

    def __init__(self, emissions: Emissions):
        """
        Parameters
        ----------
        emissions
            Container for emission-associated attributes.
        """
        self.emissions = emissions
        self.autocorrelation = None
        self.tau = None

    def autocorrelate_time_points(
        self,
        exp_min: int = -8,
        exp_max: int = 2,
        points_per_base: int = 4,
        base: int = 10,
        normalize: bool = True,
    ) -> Self:
        """
        Autocorrelation of emissions.event_time_points. Generally much faster than
        autocorrelation based on emissions.event_time_series.
        Based on https://opg.optica.org/ol/abstract.cfm?uri=ol-31-6-829.

        Parameters
        ----------
        exp_min
            Exponent of the minimum value.
        exp_max
            Exponent of the maximum value.
        points_per_base
            Number of points per base.
        base
            The base of the exponentiation.
        normalize
            Whether to normalize the autocorrelation.

        Returns
        -------
        Self
        """
        if self.emissions.event_time_points is None:
            raise ValueError("event_time_points is None.")
        if base**exp_max > self.emissions.event_time_points[-1]:
            last_time_point = self.emissions.event_time_points[-1]
            exp_max_adjusted = np.int64(
                np.floor(np.log(last_time_point) / np.log(base))
            )
            logger.warning(
                f"The exp_max {exp_max} yields a base to the power of exp_max {base**exp_max} that is larger than the last time "
                f"point {last_time_point}. Therefore, exp_max is adjusted to {exp_max_adjusted}.",
                stacklevel=2,
            )
            exp_max = exp_max_adjusted
        bins = make_loglags(
            exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base
        )
        self.autocorrelation = pcorrelate(
            t=self.emissions.event_time_points,
            u=self.emissions.event_time_points,
            bins=bins,
            normalize=normalize,
        )
        self.tau = np.mean([bins[1:], bins[:-1]], 0)

        return self

    def autocorrelate_time_series(
        self, log: bool = True, m: int = 4, normalize: bool = True
    ) -> Self:
        """
        Autocorrelation of emissions.event_time_series. The minimum lag time is equal
        to resample value of series.

        Parameters
        ----------
        log
            Whether to compute the autocorrelation on a logarithmic scale. As time
            steps increase, correlation signals are getting noisier, fluctuating around
            0. Hence, log should usually be True.
        m
            Defines the number of points on each log level. E.g., m=4 leads to
            |1, 2, 3, 4| |2, 4, 6, 8| |4, 8, 12, 16| ..., hence
            |1, 2, 3, 4, 6, 8, 12, 16, ...|. Only used if log ist True.
        normalize
            Whether to normalize the autocorrelation.

        Returns
        -------
        Self
        """
        if self.emissions.event_time_series is None:
            raise ValueError("event_time_series is None.")
        event_time_series = self.emissions.event_time_series.astype(float)
        deltat = event_time_series.index[1]
        if normalize and log:
            autocorrelation = mp.autocorrelate(
                a=event_time_series.values, m=m, deltat=deltat, normalize=True
            )
            self.tau, autocorrelation = np.transpose(autocorrelation[1:])

        elif log and not normalize:
            autocorrelation = mp.autocorrelate(
                a=event_time_series.values, m=m, deltat=deltat, normalize=False
            )
            self.tau, autocorrelation = np.transpose(autocorrelation[1:])

        elif normalize and not log:
            mean = np.mean(event_time_series.values)
            deviation = (
                event_time_series.values - mean
            )  # delta I(t) (wiki) - fluctuation around the mean value
            autocorrelation = np.correlate(deviation, deviation, mode="full")
            autocorrelation = autocorrelation[autocorrelation.size // 2 :]
            autocorrelation = np.divide(
                autocorrelation, np.arange(autocorrelation.size, 0, -1)
            )
            # averaging - in multipletau, this is included in normalize=True (denoted
            # as M-k in documentation)
            autocorrelation = (
                autocorrelation / mean**2
            )  # normalization with mean squared
            autocorrelation = autocorrelation[1:1000]
            self.tau = event_time_series.index.values[1:1000]

        else:
            autocorrelation = np.correlate(
                event_time_series.values, event_time_series.values, mode="full"
            )
            # note that this version is the autocorrelation in the sense of signal
            # processing and differs from the statistical definition of autocorrelation.
            autocorrelation = autocorrelation[autocorrelation.size // 2 :][1:1000]
            self.tau = event_time_series.index.values[1:1000]

        self.autocorrelation = autocorrelation + 1

        return self

    def plot_matplotlib(
        self,
        normalize_to: int | None = None,
        unit: str = "s",
        ax: mplAxes | None = None,
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot FCS data.

        Parameters
        ----------
        normalize_to
            Index of datapoint to which the data is normalized.
        unit
            One of 's', 'ms', 'us'. Influences the unit of the x-axis.
        ax
            Axis to plot on.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        tau_data, correl_data = np.copy(self.tau), np.copy(self.autocorrelation)
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit=unit).total_seconds()
        tau_data = tau_data / adjust_unit

        ax.plot(tau_data, correl_data, **kwargs)
        ax.set_title(rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        ax.set_xlabel(rf"$\tau \ ({unit})$")
        ax.set_xscale("log")
        ax.set_ylabel(r"$G(\tau)$")

        return ax

    def plot(
        self, normalize_to: int | None = None, unit: str = "s", **kwargs: Any
    ) -> npt.NDArray[mplAxes]:
        """
        Plot FCS data.

        Parameters
        ----------
        normalize_to
            Index of datapoint to which the data is normalized.
        unit
            One of 's', 'ms', 'us'. Influences the unit of the x-axis.
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        tau_data, correl_data = np.copy(self.tau), np.copy(self.autocorrelation)
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit=unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault("title", rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xscale", "log")
        kwargs.setdefault("xlabel", rf"$\tau \ ({unit})$")
        kwargs.setdefault("ylabel", r"$G(\tau)$")

        axes = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

        return axes


def fit_dark(
    tau: npt.ArrayLike, dark_lifetime: float, dark_occupation: float
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Fit function of dark states (e.g., triplet).

    Parameters
    ----------
    tau
        Time differences (i.e., τ, lag times).
    dark_lifetime
        Mean lifetime of the dark state.
    dark_occupation
        Steady state fraction of the dark state. Number between 0 and 1.

    Returns
    -------
    autocorrelation : : npt.NDArray[np.float64]
        Autocorrelation values.
    norm : float
        Steady state fraction of other states. Number between 0 and 1.
    """
    if dark_occupation < 0 or dark_occupation >= 1:
        raise ValueError("dark_occupation is bound to be between 0 and 1.")
    tau = np.asarray(tau)
    autocorrelation = dark_occupation * np.exp(-tau / dark_lifetime)
    norm = 1 - dark_occupation

    return autocorrelation, norm


def fit_antibunching(
    tau: npt.ArrayLike, excitation_rate: float, s1_lifetime: float
) -> npt.NDArray[np.float64]:
    """
    Fit function of antibunching.

    Parameters
    ----------
    tau
        Time differences (i.e., τ, lag times).
    excitation_rate
        Rate constant of excitation.
    s1_lifetime
        Mean lifetime of the S1 state.

    Returns
    -------
    npt.NDArray[np.float64]
        Autocorrelation values.
    """
    tau = np.asarray(tau)
    lifetime_s0_s1 = 1 / (1 / s1_lifetime + excitation_rate)
    autocorrelation = -np.exp(-tau / lifetime_s0_s1)

    return autocorrelation


def fit_triplet_cis(
    tau: npt.ArrayLike,
    k_isc: float,
    k_T: float,
    k_01: float,
    k_10: float,
    k_iso: float,
    k_biso_eff: float,
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Fit function of triplet and cis as two non-independent dark states.

    Parameters
    ----------
    tau
        Time differences (i.e., τ, lag times).
    k_isc
        Rate constant of intersystem crossing to the triplet state.
    k_T
        Rate constant of intersystem crossing out of the triplet state.
    k_01
        Rate constant of excitation.
    k_10
        Inverse of fluorescence lifetime considering all rates from S1 (not just IC and
        FL). Note: in the PAPER it is not clear which one they mean but the fit is
        significantly better if using this version.
    k_iso
        Rate constant of isomerization from trans to cis.
    k_biso_eff
        Rate constant of back isomerization from cis to trans.

    Returns
    -------
    autocorrelation : tuple[npt.NDArray[np.float64]
        Autocorrelation values.
    norm : float
        Steady state fraction of other states. Number between 0 and 1.
    """
    tau = np.asarray(tau)
    k_isc_eff = k_01 / (k_01 + k_10) * k_isc
    k_iso_eff = k_01 / (k_01 + k_10) * k_iso

    # eigen_1 = 0

    part_1 = (k_isc_eff + k_T + k_iso_eff + k_biso_eff) / 2
    part_2 = (
        (k_isc_eff + k_T + k_iso_eff + k_biso_eff) ** 2 / 4
        - k_iso_eff * k_T
        - k_isc_eff * k_biso_eff
        - k_T * k_biso_eff
    ) ** 0.5
    eigen_2 = -(part_1 + part_2)
    eigen_3 = -(part_1 - part_2)

    alpha = k_iso_eff * k_T + k_isc_eff * k_biso_eff + k_T * k_biso_eff
    beta = k_isc_eff + k_iso_eff + k_T - k_biso_eff
    gamma = (
        (k_isc_eff + k_iso_eff) ** 2
        + (k_biso_eff - k_T) ** 2
        + 2 * (k_iso_eff - k_isc_eff) * (k_biso_eff - k_T)
    ) ** 0.5
    delta = (
        k_T * (k_iso_eff + k_biso_eff - k_T - k_isc_eff) + 2 * k_isc_eff * k_biso_eff
    )

    z_1 = k_T * k_biso_eff / alpha
    z_2 = (beta + gamma) * (k_T * gamma + delta) / (4 * alpha * gamma)
    z_3 = (beta - gamma) * (k_T * gamma - delta) / (4 * alpha * gamma)

    autocorrelation = z_2 * np.exp(tau * eigen_2) + z_3 * np.exp(tau * eigen_3)
    norm = z_1

    return autocorrelation, norm


def make_loglags(
    exp_min: int,
    exp_max: int,
    points_per_base: int,
    base: int = 10,
    return_int: bool = False,
) -> npt.NDArray[Any]:
    """Make a log-spaced array useful as lag bins for cross-correlation.

    This function creates an arrays of log-spaced time-lag bins to be used
    with :func:`pcorrelate`. By default it returns integer time-lag bins
    to avoid floating point inaccuracies in the correlation
    (showing up as higher noise at small time-lags).

    Example:

        Compute log10-spaced bins with 5 bins per decade, starting from 1
        (10⁰) and stopping at 10⁶::

            >>> make_loglags(0, 6, 5)
            array([      1,       2,       3,       4,       6,      10,      16,
                        25,      40,      63,     100,     158,     251,     398,
                       631,    1000,    1585,    2512,    3981,    6310,   10000,
                     15849,   25119,   39811,   63096,  100000,  158489,  251189,
                    398107,  630957, 1000000])

        Compute log10-spaced bins with 2 bins per decade, starting
        from 10⁻¹ and stopping at 10³::

            >>> make_loglags(-1, 3, 2, return_int=False)
            array([  1.00000000e-01,   3.16227766e-01,   1.00000000e+00,
                     3.16227766e+00,   1.00000000e+01,   3.16227766e+01,
                     1.00000000e+02,   3.16227766e+02,   1.00000000e+03])

    See Also
    ---------
    :func:`pcorrelate`

    Parameters
    ----------
    exp_min
        exponent of the minimum value
    exp_max
        exponent of the maximum value
    points_per_base
        number of points per base
        (i.e. in a decade when `base = 10`)
    base
        base of the exponent. Default 10.
    return_int
        if True return integer bin edges to avoid floating point inaccuracies.
        If False, returned bin edges are float.

    Returns
    --------
    npt.NDArray[Any]
        Array of log-spaced values with specified range and spacing.
    """
    num_points = points_per_base * (exp_max - exp_min) + 1
    bins = np.logspace(exp_min, exp_max, num_points, base=base)
    if return_int:
        # using `unique` because rounding to int may create duplicates
        bins = np.unique(np.round(bins).astype("int64"))
    return bins


@numba.jit(nopython=True)
def pcorrelate(
    t: npt.ArrayLike, u: npt.ArrayLike, bins: npt.ArrayLike, normalize: bool = False
) -> npt.NDArray[np.float64]:
    """Compute correlation of two arrays of discrete events (Point-process).

    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.

    See Also
    ---------
    :func:`make_loglags` to genetate log-spaced lag bins.

    Parameters
    ----------
    t
        first array of "points" to correlate. The array needs
        to be monothonically increasing.
    u
        second array of "points" to correlate. The array needs
        to be monothonically increasing.
    bins
        bin edges for lags where correlation is computed.
    normalize
        if True, normalize the correlation function
        as typically done in FCS using :func:`pnormalize`. If False,
        return the unnormalized correlation function.

    Returns
    --------
    npt.NDArray[np.float64]
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    if normalize:
        G = pnormalize(G, t, u, bins)
    return G


@numba.jit(nopython=True)
def pnormalize(
    G: npt.ArrayLike, t: npt.ArrayLike, u: npt.ArrayLike, bins: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""Normalize point-process cross-correlation function.

    This normalization is usually employed for fluorescence correlation
    spectroscopy (FCS) analysis.
    The normalization is performed according to
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.
    Basically, the input argument `G` is multiplied by:

    .. math::
        \frac{T-\tau}{n(\{i \ni t_i \le T - \tau\})n(\{j \ni u_j \ge \tau\})}

    where `n({})` is the operator counting the elements in a set, *t* and *u*
    are the input arrays of the correlation, *τ* is the time lag and *T*
    is the measurement duration.

    Parameters
    ----------
        G (array): raw cross-correlation to be normalized.
        t (array): first input array of "points" used to compute `G`.
        u (array): second input array of "points" used to compute `G`.
        bins (array): array of bins used to compute `G`. Needs to have the
            same units as input arguments `t` and `u`.

    Returns
    --------
    npt.NDArray[np.float64]
        Array of normalized values for the cross-correlation function,
        same size as the input argument `G`.
    """
    duration = max((t.max(), u.max())) - min((t.min(), u.min()))
    Gn = G.copy()
    for i, tau in enumerate(bins[1:]):
        Gn[i] *= (duration - tau) / (
            float((t >= tau).sum()) * float((u <= (u.max() - tau)).sum())
        )
    return Gn


def coincidence_numpy(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike,
    tau_max: float,
    bin_width: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the coincidence histogram of photon arrival times using NumPy. Suitable for
    smaller datasets.

    Parameters
    ----------
    arr1
        Photon arrival times from the first detector.
    arr2
        Photon arrival times from the second detector.
    tau_max
        Maximum time difference to consider for the histogram.
    bin_width
        Width of the histogram bins.

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bins : npt.NDArray[np.float64]
        Bin edges of the histogram.
    """
    bins = np.arange(-tau_max, tau_max + bin_width, bin_width)
    hist = np.zeros(bins.size - 1)

    for t in arr1:
        mask = np.abs(arr2 - t) < tau_max
        delays = arr2[mask] - t
        h, _ = np.histogram(delays, bins=bins)
        hist += h

    return hist, bins


@numba.jit(nopython=True)
def coincidence_numba(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike,
    tau_max: float,
    bin_width: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the coincidence histogram of photon arrival times using Numba. Suitable for
    larger datasets.
    Uses similar algorithm to pcorrelate but optimized for coincidence calculations.

    Parameters
    ----------
    arr1
        Photon arrival times from the first detector.
    arr2
        Photon arrival times from the second detector.
    tau_max
        Maximum time difference to consider.
    bin_width
        Width of histogram bins.

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bins : npt.NDArray[np.float64]
        Bin edges of the histogram.
    """
    bins = np.arange(-tau_max, tau_max + bin_width, bin_width)
    hist = np.zeros(len(bins) - 1, dtype=np.float64)

    for t1 in arr1:
        # Find range of arr2 values within tau_max of t1
        # Using binary search-like approach for efficiency (log(n) complexity)
        start_idx = 0
        end_idx = len(arr2)

        # Find first arr2 value >= t1 - tau_max
        while start_idx < len(arr2) and arr2[start_idx] < t1 - tau_max:
            start_idx += 1

        # Find first arr2 value > t1 + tau_max
        while end_idx > 0 and arr2[end_idx - 1] > t1 + tau_max:
            end_idx -= 1

        # Process valid range
        for j in range(start_idx, end_idx):
            delay = arr2[j] - t1
            if abs(delay) < tau_max:
                # Find which bin this delay belongs to
                bin_idx = int((delay + tau_max) / bin_width)
                if 0 <= bin_idx < len(hist):
                    hist[bin_idx] += 1.0

    return hist, bins


def coincidence(
    photon_arrival_times: npt.ArrayLike,
    tau_max: float,
    bin_width: float,
    seed: RandomGeneratorSeed = 1,
    method: str = "numba",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the coincidence histogram of photon arrival times. Here, the Hanbury Brown
    Twiss experiment is mimicked by randomly splitting the photon arrival times into two
    detectors. The histogram is then computed for the time difference between the two
    detectors. Second order coherence is then given by a normalization routine.

    Parameters
    ----------
    photon_arrival_times
        Photon arrival times.
    tau_max
        Maximum time difference to consider for the histogram.
    bin_width
        Width of the histogram bins.
    seed
        A seed to initialize the BitGenerator.
    method
        Method to use: "numpy" or "numba".

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bins : npt.NDArray[np.float64]
        Bin edges of the histogram.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(photon_arrival_times.size) < 0.5
    arr1 = np.sort(photon_arrival_times[mask])
    arr2 = np.sort(photon_arrival_times[~mask])

    if method == "numpy":
        hist, bins = coincidence_numpy(arr1, arr2, tau_max, bin_width)
    elif method == "numba":
        hist, bins = coincidence_numba(arr1, arr2, tau_max, bin_width)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'numpy' or 'numba'.")

    hist = hist[:-1]
    # normalization to avoid finite window effects
    hist = hist / (np.max(photon_arrival_times) - np.abs(bins[1:-1]))
    # standard normalization
    average_signal_1 = arr1.size / np.max(arr1) if arr1.size > 0 else 0
    average_signal_2 = arr2.size / np.max(arr2) if arr2.size > 0 else 0
    if average_signal_1 > 0 and average_signal_2 > 0:
        hist = hist / (average_signal_1 * average_signal_2 * bin_width)

    return hist, bins
