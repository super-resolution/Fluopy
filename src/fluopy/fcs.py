"""
Fluorescence correlation spectroscopy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Self

import matplotlib.pyplot as plt
import multipletau as mp  # type: ignore[import-untyped]
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import plotting as fi

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.emissions import Emissions
    from fluopy.fluopy_types import RandomGeneratorSeed


__all__: list[str] = ["FCS"]

logger = logging.getLogger(__name__)


class FCS:
    """
    Container of FCS-associated attributes and methods.

    Attributes
    ----------
    emissions : fluopy.emissions.Emissions
        Container for emission-associated attributes.
    channel : str
        Detection channel used for autocorrelation.
    autocorrelation : npt.NDArray[np.float64]
        Autocorrelation values.
    tau : npt.NDArray[np.float64]
        Time differences (i.e., τ, lag times).
    """

    def __init__(self, emissions: Emissions, channel: str | None = None):
        """
        Parameters
        ----------
        emissions
            Container for emission-associated attributes.
        channel
            Detection channel used for autocorrelation. If None, the only configured
            channel is used.
        """
        self.emissions = emissions
        self.channel = emissions.resolve_channel(channel)
        self.autocorrelation: npt.NDArray[np.float64] | None = None
        self.tau: npt.NDArray[np.float64] | None = None

    def autocorrelate_time_points(
        self,
        exp_min: int = -8,
        exp_max: int = 2,
        points_per_base: int = 4,
        base: int = 10,
        normalize: bool = True,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Self:
        """
        Compute the autocorrelation directly from photon arrival times.

        Photon-pair delays are counted in logarithmically spaced lag intervals. This
        avoids first converting the sparse arrival times into a uniformly sampled time
        series.

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
        start_time
            The time the measurement started. This is used to normalize to the correct
            measurement duration.
        end_time
            The time the measurement ended. This is used to normalize to the correct
            measurement duration. If None, the last photon arrival time in the selected
            channel is used. Supply end_time explicitly to normalize to the full
            acquisition duration or to compare channels over the same duration.

        Returns
        -------
        Self

        Notes
        -----
        Implements the photon-pair counting formulation described by
        `Laurence et al. (2006) <https://doi.org/10.1364/OL.31.000829>`_.
        """
        if self.emissions.event_time_points is None:
            raise ValueError("event_time_points is None.")

        event_time_points = self.emissions.select_event_time_points(self.channel)
        if event_time_points.size == 0:
            raise ValueError("selected channel contains no photon arrival times.")
        if end_time is None:
            end_time = float(event_time_points[-1])
        duration = end_time - start_time

        if base**exp_max > duration:
            exp_max_adjusted = np.int64(np.floor(np.log(duration) / np.log(base)))
            logger.warning(
                f"The exp_max {exp_max} yields a base to the power of exp_max {base**exp_max} that is larger than the duration of the measurement: "
                f"{duration}. Therefore, exp_max is adjusted to {exp_max_adjusted}.",
                stacklevel=2,
            )
            exp_max = int(exp_max_adjusted)
        number_of_edges = points_per_base * (exp_max - exp_min) + 1
        bins = np.logspace(
            exp_min, exp_max, number_of_edges, base=base, dtype=np.float64
        )
        self.autocorrelation = _event_time_correlation(
            t=event_time_points,
            u=event_time_points,
            bins=bins,
            normalize=normalize,
            start_time=start_time,
            end_time=end_time,
        )
        self.tau = np.mean([bins[1:], bins[:-1]], axis=0)

        return self

    def autocorrelate_time_series(
        self, log: bool = True, m: int = 4, normalize: bool = True
    ) -> Self:
        """
        Autocorrelation of emissions.event_time_series. The minimum lag time is equal
        to sampling interval of series.

        Parameters
        ----------
        log
            Whether to compute the autocorrelation on a logarithmic scale. As time
            steps increase, correlation signals are getting noisier, fluctuating around
            0. Hence, log should usually be True.
        m
            Defines the number of points on each log level. E.g., m=4 leads to
            |1, 2, 3, 4| |2, 4, 6, 8| |4, 8, 12, 16| ..., hence
            |1, 2, 3, 4, 6, 8, 12, 16, ...|. Only used if log is True.
        normalize
            Whether to normalize the autocorrelation.

        Returns
        -------
        Self
        """
        if self.emissions.event_time_series is None:
            raise ValueError("event_time_series is None.")
        event_time_series = self.emissions.select_event_time_series(
            self.channel
        ).astype(float)
        event_values = event_time_series.to_numpy(dtype=np.float64)
        deltat = float(event_time_series.index[1] - event_time_series.index[0])
        if normalize and log:
            autocorrelation = mp.autocorrelate(
                a=event_values, m=m, deltat=deltat, normalize=True
            )
            self.tau, autocorrelation = np.transpose(autocorrelation[1:])

        elif log and not normalize:
            autocorrelation = mp.autocorrelate(
                a=event_values, m=m, deltat=deltat, normalize=False
            )
            self.tau, autocorrelation = np.transpose(autocorrelation[1:])

        elif normalize and not log:
            mean = np.mean(event_values)
            deviation = (
                event_values - mean
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
            autocorrelation = autocorrelation[1:]
            self.tau = np.arange(1, autocorrelation.size + 1, dtype=np.float64) * deltat

        else:
            autocorrelation = np.correlate(event_values, event_values, mode="full")
            # note that this version is the autocorrelation in the sense of signal
            # processing and differs from the statistical definition of autocorrelation.
            autocorrelation = autocorrelation[autocorrelation.size // 2 :][1:]
            self.tau = np.arange(1, autocorrelation.size + 1, dtype=np.float64) * deltat

        if normalize:
            autocorrelation = autocorrelation + 1

        self.autocorrelation = autocorrelation

        return self

    def plot_matplotlib(
        self,
        normalize_to: int | None = None,
        unit: Literal["s", "ms", "us"] = "s",
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

        if self.tau is None or self.autocorrelation is None:
            raise RuntimeError("correlation data has not been calculated.")

        tau_data = self.tau.copy()
        correl_data = self.autocorrelation.copy()
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
        self,
        normalize_to: int | None = None,
        unit: Literal["s", "ms", "us"] = "s",
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
        kwargs
            fluopy.plotting.universal_figure arguments

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.tau is None or self.autocorrelation is None:
            raise RuntimeError("correlation data has not been calculated.")

        tau_data = self.tau.copy()
        correl_data = self.autocorrelation.copy()
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit=unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault("title", rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("xscale", "log")
        kwargs.setdefault("xlabel", rf"$\tau \ ({unit})$")
        kwargs.setdefault("ylabel", r"$G(\tau)$")

        ax = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

        return ax


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
    autocorrelation : npt.NDArray[np.float64]
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
    s0_s1_cycle = 1 / (1 / s1_lifetime + excitation_rate)
    autocorrelation = -np.exp(-tau / s0_s1_cycle)

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
        FL).
    k_iso
        Rate constant of isomerization from trans to cis.
    k_biso_eff
        Rate constant of back isomerization from cis to trans.

    Returns
    -------
    autocorrelation : npt.NDArray[np.float64]
        Autocorrelation values.
    norm : float
        Steady state fraction of other states. Number between 0 and 1.

    Notes
    -----
    Implements the triplet-state and photoisomerization model described by
    `Widengren and Schwille (2000) <https://doi.org/10.1021/jp000059s>`_.
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


@numba.jit(nopython=True)
def _event_time_correlation(
    t: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    bins: npt.NDArray[np.float64],
    normalize: bool,
    start_time: float,
    end_time: float,
) -> npt.NDArray[np.float64]:
    """
    Correlate two sorted series of event times over arbitrary lag intervals.

    For every bin [bins[k], bins[k + 1]), the function counts pairs (t[i], u[j])
    whose delay u[j] - t[i] lies in that interval and divides the count by the
    interval width. Pair counts below successive bin edges are accumulated with a
    monotonic sweep through u; adjacent cumulative counts then give the count in
    each interval.

    Parameters
    ----------
    t
        Sorted event times for the first signal.
    u
        Sorted event times for the second signal.
    bins
        Increasing lag-bin edges in the same units as the event times.
    normalize
        Whether to normalize for event counts and the finite measurement duration.
    start_time
        Beginning of the measurement interval.
    end_time
        End of the measurement interval.

    Returns
    -------
    npt.NDArray[np.float64]
        Correlation values for the intervals between consecutive bin edges.

    Notes
    -----
    Implements the photon-pair counting formulation described by
    `Laurence et al. (2006) <https://doi.org/10.1364/OL.31.000829>`_.
    """
    pairs_before_edge = np.zeros(bins.size, dtype=np.int64)

    for edge_index in range(bins.size):
        u_index = 0
        pair_count = 0
        edge = bins[edge_index]
        for t_value in t:
            threshold = t_value + edge
            while u_index < u.size and u[u_index] < threshold:
                u_index += 1
            pair_count += u_index
        pairs_before_edge[edge_index] = pair_count

    correlation = np.diff(pairs_before_edge) / np.diff(bins)
    if not normalize:
        return correlation

    duration = end_time - start_time
    for bin_index in range(correlation.size):
        tau = bins[bin_index + 1]
        usable_u = u.size - np.searchsorted(u, start_time + tau, side="left")
        usable_t = np.searchsorted(t, end_time - tau, side="right")
        correlation[bin_index] *= (duration - tau) / (usable_u * usable_t)

    return correlation


def coincidence_numpy(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike,
    tau_max: float,
    bin_width: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the coincidence histogram of photon arrival times using NumPy. Based on
    binary search.

    Parameters
    ----------
    arr1
        Photon arrival times from the first detector.
    arr2
        Photon arrival times from the second detector.
    tau_max
        Maximum time difference to consider for the histogram.
        Shares units with arr1 and arr2.
    bin_width
        Width of the histogram bins.
        Shares units with arr1 and arr2.

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bins : npt.NDArray[np.float64]
        Bin edges of the histogram.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    bins = np.arange(-tau_max, tau_max + bin_width, bin_width)
    hist = np.zeros(bins.size - 1)

    for t in arr1:
        left = np.searchsorted(arr2, t - tau_max, side="left")
        right = np.searchsorted(arr2, t + tau_max, side="right")
        delays = arr2[left:right] - t
        h, _ = np.histogram(delays, bins=bins)
        hist += h

    return hist, bins


@numba.jit(nopython=True)
def coincidence_numba(
    arr1: npt.NDArray[np.float64],
    arr2: npt.NDArray[np.float64],
    tau_max: float,
    bin_width: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the coincidence histogram of photon arrival times using Numba. Based on
    two-pointer technique.

    Parameters
    ----------
    arr1
        Photon arrival times from the first detector.
    arr2
        Photon arrival times from the second detector.
    tau_max
        Maximum time difference to consider for the histogram.
        Shares units with arr1 and arr2.
    bin_width
        Width of the histogram bins.
        Shares units with arr1 and arr2.

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bins : npt.NDArray[np.float64]
        Bin edges of the histogram.
    """
    bins = np.arange(-tau_max, tau_max + bin_width, bin_width)
    hist = np.zeros(len(bins) - 1, dtype=np.float64)

    n1 = len(arr1)
    n2 = len(arr2)

    i2_start = 0
    i2_end = 0

    # e.g., t1 = 5, tau_max = 2
    # find arr2 elements in [3, 7]
    # pointers only move forward
    # pointer start will find first element >= 3
    # pointer end will find first element > 7
    # next t1 will be larger, so pointers only move forward

    for i in range(n1):
        t1 = arr1[i]

        while i2_start < n2 and arr2[i2_start] < t1 - tau_max:
            i2_start += 1

        if i2_end < i2_start:
            i2_end = i2_start
        while i2_end < n2 and arr2[i2_end] <= t1 + tau_max:
            i2_end += 1

        for j in range(i2_start, i2_end):
            delay = arr2[j] - t1
            bin_idx = int((delay + tau_max) / bin_width)

            if 0 <= bin_idx < len(hist):
                hist[bin_idx] += 1.0

    return hist, bins


def coincidence(
    photon_arrival_times: npt.NDArray[np.float64],
    tau_max: float,
    bin_width: float,
    seed: RandomGeneratorSeed = None,
    method: str = "numba",
    start_time: float = 0.0,
    end_time: float | None = None,
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
    start_time
        The time the measurement started. This is used to normalize to the correct
        measurement duration. Default is 0.0.
    end_time
        The time the measurement ended. This is used to normalize to the correct
        measurement duration. Default is None, which means the end time is the last
        element in the 'photon_arrival_times' array.

    Returns
    -------
    hist : npt.NDArray[np.float64]
        Coincidence histogram.
    bin_centers : npt.NDArray[np.float64]
        Bin centers of the histogram.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(photon_arrival_times.size) < 0.5
    arr1 = np.sort(photon_arrival_times[mask])
    arr2 = np.sort(photon_arrival_times[~mask])
    end_time = end_time if end_time is not None else photon_arrival_times.max()
    duration = end_time - start_time
    if method == "numpy":
        hist, bins = coincidence_numpy(
            arr1=arr1, arr2=arr2, tau_max=tau_max, bin_width=bin_width
        )
    elif method == "numba":
        hist, bins = coincidence_numba(
            arr1=arr1, arr2=arr2, tau_max=tau_max, bin_width=bin_width
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'numpy' or 'numba'.")

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # normalization to avoid finite window effects
    hist = hist / (duration - np.abs(bin_centers))
    # standard normalization
    # get the number of photons per time as if coming from a Poisson process
    average_signal_1 = arr1.size / duration if arr1.size > 0 else 0
    average_signal_2 = arr2.size / duration if arr2.size > 0 else 0
    if average_signal_1 > 0 and average_signal_2 > 0:
        hist = hist / (average_signal_1 * average_signal_2 * bin_width)

    return hist, bin_centers
