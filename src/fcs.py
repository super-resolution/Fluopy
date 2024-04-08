"""
Module fcs
"""

import numpy as np
import pandas as pd
import multipletau as mp
import pycorrelate as pc
import src.figure as fi


class FCS:
    """
    Container of FCS-assocciated attributes and methods.

    Attributes
    ----------
    emissions : src.emissions.Emissions
        Container for emission-associated attributes.
    autocorrelation : 1-D array_like
        Autocorrelation values.
    tau : 1-D array_like
        Time differences (i.e., τ, lag times).
    """

    def __init__(self, emissions):
        """
        Parameters
        ----------
        emissions : src.emissions.Emissions
            Container for emission-associated attributes.
        """
        self.emissions = emissions
        self.autocorrelation = None
        self.tau = None

    def autocorrelate_time_points(
        self, exp_min=-8, exp_max=2, points_per_base=4, base=10, normalize=True
    ):
        """
        Autocorrelation of emissions.event_time_points. Generally much faster than
        autocorrelation based on emissions.event_time_series.
        Based on https://opg.optica.org/ol/abstract.cfm?uri=ol-31-6-829.

        Parameters
        ----------
        exp_min : int
            Exponent of the minimum value.
        exp_max : int
            Exponent of the maximum value.
        points_per_base : int
            Number of points per base.
        base : int
            The base of the exponentiation.
        normalize : bool
            Whether to normalize the autocorrelation.

        Returns
        -------
        self
        """
        if self.emissions.event_time_points is None:
            raise ValueError("event_time_points is None.")
        if base**exp_max > self.emissions.event_time_points[-1]:
            raise ValueError(
                "Base to the power of exp_max cannot be larger than the last time "
                "point."
            )
        bins = pc.make_loglags(
            exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base
        )
        self.autocorrelation = pc.pcorrelate(
            t=self.emissions.event_time_points,
            u=self.emissions.event_time_points,
            bins=bins,
            normalize=normalize,
        )
        self.tau = np.mean([bins[1:], bins[:-1]], 0)

        return self

    def autocorrelate_time_series(self, log=True, m=4, normalize=True):
        """
        Autocorrelation of emissions.event_time_series. The minimum lag time is equal
        to resample value of series.

        Parameters
        ----------
        log : bool
            Whether to compute the autocorrelation on a logarithmic scale. As time
            steps increase, correlation signals are getting noisier, fluctuating around
            0. Hence, log should usually be True.
        m : int
            Defines the number of points on each log level. E.g., m=4 leads to
            |1, 2, 3, 4| |2, 4, 6, 8| |4, 8, 12, 16| ..., hence
            |1, 2, 3, 4, 6, 8, 12, 16, ...|. Only used if log ist True.
        normalize : bool
            Whether to normalize the autocorrelation.

        Returns
        -------
        self
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

    def plot(self, normalize_to=None, unit="s", **kwargs):
        """
        Plot FCS data.

        Parameters
        ----------
        normalize_to : None, int
            Index of datapoint to which the data is normalized.
        unit : str
            One of 's', 'ms', 'us'. Influences the unit of the x-axis.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
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
        kwargs.setdefault("xlabel", rf"$\tau \ [{unit}]$")
        kwargs.setdefault("ylabel", r"$G(\tau)$")

        axes = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

        return axes


def fit_dark(tau, dark_lifetime, dark_occupation):
    """
    Fit function of dark states (e.g., triplet).

    Parameters
    ----------
    tau : 1-D array_like
        Time differences (i.e., τ, lag times).
    dark_lifetime : float
        Mean lifetime of the dark state.
    dark_occupation : float
        Steady state fraction of the dark state. Number between 0 and 1.

    Returns
    -------
    autocorrelation : 1-D array_like
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


def fit_antibunching(tau, excitation_rate, s1_lifetime):
    """
    Fit function of antibunching.

    Parameters
    ----------
    tau : 1-D array_like
        Time differences (i.e., τ, lag times).
    excitation_rate : float
        Rate constant of excitation.
    s1_lifetime : float
        Mean lifetime of the S1 state.

    Returns
    -------
    autocorrelation : 1-D array_like
        Autocorrelation values.
    """
    tau = np.asarray(tau)
    lifetime_s0_s1 = 1 / (1 / s1_lifetime + excitation_rate)
    autocorrelation = -np.exp(-tau / lifetime_s0_s1)

    return autocorrelation


def fit_triplet_cis(tau, k_isc, k_T, k_01, k_10, k_iso, k_biso_eff):
    """
    Fit function of triplet and cis as two non-independent dark states.

    Parameters
    ----------
    tau : 1-D array_like
        Time differences (i.e., τ, lag times).
    k_isc : float
        Rate constant of intersystem crossing to the triplet state.
    k_T : float
        Rate constant of intersystem crossing out of the triplet state.
    k_01 : float
        Rate constant of excitation.
    k_10 : float
        Inverse of fluorescence lifetime considering all rates from S1 (not just IC and
        FL). Note: in the PAPER it is not clear which one they mean but the fit is
        significantly better if using this version.
    k_iso : float
        Rate constant of isomerization from trans to cis.
    k_biso_eff : float
        Rate constant of back isomerization from cis to trans.

    Returns
    -------
    autocorrelation : 1-D array_like
        Autocorrelation values.
    norm : float
        Steady state fraction of other states. Number between 0 and 1.
    """
    tau = np.asarray(tau)
    k_isc_eff = k_01 / (k_01 + k_10) * k_isc
    k_iso_eff = k_01 / (k_01 + k_10) * k_iso

    eigen_1 = 0

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
