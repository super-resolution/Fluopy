"""Contains functions related to fluorescence correlation spectroscopy."""
import numpy as np
import multipletau as mp
from pycorrelate import pcorrelate, make_loglags


def autocorrelate_time_series(event_time_series, normalize=False, log=False, m=2, deltat=5e-3):
    """
    Autocorrelation of event_time_series.values as exploited in typical fluorescence correlation spectroscopy setups.

    Parameters
    ----------
    event_time_series : pd.Series
        The return value of emitting_transitions.construct_event_time_series.
        Contains the time points in seconds as index and the number of events as values.
    normalize : bool
        Whether to normalize the correlation. Note that this involves more than just dividing by mean squared.
    log : bool
        Whether to compute the autocorrelation on a logarithmic scale. Uses multipletau package.
    m : int
        Defines the number of points on one level (i.e., 1, 2, 4, 8, etc.), E.g., m=4 leads to 1,2,3,4; 2,4,6,8;
        4,8,12,16; ... .
        Only needed if log is True.
    deltat : float
        The time between each entry of event_time_series. In seconds.
        Only needed if log is True.

    Returns
    -------
    tau : np.ndarray
        Time differences (i.e., tau or lag times) that correspond to the autocorrelation values.
    autocorrelation : np.ndarray
        Autocorrelation values.
    """
    if normalize and log:
        autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat, normalize=True)
        tau, autocorrelation = np.transpose(autocorrelation)

    elif log and not normalize:
        autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat)
        tau, autocorrelation = np.transpose(autocorrelation)

    elif normalize and not log:
        mean = np.mean(event_time_series.values)
        deviation = (event_time_series.values - mean)  # delta I(t) (wiki) - fluctuation around the mean value
        autocorrelation = np.correlate(deviation, deviation, mode="full")
        autocorrelation = autocorrelation[autocorrelation.size // 2:]
        autocorrelation = np.divide(autocorrelation, np.arange(autocorrelation.size, 0, -1))
        # averaging - in multipletau, this is included in normalize=True (denoted as M-k in documentation)
        autocorrelation = autocorrelation / mean**2  # normalization with mean squared
        tau = event_time_series.index.values

    else:
        autocorrelation = np.correlate(event_time_series.values, event_time_series.values, mode="full")
        # note that this version is the autocorrelation in the sense of signal processing and differs from the
        # statistical definition of autocorrelation.
        autocorrelation = autocorrelation[autocorrelation.size // 2:]
        tau = event_time_series.index.values

    return tau, autocorrelation


def autocorrelate_time_points(time_points_events, exp_min, exp_max, points_per_base=4, base=10, normalize=True):
    """
    Autocorrelation of time_points_events as exploited in typical fluorescence correlation spectroscopy setups.

    Parameters
    ----------
    time_points_events : np.ndarray
        Contains the time points at which an event occurs.
    exp_min : int
        Exponent of the minimum value.
    exp_max : int
        Exponent of the maximum value.
    points_per_base : int
        Number of points per base.
    base : int
        The base of the exponent
    normalize : bool
        Whether to normalize the correlation.

    Returns
    -------
    tau : np.ndarray
        Time differences (i.e., tau or lag times) that correspond to the autocorrelation values.
    autocorrelation : np.ndarray
        Autocorrelation values.
    """
    bins = make_loglags(exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base)
    autocorrelation = pcorrelate(time_points_events, time_points_events, bins=bins, normalize=normalize)
    tau = np.mean([bins[1:], bins[:-1]], 0)

    return tau, autocorrelation
