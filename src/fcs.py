import numpy as np
import multipletau as mp


def autocorrelate(event_time_series, normalize=False, log=False, m=2, deltat=5e-3):
    """
    Autocorrelation of event_time_series.values as exploited in typical fluorescence correlation spectroscopy setups.

    Parameters
    ----------
    event_time_series : pd.Series
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
        The time between each entry of event_time_series. Only needed if log is True.

    Returns
    -------
    tau : np.ndarray
        Time differences (i.e., lag times) that correspond to the autocorrelation values.
    autocorrelation : np.ndarray
        Autocorrelation values.
    """
    if normalize and log:
        autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat, normalize=True)
        tau, autocorrelation = np.transpose(autocorrelation)

        return tau, autocorrelation

    elif log and not normalize:
        autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat)
        tau, autocorrelation = np.transpose(autocorrelation)

        return tau, autocorrelation

    elif normalize and not log:
        mean = np.mean(event_time_series.values)
        deviation = event_time_series.values - mean  # delta I(t) (wiki) - fluctuation around the mean value
        autocorrelation = np.correlate(deviation, deviation, mode="full")
        autocorrelation = autocorrelation[autocorrelation.size // 2:]
        autocorrelation = np.divide(autocorrelation, np.arange(autocorrelation.size, 0, -1))  # averaging - in multipletau, this is included
        # in normalize=True (denoted as M-k in documentation)
        autocorrelation = autocorrelation / mean ** 2  # normalization with mean squared
        tau = event_time_series.index.values

        return tau, autocorrelation

    else:
        autocorrelation = np.correlate(event_time_series.values, event_time_series.values, mode="full")  # note that this version is the
        # autocorrelation in the sense of signal processing and differs from the statistical definition of
        # autocorrelation.
        autocorrelation = autocorrelation[autocorrelation.size // 2:]
        tau = event_time_series.index.values

        return tau, autocorrelation
