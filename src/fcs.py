import numpy as np
import multipletau as mp


def autocorrelate(pandas_series, normalize=False, log=False, m=2, deltat=5e-3):
    """
    Autocorrelation of pandas_series.values as exploited in typical fluorescence correlation spectroscopy setups.

    Parameters
    ----------
    pandas_series : pd.Series
        Contains the time step in seconds as index and the number of events as values.
    normalize : bool
        Whether to normalize the correlation. Note that this involves more than just dividing by mean squared.
    log : bool
        Whether to compute the autocorrelation on a logarithmic scale. Uses multipletau package.
    m : int
        Defines the number of points on one level (i.e., 1, 2, 4, 8, etc.), E.g., m=4 leads to 1,2,3,4; 2,4,6,8;
        4,8,12,16; ... .
        Only needed if log is True.
    deltat : float
        The time between each entry of pandas_series. Only needed if log is True.

    Returns
    -------
    tau : np.ndarray
        Time differences that correspond to the autocorrelation values.
    correl : np.ndarray
        Autocorrelation values.
    """
    if normalize and log:
        correl = mp.autocorrelate(pandas_series.values, m=m, deltat=deltat, normalize=True)
        tau, correl = np.transpose(correl)

        return tau, correl

    elif log and not normalize:
        correl = mp.autocorrelate(pandas_series.values, m=m, deltat=deltat)
        tau, correl = np.transpose(correl)

        return tau, correl

    elif normalize and not log:
        mean = np.mean(pandas_series.values)
        deviation = pandas_series.values - mean  # delta I(t) (wiki) - fluctuation around the mean value
        correl = np.correlate(deviation, deviation, mode="full")
        correl = correl[correl.size // 2:]
        correl = np.divide(correl, np.arange(correl.size, 0, -1))  # averaging - in multipletau, this is included
        # in normalize=True (denoted as M-k in documentation)
        correl = correl / mean ** 2  # normalization with mean squared
        tau = pandas_series.index.values

        return tau, correl

    else:
        correl = np.correlate(pandas_series.values, pandas_series.values, mode="full")  # note that this version is the
        # autocorrelation in the sense of signal processing and differs from the statistical definition of
        # autocorrelation.
        correl = correl[correl.size // 2:]
        tau = pandas_series.index.values

        return tau, correl
