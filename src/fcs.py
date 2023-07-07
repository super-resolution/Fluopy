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
        self.emissions = emissions
        self.autocorrelation = None
        self.tau = None

    def autocorrelate_time_points(self, exp_min=-8, exp_max=2, points_per_base=4, base=10, normalize=True):
        """
        Autocorrelation of emissions.event_time_points.
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
            Whether to normalize the correlation.

        Returns
        -------
        self
        """
        # generally much faster than autocorrelation based on time series
        bins = pc.make_loglags(exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base)
        self.autocorrelation = pc.pcorrelate(t=self.emissions.event_time_points, u=self.emissions.event_time_points,
                                             bins=bins, normalize=normalize)
        self.tau = np.mean([bins[1:], bins[:-1]], 0)

        return self

    def autocorrelate_time_series(self, log=True, m=4, normalize=True):
        """
        Autocorrelation of emissions.event_time_series.

        Parameters
        ----------
        log : bool
            Whether to compute the autocorrelation on a logarithmic scale.
        m : int
            Defines the number of points on each level. E.g., m=4 leads to |1, 2, 3, 4| |2, 4, 6, 8| |4, 8, 12, 16| ...,
            hence |1, 2, 3, 4, 6, 8, 12, 16, ...|.
        normalize : bool
            Whether to normalize the autocorrelation. Normalization is

        Returns
        -------
        self
        """
        event_time_series = self.emissions.event_time_series
        deltat = event_time_series.index[1]
        if normalize and log:
            autocorrelation = mp.autocorrelate(a=event_time_series.values, m=m, deltat=deltat, normalize=True)
            self.tau, self.autocorrelation = np.transpose(autocorrelation[1:])

        elif log and not normalize:
            autocorrelation = mp.autocorrelate(a=event_time_series.values, m=m, deltat=deltat, normalize=False)
            self.tau, self.autocorrelation = np.transpose(autocorrelation[1:])

        elif normalize and not log:
            mean = np.mean(event_time_series.values)
            deviation = (event_time_series.values - mean)  # delta I(t) (wiki) - fluctuation around the mean value
            autocorrelation = np.correlate(deviation, deviation, mode="full")
            autocorrelation = autocorrelation[autocorrelation.size // 2:]
            autocorrelation = np.divide(autocorrelation, np.arange(autocorrelation.size, 0, -1))
            # averaging - in multipletau, this is included in normalize=True (denoted as M-k in documentation)
            autocorrelation = autocorrelation / mean ** 2  # normalization with mean squared
            self.autocorrelation = autocorrelation[1:1000]
            self.tau = event_time_series.index.values[1:1000]

        else:
            autocorrelation = np.correlate(event_time_series.values, event_time_series.values, mode="full")
            # note that this version is the autocorrelation in the sense of signal processing and differs from the
            # statistical definition of autocorrelation.
            self.autocorrelation = autocorrelation[autocorrelation.size // 2:][1:1000]
            self.tau = event_time_series.index.values[1:1000]

        return self

    def plot(self, normalize_to=None, unit='s', **kwargs):
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
        kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', fr"$\tau [{unit}]$")
        kwargs.setdefault('ylabel', r"$G(\tau)$")

        axes = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

        return axes
