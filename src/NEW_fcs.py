import numpy as np
import pandas as pd
import multipletau as mp
import pycorrelate as pc
import src.custom_plot as cp


class FCS:
    def __init__(self):
        self.autocorrelation = None
        self.tau = None

    def autocorrelate_time_points(self, event_time_points, exp_min=-8, exp_max=2, points_per_base=4, base=10,
                                  normalize=True):
        # generally much faster than autocorrelation based on time series
        bins = pc.make_loglags(exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base)
        self.autocorrelation = pc.pcorrelate(event_time_points, event_time_points, bins=bins, normalize=normalize)
        self.tau = np.mean([bins[1:], bins[:-1]], 0)

        return self.autocorrelation, self.tau

    def autocorrelate_time_series(self, event_time_series, log=True, m=4, deltat=5e-3, normalize=True):
        if normalize and log:
            autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat, normalize=True)
            self.tau, self.autocorrelation = np.transpose(autocorrelation[1:])

        elif log and not normalize:
            autocorrelation = mp.autocorrelate(event_time_series.values, m=m, deltat=deltat)
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

        return self.autocorrelation, self.tau

    def plot(self, normalize_to=None, unit='s', **kwargs):
        tau_data, correl_data = np.copy(self.tau), np.copy(self.autocorrelation)
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', fr"$\tau [{unit}]$")
        kwargs.setdefault('ylabel', r"$G(\tau)$")

        fig, ax = cp.universal_figure(data=[tau_data, correl_data], **kwargs)

        return fig, ax
