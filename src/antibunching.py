import numpy as np
import pycorrelate as pc
import src.figure as fi
import pandas as pd


class Antibunching:
    def __init__(self, simulation, seed):
        self.simulation = simulation
        rng = np.random.default_rng(seed)
        emission_indices = self.get_emission_indices()
        rng.shuffle(emission_indices)
        self.first_channel = np.sort(self.simulation.time_series[emission_indices[:int(emission_indices.size/2)]])
        self.second_channel = np.sort(self.simulation.time_series[emission_indices[int(emission_indices.size/2):]])
        self.correlation = None
        self.tau = None

    def get_emission_indices(self):
        """
        Get indices to apply to simulation.transition_series to yield emitting transitions.

        Returns
        -------
        emission_indices : 1-D array_like
            Indices of emitting transitions to apply to simulation.transition_series.
        """
        df = self.simulation.transitions.combined_state_transitions_df
        emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
        emission_indices = np.in1d(self.simulation.transition_series, emitting_transition_ids).nonzero()[0]

        return emission_indices

    def correlate_time_points(self, exp_min=-8, exp_max=2, points_per_base=4, base=10, normalize=True):
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
            Whether to normalize the autocorrelation.

        Returns
        -------
        self
        """
        # generally much faster than autocorrelation based on time series
        if base ** exp_max > self.first_channel[-1]:
            raise ValueError('Base to the power of exp_max cannot be larger than the last time point.')
        bins = pc.make_loglags(exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base)
        self.correlation = pc.pcorrelate(t=self.first_channel, u=self.second_channel,
                                             bins=bins, normalize=normalize)
        self.tau = np.mean([bins[1:], bins[:-1]], 0)

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
        tau_data, correl_data = np.copy(self.tau), np.copy(self.correlation)
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit=unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', fr"$\tau \ [{unit}]$")
        kwargs.setdefault('ylabel', r"$G(\tau)$")

        axes = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

        return axes
