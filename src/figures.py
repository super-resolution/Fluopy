import src.custom_plot as cp
import numpy as np


class FigureCollection:
    """
    Collection of figure creating functions whose input data solely depends on system_object.

    Attributes
    ----------
    object : fluorophore_systems.FluorophoreSystem or its subclasses
        The input system_object (see __init__).
    """
    def __init__(self, system_object):
        """
        Parameters
        ----------
        system_object : fluorophore_systems.FluorophoreSystem or its subclasses
        """
        self.object = system_object

    def state_population(self, use_unique=True, **kwargs):
        """
        Shows the populations (e.g., as probability densities) of states during a Markov chain.

        Parameters
        ----------
        use_unique : bool
            Whether to use only unique states (e.g., S0_S1 and S1_S0 are combined).
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if use_unique:
            kwargs.setdefault('data', self.object.unique_series_converted)  # kwargs.setdefault does not overwrite a
            # potentially already set keyword argument but sets a default in case the keyword was not set
            kwargs.setdefault('xlim', [0, len(self.object.unique_states)])
            kwargs.setdefault('xticks', range(len(self.object.unique_states)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.unique_names, rotation=70))
            kwargs.setdefault('bins', np.arange(0, len(self.object.unique_states) + 1, 1))
        else:
            kwargs.setdefault('data', self.object.state_series)
            kwargs.setdefault('xlim', [0, len(self.object.Joined_States)])
            kwargs.setdefault('xticks', range(len(self.object.Joined_States)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.state_names, rotation=70))
            kwargs.setdefault('bins', np.arange(0, len(self.object.Joined_States) + 1, 1))

        kwargs.setdefault('density', True)
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('log', True)

        fig, ax = cp.universal_figure(type_="hist", **kwargs)

        return fig, ax

    def time_steps(self, only_emitting_transitions=True, **kwargs):
        """
        Shows the frequency of time steps between transitions.

        Parameters
        ----------
        only_emitting_transitions : bool
            Whether to use only emitting transitions.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if only_emitting_transitions:
            kwargs.setdefault('data', self.object.time_step_series[self.object.emitting_mask])
        else:
            kwargs.setdefault('data', self.object.time_step_series)

        kwargs.setdefault('density', True)
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('log', True)

        fig, ax = cp.universal_figure(type_="hist", **kwargs)

        return fig, ax

    def occupation_time(self, total=False, **kwargs):
        """
        Shows the occupation time of each state.

        Parameters
        ----------
        total : bool
            Whether to display the total or the mean occupation time of a state.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if total:
            kwargs.setdefault('data', [np.arange(len(self.object.unique_states)), self.object.occupation_time_total])
            kwargs.setdefault('ylabel', "Total time [s]")
        else:
            kwargs.setdefault('data', [np.arange(len(self.object.unique_states)), self.object.occupation_time_mean])
            kwargs.setdefault('ylabel', "Mean time [s]")

        kwargs.setdefault('yscale', "log")
        kwargs.setdefault('xlabel', None)
        kwargs.setdefault('xticks', range(len(self.object.unique_states)))
        kwargs.setdefault('xticklabels', dict(labels=self.object.unique_names, rotation=70))

        fig, ax = cp.universal_figure(type_="bar", **kwargs)

        return fig, ax

    def emission_events(self, time_series=True, **kwargs):
        """
        Shows the photon / emission counts.

        Parameters
        ----------
        time_series : bool
            Whether to display a time series or the frequency (i.e., histogram).
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if time_series:
            kwargs.setdefault('type_', "line")
            kwargs.setdefault('data', [self.object.pandas_series.index, self.object.pandas_series.values])
            kwargs.setdefault('ylabel', "Photon count")
            kwargs.setdefault('xlabel', "time [s]")
        else:
            kwargs.setdefault('type_', "hist")
            kwargs.setdefault('data', self.object.pandas_series[self.object.pandas_series != 0])
            kwargs.setdefault('ylabel', "PD")
            kwargs.setdefault('xlabel', "Photon counts")
            kwargs.setdefault('density', True)

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def fcs(self, log=True, **kwargs):
        """
        Shows the fluorescence correlation spectroscopy (FCS) curve with the autocorrelation as a function of time.

        Parameters
        ----------
        log : bool
            Set True if log was True in processing.autocorrelate.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if log:
            kwargs.setdefault('data', [self.object.autocorrelation[0][1:], self.object.autocorrelation[1][1:]])
        else:
            kwargs.setdefault('data', [self.object.autocorrelation[0][1:1000], self.object.autocorrelation[1][1:1000]])

        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', "time [s]")
        kwargs.setdefault('ylabel', "G(t)")

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def on_off(self, on=True, time_series=False, resample="5 ms", display_mean=True, **kwargs):
        """
        Shows the on or off periods.

        Parameters
        ----------
        on : bool
            Whether to display on periods or off periods.
        time_series : bool
            Whether to display a time series or the frequency (i.e., histogram).
        resample : str
            Set equal to the resample parameter of emitting_transitions.pandas_event_time_series.
        display_mean : bool
            Whether to display the mean value in the top right corner.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        if time_series:
            kwargs.setdefault('type_', "line")
            kwargs.setdefault('xlabel', "number")
            if on:
                kwargs.setdefault('data', [np.arange(0, self.object.on_periods.size), self.object.on_periods])
                kwargs.setdefault('ylabel', "ON periods [frames à " + resample + "]")
                mean = np.mean(self.object.on_periods)
            else:
                kwargs.setdefault('data', [np.arange(0, self.object.off_periods.size), self.object.off_periods])
                kwargs.setdefault('ylabel', "OFF periods [frames à " + resample + "]")
                mean = np.mean(self.object.off_periods)
        else:
            kwargs.setdefault('type_', "hist")
            kwargs.setdefault('ylabel', "PD")
            kwargs.setdefault('density', True)
            if on:
                kwargs.setdefault('data', self.object.on_periods)
                kwargs.setdefault('xlabel', "ON periods [frames à " + resample + "]")
                mean = np.mean(self.object.on_periods)
            else:
                kwargs.setdefault('data', self.object.off_periods)
                kwargs.setdefault('xlabel', "OFF periods [frames à " + resample + "]")
                mean = np.mean(self.object.off_periods)

        fig, ax = cp.universal_figure(**kwargs)
        if display_mean:
            ax[0].text(x=0.7, y=0.85, s=f"mean: {mean:.2f}", transform=ax[0].transAxes, fontsize=16)

        return fig, ax
