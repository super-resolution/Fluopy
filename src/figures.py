import numpy as np
import pandas as pd
import src.custom_plot as cp
import src.helper as hp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


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

    def network(self, seed):
        """
        Shows photophysical system as graph.
        Adapted from
        https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx

        Parameters
        ----------
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        g = self.object.graph
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g, seed)
        nx.draw_networkx_nodes(g, pos, ax=ax)
        nx.draw_networkx_labels(g, pos, ax=ax)
        curved_edges = []
        for edge in g.edges:
            if list(reversed(edge[:2])) + [edge[2]] in g.edges:
                curved_edges.append(edge)
        straight_edges = list(set(g.edges) - set(curved_edges))
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=straight_edges)
        arc_rad = 0.25
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
        edge_weights = nx.get_edge_attributes(g, 'w')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        hp.draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
        hp.draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, rad=0)

        return fig, ax

    def populations(self, mode="single_states", single_fluorophores=True, **kwargs):
        """
        Shows frequency of single states or transitions.

        Parameters
        ----------
        mode : str
            One of "single_states" (e.g., S0, S1, ...), "transitions" (e.g., excitation, ...).
        single_fluorophores : bool
            Whether to show the result for each fluorophore individually or summarized.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        kwargs.setdefault('type_', "hist")
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('xlabel', None)
        kwargs.setdefault('density', True)
        kwargs.setdefault('yscale', "log")
        if single_fluorophores:
            colors = cm.rainbow(np.linspace(0, 1, self.object.number))
            kwargs.setdefault('color', colors)
            key1, key2 = 'single_state_occurrences', 'transition_occurrences'
        else:
            key1, key2 = 'single_state_occurrences_all', 'transition_occurrences_all'
        if mode == "single_states":
            kwargs.setdefault('data', self.object.single_state_lifetimes[key1])
            kwargs.setdefault('xticks', range(len(self.object.single_states)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.single_states, rotation=70))
        elif mode == "transitions":
            labels = [' '.join([y[:3] for y in x.split(" ")]) for x in self.object.transition_dict.values()]
            kwargs.setdefault('data', self.object.transition_lifetimes[key2])
            kwargs.setdefault('xticks', range(len(self.object.rates)))
            kwargs.setdefault('xticklabels', dict(labels=labels, rotation=70))

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def lifetimes(self, mode="single_states", statistic="mean", single_fluorophores=True, **kwargs):
        """
        Shows all single state lifetimes or all times to transition. In case of single states, either means or totals.

        Parameters
        ----------
        mode : str
            One of "single_states" (e.g., S0, S1, ...), "transitions" (e.g., excitation, ...).
        statistic : str
            One of "mean", "total".
        single_fluorophores : bool
            Whether to show the result for each fluorophore individually or summarized.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        kwargs.setdefault('type_', "bar")
        kwargs.setdefault('yscale', "log")
        kwargs.setdefault('xlabel', None)
        if single_fluorophores:
            colors = cm.rainbow(np.linspace(0, 1, self.object.number))
            kwargs.setdefault('color', colors)
            kwargs.setdefault('width', 1/self.object.number)
            key1, key2, key3 = 'mean_lifetimes', 'total_lifetimes', 'mean_transition_times'
        else:
            key1, key2, key3 = 'mean_lifetimes_all', 'total_lifetimes_all', 'mean_transition_times_all'

        if mode == "single_states":
            kwargs.setdefault('xticks', range(len(self.object.single_states)))
            kwargs.setdefault('xticklabels', dict(labels=self.object.single_states, rotation=70))
            if statistic == "mean":
                kwargs.setdefault('data', [np.arange(len(self.object.single_states)),
                                           self.object.single_state_lifetimes[key1]])
                kwargs.setdefault('ylabel', "Mean lifetime [s]")
            elif statistic == "total":
                kwargs.setdefault('data', [np.arange(len(self.object.single_states)),
                                           self.object.single_state_lifetimes[key2]])
                kwargs.setdefault('ylabel', "Total lifetime [s]")
        elif mode == "transitions":
            labels = [' '.join([y[:3] for y in x.split(" ")]) for x in self.object.transition_dict.values()]
            kwargs.setdefault('xticks', range(len(self.object.rates)))
            kwargs.setdefault('xticklabels', dict(labels=labels, rotation=70))
            kwargs.setdefault('data', [np.arange(len(self.object.rates)),
                                       self.object.transition_lifetimes[key3]])
            kwargs.setdefault('ylabel', "Mean time to transition [s]")

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def individual_lifetimes(self, fluorophore_id=None, single_state_id=None,
                             transition_id=None, **kwargs):
        """
        Shows probability densities of the lifetimes of a particular state or times until particular transition occurs.
        Note that each transition starting from the same state (no matter its rate) will, on average, occur after
        the same time has elapsed:
        tau_state = 1 / sum(k_all_outgoing_transitions).

        fluorophore_id : None, int
            The id of the fluorophore of interest. Cannot be looked up. If None, all fluorophores are summarized.
        single_state_id : None, int
            The id of the single state of interest. Can be looked up in single_state_id.
        transition_id : None, int
            The id of the transition of interest. Can be looked up in transition_dict.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        kwargs.setdefault('type_', "hist")
        kwargs.setdefault('ylabel', "PD")
        kwargs.setdefault('yscale', "log")
        index2_1, index2_2 = single_state_id, transition_id
        if fluorophore_id is not None:
            key1, key2 = 'lifetimes_single_states', 'transition_times'
            index1 = fluorophore_id
        else:
            key1, key2 = 'lifetimes_single_states_all', 'transition_times_all'
            index1 = None
        if index2_1 is not None and index2_2 is not None:
            raise ValueError("Only one of single_state_id or transition_id must be defined.")
        elif index2_1 is None and index2_2 is None:
            raise ValueError("One of single_state_id or transition_id has to be defined.")
        if index2_1 is not None:
            kwargs.setdefault('xlabel', f"Lifetimes of {self.object.single_states[single_state_id]} [s]")
            if index1 is not None:
                kwargs.setdefault('data', (self.object.single_state_lifetimes[key1][index1][index2_1]))
            else:
                kwargs.setdefault('data', (self.object.single_state_lifetimes[key1][index2_1]))
        else:
            kwargs.setdefault('xlabel', f"Time until {self.object.transition_dict[transition_id]} [s]")
            if index1 is not None:
                kwargs.setdefault('data', (self.object.transition_lifetimes[key2][index1][index2_2]))
            else:
                kwargs.setdefault('data', (self.object.transition_lifetimes[key2][index2_2]))

        fig, ax = cp.universal_figure(**kwargs)

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
        kwargs.setdefault('title', f"frames: {self.object.last_parameters['resample']}")
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

    def fcs(self, log=True, normalize_to=None, unit="s", **kwargs):
        """
        Shows the fluorescence correlation spectroscopy (FCS) curve with the autocorrelation as a function of time.

        Parameters
        ----------
        log : bool
            Set True if log was True in processing.autocorrelate.
        normalize_to : None, int
            Index of datapoint to which the data is normalized.
        unit : str
            One of "s", "ms", "us".
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        tau_data, correl_data = np.copy(self.object.autocorrelation[0]), np.copy(self.object.autocorrelation[1])
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[1]:.2e}$ {unit}")
        if log:
            kwargs.setdefault('data', [tau_data[1:], correl_data[1:]])
        else:
            kwargs.setdefault('data', [tau_data[1:1000], correl_data[1:1000]])

        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', fr"$\tau [{unit}]$")
        kwargs.setdefault('ylabel', r"$G(\tau)$")

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def on_off(self, mode="on", time_series=False, display_mean=True, **kwargs):
        """
        Shows the on or off periods.

        Parameters
        ----------
        mode : str
            One of "on", "off", "mesh".
        time_series : bool
            Whether to display a time series or the frequency (i.e., histogram).
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
        kwargs.setdefault('title', f"frames: {self.object.last_parameters['resample']}")
        if time_series:
            kwargs.setdefault('xlabel', "number")
            if mode == "mesh":
                kwargs.setdefault('type_', "multiple_line")
                kwargs.setdefault('data', [[np.arange(0, self.object.on_periods.size*2, 2), self.object.on_periods],
                                  [np.arange(1, self.object.off_periods.size*2, 2), self.object.off_periods]])
                colors = cm.rainbow(np.linspace(0, 1, 2))
                kwargs.setdefault('color', colors)
                kwargs.setdefault('ylabel', "ON/OFF periods")
            elif mode == "on":
                kwargs.setdefault('type_', "line")
                kwargs.setdefault('data', [np.arange(0, self.object.on_periods.size), self.object.on_periods])
                kwargs.setdefault('ylabel', "ON periods")
            else:
                kwargs.setdefault('type_', "line")
                kwargs.setdefault('data', [np.arange(0, self.object.off_periods.size), self.object.off_periods])
                kwargs.setdefault('ylabel', "OFF periods")

        else:
            kwargs.setdefault('ylabel', "PD")
            kwargs.setdefault('density', True)
            if mode == "mesh":
                kwargs.setdefault('type_', "multiple_hist")
                kwargs.setdefault('data', [self.object.on_periods, self.object.off_periods])
                kwargs.setdefault('xlabel', "ON/OFF periods")
                colors = cm.rainbow(np.linspace(0, 1, 2))
                kwargs.setdefault('color', colors)
            elif mode == "on":
                kwargs.setdefault('type_', "hist")
                kwargs.setdefault('data', self.object.on_periods)
                kwargs.setdefault('xlabel', "ON periods")
            else:
                kwargs.setdefault('type_', "hist")
                kwargs.setdefault('data', self.object.off_periods)
                kwargs.setdefault('xlabel', "OFF periods")

        fig, ax = cp.universal_figure(**kwargs)
        if display_mean:
            if mode == "mesh":
                mean_on = np.mean(self.object.on_periods)
                mean_off = np.mean(self.object.off_periods)
                ax[0].text(x=0.7, y=0.85, s=f"ON mean: {mean_on:.2f}", transform=ax[0].transAxes, fontsize=16)
                ax[0].text(x=0.7, y=0.75, s=f"OFF mean: {mean_off:.2f}", transform=ax[0].transAxes, fontsize=16)
            else:
                if mode == "on":
                    mean = np.mean(self.object.on_periods)
                else:
                    mean = np.mean(self.object.off_periods)
                ax[0].text(x=0.7, y=0.85, s=f"mean: {mean:.2f}", transform=ax[0].transAxes, fontsize=16)

        return fig, ax
