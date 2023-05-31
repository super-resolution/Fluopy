import numpy as np
import pandas as pd
import src.custom_plot as cp
import src.miscellaneous as mi
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class FigureCollection:
    """
    Collection of figure-creating functions whose input data solely depends on system.

    Attributes
    ----------
    system : fluorophore_systems.FluorophoreSystem or its subclasses
        The input system (see __init__).
    """
    def __init__(self, system):
        """
        Parameters
        ----------
        system : fluorophore_systems.FluorophoreSystem or its subclasses
        """
        self.system = system

    def network(self, type='shell'):
        """
        Shows photophysical system as graph.
        Adapted from
        https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx.

        Parameters
        ----------
        type : str
            Specifies networkx layout. One of 'shell', 'planar', 'circular', 'kamada'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        g = self.system.graph
        fig, ax = plt.subplots()
        if type == 'circular':
            pos = nx.circular_layout(g)
        elif type == 'planar':
            pos = nx.planar_layout(g)
        elif type == 'shell':
            pos = nx.shell_layout(g)
        else:
            pos = nx.kamada_kawai_layout(g)

        labels = {}
        colormap = []
        for i, node in enumerate(g):
            if '(2)' in node:
                colormap.append('red')
                labels[node] = node.replace('(2)', '')
            else:
                colormap.append('blue')
                labels[node] = node
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colormap)
        nx.draw_networkx_labels(g, pos, ax=ax, labels=labels)

        edge_weights = nx.get_edge_attributes(g, 'w')
        straight_edges = []
        arc_rad = 0
        arc_rad_reversed = 0
        for new_edge in g.edges:
            nothing_found = True
            for old_edge in straight_edges:
                if new_edge[:2] == old_edge[:2]:
                    arc_rad += 0.25
                    nothing_found = False
                    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[new_edge],
                                           connectionstyle=f'arc3, rad = {arc_rad}')
                    mi.draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                        rotate=False, rad=arc_rad)
                    break
                elif list(reversed(new_edge[:2])) == list(old_edge[:2]):
                    arc_rad_reversed += 0.25
                    nothing_found = False
                    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[new_edge],
                                           connectionstyle=f'arc3, rad = {arc_rad_reversed}')
                    mi.draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                        rotate=False, rad=arc_rad_reversed)
                    break
            if nothing_found:
                arc_rad = 0
                arc_rad_reversed = 0
                straight_edges.append(new_edge)
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=straight_edges)
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        mi.draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, rad=0)

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
            colors = cm.rainbow(np.linspace(0, 1, self.system.parameter_collection.loc[
                ('init', 'number_fluorophores'), 0]))
            kwargs.setdefault('color', colors)
            kwargs.setdefault('legend', True)
            kwargs.setdefault('label', ['fluo_' + f'{id}' for id in
                                        np.arange(0, self.system.parameter_collection.loc[
                                            ('init', 'number_fluorophores'), 0], 1)])
            key1, key2 = 'single_state_occurrences', 'transition_occurrences'
        else:
            key1, key2 = 'single_state_occurrences_all', 'transition_occurrences_all'
            kwargs.setdefault('edgecolor', 'black')
        if mode == "single_states":
            kwargs.setdefault('title', 'single state population')
            kwargs.setdefault('data', self.system.single_state_lifetimes[key1])
            kwargs.setdefault('xticks', range(len(self.system.single_states)))
            kwargs.setdefault('bins', np.arange(0, len(self.system.single_states)+1)-0.5)
            kwargs.setdefault('xticklabels', dict(labels=self.system.single_states.values(), rotation=70))
            if self.system.single_state_lifetimes['single_state_occurrences_pred'] is not None:
                kwargs.setdefault('draw_marker', [range(len(self.system.single_states)),
                                                  self.system.single_state_lifetimes['single_state_occurrences_pred']])
        elif mode == "transitions":
            labels = self.system.unique_transitions['abbreviation'].values
            kwargs.setdefault('title', 'transition occurrences')
            kwargs.setdefault('data', self.system.transition_lifetimes[key2])
            kwargs.setdefault('xticks', range(self.system.unique_transitions.index.size))
            kwargs.setdefault('bins', np.arange(0, self.system.unique_transitions.index.size+1)-0.5)
            kwargs.setdefault('xticklabels', dict(labels=labels, rotation=90, va='bottom'))
            kwargs.setdefault('tick_params', dict(axis='x', direction='in', pad=-10))
            if self.system.transition_lifetimes['transition_occurrences_pred'] is not None:
                kwargs.setdefault('draw_marker', [range(self.system.unique_transitions.index.size),
                                                  self.system.transition_lifetimes['transition_occurrences_pred']])

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
            colors = cm.rainbow(np.linspace(0, 1,
                                            self.system.parameter_collection.loc[('init', 'number_fluorophores'), 0]))
            kwargs.setdefault('color', colors)
            kwargs.setdefault('width', 1/self.system.parameter_collection.loc[('init', 'number_fluorophores'), 0])
            kwargs.setdefault('legend', True)
            kwargs.setdefault('label', ['fluo_' + f'{id}' for id in
                                        np.arange(0, self.system.parameter_collection.loc[
                                            ('init', 'number_fluorophores'), 0], 1)])
            key1, key2, key3 = 'mean_lifetimes', 'total_lifetimes', 'mean_transition_times'
        else:
            key1, key2, key3 = 'mean_lifetimes_all', 'total_lifetimes_all', 'mean_transition_times_all'

        if mode == "single_states":
            kwargs.setdefault('title', 'single state lifetimes')
            kwargs.setdefault('xticks', range(len(self.system.single_states)))
            kwargs.setdefault('xticklabels', dict(labels=self.system.single_states.values(), rotation=70))
            if statistic == "mean":
                kwargs.setdefault('data', [np.arange(len(self.system.single_states)),
                                           self.system.single_state_lifetimes[key1]])
                kwargs.setdefault('ylabel', "mean [s]")
                if self.system.single_state_lifetimes['mean_lifetimes_pred'] is not None:
                    kwargs.setdefault('draw_marker', [np.arange(len(self.system.single_states)),
                                                      self.system.single_state_lifetimes['mean_lifetimes_pred']])
            elif statistic == "total":
                kwargs.setdefault('data', [np.arange(len(self.system.single_states)),
                                           self.system.single_state_lifetimes[key2]])
                kwargs.setdefault('ylabel', "total [s]")
                if self.system.single_state_lifetimes['total_lifetimes_pred'] is not None:
                    kwargs.setdefault('draw_marker', [np.arange(len(self.system.single_states)),
                                                      self.system.single_state_lifetimes['total_lifetimes_pred']])
        elif mode == "transitions":
            kwargs.setdefault('title', 'time to transition')
            labels = self.system.unique_transitions['abbreviation'].values
            kwargs.setdefault('xticks', range(self.system.unique_transitions.index.size))
            kwargs.setdefault('xticklabels', dict(labels=labels, rotation=90, va='bottom'))
            kwargs.setdefault('data', [np.arange(self.system.unique_transitions.index.size),
                                       self.system.transition_lifetimes[key3]])
            kwargs.setdefault('ylabel', "mean [s]")
            kwargs.setdefault('tick_params', dict(axis='x', direction='in', pad=-10))
            if self.system.transition_lifetimes['mean_transition_times_pred'] is not None:
                kwargs.setdefault('draw_marker', [range(self.system.unique_transitions.index.size),
                                                  self.system.transition_lifetimes['mean_transition_times_pred']])

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def individual_lifetimes(self, fluorophore_id=None, single_state_id=None, transition_id=None, **kwargs):
        """
        Shows probability densities of the lifetimes of a particular state or times until particular transition occurs.

        Parameters
        ----------
        fluorophore_id : None, int
            The id of the fluorophore of interest. Cannot be looked up. If None, all fluorophores are summarized.
        single_state_id : None, int
            The id of the single state of interest. Can be looked up in single_states (keys).
        transition_id : None, int
            The id of the transition of interest. Can be looked up in unique_transitions (index).
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
        kwargs.setdefault('density', True)
        kwargs.setdefault('label', 'data')
        kwargs.setdefault('legend', True)
        if fluorophore_id is not None:
            key1, key2 = 'lifetimes_single_states', 'transition_times'
            index1 = fluorophore_id
        else:
            key1, key2 = 'lifetimes_single_states_all', 'transition_times_all'
            index1 = None
        if single_state_id is not None and transition_id is not None:
            raise ValueError("Only one of single_state_id or transition_id must be defined.")
        elif single_state_id is None and transition_id is None:
            raise ValueError("One of single_state_id or transition_id has to be defined.")
        if single_state_id is not None:
            kwargs.setdefault('title', 'lifetime distribution')
            if index1 is not None:
                kwargs.setdefault('data', (self.system.single_state_lifetimes[key1][index1][single_state_id]))
                kwargs.setdefault('xlabel', f"{self.system.single_states[single_state_id]} of fluo_{index1} [s]")
            else:
                kwargs.setdefault('data', (self.system.single_state_lifetimes[key1][single_state_id]))
                kwargs.setdefault('xlabel', f"{self.system.single_states[single_state_id]} [s]")
            if self.system.single_state_lifetimes['lifetimes_single_states_pred'] is not None:
                kwargs.setdefault('plot_distribution',
                                  [self.system.single_state_lifetimes['lifetimes_single_states_pred'][single_state_id],
                                   'pred'])
        else:
            kwargs.setdefault('title', 'time to transition distribution')
            if index1 is not None:
                kwargs.setdefault('data', (self.system.transition_lifetimes[key2][index1][transition_id]))
                kwargs.setdefault('xlabel', f'{self.system.unique_transitions.loc[transition_id, "abbreviation"]} '
                                            f'of fluo_{index1} [s]')
            else:
                kwargs.setdefault('data', (self.system.transition_lifetimes[key2][transition_id]))
                kwargs.setdefault('xlabel', f'{self.system.unique_transitions.loc[transition_id, "abbreviation"]} [s]')
            if self.system.transition_lifetimes['transition_times_pred'] is not None:
                kwargs.setdefault('plot_distribution',
                                  [self.system.transition_lifetimes['transition_times_pred'][transition_id], 'pred'])

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def emission_events(self, time_series=True, density=True, include_0=False, **kwargs):
        """
        Shows the photon / emission counts.

        Parameters
        ----------
        time_series : bool
            Whether to display a time series or the frequency (i.e., histogram).
        density : bool
            Whether to display the histogram with probability densities (True) or probabilities (False). Only needed
            if time_series is False.
        include_0 : bool
            Whether to include counts of 0 events. Only needed if time_series is False.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        kwargs.setdefault('label', 'data')
        if time_series:
            kwargs.setdefault('title', 'fluorescence trajectory')
            kwargs.setdefault('type_', "line")
            kwargs.setdefault('data', [self.system.event_time_series.index, self.system.event_time_series.values])
            kwargs.setdefault('ylabel', "photon count")
            kwargs.setdefault('xlabel', "time [s]")
        else:
            if include_0:
                data = self.system.event_time_series
            else:
                data = self.system.event_time_series[self.system.event_time_series != 0]
            kwargs.setdefault('title', 'intensity distribution')
            kwargs.setdefault('type_', "hist")
            kwargs.setdefault('data', data)
            kwargs.setdefault('xlabel', "photon count")
            if density:
                kwargs.setdefault('ylabel', 'PD')
                kwargs.setdefault('density', True)
            else:
                kwargs.setdefault('ylabel', "Pr")
                kwargs.setdefault('weights', np.zeros_like(data) + 1 / data.size)

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def fcs(self, normalize_to=None, unit="s", **kwargs):
        """
        Shows the fluorescence correlation spectroscopy (FCS) curve with the autocorrelation as a function of time.

        Parameters
        ----------
        normalize_to : None, int
            Index of datapoint to which the data is normalized.
        unit : str
            One of "s", "ms", "us". Influences the unit the data is displayed in.
        kwargs : custom_plot.universal_figure arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        ax : matplotlib.axes.Axes
            Contains most of the figure elements.
        """
        tau_data, correl_data = np.copy(self.system.autocorrelation[0]), np.copy(self.system.autocorrelation[1])
        if normalize_to is not None:
            correl_data /= correl_data[normalize_to]

        adjust_unit = pd.to_timedelta(1, unit).total_seconds()
        tau_data = tau_data / adjust_unit
        kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[1]:.2e}$ {unit}")
        if self.system.parameter_collection.loc[('fcs', 'log'), 0]:
            kwargs.setdefault('data', [tau_data[1:], correl_data[1:]])
        else:
            kwargs.setdefault('data', [tau_data[1:1000], correl_data[1:1000]])

        kwargs.setdefault('type_', "line")
        kwargs.setdefault('xscale', "log")
        kwargs.setdefault('xlabel', fr"$\tau [{unit}]$")
        kwargs.setdefault('ylabel', r"$G(\tau)$")

        fig, ax = cp.universal_figure(**kwargs)

        return fig, ax

    def on_off(self, mode="on", time_series=False, **kwargs):
        """
        Shows the on or off periods.

        Parameters
        ----------
        mode : str
            One of "on", "off", "mesh".
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
        mean = None
        if time_series:
            kwargs.setdefault('xlabel', "frame number")
            if mode == "mesh":
                kwargs.setdefault('title', 'ON OFF periods')
                kwargs.setdefault('legend', True)
                kwargs.setdefault('label', ['ON periods', 'OFF periods'])
                kwargs.setdefault('type_', "multiple_line")
                kwargs.setdefault('data', [[np.arange(0, self.system.on_periods.size*2, 2), self.system.on_periods],
                                  [np.arange(1, self.system.off_periods.size*2, 2), self.system.off_periods]])
                colors = cm.rainbow(np.linspace(0, 1, 2))
                kwargs.setdefault('color', colors)
                kwargs.setdefault('ylabel', "consecutive frames")
            elif mode == "on":
                kwargs.setdefault('title', 'ON periods')
                kwargs.setdefault('type_', "line")
                kwargs.setdefault('data', [np.arange(0, self.system.on_periods.size), self.system.on_periods])
                kwargs.setdefault('ylabel', "consecutive frames")
            else:
                kwargs.setdefault('title', 'OFF periods')
                kwargs.setdefault('type_', "line")
                kwargs.setdefault('data', [np.arange(0, self.system.off_periods.size), self.system.off_periods])
                kwargs.setdefault('ylabel', "consecutive frames")

        else:
            kwargs.setdefault('ylabel', "PD")
            kwargs.setdefault('density', True)
            if mode == "mesh":
                raise ValueError("'mesh' not possible for histogram.")
            elif mode == "on":
                kwargs.setdefault('title', 'ON periods')
                kwargs.setdefault('type_', "hist")
                kwargs.setdefault('data', self.system.on_periods)
                kwargs.setdefault('bins', np.arange(1, np.max(self.system.on_periods) + 1) - 0.5)
                kwargs.setdefault('xlabel', "consecutive frames")
                mean = np.mean(self.system.on_periods)
            else:
                kwargs.setdefault('title', 'OFF periods')
                kwargs.setdefault('type_', "hist")
                kwargs.setdefault('data', self.system.off_periods)
                kwargs.setdefault('xlabel', "consecutive frames")
                mean = np.mean(self.system.off_periods)

        fig, ax = cp.universal_figure(**kwargs)
        if mean is not None:
            ax[0][0].text(x=0.7, y=0.85, s=f"mean: {mean:.2f}", transform=ax[0][0].transAxes, fontsize=16)

        return fig, ax

    def add_table(self, fig, grid, level_0='emitters', level_1=None, scale=(1, 1), fontsize=20):
        """
        Adds a table to a subplot figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        grid : int
            Divide the figure subplots into an a x b grid and place this table at position c. Then the grid value should
            be abc. E.g., suppose a subplot with 2 rows and 3 columns and the table should span the entire lower row -
            then we want to use half the figure, hence divide it into two parts (a=2, b=1). We want to place it in the
            second row, hence the position c is 2 (it starts at 1, not at 0). The value to insert is then 212.
        level_0 : list
            List of names of the 0th level index of the multiindex dataframe parameter_collection.
        level_1 : list
            List of lists of names of the 1th level index of the multiindex dataframe parameter_collection.
        scale : tuple
            To scale the table in x and y direction.
        fontsize : float
            Size of text.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The top level container for all the plot elements.
        """
        new_ax = fig.add_subplot(grid)
        new_ax.axis('off')
        new_table = pd.concat([self.system.parameter_collection.loc[(level_0_atm, level_1_atm), :]
                               for level_0_atm, level_1_atm in zip(level_0, level_1)])
        table = new_ax.table(cellText=new_table.values,
                             rowLabels=[index_name[0] + '  ' + index_name[1] for index_name in new_table.index],
                             loc='center')
        table.scale(scale[0], scale[1])
        table.set_fontsize(fontsize)

        return fig
