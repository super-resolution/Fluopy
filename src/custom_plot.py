import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams


def universal_figure(nrows=1, ncols=1, width=10, height=5, type_="line", data=(0, 0), color="blue", ylabel="y",
                     xlabel="x", legend=False, xlim=None, ylim=None, xscale=None, yscale=None, xticks=None, yticks=None,
                     xticklabels=None, yticklabels=None, tick_spacing_x=None, tick_spacing_y=None, tick_style_x=None,
                     tick_style_y=None, second_axis_x=True, second_axis_y=True, **type_specific_kwargs):
    """
    Constructs a figure in versatile types and designs.

    Parameters
    ----------
    nrows : int
        Number of rows of plt.subplots.
    ncols : int
        Number of columns of plt.subplots.
    width : float
        Width of the figure.
    height : float
        Height of the figure.
    type_ : str
        Type of the plot. One of "line", "hist", "bar".
    data : np.ndarray, Collection
        Data to be plotted. Required formation depends on input parameter type_.
    color : str
        Color.
    ylabel : str
        The label text of the y-axis.
    xlabel : str
        The label text of the x-axis.
    legend : bool
        Whether to display a legend.
    xlim : float, Collection
        Left and right limit of the x-axis.
    ylim : float, Collection
        Lower and upper limit of the y-axis.
    xscale : str
        One of "linear", "log", "symlog", "logit".
    yscale : str
        One of "linear", "log", "symlog", "logit".
    xticks :  array-like
        xtick locations.
    yticks : array-like
        ytick locations.
    xticklabels : dict
        Keyword 'labels' with labels to place at the given tick locations. Keyword 'rotation' to rotate text.
    yticklabels : dict
        Keyword 'labels' with labels to place at the given tick locations. Keyword 'rotation' to rotate text.
    tick_spacing_x : float
        Set a tick on each integer multiple of tick_spacing_x.
    tick_spacing_y : float
        Set a tick on each integer multiple of tick_spacing_y.
    tick_style_x : str
        One of "sci", "plain".
    tick_style_y : str
        One of "sci", "plain".
    second_axis_x : bool
        Whether to plot a second x-axis.
    second_axis_y : bool
        Whether to plot a second y-axis.
    type_specific_kwargs : type_ properties

    Returns
    -------
    fig : matplotlib.figure.Figure
        The top level container for all the plot elements.
    axes : matplotlib.axes.Axes or array of Axes
        Contains most of the figure elements.
    """
    # initialize figure
    rcParams["axes.linewidth"] = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    if type(axes) == np.ndarray:
        axes = axes.ravel()
    else:
        axes = [axes]
        data = [data]
    for i, ax in enumerate(axes):
        dat = data[i]
        # texts
        ax.set_ylabel(ylabel, fontsize=21)
        ax.set_xlabel(xlabel, fontsize=21)
        if legend:
            ax.legend()

        # x-axis
        if xlim:
            ax.set_xlim(xlim)
        if xscale:
            ax.set_xscale(xscale)

        # x-axis ticks
        if xticks:
            ax.set_xticks(xticks)
        if xticklabels:
            ax.set_xticklabels(**xticklabels)
        if tick_spacing_x:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))

        # y-axis
        if ylim:
            ax.set_ylim(ylim)
        if yscale:
            ax.set_yscale(yscale)

        # y-axis ticks
        if yticks:
            ax.set_yticks(yticks)
        if yticklabels:
            ax.set_yticklabels(**yticklabels)
        if tick_spacing_y:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))

        # general tick formatting
        ax.tick_params(labelsize=20, width=2, length=6)
        ax.tick_params(which="minor", width=2, length=4, labelleft=False, left=True)
        if tick_style_x:
            ax.ticklabel_format(style=tick_style_x, axis="x", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_visible(False)
        if tick_style_y:
            ax.ticklabel_format(style=tick_style_y, axis="y", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_visible(False)

        # data incorporation
        if type_ == "hist":
            ax.hist(x=dat, color=color, **type_specific_kwargs)
        elif type_ == "bar":
            ax.bar(x=dat[0], height=dat[1], color=color, **type_specific_kwargs)
        elif type_ == "line":
            ax.plot(dat[0], dat[1], color=color, **type_specific_kwargs)

        # second x-axis
        if second_axis_x:
            ticks = ax.get_xticks()
            sec_ax = ax.secondary_xaxis("top")
            sec_ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            sec_ax.tick_params(axis="x", width=2, direction="in", labeltop=False, length=6)
            sec_ax.tick_params(which="minor", axis="x", direction="in", width=2, length=4, labeltop=False)
        # second y-axis
        if second_axis_y:
            ticks = ax.get_yticks()
            sec_ax = ax.secondary_yaxis("right")
            sec_ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            sec_ax.tick_params(axis="y", width=2, direction="in", labelright=False, length=6)
            sec_ax.tick_params(which="minor", axis="y", direction="in", width=2, length=4, labelright=False)

    return fig, axes
