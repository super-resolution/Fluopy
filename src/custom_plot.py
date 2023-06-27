"""Contains functions that make many figure modifications more accessible."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, rcParamsDefault


def universal_figure(nrows=1, ncols=1, fig_width=6, fig_height=3, scale=1, type_="line", data=(0, 0), color="blue",
                     ylabel="y", xlabel="x", legend=False, label=None, title=None, xlim=None, ylim=None, xscale=None,
                     yscale=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, tick_params=None,
                     tick_spacing_x=None, tick_spacing_y=None, tick_style_x=None, tick_style_y=None, second_axis_x=True,
                     second_axis_y=True, draw_marker=None, plot_distribution=None, fig=None, axes=None,
                     **type_specific_kwargs):
    """
    Constructs a figure in versatile types and designs.

    Parameters
    ----------
    nrows : int
        Number of rows of plt.subplots.
    ncols : int
        Number of columns of plt.subplots.
    fig_width : float
        Width of the figure.
    fig_height : float
        Height of the figure.
    scale : float
        Factor to scale the figure.
    type_ : str
        Type of the plot. One of "hist", "bar", "line", "multiple_line", "scatter".
    data : np.ndarray, Collection
        Data to be plotted. Required formation depends on input parameter type_.
    color : str, list
        Color.
    ylabel : str
        The label text of the y-axis.
    xlabel : str
        The label text of the x-axis.
    legend : bool
        Whether to display a legend.
    label : str, list
        Label to pass to legend.
    title : str
        The title of the plot.
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
    tick_params : dict
        Parameters to pass to .tick_params().
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
    draw_marker : Collection
        The data positions, consists of x and y.
    plot_distribution : Collection
        Contains distr.rv_continuous_frozen or distr.rv_discrete_frozen or distr.rv_frozen and label.
    fig : matplotlib.figure.Figure
        The top level container for all the plot elements.
    axes : list
        Contains matplotlib.axes.Axes objects.
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
    rcParams['figure.dpi'] = rcParamsDefault['figure.dpi'] * scale
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    else:
        fig, axes = fig, axes
    if type(axes) == np.ndarray:
        axes = axes.ravel()  # convert to 1d array
    else:
        axes = np.array([axes])

    data = [data]
    for i, dat in enumerate(data):
        ax = axes[i]
        # texts
        ax.set_ylabel(ylabel, fontsize=21)
        ax.set_xlabel(xlabel, fontsize=21)
        if title is not None:
            ax.set_title(title, fontsize=21)

        # x-axis
        if xlim is not None:
            ax.set_xlim(xlim)
        if xscale is not None:
            ax.set_xscale(xscale)

        # x-axis ticks
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(**xticklabels)
        if tick_spacing_x is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))

        # y-axis
        if ylim is not None:
            ax.set_ylim(ylim)
        if yscale is not None:
            ax.set_yscale(yscale)

        # y-axis ticks
        if yticks is not None:
            ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(**yticklabels)
        if tick_spacing_y is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))

        # general tick formatting
        ax.tick_params(labelsize=20, width=2, length=6)
        if tick_params is not None:
            ax.tick_params(**tick_params)
        ax.tick_params(which="minor", width=2, length=4, labelleft=False, left=True)
        if tick_style_x is not None:
            ax.ticklabel_format(style=tick_style_x, axis="x", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_visible(False)
        if tick_style_y is not None:
            ax.ticklabel_format(style=tick_style_y, axis="y", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_visible(False)

        # data incorporation
        if type_ == "hist":
            _, bins, _ = ax.hist(x=dat, color=color, label=label, **type_specific_kwargs)
            if plot_distribution is not None:
                try:
                    plot_distribution[0].pmf(0)  # check if the distribution is discrete
                    if np.min(bins) < 0:
                        minimum = 0
                    else:
                        minimum = int(np.min(bins))
                    x = np.linspace(minimum, int(np.max(bins)), int(np.max(bins)) - minimum + 1)
                    ax.plot(x, plot_distribution[0].pmf(x), c='k', label=plot_distribution[1])
                except AttributeError:
                    x = np.linspace(np.min(bins), np.max(bins), 100)
                    ax.plot(x, plot_distribution[0].pdf(x), c='k', label=plot_distribution[1])

        elif type_ == "multiple_hist":
            for j, dat_ in enumerate(dat):
                _, bins, _ = ax.hist(x=dat_, color=color(j), label=label[j], **type_specific_kwargs)
                if plot_distribution is not None:
                    plot_distr = plot_distribution[j]
                    try:
                        plot_distr.pmf(0)  # check if the distribution is discrete
                        if np.min(bins) < 0:
                            minimum = 0
                        else:
                            minimum = int(np.min(bins))
                        x = np.linspace(minimum, int(np.max(bins)), int(np.max(bins)) - minimum + 1)
                        ax.plot(x, plot_distr.pmf(x), c='k', label='pred')
                    except AttributeError:
                        x = np.linspace(np.min(bins), np.max(bins), 100)
                        ax.plot(x, plot_distr.pdf(x), c='k', label='pred')

        elif type_ == "bar":
            if dat[1].ndim > 1:
                for j, dat_ in enumerate(dat[1]):
                    if 'width' in type_specific_kwargs:
                        width = type_specific_kwargs['width']
                        dat_x = dat[0] + j*width
                    else:
                        dat_x = dat[0]
                    ax.bar(x=dat_x, height=dat_, color=color[j], label=label[j], **type_specific_kwargs)
            else:
                ax.bar(x=dat[0], height=dat[1], color=color, label=label, **type_specific_kwargs)
        elif type_ == "line":
            ax.plot(dat[0], dat[1], color=color, label=label, **type_specific_kwargs)
        elif type_ == "multiple_line":
            for j, dat_ in enumerate(dat):
                ax.plot(dat_[0], dat_[1], color=color(j), label=label[j], **type_specific_kwargs)
        elif type_ == "scatter":
            ax.scatter(dat[0], dat[1], color=color, label=label, **type_specific_kwargs)

        # second x-axis
        if second_axis_x is not None:
            ticks = ax.get_xticks()
            sec_ax = ax.secondary_xaxis("top")
            sec_ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            sec_ax.tick_params(axis="x", width=2, direction="in", labeltop=False, length=6)
            sec_ax.tick_params(which="minor", axis="x", direction="in", width=2, length=4, labeltop=False)
        # second y-axis
        if second_axis_y is not None:
            ticks = ax.get_yticks()
            sec_ax = ax.secondary_yaxis("right")
            sec_ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            sec_ax.tick_params(axis="y", width=2, direction="in", labelright=False, length=6)
            sec_ax.tick_params(which="minor", axis="y", direction="in", width=2, length=4, labelright=False)

        if draw_marker is not None:
            ax.scatter(*draw_marker, marker='x', c='k', label='pred')

        if legend:
            ax.legend()

    axes = axes.reshape(nrows, ncols)

    return fig, axes
