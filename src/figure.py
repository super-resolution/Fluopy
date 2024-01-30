"""
Module custom_plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, rcParamsDefault


def universal_figure(nrows=1, ncols=1, fig_width=6, fig_height=3, scale=1, type_="line", data=(0, 0), color="blue",
                     ylabel="y", xlabel="x", fontsize=21, legend=False, label=None, legendcolor='black', title=None, xlim=None, ylim=None,
                     xscale=None, yscale=None, adjust_x=None, adjust_y=None, xticks=None, yticks=None, xticklabels=None,
                     yticklabels=None, tick_params=None, tick_spacing_x=None, tick_spacing_y=None, tick_style_x=None,
                     tick_style_y=None, second_axis_x=False, second_axis_y=False, draw_marker=None,
                     plot_distribution=None, axes=None, **type_specific_kwargs):
    """
    Constructs a figure or modifies axes.

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
    color : str, list, callable
        Color.
    ylabel : str
        The label text of the y-axis.
    xlabel : str
        The label text of the x-axis.
    fontsize : float
        Size of font.
    legend : bool
        Whether to display a legend.
    label : str, list
        Label to pass to legend.
    legendcolor : str
        Color of text in legend.
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
    adjust_x : float
        Factor with which the x data is multiplicated.
    adjust_y : float
        Factor with which the y data is multiplicated.
    xticks : array-like
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
    axes : np.ndarray
        Contains matplotlib.axes.Axes objects.
    type_specific_kwargs : type_ properties

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes.Axes.
    """
    # initialize figure
    rcParams["axes.linewidth"] = 2
    rcParams['figure.dpi'] = rcParamsDefault['figure.dpi'] * scale
    rcParams['figure.facecolor'] = 'white'

    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    else:
        axes = axes

    axes = np.asarray(axes)
    if axes.ndim > 1:
        axes = axes.ravel()
    elif axes.ndim == 0:
        axes = axes[np.newaxis]

    ax = axes[0]
    # texts
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # x-axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if adjust_x is not None:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda old_x, _: '{0:g}'.format(old_x * adjust_x)))
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
    if adjust_y is not None:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda old_y, _: '{0:g}'.format(old_y * adjust_y)))
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
    ax.tick_params(labelsize=fontsize, width=2, length=6)
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
        dot = False
        if 'histtype' in type_specific_kwargs and type_specific_kwargs['histtype'] == 'dot':
            type_specific_kwargs.pop('histtype', None)
            dot = True
        n, bins, patches = ax.hist(x=data, color=color, label=label, **type_specific_kwargs)

        if dot:
            patches.remove()
            ax.scatter(bins[:-1] + 0.5*(bins[1:] - bins[:-1]), n, marker='o', color=color, label=label, s=4)
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
        for j, dat_ in enumerate(data):
            if dat_.size != 0:
                if callable(color):
                    use_color = color(j)
                elif isinstance(color, str):
                    use_color = color
                else:
                    use_color = color[j]
                if isinstance(label, str):
                    use_label = label
                else:
                    use_label = label[j]
                _, bins, _ = ax.hist(x=dat_, color=use_color, label=use_label, **type_specific_kwargs)
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
        if data[1].ndim > 1:
            for j, dat_ in enumerate(data[1]):
                if 'width' in type_specific_kwargs:
                    width = type_specific_kwargs['width']
                    dat_x = data[0] + j*width
                else:
                    dat_x = data[0]
                ax.bar(x=dat_x, height=dat_, color=color[j], label=label[j], **type_specific_kwargs)
        else:
            ax.bar(x=data[0], height=data[1], color=color, label=label, **type_specific_kwargs)
    elif type_ == "line":
        ax.plot(data[0], data[1], color=color, label=label, **type_specific_kwargs)
    elif type_ == "errorbar":
        ax.errorbar(data[0], data[1], yerr=data[2], color=color, label=label, **type_specific_kwargs)
    elif type_ == "multiple_line":
        for j, dat_ in enumerate(data):
            if callable(color):
                use_color = color(j)
            else:
                use_color = color
            ax.plot(dat_[0], dat_[1], color=use_color, label=label[j], **type_specific_kwargs)
    elif type_ == "scatter":
        ax.scatter(data[0], data[1], color=color, label=label, **type_specific_kwargs)

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

    if draw_marker is not None:
        ax.scatter(*draw_marker, marker='x', c='k', label='pred')
    
    if legend:
        ax.legend(labelcolor=legendcolor)

    axes = axes.reshape(nrows, ncols)

    return axes
