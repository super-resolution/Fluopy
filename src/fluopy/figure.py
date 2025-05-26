"""
Module custom_plot
"""

import io
import cairosvg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, rcParamsDefault
from .miscellaneous import format_axis_labels


__version__ = "0.1.0"


def universal_figure(
    nrows=1,
    ncols=1,
    fig_width=6,
    fig_height=3,
    scale=1,
    rc_linewidth=2,
    type_="line",
    data=(0, 0),
    label=None,
    color="blue",
    title=None,
    xlabel="x",
    ylabel="y",
    ylabelcolor="black",
    xlim=None,
    ylim=None,
    xscale=None,
    yscale=None,
    xminor=False,
    yminor=False,
    adjust_x=None,
    adjust_y=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    tick_params=None,
    tick_spacing_x=None,
    tick_spacing_y=None,
    tick_style_x=None,
    tick_style_y=None,
    second_axis_x=False,
    second_axis_y=False,
    fontsize=21,
    legend=False,
    legendhandles=None,
    legendcolor="black",
    legendargs=None,
    draw_marker=None,
    draw_marker_param=None,
    plot_distribution=None,
    plot_distribution_label=None,
    axes=None,
    **type_specific_kwargs,
):
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
        Type of the plot. One of "hist", "multiple_hist", "bar", "line",
        "multiple_line", "scatter", "errorbar", "step".
    data : np.ndarray, Collection
        Data to be plotted. Required formation depends on input parameter type_.
    label : str, list
        Label to pass to legend.
    color : str, list, callable
        Color.
    title : str
        The title of the plot.
    xlabel : str
        The label text of the x-axis.
    ylabel : str
        The label text of the y-axis.
    ylabelcolor : str
        The color of the y-axis label.
    xlim : float, Collection
        Left and right limit of the x-axis.
    ylim : float, Collection
        Lower and upper limit of the y-axis.
    xscale : str
        One of "linear", "log", "symlog", "logit".
    yscale : str
        One of "linear", "log", "symlog", "logit".
    xminor : bool
        Whether to plot minor ticks on the x-axis.
    yminor : bool
        Whether to plot minor ticks on the y-axis.
    adjust_x : float
        Factor with which the x data is multiplicated.
    adjust_y : float
        Factor with which the y data is multiplicated.
    xticks : array-like
        xtick locations.
    yticks : array-like
        ytick locations.
    xticklabels : dict
        Keyword 'labels' with labels to place at the given tick locations. Keyword
        'rotation' to rotate text.
    yticklabels : dict
        Keyword 'labels' with labels to place at the given tick locations. Keyword
        'rotation' to rotate text.
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
    fontsize : float
        Size of font.
    legend : bool
        Whether to display a legend.
    legendhandles : Collection
        If not None, collection of handles (e.g., matplotlib.patches.Patch).
    legendcolor : str
        Color of text in legend.
    legendargs : dict
        Additional arguments to pass to legend.
    draw_marker : Collection
        The data positions, consists of x and y.
    draw_marker_param : dict
        Parameters to pass to .scatter
    plot_distribution : distr.rv_frozen
        Additional distribution to be plotted.
    plot_distribution_label : str
        Label of plot_distribution.
    axes : np.ndarray
        Contains matplotlib.axes.Axes objects.
    type_specific_kwargs : type_ properties

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes.Axes.
    """
    # initialize figure
    rcParams["axes.linewidth"] = rc_linewidth
    rcParams["figure.dpi"] = rcParamsDefault["figure.dpi"] * scale
    rcParams["figure.facecolor"] = "white"

    if axes is None:
        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height)
        )
    else:
        axes = axes

    axes = np.asarray(axes)
    if axes.ndim > 1:
        axes = axes.ravel()
    elif axes.ndim == 0:
        axes = axes[np.newaxis]

    ax = axes[0]

    # data incorporation
    match type_:
        case "hist":
            dot = False
            if (
                "histtype" in type_specific_kwargs
                and type_specific_kwargs["histtype"] == "dot"
            ):
                type_specific_kwargs.pop("histtype", None)
                dot = True
            n, bins, patches = ax.hist(
                x=data, color=color, label=label, **type_specific_kwargs
            )

            if dot:
                patches.remove()
                ax.scatter(
                    bins[:-1] + 0.5 * (bins[1:] - bins[:-1]),
                    n,
                    marker="o",
                    color=color,
                    label=label,
                    s=4,
                )
            if plot_distribution is not None:
                try:
                    plot_distribution.pmf(0)  # check if the distribution is discrete
                    if np.min(bins) < 0:
                        minimum = 0
                    else:
                        minimum = int(np.min(bins))
                    x = np.linspace(
                        minimum, int(np.max(bins)), int(np.max(bins)) - minimum + 1
                    )
                    ax.plot(
                        x,
                        plot_distribution.pmf(x),
                        c="k",
                        label=plot_distribution_label,
                    )

                except AttributeError:
                    x = np.linspace(np.min(bins), np.max(bins), 100)
                    ax.plot(
                        x,
                        plot_distribution.pdf(x),
                        c="k",
                        label=plot_distribution_label,
                    )

        case "multiple_hist":
            for j, dat_ in enumerate(data):
                if "weights" in type_specific_kwargs:
                    type_specific_kwargs["weights"] = np.ones_like(dat_) / dat_.size
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
                    _, bins, _ = ax.hist(
                        x=dat_, color=use_color, label=use_label, **type_specific_kwargs
                    )
                    if plot_distribution is not None:
                        plot_distr = plot_distribution[j]
                        try:
                            plot_distr.pmf(0)  # check if the distribution is discrete
                            if np.min(bins) < 0:
                                minimum = 0
                            else:
                                minimum = int(np.min(bins))
                            x = np.linspace(
                                minimum,
                                int(np.max(bins)),
                                int(np.max(bins)) - minimum + 1,
                            )
                            ax.plot(x, plot_distr.pmf(x), c="k", label="pred")
                        except AttributeError:
                            x = np.linspace(np.min(bins), np.max(bins), 100)
                            ax.plot(x, plot_distr.pdf(x), c="k", label="pred")
        case "2d_hist":
            h, xedges, yedges, _ = ax.hist2d(data[0], data[1], **type_specific_kwargs)
        case "bar":
            if data[1].ndim > 1:
                for j, dat_ in enumerate(data[1]):
                    if "width" in type_specific_kwargs:
                        width = type_specific_kwargs["width"]
                        dat_x = data[0] + j * width
                    else:
                        dat_x = data[0]
                    ax.bar(
                        x=dat_x,
                        height=dat_,
                        color=color[j],
                        label=label[j],
                        **type_specific_kwargs,
                    )
            else:
                ax.bar(
                    x=data[0],
                    height=data[1],
                    color=color,
                    label=label,
                    **type_specific_kwargs,
                )
        case "line":
            ax.plot(data[0], data[1], color=color, label=label, **type_specific_kwargs)
        case "step":
            ax.step(data[0], data[1], color=color, label=label, **type_specific_kwargs)
        case "errorbar":
            ax.errorbar(
                data[0],
                data[1],
                yerr=data[2],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "multiple_line":
            for j, dat_ in enumerate(data):
                if callable(color):
                    use_color = color(j)
                else:
                    use_color = color[j]
                ax.plot(
                    dat_[0],
                    dat_[1],
                    color=use_color,
                    label=label[j],
                    **type_specific_kwargs,
                )
        case "scatter":
            ax.scatter(
                data[0], data[1], color=color, label=label, **type_specific_kwargs
            )
        case "boxplot":
            ax.boxplot(data, labels=label, **type_specific_kwargs)
        case _:
            raise ValueError('Invalid type_ argument.')

    # x-axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if adjust_x is not None:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda old_x, _: "{0:g}".format(old_x * adjust_x))
        )
    if xscale is not None:
        ax.set_xscale(xscale)
        if xminor:
            ax.xaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs="auto", numticks=10)
            )
        else:
            ax.xaxis.minorticks_off()

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
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda old_y, _: "{0:g}".format(old_y * adjust_y))
        )
    if yscale is not None:
        ax.set_yscale(yscale)
        if yminor:
            ax.yaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs="auto", numticks=10)
            )
        else:
            ax.yaxis.minorticks_off()

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

    if tick_style_x is not None:
        plt.draw()
        offset_x = ax.xaxis.get_offset_text().get_text()
        xlabel = format_axis_labels(xlabel, offset_x)
    if tick_style_y is not None:
        plt.draw()
        offset_y = ax.yaxis.get_offset_text().get_text()
        ylabel = format_axis_labels(ylabel, offset_y)

    # texts
    ax.set_ylabel(ylabel, fontsize=fontsize, color=ylabelcolor)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # second x-axis
    if second_axis_x:
        ticks = ax.get_xticks()
        sec_ax = ax.secondary_xaxis("top")
        sec_ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        sec_ax.tick_params(axis="x", width=2, direction="in", labeltop=False, length=6)
        sec_ax.tick_params(
            which="minor", axis="x", direction="in", width=2, length=4, labeltop=False
        )

    # second y-axis
    if second_axis_y:
        ticks = ax.get_yticks()
        sec_ax = ax.secondary_yaxis("right")
        sec_ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        sec_ax.tick_params(
            axis="y", width=2, direction="in", labelright=False, length=6
        )
        sec_ax.tick_params(
            which="minor", axis="y", direction="in", width=2, length=4, labelright=False
        )

    if draw_marker is not None:
        if draw_marker_param is None:
            draw_marker_param = {
                "marker": "x",
                "c": "k",
                "label": "Prediction",
                "s": 100,
            }
        ax.scatter(*draw_marker, **draw_marker_param)

    if legend:
        if legendargs is None:
            legendargs = {}
        if legendhandles is not None:
            ax.legend(labelcolor=legendcolor, handles=legendhandles, **legendargs)
        else:
            ax.legend(labelcolor=legendcolor, **legendargs)

    axes = axes.reshape(nrows, ncols)

    return axes


def multi_plot(
    svg_files,
    dpi=300,
    dims=None,
    figsize=(10, 10),
    width_ratios=None,
    height_ratios=None,
    spans=None,
):
    """
    Constructs a figure with multiple plots.

    Parameters
    ----------
    svg_files : list
        List of .svg files to include in the figure.
    dpi : int
        Dots per inch.
    dims : tuple
        Dimensions of the grid. If None, the grid will be len(svg_files)/2 x 2.
    figsize : tuple
        Size of the figure.

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes.Axes.
    """
    images = []
    for svg_file in svg_files:
        image = cairosvg.svg2png(url=svg_file, dpi=dpi)
        img = Image.open(io.BytesIO(image))
        images.append(img)
    if dims is None:
        dims = (len(svg_files) // 2 + 1, 2)
    gs = plt.GridSpec(
        dims[0], dims[1], width_ratios=width_ratios, height_ratios=height_ratios
    )
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        if spans:
            ax = fig.add_subplot(gs[spans[i][0] : spans[i][1]])
        else:
            ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")

    return fig
