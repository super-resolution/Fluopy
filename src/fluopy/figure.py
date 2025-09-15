"""
Module custom_plot

A universal figure is defined for plotting simulation results with matplotlib.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
from matplotlib import rcParams, rcParamsDefault

from .miscellaneous import format_axis_labels

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes
    from scipy.stats.distributions import rv_frozen


def universal_figure(
    nrows: int = 1,
    ncols: int = 1,
    fig_width: float = 6,
    fig_height: float = 3,
    scale: float = 1,
    rc_linewidth: float = 2,
    type_: str = "line",
    data: npt.ArrayLike | Sequence[Any] = (0, 0),
    label: str | list[str] | None = None,
    color: str | list[str] | Callable[Any, Any] = "blue",
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    ylabelcolor: str = "black",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xminor: bool = False,
    yminor: bool = False,
    adjust_x: float | None = None,
    adjust_y: float | None = None,
    xticks: npt.ArrayLike | None = None,
    yticks: npt.ArrayLike | None = None,
    xticklabels: dict[str, Any] | None = None,
    yticklabels: dict[str, Any] | None = None,
    tick_params: dict[str, Any] | None = None,
    tick_spacing_x: float | None = None,
    tick_spacing_y: float | None = None,
    tick_style_x: str | None = None,
    tick_style_y: str | None = None,
    second_axis_x: bool = False,
    second_axis_y: bool = False,
    fontsize: float = 21,
    legend: bool = False,
    legendhandles: Sequence[Any] | None = None,
    legendcolor: str = "black",
    legendargs: dict[str, Any] | None = None,
    draw_marker: Sequence[Any] | None = None,
    draw_marker_param: dict[str, Any] | None = None,
    plot_distribution: rv_frozen | None = None,
    plot_distribution_label: str | None = None,
    axes: npt.NDArray[mplAxes] | None = None,
    **type_specific_kwargs: Any,
) -> npt.NDArray[mplAxes]:
    """
    Constructs a figure or modifies axes.

    Parameters
    ----------
    nrows
        Number of rows of plt.subplots.
    ncols
        Number of columns of plt.subplots.
    fig_width
        Width of the figure.
    fig_height
        Height of the figure.
    scale
        Factor to scale the figure.
    type_
        Type of the plot. One of "hist", "multiple_hist", "bar", "line",
        "multiple_line", "scatter", "errorbar", "step", "stair".
    data
        Data to be plotted. Required formation depends on input parameter type_.
    label
        Label to pass to legend.
    color
        Color.
    title
        The title of the plot.
    xlabel
        The label text of the x-axis.
    ylabel
        The label text of the y-axis.
    ylabelcolor
        The color of the y-axis label.
    xlim
        Left and right limit of the x-axis.
    ylim
        Lower and upper limit of the y-axis.
    xscale
        One of "linear", "log", "symlog", "logit".
    yscale
        One of "linear", "log", "symlog", "logit".
    xminor
        Whether to plot minor ticks on the x-axis.
    yminor
        Whether to plot minor ticks on the y-axis.
    adjust_x
        Factor with which the x data is multiplicated.
    adjust_y
        Factor with which the y data is multiplicated.
    xticks
        xtick locations.
    yticks
        ytick locations.
    xticklabels
        Keyword 'labels' with labels to place at the given tick locations. Keyword
        'rotation' to rotate text.
    yticklabels
        Keyword 'labels' with labels to place at the given tick locations. Keyword
        'rotation' to rotate text.
    tick_params
        Parameters to pass to .tick_params().
    tick_spacing_x
        Set a tick on each integer multiple of tick_spacing_x.
    tick_spacing_y
        Set a tick on each integer multiple of tick_spacing_y.
    tick_style_x
        One of "sci", "plain".
    tick_style_y
        One of "sci", "plain".
    second_axis_x
        Whether to plot a second x-axis.
    second_axis_y
        Whether to plot a second y-axis.
    fontsize
        Size of font.
    legend
        Whether to display a legend.
    legendhandles
        If not None, collection of handles (e.g., matplotlib.patches.Patch).
    legendcolor
        Color of text in legend.
    legendargs
        Additional arguments to pass to legend.
    draw_marker
        The data positions, consists of x and y.
    draw_marker_param
        Parameters to pass to .scatter
    plot_distribution
        Additional distribution to be plotted.
    plot_distribution_label
        Label of plot_distribution.
    axes
        Contains matplotlib.axes.Axes objects.
    type_specific_kwargs
        type_ properties

    Returns
    -------
    npt.NDArray[matplotlib.axes.Axes]
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
        case "stair":
            ax.stairs(
                data[1],
                data[0],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
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
            raise ValueError("Invalid type_ argument.")

    # x-axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if adjust_x is not None:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda old_x, _: f"{old_x * adjust_x:g}")
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
            ticker.FuncFormatter(lambda old_y, _: f"{old_y * adjust_y:g}")
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
