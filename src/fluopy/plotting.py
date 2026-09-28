"""Plot simulation results with consistent Matplotlib formatting."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import rcParamsDefault
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mplFigure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from scipy.stats.distributions import rv_frozen

    from .transitions import TransitionSet


__all__: list[str] = ["plot_data"]


def delete_subplots(
    axes: npt.NDArray[Any],
    keep_number: int | None = None,
    del_positions: npt.ArrayLike | None = None,
) -> None:
    """
    Deletes subplots from figure object.

    Parameters
    ----------
    axes
        Contains matplotlib.axes._subplots.AxesSubplots.
    keep_number
        Number of subplots to keep. Assumes them to be in the first keep_number
        positions of the flattened ax array.
    del_positions
        An array that contains a 1-D array of shape (2,) for each ax to be deleted like
        [row, column].

    """
    flattened = axes.ravel()
    fig = flattened[0].get_figure()
    if keep_number is not None and del_positions is not None:
        raise ValueError("Only one of keep_number or del_positions must be provided.")
    elif keep_number is not None:
        for i in range(flattened.size - keep_number):
            fig.delaxes(flattened[-1 - i])
    elif del_positions is not None:
        positions = np.asarray(del_positions, dtype=np.int64)
        for position in positions:
            fig.delaxes(axes[position[0], position[1]])
    else:
        raise ValueError("Either keep_number or del_positions must be provided.")


def create_row_subtitles(
    axes: npt.NDArray[Any],
    nrows: int = 1,
    ncols: int = 1,
    titles: Sequence[str] | None = None,
) -> None:
    """
    Creates subtitles of figure displayed in the middle of each row.

    Parameters
    ----------
    axes
        Contains matplotlib.axes._subplots.AxesSubplots.
    nrows
        Number of rows in the figure.
    ncols
        Number of columns in the figure.
    titles
        Contains elements of type str. Must have the same length as nrows. If None,
        ['default_title'] is used.

    """
    if titles is None:
        titles = ["default_title"] * nrows

    fig = get_figure(axes=axes)
    grid = GridSpec(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        row = fig.add_subplot(grid[i, ::])
        row.set_title(titles[i], fontsize=22, pad=20, fontweight="bold")
        row.set_frame_on(False)
        row.axis("off")


def add_table(
    axes: Axes | None,
    data: npt.ArrayLike | pd.Series,
    labels: npt.ArrayLike | None = None,
    grid: int = 111,
    xscale: float = 1,
    yscale: float = 1,
    fontsize: float = 12,
) -> Axes:
    """
    Adds a table to a subplot figure.

    Parameters
    ----------
    axes
        matplotlib.axes._subplots.AxesSubplots.
    data
        If pd.Series, values to display in table with index as labels.
    labels
        Labels of table rows.
        Only used if data is not pd.Series. Otherwise, index of pd.Series is used.
    grid
        Divide the figure subplots into an a x b grid. Choose a position c for the
        table such that it corresponds to the index + 1 of the flattened grid.
        Example: suppose a subplot with 2 rows and 3 columns. The table should span the
        entire lower row, hence half of the figure. Divide the figure into 2 rows and 1
        column (a = 2, b = 1). The position c is 2. The value to use for grid is abc,
        hence in the example 212.
    xscale
        Scale table in x direction.
    yscale
        Scale table in y direction.
    fontsize
        Set the font size.

    Returns
    -------
    Axes
        The input axes object.
    """
    if axes is None:
        axes = plt.gca()

    row_labels: list[str] | None
    if isinstance(data, pd.Series):
        cells = data.to_numpy()[:, np.newaxis]
        row_labels = [str(label) for label in data.index]
    else:
        cells = np.asarray(data)
        row_labels = (
            None if labels is None else [str(label) for label in np.asarray(labels)]
        )

    fig = get_figure(axes=axes)
    new_ax = fig.add_subplot(grid)
    new_ax.axis("off")
    table = new_ax.table(cellText=cells.tolist(), rowLabels=row_labels, loc="center")
    table.scale(xscale=xscale, yscale=yscale)
    table.set_fontsize(size=fontsize)

    return axes


def get_figure(axes: Axes | npt.NDArray[Any] | None = None) -> mplFigure:
    """
    Get the figure object based on axes, where axes is either an axes object or a
    np.ndarray.

    Parameters
    ----------
    axes
        In the case of axes being np.ndarray, it contains
        matplotlib.axes._subplots.AxesSubplots

    Returns
    -------
    matplotlib.figure.Figure
        The figure object that corresponds to axes.
    """
    if axes is None:
        ax = plt.gca()
    elif isinstance(axes, np.ndarray):
        flattened = axes.ravel()
        ax = flattened[0]
    else:
        ax = axes
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("axes is not attached to a figure.")

    return cast(mplFigure, fig)


def format_electronic_state(label: str) -> str:
    """
    Format label for LaTeX.

    Parameters
    ----------
    label
        Label to format.

    Returns
    -------
    str
        Formatted label.
    """
    if re.match(pattern=r"^[A-Z]\d$", string=label):
        return label[0] + r"$_{" + label[1:] + r"}$"
    return label


def format_transition(label: str) -> str:
    """
    Format label for LaTeX.

    Parameters
    ----------
    label
        Label to format.

    Returns
    -------
    str
        Formatted label.
    """
    if "_" in label:
        parts = label.split(sep="_", maxsplit=1)
        return parts[0] + r"$_{" + parts[1] + r"}$"
    return label


def _flatten_state_values(
    transition_set: TransitionSet,
    values_by_fluorophore: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Merge state values in the fluorophore order of a transition set.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes.
    values_by_fluorophore
        State values grouped by fluorophore name.

    Returns
    -------
    npt.NDArray[np.float64]
        Merged state values.
    """
    return np.concatenate(
        [
            values_by_fluorophore[fluorophore]
            for fluorophore in transition_set.single_states
        ]
    )


def _plot_transition_bars(
    transition_df: pd.DataFrame,
    values: npt.NDArray[np.float64],
    default_ylabel: str,
    draw_marker: list[npt.NDArray[np.float64]] | None = None,
    legend_labels: Sequence[str] | None = None,
    **kwargs: Any,
) -> Axes:
    """
    Plot values associated with transitions as bars.

    Parameters
    ----------
    transition_df
        Dataframe of transitions with transition-group labels and transition identities
        as its index.
    values
        Values ordered by transition identity.
    default_ylabel
        Y-axis label used unless ylabel is supplied in kwargs.
    draw_marker
        Marker coordinates to draw over the bars.
    legend_labels
        Transition-group labels to display in the legend. By default, the labels from
        transition_df are used.
    kwargs
        kwargs for fluopy.plotting.plot_data.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis.
    """
    data = [np.arange(transition_df.shape[0]), values]
    kwargs.setdefault("type_", "bar")
    kwargs.setdefault("xlabel", None)
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("xticks", range(transition_df.shape[0]))
    kwargs.setdefault(
        "xticklabels",
        dict(
            labels=transition_df["abbreviation"].apply(format_transition),
            rotation=70,
        ),
    )
    group_labels = transition_df.index.get_level_values(0)
    unique_group_labels = group_labels.unique()
    colormap = mpl.colors.ListedColormap(
        [
            mpl.colormaps["Spectral"](value)
            for value in np.linspace(0, 1, unique_group_labels.size)
        ]
    )
    kwargs.setdefault(
        "color",
        [
            colormap(i)
            for i, size in enumerate(transition_df.groupby(level=0, sort=False).size())
            for _ in range(size)
        ],
    )
    kwargs.setdefault("ylabel", default_ylabel)
    kwargs.setdefault("legend", True)
    labels = unique_group_labels if legend_labels is None else legend_labels
    kwargs.setdefault(
        "legendhandles",
        [
            mpl.patches.Patch(color=colormap(i), label=label)
            for i, label in enumerate(labels)
        ],
    )
    return plot_data(data=data, draw_marker=draw_marker, **kwargs)


def _plot_state_bars(
    transition_set: TransitionSet,
    values: npt.NDArray[np.float64],
    default_ylabel: str,
    draw_marker: list[npt.NDArray[np.float64]] | None = None,
    full_xlim: bool = False,
    **kwargs: Any,
) -> Axes:
    """
    Plot values associated with fluorophore states as bars.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes.
    values
        State values ordered by fluorophore and state identity.
    default_ylabel
        Y-axis label used unless ylabel is supplied in kwargs.
    draw_marker
        Marker coordinates to draw over the bars.
    full_xlim
        Whether to extend the x-axis limits beyond the first and last bars.
    kwargs
        kwargs for fluopy.plotting.plot_data.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis.
    """
    single_states = transition_set.single_states
    colormap = mpl.colors.ListedColormap(
        [
            mpl.colormaps["Spectral"](value)
            for value in np.linspace(0, 1, len(single_states))
        ]
    )
    colors: list[Any] = []
    patches: list[Any] = []
    labels: list[str] = []
    for i, (fluorophore, states) in enumerate(single_states.items()):
        colors.extend([colormap(i) for _ in range(states.size)])
        patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
        labels.extend(
            [
                format_electronic_state(transition_set.states_by_value[identity].name)
                for identity in states
            ]
        )

    positions = np.arange(values.size)
    data = [positions, values]
    kwargs.setdefault("type_", "bar")
    kwargs.setdefault("xlabel", None)
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("xticks", range(values.size))
    if full_xlim:
        kwargs.setdefault("xlim", [-1, values.size])
    kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
    kwargs.setdefault("ylabel", default_ylabel)
    kwargs.setdefault("color", colors)
    kwargs.setdefault("legend", True)
    kwargs.setdefault("legendhandles", patches)
    return plot_data(data=data, draw_marker=draw_marker, **kwargs)


def format_axis_labels(label: str, offset: str) -> str:
    """
    Format axis labels for LaTeX.

    Parameters
    ----------
    label
        Label to format.
    offset
        Offset to multiply label with. Format: "1eX" with X being an integer.

    Returns
    -------
    str
        Formatted label
    """
    _, exponent = offset.split("e")
    offset = rf"$10^{{{exponent}}} \\times$"
    if "(" in label and ")" in label:
        label = re.sub(pattern=r"\((.*?)\)", repl=rf"({offset} \1)", string=label)
    elif "[" in label and "]" in label:
        label = re.sub(pattern=r"\[(.*?)\]", repl=rf"[{offset} \1]", string=label)
    else:
        offset = rf"$ \times 10^{{{exponent}}}$"
        label = rf"{label} ({offset})"

    return label


def compute_tight_bbox(fig: mplFigure, pad_inches: float = 0.0) -> Bbox:
    """
    Compute tight bounding box of a figure with specified padding. The width is not
    changed.
    """
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    tight = fig.get_tightbbox(renderer)
    not_tight = fig.bbox
    width, _ = fig.get_size_inches()

    bbox = Bbox.from_bounds(
        not_tight.x0,
        tight.y0 - pad_inches,
        width,
        tight.height + 2 * pad_inches,
    )

    return bbox


def plot_data(
    fig_width: float = 6,
    fig_height: float = 3,
    scale: float = 1,
    rc_linewidth: float = 2,
    type_: str = "line",
    data: npt.ArrayLike | Sequence[Any] = (0, 0),
    label: str | Sequence[str] | None = None,
    color: str | Sequence[str] | Callable[[int], Any] = "blue",
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
    tick_style_x: Literal["", "sci", "scientific", "plain"] | None = None,
    tick_style_y: Literal["", "sci", "scientific", "plain"] | None = None,
    second_axis_x: bool = False,
    second_axis_y: bool = False,
    fontsize: float = 21,
    legend: bool = False,
    legendhandles: Sequence[Any] | None = None,
    legendcolor: str = "black",
    legendargs: dict[str, Any] | None = None,
    draw_marker: Sequence[Any] | None = None,
    draw_marker_param: dict[str, Any] | None = None,
    plot_distribution: rv_frozen | Sequence[rv_frozen] | None = None,
    plot_distribution_label: str | None = None,
    ax: Axes | None = None,
    **type_specific_kwargs: Any,
) -> Axes:
    """
    Constructs a figure or modifies an axis.

    Parameters
    ----------
    fig_width
        Width of the figure.
    fig_height
        Height of the figure.
    scale
        Factor applied to Matplotlib's default figure DPI.
    rc_linewidth
        Linewidth of the axes.
    type_
        Type of the plot. One of "hist", "multiple_hist", "2d_hist", "bar", "line",
        "multiple_line", "scatter", "errorbar", "step", "stair", "boxplot".
    data
        Data to be plotted. Required formation depends on input parameter type_.
    label
        Label to pass to legend. For multiple_line, multiple_hist, and bar, a list of
        labels.
    color
        Color. For multiple_line and multiple_hist, either a callable function or a list
        of colors. For bar, a color or a list of colors.
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
        Whether to plot minor ticks on the x-axis. Only relevant if xscale is log.
    yminor
        Whether to plot minor ticks on the y-axis. Only relevant if yscale is log.
    adjust_x
        Factor applied to tick label formatter.
    adjust_y
        Factor applied to tick label formatter.
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
        "sci" - scientific notation. The offset is added to the xlabel.
    tick_style_y
        "sci" - scientific notation. The offset is added to the ylabel.
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
        Parameters to pass to .scatter. Default is {"marker": "x", "c": "k",
        "label": "prediction", "s": 100}.
    plot_distribution
        Additional distribution to be plotted. For multiple_hist, a list of
        distributions.
    plot_distribution_label
        Label of plot_distribution. For multiple_hist, label is 'pred'.
    ax
        matplotlib.axes.Axes to modify. If None, a new figure and axis are created.
    type_specific_kwargs
        type_ properties

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(fig_width, fig_height),
            dpi=rcParamsDefault["figure.dpi"] * scale,
            facecolor="white",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(rc_linewidth)
    data_items = cast(Sequence[Any], data)

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
                x=data,
                color=cast(Any, color),
                label=label,
                **type_specific_kwargs,
            )

            if dot:
                cast(Any, patches).remove()
                ax.scatter(
                    bins[:-1] + 0.5 * (bins[1:] - bins[:-1]),
                    n,
                    marker="o",
                    color=color,
                    label=label,
                    s=4,
                )
            if plot_distribution is not None:
                distribution = cast(Any, plot_distribution)
                try:
                    distribution.pmf(0)  # check if the distribution is discrete
                    if np.min(bins) < 0:
                        minimum = 0
                    else:
                        minimum = int(np.min(bins))
                    x = np.linspace(
                        minimum, int(np.max(bins)), int(np.max(bins)) - minimum + 1
                    )
                    ax.plot(
                        x,
                        distribution.pmf(x),
                        c="k",
                        label=plot_distribution_label,
                    )

                except AttributeError:
                    x = np.linspace(np.min(bins), np.max(bins), 100)
                    ax.plot(
                        x,
                        distribution.pdf(x),
                        c="k",
                        label=plot_distribution_label,
                    )

        case "multiple_hist":
            distributions = cast(Sequence[Any], plot_distribution)
            for j, values in enumerate(data_items):
                dat_ = np.asarray(values)
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
                    elif label is None:
                        use_label = None
                    else:
                        use_label = label[j]
                    _, bins, _ = ax.hist(
                        x=dat_, color=use_color, label=use_label, **type_specific_kwargs
                    )
                    if plot_distribution is not None:
                        plot_distr = distributions[j]
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
            h, xedges, yedges, _ = ax.hist2d(
                data_items[0], data_items[1], **type_specific_kwargs
            )
        case "bar":
            data_y = np.asarray(data_items[1])
            if data_y.ndim > 1:
                colors = cast(Sequence[str], color)
                labels = cast(Sequence[str], label)
                for j, dat_ in enumerate(data_y):
                    if "width" in type_specific_kwargs:
                        width = type_specific_kwargs["width"]
                        dat_x = np.asarray(data_items[0]) + j * width
                    else:
                        dat_x = data_items[0]
                    ax.bar(
                        x=dat_x,
                        height=dat_,
                        color=colors[j],
                        label=labels[j],
                        **type_specific_kwargs,
                    )
            else:
                ax.bar(
                    x=data_items[0],
                    height=data_y,
                    color=color,
                    label=label,
                    **type_specific_kwargs,
                )
        case "line":
            ax.plot(
                data_items[0],
                data_items[1],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "step":
            ax.step(
                data_items[0],
                data_items[1],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "stair":
            ax.stairs(
                data_items[1],
                data_items[0],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "errorbar":
            ax.errorbar(
                data_items[0],
                data_items[1],
                yerr=data_items[2],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "multiple_line":
            for j, dat_ in enumerate(data_items):
                if callable(color):
                    use_color = color(j)
                elif isinstance(color, str):
                    use_color = color
                else:
                    use_color = color[j]
                if isinstance(label, str):
                    use_label = label
                elif label is None:
                    use_label = None
                else:
                    use_label = label[j]
                ax.plot(
                    dat_[0],
                    dat_[1],
                    color=use_color,
                    label=use_label,
                    **type_specific_kwargs,
                )
        case "scatter":
            ax.scatter(
                data_items[0],
                data_items[1],
                color=color,
                label=label,
                **type_specific_kwargs,
            )
        case "boxplot":
            tick_labels = [label] if isinstance(label, str) else label
            ax.boxplot(data, tick_labels=tick_labels, **type_specific_kwargs)
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
        sec_ax.xaxis.set_major_locator(ticker.FixedLocator(ticks.tolist()))
        sec_ax.tick_params(axis="x", width=2, direction="in", labeltop=False, length=6)
        sec_ax.tick_params(
            which="minor", axis="x", direction="in", width=2, length=4, labeltop=False
        )

    # second y-axis
    if second_axis_y:
        ticks = ax.get_yticks()
        sec_ax = ax.secondary_yaxis("right")
        sec_ax.yaxis.set_major_locator(ticker.FixedLocator(ticks.tolist()))
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

    return ax
