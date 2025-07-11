"""
Module miscellaneous
"""

from __future__ import annotations

import re
import reprlib
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes
    from matplotlib.figure import Figure as mplFigure


def delete_subplots(
    axes: npt.NDArray[mplAxes],
    keep_number: int | None = None,
    del_positions: npt.ArrayLike = None,
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

    Returns
    -------
    None
    """
    flattened = axes.ravel()
    fig = flattened[0].get_figure()
    if keep_number is not None:
        for i in range(flattened.size - keep_number):
            fig.delaxes(flattened[-1 - i])
    elif del_positions is not None:
        for position in del_positions:
            fig.delaxes(axes[position[0], position[1]])


def create_row_subtitles(
    axes: npt.NDArray[mplAxes],
    nrows: int = 1,
    ncols: int = 1,
    titles: Sequence[str] = None,
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
        Containes elements of type str. Must have the same length as nrows.

    Returns
    -------
    None
    """
    if titles is None:
        titles = ["default_title"]

    fig = get_figure(axes)
    grid = plt.GridSpec(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        row = fig.add_subplot(grid[i, ::])
        row.set_title(titles[i], fontsize=22, pad=20, fontweight="bold")
        row.set_frame_on(False)
        row.axis("off")


def add_table(
    axes: mplAxes,
    data: npt.ArrayLike | pd.Series,
    labels: npt.ArrayLike | None = None,
    grid: int = 111,
    xscale: float = 1,
    yscale: float = 1,
    fontsize: float = 12,
) -> mplAxes:
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
        Only used if data is not pd.Series.
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
    None
    """
    if axes is None:
        axes = plt.gca()

    if isinstance(data, pd.Series):
        cells = data.values[:, np.newaxis]
        labels = data.index
    else:
        cells = data

    fig = get_figure(axes)
    new_ax = fig.add_subplot(grid)
    new_ax.axis("off")
    table = new_ax.table(cellText=cells, rowLabels=labels, loc="center")
    table.scale(xscale=xscale, yscale=yscale)
    table.set_fontsize(size=fontsize)

    return axes


def get_figure(axes: mplAxes | npt.NDArray[mplAxes] | None = None) -> mplFigure:
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

    return fig


def print_class(class_instance: Any) -> None:
    """
    Print all class attributes.

    Parameters
    ----------
    class_instance
        Instance of a class.
    """
    aRepr = reprlib.Repr()
    aRepr.maxlevel = 6
    aRepr.maxtuple = 6
    aRepr.maxlist = 6
    aRepr.maxarray = 5
    aRepr.maxdict = 4
    aRepr.maxset = 6
    aRepr.maxfrozenset = 6
    aRepr.maxdeque = 6
    aRepr.maxstring = 30
    aRepr.maxlong = 40
    aRepr.maxother = 100

    print(f"Attributes of {class_instance}:")
    print("." * 65)

    if is_dataclass(class_instance):
        for field in fields(class_instance):
            field_value = getattr(class_instance, field.name)
            print(field.name, "=", aRepr.repr(field_value))
            print("_" * 65)
    else:
        for attr_name, attr_value in vars(class_instance).items():
            if isinstance(attr_value, pd.DataFrame) or isinstance(
                attr_value, pd.Series
            ):
                print(attr_name + "[:6]", "=", attr_value.iloc[:6])
            else:
                print(attr_name, "=", aRepr.repr(attr_value))
            print("_" * 65)
    print("\n")


def find_key_in_list(row: pd.Series, key: Any) -> int | None:
    """
    If key is in row['fluorophore_ids'], return the index of the row.

    Parameters
    ----------
    row
        Row of a DataFrame.
    key
        Key to search for in row['fluorophore_ids'].

    Returns
    -------
    int | None
    """
    if key in row["fluorophore_ids"]:
        return row.name
    return None


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
    if re.match(r"^[A-Z]\d$", label):
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
        parts = label.split("_", 1)
        return parts[0] + r"$_{" + parts[1] + r"}$"
    return label


def format_axis_labels(label: str, offset: str) -> str:
    """
    Format axis labels for LaTeX.

    Parameters
    ----------
    label
        Label to format.
    offset
        Offset to multiply label with.

    Returns
    -------
    str
        Formatted label
    """
    _, exponent = offset.split("e")
    offset = rf"$10^{{{exponent}}}$"
    if "(" in label and ")" in label:
        label = re.sub(r"\((.*?)\)", rf"({offset} x \1)", label)
    elif "[" in label and "]" in label:
        label = re.sub(r"\[(.*?)\]", rf"[{offset} × \1]", label)
    else:
        label = f"{label} (× {offset})"

    return label
