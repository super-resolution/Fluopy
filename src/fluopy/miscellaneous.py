"""
Miscellaneous tools to format plots and labels.
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


__all__: list[str] = []


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
    if keep_number is not None and del_positions is not None:
        raise ValueError("Only one of keep_number or del_positions must be provided.")
    elif keep_number is not None:
        for i in range(flattened.size - keep_number):
            fig.delaxes(flattened[-1 - i])
    elif del_positions is not None:
        for position in del_positions:
            fig.delaxes(axes[position[0], position[1]])
    else:
        raise ValueError("Either keep_number or del_positions must be provided.")


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
        Contains elements of type str. Must have the same length as nrows. If None,
        ['default_title'] is used.

    Returns
    -------
    None
    """
    if titles is None:
        titles = ["default_title"]

    fig = get_figure(axes=axes)
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
    axes : mplAxes
        The input axes object.
    """
    if axes is None:
        axes = plt.gca()

    if isinstance(data, pd.Series):
        cells = data.values[:, np.newaxis]
        labels = data.index
    else:
        cells = data

    fig = get_figure(axes=axes)
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
    Print all class attributes. Values are truncated for readability.

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


def compute_tight_bbox(fig, pad_inches: float = 0.0):
    """
    Compute tight bounding box of a figure with specified padding. The width is not
    changed.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    not_tight = fig.bbox
    width, current_height = fig.get_size_inches()

    from matplotlib.transforms import Bbox

    bbox = Bbox.from_bounds(
        not_tight.x0,
        tight.y0 - pad_inches,
        width,
        tight.height + 2 * pad_inches,
    )

    return bbox


def crop_to_content_with_padding(
    in_file, out_file, 
    dpi: int = 300, 
    pad_inches: float = 2/72, 
    threshold: int = 255
) -> None:
    """
    Crops the image to the content and adds padding, then saves the image.
    
    Parameters
    ----------
    in_file
        Path to input image file.
    out_file
        Path to output image file.
    dpi
        DPI for the output image.
    pad_inches
        Padding in inches.
    threshold
        Threshold for determining content.
    """
    im = Image.open(in_file).convert("RGB")

    gray = ImageOps.grayscale(im)

    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()

    pad_px = round(pad_inches * dpi)

    left, upper, right, lower = bbox
    left = max(left - pad_px, 0)
    upper = max(upper - pad_px, 0)
    right = min(right + pad_px, im.width)
    lower = min(lower + pad_px, im.height)

    cropped = im.crop((left, upper, right, lower))

    cropped.save(out_file, compression="tiff_lzw", dpi=(dpi, dpi))
