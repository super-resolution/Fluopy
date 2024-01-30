"""
Module miscellaneous
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def delete_subplots(axes, keep_number=None, del_positions=None):
    """
    Deletes subplots from figure object.

    Parameters
    ----------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    keep_number : None, int
        Number of subplots to keep. Assumes them to be in the first keep_number positions of the flattened ax array.
    del_positions : None, np.ndarray
        An array that contains a 1-D array of shape (2,) for each ax to be deleted like [row, column].

    Returns
    -------
    None
    """
    flattened = axes.flatten()
    fig = flattened[0].get_figure()
    if keep_number is not None:
        for i in range(flattened.size - keep_number):
            fig.delaxes(flattened[-1 - i])
    elif del_positions is not None:
        for position in del_positions:
            fig.delaxes(axes[position[0], position[1]])


def create_row_subtitles(axes, nrows=1, ncols=1, titles=None):
    """
    Creates subtitles of figure displayed in the middle of each row.

    Parameters
    ----------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    nrows : int
        Number of rows in the figure.
    ncols : int
        Number of columns in the figure.
    titles : collection
        Containes elements of type str. Must have the same length as nrows.

    Returns
    -------
    None
    """
    if titles is None:
        titles = ['default_title']

    fig = get_figure(axes)
    grid = plt.GridSpec(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        row = fig.add_subplot(grid[i, ::])
        row.set_title(titles[i], fontsize=22, pad=20, fontweight='bold')
        row.set_frame_on(False)
        row.axis('off')


def add_table(axes, data, labels=None, grid=111, xscale=1, yscale=1, fontsize=12):
    """
    Adds a table to a subplot figure.

    Parameters
    ----------
    axes : np.ndarray
        matplotlib.axes._subplots.AxesSubplots.
    data : 2-D array_like, pd.Series
        If pd.Series, values to display in table with index as labels.
    labels : None, 1-D array_like
        Labels of table rows.
        Only used if data is not pd.Series.
    grid : int
        Divide the figure subplots into an a x b grid. Choose a position c for the table such that it corresponds to
        the index + 1 of the flattened grid.
        Example: suppose a subplot with 2 rows and 3 columns. The table should span the entire lower row, hence half
        of the figure. Divide the figure into 2 rows and 1 column (a = 2, b = 1). The position c is 2. The value to use
        for grid is abc, hence in the example 212.
    xscale : float
        Scale table in x direction.
    yscale : float
        Scale table in y direction.
    fontsize : float
        Set the font size.

    Returns
    -------
    None
    """
    if isinstance(data, pd.Series):
        cells = data.values[:, np.newaxis]
        labels = data.index
    else:
        cells = data

    fig = get_figure(axes)
    new_ax = fig.add_subplot(grid)
    new_ax.axis('off')
    table = new_ax.table(cellText=cells, rowLabels=labels, loc='center')
    table.scale(xscale=xscale, yscale=yscale)
    table.set_fontsize(size=fontsize)


def get_figure(axes):
    if isinstance(axes, np.ndarray):
        flattened = axes.flatten()
        ax = flattened[0]
    else:
        ax = axes
    fig = ax.get_figure()

    return fig
