import matplotlib.pyplot as plt


def delete_subplots(fig, ax, keep_number=None, del_positions=None):
    flattened = ax.flatten()
    if keep_number is not None:
        for i in range(flattened.size - keep_number):
            fig.delaxes(flattened[-1 - i])
    elif del_positions is not None:
        for position in del_positions:
            fig.delaxes(ax[position[0], position[1]])

    return fig


def create_row_subtitles(fig, nrows, ncols, titles):
    grid = plt.GridSpec(nrows, ncols)
    for i in range(nrows):
        row = fig.add_subplot(grid[i, ::])
        row.set_title(titles[i], fontsize=22, pad=20, fontweight='bold')
        row.set_frame_on(False)
        row.axis('off')


def add_table(fig, grid, series, labels, x_scale, y_scale, fontsize):
    new_ax = fig.add_subplot(grid)
    new_ax.axis('off')
    table = new_ax.table(cellText=series.values, rowLabels=labels, loc='center')
    table.scale(x_scale, y_scale)
    table.set_fontsize(fontsize)

    return fig
