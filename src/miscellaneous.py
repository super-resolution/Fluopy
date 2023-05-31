import matplotlib.pyplot as plt
import numpy as np
import time
import src.fluorophore_systems as fs


def draw_networkx_curved_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color="k",
                                     font_family="sans-serif", font_weight="normal", alpha=None, bbox=None,
                                     horizontalalignment="center", verticalalignment="center", ax=None, rotate=True,
                                     clip_on=True, rad=0):
    """
    Adapted from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx.
    For documentation, see draw_networkx_edge_labels on networkx.org/documentation. In this adaption, the rad parameter
    was added.

    """
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2, _), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos), y1 * label_pos + y2 * (1.0 - label_pos),)
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha,
                    horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=trans_angle,
                    transform=ax.transData, bbox=bbox, zorder=1, clip_on=clip_on)
        text_items[(n1, n2)] = t

    ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    return text_items


def time_complexity(mode, number_fluorophores, simulation_steps_exp, seed, transitions):
    """
    Measures the time durations of simulations and their post-processing steps. Mode 'simulation_steps' runs the
    simulation using 10**3 up to 10**simulation_steps_exp steps. Mode 'number_fluorophores' runs the simulation using
    1 up to number_fluorophores as number of fluorophores.

    Parameters
    ----------
    mode : str
        One of 'simulation_steps', 'number_fluorophores'.
    number_fluorophores : int
        Specifies (maximum) number of fluorophores.
    simulation_steps_exp : int
        Specifies (maximum) 10 ** simulation_steps_exp simulation steps.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.
    transitions : list
            Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
            (str), abbreviation (str), fluorescence (bool)]. In the case of energy transfers, the first entry is
            k_singlestate1_singlestate2__singlestate1_singlestate2, where the first part represents one fluorophore and
            the second part the other fluorophore.
    Returns
    -------
    times : list
        Contains the measured time durations.
    """
    rng = np.random.default_rng(seed)
    times = []
    if mode == 'simulation_steps':
        steps = np.logspace(3, simulation_steps_exp, simulation_steps_exp - 2)
        for step in steps:
            start = time.time()
            system = fs.FluorophoreSystem(number_fluorophores, distances=1, transitions=transitions)
            system.simulate(n_steps=int(step), seed=rng)
            system.process()
            system.emitters(photon_collection_rate=0.5, resample='5ms', emccd_gain=10)
            end = time.time()
            times.append(end - start)
    elif mode == 'number_fluorophores':
        n_fluo = np.arange(1, number_fluorophores + 1, 1)
        for n in n_fluo:
            start = time.time()
            system = fs.FluorophoreSystem(n, distances=1, transitions=transitions)
            system.simulate(n_steps=int(10 ** simulation_steps_exp), seed=rng)
            system.process()
            system.emitters(photon_collection_rate=0.5, resample='5ms', emccd_gain=10)
            end = time.time()
            times.append(end - start)

    return times


def delete_subplots(fig, ax, keep_number=None, del_positions=None):
    """
    Deletes subplots from figure object.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The top level container for all the plot elements.
    ax : matplotlib.axes.Axes
        Contains most of the figure elements.
    keep_number : None, int
        Number of subplots to keep. Assumes them to be in the first keep_number positions of the flattened ax array.
    del_positions : None, np.ndarray
        An array that contains a 1d array of shape (2,) for each ax to be deleted like [row, column].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The altered top level container for all the plot elements.
    """
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
