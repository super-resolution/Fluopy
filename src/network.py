"""
Module network
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def construct_network(transition_df):
    """
    Constructs a network of transitions (edges) and their involved states (nodes).

    Parameters
    ----------
    transition_df : pd.DataFrame
        Dataframe of transitions containing their id as index and their other attributes as columns
        (see src.transitions.Transition).

    Returns
    -------
    network : nx.MultiDiGraph
        Markov Chain representation by nodes and edges.
    """
    from src.transitions import SingleState  # avoids circular import
    network = nx.MultiDiGraph()
    edges = []
    for id, row in transition_df.iterrows():
        abbreviation = row['abbreviation']
        initial_state = row['initial_state']
        if isinstance(initial_state, SingleState):
            final_state = row['final_state']
            source = initial_state.name
            destination = final_state.name
            edge = (source, destination, {'w': abbreviation})
            edges.append(edge)
        else:
            source_1 = initial_state.value[0].name
            source_2 = initial_state.value[1].name
            edge = (source_1, source_2 + '(2)', {'w': abbreviation})
            edges.append(edge)

    network.add_edges_from(edges)

    return network


def plot_network(network, graph_type='shell', colors=None):
    """
    Plot network.
    Adapted from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx.

    Parameters
    ----------
    network : nx.MultiDiGraph
        Markov Chain representation by nodes and edges.
    graph_type : str
        Specifies network layout. One of 'shell', 'circular', 'planar' or 'kamada'.
    colors : Collection
        Contains two colors as Hex values of type str.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    if colors is None:
        colors = ['#ADD8E6', '#FFF0C8']

    g = network
    _, ax = plt.subplots()
    if graph_type == 'circular':
        pos = nx.circular_layout(g)
    elif graph_type == 'planar':
        pos = nx.planar_layout(g)
    elif graph_type == 'shell':
        pos = nx.shell_layout(g)
    else:
        pos = nx.kamada_kawai_layout(g)

    labels = {}
    colormap = []
    for i, node in enumerate(g):
        if '(2)' in node:
            colormap.append(colors[1])
            labels[node] = node.replace('(2)', '')
        else:
            colormap.append(colors[0])
            labels[node] = node
    nx.draw_networkx_nodes(G=g, pos=pos, ax=ax, node_color=colormap)
    nx.draw_networkx_labels(G=g, pos=pos, ax=ax, labels=labels)

    edge_weights = nx.get_edge_attributes(g, name='w')
    straight_edges = []
    arc_rad = 0
    arc_rad_reversed = 0
    for new_edge in g.edges:
        nothing_found = True
        for old_edge in straight_edges:
            if new_edge[:2] == old_edge[:2]:
                arc_rad += 0.25
                nothing_found = False
                nx.draw_networkx_edges(G=g, pos=pos, ax=ax, edgelist=[new_edge],
                                       connectionstyle=f'arc3, rad = {arc_rad}')
                draw_networkx_curved_edge_labels(G=g, pos=pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                 rad=arc_rad)
                break
            elif list(reversed(new_edge[:2])) == list(old_edge[:2]):
                arc_rad_reversed += 0.25
                nothing_found = False
                nx.draw_networkx_edges(G=g, pos=pos, ax=ax, edgelist=[new_edge],
                                       connectionstyle=f'arc3, rad = {arc_rad_reversed}')
                draw_networkx_curved_edge_labels(G=g, pos=pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                rad=arc_rad_reversed)
                break
        if nothing_found:
            arc_rad = 0
            arc_rad_reversed = 0
            straight_edges.append(new_edge)
    nx.draw_networkx_edges(G=g, pos=pos, ax=ax, edgelist=straight_edges)
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    draw_networkx_curved_edge_labels(G=g, pos=pos, ax=ax, edge_labels=straight_edge_labels, rad=0)

    return ax


def draw_networkx_curved_edge_labels(G, pos, ax=None, edge_labels=None, rad=0):
    """
    Draws labels to curved edges.
    Adapted from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx.

    Parameters
    ----------
    G : graph
        A networkx graph.
    pos : dict
        Nodes as keys and positions as values.
    ax : matplotlib.axes._subplots.AxesSubplot
    edge_labels : dict
        Edges (tuples) as keys and labels as values.
    rad : float
        Rounding radius of curved edge.

    Returns
    -------
    None
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

        trans_angle = 0.0
        # use default box of white with white border
        bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(x, y, label, size=10, color='k', family="sans-serif", weight="normal", alpha=None,
                    horizontalalignment="center", verticalalignment="center", rotation=trans_angle,
                    transform=ax.transData, bbox=bbox, zorder=1, clip_on=True)
        text_items[(n1, n2)] = t

    ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
