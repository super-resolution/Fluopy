"""
Module network
"""

import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault


def construct_state_graphs(transition_df):
    """
    Constructs graphs of states (nodes) and their transitions (edges). Each fluorophore
    or fluorophore combination gets a separate graph.

    Parameters
    ----------
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as
        second level index and their other attributes as columns. Name of fluorophores
        as first level index.

    Returns
    -------
    graphs : list
        Contains objects of type nx.MultiDiGraph.
    """
    graphs = []
    grouped = transition_df.groupby(level=0)
    for fluorophore, f_transitions in grouped:
        G = nx.MultiDiGraph()
        edges = []
        for (_, _), transition in f_transitions.iterrows():
            abbr = transition["abbreviation"]
            if not "dist" in fluorophore:
                source = transition["initial_state"].name
                destination = transition["final_state"].name
                edge = (
                    fluorophore + "_" + source,
                    fluorophore + "_" + destination,
                    {"w": abbr, "dist": ""},
                )
                edges.append(edge)
            else:
                pattern = re.compile(r"D: (\w+), A: (\w+), dist: (\w+)")
                d, a, dist = pattern.findall(fluorophore)[0]
                source_1 = transition["initial_state"].value[0].name
                source_2 = transition["initial_state"].value[1].name
                edge = (
                    d + "_" + source_1,
                    a + "_" + source_2 + "(2)",
                    {"w": abbr, "dist": f"distance: {dist} nm"},
                )
                edges.append(edge)
        G.add_edges_from(edges)
        graphs.append(G)

    return graphs


def construct_transition_graph(transition_df):
    """
    Constructs a graph of transitions (nodes) and their involved states (edges).

    Parameters
    ----------
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as
        second level index and their other attributes as columns. Name of fluorophores
        as first level index.

    Returns
    -------
    G : nx.MultiDiGraph
        Markov chain representation by nodes and edges.
    """
    if transition_df.index.get_level_values(0).nunique() > 1:
        raise ValueError(
            "construct_transition_graph only available for single "
            "fluorophore systems."
        )
    G = nx.MultiDiGraph()
    edges = []
    for (_, id_source), row in transition_df.iterrows():
        final_state = row["final_state"]
        for (_, id_destination), row in transition_df.iterrows():
            if row["initial_state"] == final_state:
                source = id_source
                destination = id_destination
                edge = (source, destination, {"w": f"{final_state.name}"})
                edges.append(edge)
    G.add_edges_from(edges)

    return G


def check_graph_suitable(G, starting_node):
    """
    Checks whether a Markov chain is suitable for an approximation of its development
    in time. This means being acyclic (except cycles that include the starting node).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Markov Chain representation by nodes and edges.
    starting_node : int
        Numeric value representing the starting node (i.e., state).

    Returns
    -------
    graph_suited : bool
        Whether the graph is suited for the algorithms.
    cycles : list
        Contains each simple cycle of G.
    """
    # check for reversible reactions and loops that do not contain the starting node:
    graph_suited = True
    cycles = list(nx.simple_cycles(G))
    for cycle in cycles:
        if starting_node not in cycle:
            graph_suited = False

    return graph_suited, cycles


def determine_node_order(G, starting_node):
    """
    Determine the order of nodes of a graph such that each node that leads to another
    node has been visited before the other node. Requires the graph to be a DAG
    (directed acyclic graph). If the starting node is part of each cycle, it can be
    removed to convert the graph to DAG.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Markov Chain representation by nodes and edges.
    starting_node : int
        Numeric value representing the starting node (i.e., state).

    Returns
    -------
    node_order : generator
        Yields the topological sort of the graph.
    """
    G_mutated = G.copy()
    edges_to_remove = []
    for edge in G.edges:
        if edge[1] == starting_node:
            edges_to_remove.append(edge)
    G_mutated.remove_edges_from(edges_to_remove)
    node_order = nx.topological_sort(G_mutated)

    return node_order


def plot_graph(G, graph_type="shell", colors=None, scale=1):
    """
    Plot graph.
    Adapted from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-
    between-two-nodes-with-networkx.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Markov Chain representation by nodes and edges.
    graph_type : str
        Specifies network layout. One of 'shell', 'circular', 'planar' or 'kamada'.
    colors : Collection
        Contains two colors as Hex values of type str.
    scale : float
        Factor to scale the figure.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    if colors is None:
        colors = ["#ADD8E6", "#FFF0C8"]
    rcParams["figure.dpi"] = rcParamsDefault["figure.dpi"] * scale
    _, ax = plt.subplots()
    if graph_type == "circular":
        pos = nx.circular_layout(G)
    elif graph_type == "planar":
        pos = nx.planar_layout(G)
    elif graph_type == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    labels = {}
    colormap = []

    for i, node in enumerate(G):
        if isinstance(node, str) and "(2)" in node:
            colormap.append(colors[1])
            labels[node] = node.replace("(2)", "")
        else:
            colormap.append(colors[0])
            labels[node] = node
    nx.draw_networkx_nodes(G=G, pos=pos, ax=ax, node_color=colormap)
    nx.draw_networkx_labels(G=G, pos=pos, ax=ax, labels=labels)

    edge_weights = nx.get_edge_attributes(G, name="w")
    straight_edges = []
    arc_rad = 0
    arc_rad_reversed = 0
    for i, new_edge in enumerate(G.edges):
        if i == 0:
            distance = nx.get_edge_attributes(G, name="dist")[new_edge]
            ax.set_title(distance)
        nothing_found = True
        for old_edge in straight_edges:
            if new_edge[:2] == old_edge[:2]:
                arc_rad += 0.25
                nothing_found = False
                nx.draw_networkx_edges(
                    G=G,
                    pos=pos,
                    ax=ax,
                    edgelist=[new_edge],
                    connectionstyle=f"arc3, rad = {arc_rad}",
                )
                draw_networkx_curved_edge_labels(
                    G=G,
                    pos=pos,
                    ax=ax,
                    edge_labels={new_edge: edge_weights[new_edge]},
                    rad=arc_rad,
                )
                break
            elif list(reversed(new_edge[:2])) == list(old_edge[:2]):
                arc_rad_reversed += 0.25
                nothing_found = False
                nx.draw_networkx_edges(
                    G=G,
                    pos=pos,
                    ax=ax,
                    edgelist=[new_edge],
                    connectionstyle=f"arc3, rad = {arc_rad_reversed}",
                )
                draw_networkx_curved_edge_labels(
                    G=G,
                    pos=pos,
                    ax=ax,
                    edge_labels={new_edge: edge_weights[new_edge]},
                    rad=arc_rad_reversed,
                )
                break
        if nothing_found:
            arc_rad = 0
            arc_rad_reversed = 0
            straight_edges.append(new_edge)
    nx.draw_networkx_edges(G=G, pos=pos, ax=ax, edgelist=straight_edges)
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    draw_networkx_curved_edge_labels(
        G=G, pos=pos, ax=ax, edge_labels=straight_edge_labels, rad=0
    )

    return ax


def draw_networkx_curved_edge_labels(G, pos, ax=None, edge_labels=None, rad=0):
    """
    Draws labels to curved edges.
    Adapted from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-
    between-two-nodes-with-networkx.

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
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        trans_angle = 0.0
        # use default box of white with white border
        bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=10,
            color="k",
            family="sans-serif",
            weight="normal",
            alpha=None,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=True,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
