import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def construct_network(transition_df):
    from src.NEW_transitions import SingleState  # avoids circular import
    network = nx.MultiDiGraph()
    edges = []
    for id, row in transition_df.iterrows():
        abbreviation = row['abbreviation']
        initial_state = row['initial_state']
        final_state = row['final_state']
        if isinstance(initial_state, SingleState):
            source = initial_state.name
            destination = final_state.name
            edge = (source, destination, {'w': row['abbreviation']})
            edges.append(edge)
        else:
            source_1 = initial_state.value[0].name
            source_2 = initial_state.value[1].name
            destination_1 = final_state.value[0].name
            destination_2 = final_state.value[0].name
            edge = (source_1, source_2 + '(2)', {'w': row['abbreviation']})
            edges.append(edge)

    network.add_edges_from(edges)

    return network


def plot_network(network, type='shell', colors=None):
    if colors is None:
        colors = ['#ADD8E6', '#FFF0C8']

    g = network
    fig, ax = plt.subplots()
    if type == 'circular':
        pos = nx.circular_layout(g)
    elif type == 'planar':
        pos = nx.planar_layout(g)
    elif type == 'shell':
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
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colormap)
    nx.draw_networkx_labels(g, pos, ax=ax, labels=labels)

    edge_weights = nx.get_edge_attributes(g, 'w')
    straight_edges = []
    arc_rad = 0
    arc_rad_reversed = 0
    for new_edge in g.edges:
        nothing_found = True
        for old_edge in straight_edges:
            if new_edge[:2] == old_edge[:2]:
                arc_rad += 0.25
                nothing_found = False
                nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[new_edge],
                                       connectionstyle=f'arc3, rad = {arc_rad}')
                draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                 rotate=False, rad=arc_rad)
                break
            elif list(reversed(new_edge[:2])) == list(old_edge[:2]):
                arc_rad_reversed += 0.25
                nothing_found = False
                nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[new_edge],
                                       connectionstyle=f'arc3, rad = {arc_rad_reversed}')
                draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels={new_edge: edge_weights[new_edge]},
                                                 rotate=False, rad=arc_rad_reversed)
                break
        if nothing_found:
            arc_rad = 0
            arc_rad_reversed = 0
            straight_edges.append(new_edge)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=straight_edges)
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    draw_networkx_curved_edge_labels(g, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, rad=0)

    return fig, ax


def draw_networkx_curved_edge_labels(G, pos, edge_labels=None, font_size=10, font_color="k", font_family="sans-serif",
                                     font_weight="normal", alpha=None, bbox=None, horizontalalignment="center",
                                     verticalalignment="center", ax=None, rotate=True, clip_on=True, rad=0):
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