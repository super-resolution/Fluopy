import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from fluopy import network as net


def test_construct_state_graphs(tr_set_bl_et_3f):
    graphs = net.construct_state_graphs(tr_set_bl_et_3f.transition_df)
    states = [
        ["testfluo_1_S1", "testfluo_1_S0(2)", "testfluo_1_T1(2)"],
        ["testfluo_1_S1", "testfluo_2_S0(2)"],
        ["testfluo_1_S1", "testfluo_2_S0(2)"],
        ["testfluo_2_S1", "testfluo_1_S0(2)", "testfluo_1_T1(2)"],
        ["testfluo_2_S1", "testfluo_1_S0(2)", "testfluo_1_T1(2)"],
        [
            "testfluo_1_S0",
            "testfluo_1_S1",
            "testfluo_1_T1",
            "testfluo_1_cis",
            "testfluo_1_B",
        ],
        ["testfluo_2_S0", "testfluo_2_S1", "testfluo_2_T1", "testfluo_2_B"],
    ]
    edges = [
        [
            ("testfluo_1_S1", "testfluo_1_S0(2)", 0),
            ("testfluo_1_S1", "testfluo_1_T1(2)", 0),
        ],
        [("testfluo_1_S1", "testfluo_2_S0(2)", 0)],
        [("testfluo_1_S1", "testfluo_2_S0(2)", 0)],
        [
            ("testfluo_2_S1", "testfluo_1_S0(2)", 0),
            ("testfluo_2_S1", "testfluo_1_T1(2)", 0),
        ],
        [
            ("testfluo_2_S1", "testfluo_1_S0(2)", 0),
            ("testfluo_2_S1", "testfluo_1_T1(2)", 0),
        ],
        [
            ("testfluo_1_S0", "testfluo_1_S1", 0),
            ("testfluo_1_S1", "testfluo_1_S0", 0),
            ("testfluo_1_S1", "testfluo_1_S0", 1),
            ("testfluo_1_S1", "testfluo_1_T1", 0),
            ("testfluo_1_S1", "testfluo_1_cis", 0),
            ("testfluo_1_T1", "testfluo_1_S0", 0),
            ("testfluo_1_T1", "testfluo_1_B", 0),
            ("testfluo_1_cis", "testfluo_1_S0", 0),
        ],
        [
            ("testfluo_2_S0", "testfluo_2_S1", 0),
            ("testfluo_2_S1", "testfluo_2_S0", 0),
            ("testfluo_2_S1", "testfluo_2_S0", 1),
            ("testfluo_2_S1", "testfluo_2_T1", 0),
            ("testfluo_2_T1", "testfluo_2_S0", 0),
            ("testfluo_2_T1", "testfluo_2_B", 0),
        ],
    ]
    edges_data = [
        [
            (
                "testfluo_1_S1",
                "testfluo_1_S0(2)",
                {"w": "FRET", "dist": "distance: 1.0 nm"},
            ),
            (
                "testfluo_1_S1",
                "testfluo_1_T1(2)",
                {"w": "STA", "dist": "distance: 1.0 nm"},
            ),
        ],
        [
            (
                "testfluo_1_S1",
                "testfluo_2_S0(2)",
                {"w": "FRET", "dist": "distance: 1.0 nm"},
            )
        ],
        [
            (
                "testfluo_1_S1",
                "testfluo_2_S0(2)",
                {"w": "FRET", "dist": "distance: 2.0 nm"},
            )
        ],
        [
            (
                "testfluo_2_S1",
                "testfluo_1_S0(2)",
                {"w": "FRET", "dist": "distance: 1.0 nm"},
            ),
            (
                "testfluo_2_S1",
                "testfluo_1_T1(2)",
                {"w": "STA", "dist": "distance: 1.0 nm"},
            ),
        ],
        [
            (
                "testfluo_2_S1",
                "testfluo_1_S0(2)",
                {"w": "FRET", "dist": "distance: 2.0 nm"},
            ),
            (
                "testfluo_2_S1",
                "testfluo_1_T1(2)",
                {"w": "STA", "dist": "distance: 2.0 nm"},
            ),
        ],
        [
            ("testfluo_1_S0", "testfluo_1_S1", {"w": "EXC", "dist": ""}),
            ("testfluo_1_S1", "testfluo_1_S0", {"w": "FLU", "dist": ""}),
            ("testfluo_1_S1", "testfluo_1_S0", {"w": "IC", "dist": ""}),
            ("testfluo_1_S1", "testfluo_1_T1", {"w": "ISC_ST", "dist": ""}),
            ("testfluo_1_S1", "testfluo_1_cis", {"w": "ISO", "dist": ""}),
            ("testfluo_1_T1", "testfluo_1_S0", {"w": "ISC_TS", "dist": ""}),
            ("testfluo_1_T1", "testfluo_1_B", {"w": "BLE", "dist": ""}),
            ("testfluo_1_cis", "testfluo_1_S0", {"w": "PBISO", "dist": ""}),
        ],
        [
            ("testfluo_2_S0", "testfluo_2_S1", {"w": "EXC", "dist": ""}),
            ("testfluo_2_S1", "testfluo_2_S0", {"w": "FLU", "dist": ""}),
            ("testfluo_2_S1", "testfluo_2_S0", {"w": "IC", "dist": ""}),
            ("testfluo_2_S1", "testfluo_2_T1", {"w": "ISC_ST", "dist": ""}),
            ("testfluo_2_T1", "testfluo_2_S0", {"w": "ISC_TS", "dist": ""}),
            ("testfluo_2_T1", "testfluo_2_B", {"w": "BLE", "dist": ""}),
        ],
    ]

    for i, graph in enumerate(graphs):
        assert isinstance(graph, nx.MultiDiGraph)
        assert list(graph.nodes) == states[i]
        assert list(graph.edges) == edges[i]
        assert list(graph.edges.data()) == edges_data[i]


def test_construct_transition_graph(tr_set_bl_et_3f, tr_set_1f_bl):
    with pytest.raises(
        ValueError,
        match="construct_transition_graph only available for single fluorophore "
        "systems.",
    ):
        net.construct_transition_graph(tr_set_bl_et_3f.transition_df)

    graph = net.construct_transition_graph(tr_set_1f_bl.transition_df)
    assert isinstance(graph, nx.MultiDiGraph)
    assert list(graph.nodes) == [0, 1, 2, 4, 6, 3, 7, 5]
    assert list(graph.edges) == [
        (0, 1, 0),
        (0, 2, 0),
        (0, 4, 0),
        (0, 6, 0),
        (1, 0, 0),
        (2, 3, 0),
        (2, 7, 0),
        (4, 5, 0),
        (6, 0, 0),
        (3, 0, 0),
        (5, 0, 0),
    ]
    assert list(graph.edges.data()) == [
        (0, 1, {"w": "S1"}),
        (0, 2, {"w": "S1"}),
        (0, 4, {"w": "S1"}),
        (0, 6, {"w": "S1"}),
        (1, 0, {"w": "S0"}),
        (2, 3, {"w": "T1"}),
        (2, 7, {"w": "T1"}),
        (4, 5, {"w": "cis"}),
        (6, 0, {"w": "S0"}),
        (3, 0, {"w": "S0"}),
        (5, 0, {"w": "S0"}),
    ]


def test_check_graph_suitable():
    G = nx.MultiDiGraph([(2, 3), (3, 4), (4, 2), (2, 1), (1, 2)])
    graph_suited, cycles = net.check_graph_suitable(G, starting_node=1)
    assert not graph_suited
    assert cycles == [[1, 2], [2, 3, 4]]

    G = nx.MultiDiGraph([(2, 3), (3, 4), (4, 2), (2, 1), (1, 2)])
    graph_suited, cycles = net.check_graph_suitable(G, starting_node=2)
    assert graph_suited
    assert cycles == [[1, 2], [2, 3, 4]]


def test_determine_node_order():
    G = nx.MultiDiGraph([(2, 3), (4, 1), (3, 4), (4, 2), (2, 1), (1, 2)])
    node_order = net.determine_node_order(G, starting_node=2)
    exp_node_order = [2, 3, 4, 1]
    for i, node in enumerate(node_order):
        assert node == exp_node_order[i]


def test_plot_graph():
    G = nx.MultiDiGraph()
    ax = net.plot_graph(G=G, graph_type="shell", colors=None, scale=1)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.visual
def test_plot_graph_visual():
    G = nx.MultiDiGraph()
    ax = net.plot_graph(G=G, graph_type="shell", colors=None, scale=1)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.show()


def test_draw_networkx_curved_edge_labels():
    G = nx.MultiDiGraph()
    pos = {}
    ax = net.draw_networkx_curved_edge_labels(
        G=G, pos=pos, ax=None, edge_labels=None, rad=0
    )
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.visual
def test_draw_networkx_curved_edge_labels_visual():
    G = nx.MultiDiGraph()
    pos = {}
    ax = net.draw_networkx_curved_edge_labels(
        G=G, pos=pos, ax=None, edge_labels=None, rad=0
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.show()
