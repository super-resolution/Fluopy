import pytest
import networkx as nx
import src.network as net


def test_construct_state_graphs(tr_set_bl_et):
    graphs = net.construct_state_graphs(tr_set_bl_et.transition_df)
    states = [
        ["atto643_S1", "cy5_Cis(2)", "cy5_OFF1(2)", "cy5_S0(2)", "cy5_T1(2)"],
        ["atto643_S1", "cy5_Cis(2)", "cy5_OFF1(2)", "cy5_S0(2)", "cy5_T1(2)"],
        ["cy5_S1", "atto643_S0(2)"],
        ["cy5_S1", "atto643_S0(2)"],
        ["cy5_S1", "cy5_Cis(2)", "cy5_OFF1(2)", "cy5_S0(2)", "cy5_T1(2)"],
        ["atto643_S0", "atto643_S1", "atto643_T1", "atto643_B"],
        ["cy5_S0", "cy5_S1", "cy5_T1", "cy5_Cis", "cy5_B"],
    ]
    edges = [
        [
            ("atto643_S1", "cy5_Cis(2)", 0),
            ("atto643_S1", "cy5_OFF1(2)", 0),
            ("atto643_S1", "cy5_S0(2)", 0),
            ("atto643_S1", "cy5_T1(2)", 0),
        ],
        [
            ("atto643_S1", "cy5_Cis(2)", 0),
            ("atto643_S1", "cy5_OFF1(2)", 0),
            ("atto643_S1", "cy5_S0(2)", 0),
            ("atto643_S1", "cy5_T1(2)", 0),
        ],
        [("cy5_S1", "atto643_S0(2)", 0)],
        [("cy5_S1", "atto643_S0(2)", 0)],
        [
            ("cy5_S1", "cy5_Cis(2)", 0),
            ("cy5_S1", "cy5_OFF1(2)", 0),
            ("cy5_S1", "cy5_S0(2)", 0),
            ("cy5_S1", "cy5_T1(2)", 0),
        ],
        [
            ("atto643_S0", "atto643_S1", 0),
            ("atto643_S1", "atto643_S0", 0),
            ("atto643_S1", "atto643_S0", 1),
            ("atto643_S1", "atto643_T1", 0),
            ("atto643_T1", "atto643_S0", 0),
            ("atto643_T1", "atto643_B", 0),
        ],
        [
            ("cy5_S0", "cy5_S1", 0),
            ("cy5_S1", "cy5_S0", 0),
            ("cy5_S1", "cy5_S0", 1),
            ("cy5_S1", "cy5_T1", 0),
            ("cy5_S1", "cy5_Cis", 0),
            ("cy5_T1", "cy5_S0", 0),
            ("cy5_T1", "cy5_B", 0),
            ("cy5_Cis", "cy5_S0", 0),
        ],
    ]
    edges_data = [
        [
            ("atto643_S1", "cy5_Cis(2)", {"w": "CFRET2", "dist": "distance: 1 nm"}),
            ("atto643_S1", "cy5_OFF1(2)", {"w": "OFRET1", "dist": "distance: 1 nm"}),
            ("atto643_S1", "cy5_S0(2)", {"w": "FRET", "dist": "distance: 1 nm"}),
            ("atto643_S1", "cy5_T1(2)", {"w": "STA", "dist": "distance: 1 nm"}),
        ],
        [
            ("atto643_S1", "cy5_Cis(2)", {"w": "CFRET2", "dist": "distance: 2 nm"}),
            ("atto643_S1", "cy5_OFF1(2)", {"w": "OFRET1", "dist": "distance: 2 nm"}),
            ("atto643_S1", "cy5_S0(2)", {"w": "FRET", "dist": "distance: 2 nm"}),
            ("atto643_S1", "cy5_T1(2)", {"w": "STA", "dist": "distance: 2 nm"}),
        ],
        [("cy5_S1", "atto643_S0(2)", {"w": "FRET", "dist": "distance: 1 nm"})],
        [("cy5_S1", "atto643_S0(2)", {"w": "FRET", "dist": "distance: 2 nm"})],
        [
            ("cy5_S1", "cy5_Cis(2)", {"w": "CFRET2", "dist": "distance: 1 nm"}),
            ("cy5_S1", "cy5_OFF1(2)", {"w": "OFRET1", "dist": "distance: 1 nm"}),
            ("cy5_S1", "cy5_S0(2)", {"w": "FRET", "dist": "distance: 1 nm"}),
            ("cy5_S1", "cy5_T1(2)", {"w": "STA", "dist": "distance: 1 nm"}),
        ],
        [
            ("atto643_S0", "atto643_S1", {"w": "EXC", "dist": ""}),
            ("atto643_S1", "atto643_S0", {"w": "FLU", "dist": ""}),
            ("atto643_S1", "atto643_S0", {"w": "ICS", "dist": ""}),
            ("atto643_S1", "atto643_T1", {"w": "ISCST", "dist": ""}),
            ("atto643_T1", "atto643_S0", {"w": "ISCTS", "dist": ""}),
            ("atto643_T1", "atto643_B", {"w": "BLE1", "dist": ""}),
        ],
        [
            ("cy5_S0", "cy5_S1", {"w": "EXC", "dist": ""}),
            ("cy5_S1", "cy5_S0", {"w": "FLU", "dist": ""}),
            ("cy5_S1", "cy5_S0", {"w": "ICS", "dist": ""}),
            ("cy5_S1", "cy5_T1", {"w": "ISCST", "dist": ""}),
            ("cy5_S1", "cy5_Cis", {"w": "ISO", "dist": ""}),
            ("cy5_T1", "cy5_S0", {"w": "ISCTS", "dist": ""}),
            ("cy5_T1", "cy5_B", {"w": "BLE1", "dist": ""}),
            ("cy5_Cis", "cy5_S0", {"w": "BISO", "dist": ""}),
        ],
    ]
    for i, graph in enumerate(graphs):
        assert isinstance(graph, nx.MultiDiGraph)
        assert list(graph.nodes) == states[i]
        assert list(graph.edges) == edges[i]
        assert list(graph.edges.data()) == edges_data[i]


def test_construct_transition_graph(tr_set_bl_et, tr_set_1f_bl):
    with pytest.raises(
        ValueError,
        match="construct_transition_graph only available for single fluorophore "
        "systems.",
    ):
        net.construct_transition_graph(tr_set_bl_et.transition_df)

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
        (4, 5, {"w": "Cis"}),
        (6, 0, {"w": "S0"}),
        (3, 0, {"w": "S0"}),
        (5, 0, {"w": "S0"}),
    ]


def test_check_graph_suitable():
    G = nx.MultiDiGraph([(2, 3), (3, 4), (4, 2), (2, 1)])
    graph_suited, cycles = net.check_graph_suitable(G, starting_node=2)
    assert not graph_suited
    assert cycles == [[2, 3, 4]]

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
