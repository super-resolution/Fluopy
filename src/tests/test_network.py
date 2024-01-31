import pytest
import src.network as net
import networkx as nx


def test_construct_graph_states(transition_set_object):
    df = transition_set_object.transition_df
    result = net.construct_graph_states(df, numerical=False)
    assert isinstance(result, nx.MultiDiGraph)
    assert list(result.nodes) == ['S0', 'S1', 'S0(2)']
    assert list(result.edges) == [('S0', 'S1', 0), ('S1', 'S0', 0), ('S1', 'S0', 1), ('S1', 'S0(2)', 0),
                                  ('S1', 'S0(2)', 1)]
    assert list(result.edges(data=True)) == [('S0', 'S1', {'w': 'EXC'}), ('S1', 'S0', {'w': 'FLU'}),
                                             ('S1', 'S0', {'w': 'ICS'}), ('S1', 'S0(2)', {'w': 'HFRET(7.0)'}),
                                             ('S1', 'S0(2)', {'w': 'HFRET(9.9)'})]
