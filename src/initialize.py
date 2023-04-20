import numpy as np
import networkx as nx


def recursion(number, original_number, iterable, collector=None):
    """
    Combines all elements of iterable with elements of iterable original_number of times.

    Parameters
    ----------
    number : int
        When this function is first called, it should be equal to original_number.
    original_number : int
        The amount of combinations.
    iterable : iterable object
        Contains the elements that are to be combined with themselves.
    collector : None, bool
        When this function is first called, it should be equal to None.

    Returns
    -------
    Generator object
        To access the elements, iterate through the generator object.
    """
    if collector is None:
        collector = []
    if number >= 1:
        for i in iterable:  # the outer for loop (if number is max) still runs, even if it is interrupted by new
            # function calls with smaller numbers and their for loops
            collector.append(i)
            yield from recursion(number - 1, original_number, iterable, collector)
            collector.pop()
    else:
        if len(collector) > original_number:  # this is to shift the collector to the left
            # this is needed because .pop() only removes the last item but if n outer loops continue, collector is
            # appended n times too often (and at the wrong position, too)
            diff = len(collector) - original_number
            for pos in range(diff):
                del collector[original_number - 1 - (pos+1)]
        yield collector


def state_pairs(number, single_states):
    """
    Combines all given states with themselves, separated by an underscore, number of times.

    Parameters
    ----------
    number : int
        The amount of combinations.
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        state.

    Returns
    -------
    joined_states : dict
        Combined single states as keys, list of two arrays as values. The first array contains the corresponding
        single state ids, the second array contains the number of each single state.
    """
    state_pair_generator = recursion(number, number, single_states)

    joined_states = dict()
    single_states = np.array(single_states)
    for state_pair in state_pair_generator:
        string = ""
        blank = np.zeros(shape=len(single_states), dtype=int)
        blank2 = np.zeros(shape=number, dtype=int)
        for i, state in enumerate(state_pair):
            index = np.where(single_states == state)
            blank[index] += 1
            blank2[i] = index[0][0]
            if len(string) > 0:
                string += f"_{state}"
            else:
                string += f"{state}"
        joined_states[string] = [blank2, blank]

    return joined_states


def transition_pairs(joined_states):
    """
    Combines all entries of the 'name' column of joined_states with themselves and assigns their 'id' pair to them.

    Parameters
    ----------
    joined_states : pd.DataFrame
        Contains name (str), single_states (array) and single_state_counter (array) of each joined state, where their id
        is the index.

    Returns
    -------
    joined_transitions : dict
        Contains all combinations of joined_states (combined with double underscore) as keys and their id pairs as
        values.
    """
    joined_transitions = {f"{joined_state_1}__{joined_state_2}": (i, j)
                          for i, joined_state_1 in zip(joined_states.index, joined_states['name'])
                          for j, joined_state_2 in zip(joined_states.index, joined_states['name'])}

    return joined_transitions


def rate_assignment(transition_rate_list, joined_transitions, source, destination, rate,
                    identification, name, fluorescence):
    """
    Adds a transition in list form (see construct_transition_rate_list) to the assigned_rate_list, if source is part of the first
    joined state of the transition and destination is part of the second joined state of the transition. All other
    contributors to the first and second joined state should be constant.

    Parameters
    ----------
    transition_rate_list : iterable
        Destination of addable transitions and their rates. May be empty.
    joined_transitions : dict
        The return value of transition_pairs.
        Contains all combinations of joined_states (combined with double underscore) as keys and their id pairs as
        values.
    source : str
        Search value of the first state.
    destination : str
        Search value of the second state.
    rate : float
        The rate of the target transition.
    identification : int
        The id of the target transition.
    name : str
        The trivial name of the target transition.
    fluorescence : bool
        Whether the transition emits fluorescence.

    Returns
    -------
    transition_rate_list : list
        The possibly altered input parameter.
    """
    for transition, value_pair in joined_transitions.items():
        current_state, future_state = transition.split("__")
        current_state_split = current_state.split("_")
        future_state_split = future_state.split("_")
        if source in current_state_split:
            indices_current = [i for i, e in enumerate(current_state_split) if e == source]
            for i in indices_current:
                if destination in future_state_split[i]:
                    future_state_part = future_state_split[:i] + future_state_split[i+1:]
                    current_state_part = current_state_split[:i] + current_state_split[i+1:]
                    if not future_state_part == current_state_part:
                        break
                    else:
                        transition_rate_list.append([transition, value_pair, identification, rate, name, fluorescence])

    return transition_rate_list


def construct_transition_rate_list(single_transitions, joined_transitions):
    """
    Constructs a list that contains lists of each possible transition from one joined state to another.

    Parameters
    ----------
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.
    joined_transitions : dict
        The return value of transition_pairs.
        Contains all combinations of joined_states (combined with double underscore) as keys and their id pairs as
        values.

    Returns
    -------
    transition_rate_list : list
        Contains lists of each possible transition. These lists contain the following:
            - name of joined transition, e.g. S0_S1__S0_S0
            - joined state id pair, e.g. (1, 0)
            - single transition id, e.g. 1
            - rate of transition
            - trivial name of transition, e.g. fluorescent emission
            - fluorescence boolean, e.g. True
    """
    transition_rate_list = list()

    for identification, row in single_transitions.iterrows():
        name = row['name']
        split = name.split("_")
        source, destination = split[1], split[2]

        rate = row['rate']
        trivial_name = row['trivial_name']
        fluorescence = row['fluorescence']
        transition_rate_list = rate_assignment(transition_rate_list, joined_transitions, source, destination, rate,
                                               identification, trivial_name, fluorescence)
    return transition_rate_list


def add_absorbing_states(joined_states, joined_transitions):
    """
    Adds a column to joined_states with information whether the state is an absorbing state, i.e., the state has no
    outgoing transitions and therefore terminates the Markov chain.

    Parameters
    ----------
    joined_states : pd.DataFrame
        Contains name (str), single_states (array) and single_state_counter (array) of each joined state, where their id
        is the index.
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), single_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.

    Returns
    -------
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.
    """

    joined_states.loc[:, 'absorbing'] = True
    for joined_state_id in joined_transitions['joined_states_id']:
        present_state = joined_state_id[0]
        joined_states.loc[present_state, 'absorbing'] = False

    return joined_states


def contruct_transition_matrices(joined_transitions, joined_states):
    """
    Constructs a matrix of shape (joined_states.index.size, joined_states.index.size) with zeros in all positions except
    if the position equals a joined_states id pair listed in joined_transitions. If it does, it is set equal to the
    corresponding rate. If the id pair is listed multiple times within joined_transitions, the total rate is assigned.
    From there, the return values (see below) are created.

    Parameters
    ----------
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), single_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.

    Returns
    -------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of all transition rates of a
        state.
    """
    uni_dir_shape = joined_states.index.size
    transition_rate_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))

    for rate, index in zip(joined_transitions['rate'], joined_transitions['joined_states_id']):
        transition_rate_matrix[index] += rate  # all transitions effecting the very same
        # source/destination states are added up

    transition_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))
    row_sums = transition_rate_matrix.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            transition_matrix[i] = transition_rate_matrix[i] / row_sum

    return transition_matrix, row_sums


def construct_network(single_transitions):
    """
    Constructs network based on the states (nodes) and their transitions (edges) given.

    Parameters
    ----------
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.

    Returns
    -------
    network : nx.DiGraph
        Contains nodes and edges of the Markov chain.
    """
    network = nx.MultiDiGraph()  # each edge has entry ('node1', 'node2', id of this node combination)
    edges = []

    for identification, row in single_transitions.iterrows():
        name = row['name']
        split = name.split('_')
        source, destination = split[1], split[2]
        edge = (source, destination, {'w': row['trivial_name']})
        edges.append(edge)
    network.add_edges_from(edges)

    return network
