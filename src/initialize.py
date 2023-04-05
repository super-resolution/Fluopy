import numpy as np
from enum import Enum
from scipy.stats import expon
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
    Combines all given states with themselves number of times.

    Parameters
    ----------
    number : int
        The amount of combinations.
    single_states : iterable object
        Contains elements of type str.

    Returns
    -------
    joined_states : enum.EnumMeta
        The elements e are combined with an underscore between each element. Each combination (i.e., joined_state)
        receives a unique value. The set of combinations can therefore be considered an enumeration.
    single_state_counter : dict
        Contains the joined_states as keys and np.ndarray as values. The values contain the number of single_states
        (at the corresponding index) contained in joined_states.
    single_state_id : dict
        Contains the joined_states as keys and np.ndarray as values. The values contain the single_state indices
        in the correct order. E.g., the key 'S0_S1_S0' will have the value [0, 1, 0].
    """
    state_pair_generator = recursion(number, number, single_states)

    single_states = np.array(single_states)
    single_state_counter = dict()
    single_state_id = dict()
    strings = []
    for state_pair in state_pair_generator:
        string = ""
        blank = np.zeros(shape=len(single_states))
        blank2 = np.zeros(shape=number, dtype=np.int)
        for i, state in enumerate(state_pair):
            index = np.where(single_states == state)
            blank[index] += 1
            blank2[i] = index[0][0]
            if len(string) > 0:
                string += f"_{state}"
            else:
                string += f"{state}"
        single_state_counter[string] = blank
        single_state_id[string] = blank2
        strings.append(string)

    joined_states = Enum("Joined_States", strings, start=0)

    return joined_states, single_state_counter, single_state_id


def transition_pairs(joined_states):
    """
    Combines all of joined_states with themselves and assigns a unique value pair to it.

    Parameters
    ----------
    joined_states : enum.EnumMeta, list
        The return value of state_pairs.
        Alternatively, a list of elements of enum.EnumMeta (see unique_joined_states).

    Returns
    -------
    transitions : dict
        Contains all combinations of joined_states (combined with double underscore) as keys and their unique value pair
        as values.
    """
    trans_pairs = [(joined_state_1.name, joined_state_2.name) for joined_state_1 in joined_states
                   for joined_state_2 in joined_states]
    transitions = {f"{joined_state_1}__{joined_state_2}": (i // len(joined_states), i % len(joined_states))
                   for i, (joined_state_1, joined_state_2) in enumerate(trans_pairs)}

    return transitions


def initial_row_vector(state_ids):
    """
    Returns an array of zeros of shape (len(state_ids),) with a 1 at position 0.

    Parameters
    ----------
    state_ids : Collection
        Contains all state's identification numbers.

    Returns
    -------
    vector : np.ndarray
        Of shape (len(transitions),) with a 1 at position 0 (else 0).
    """
    uni_dir_shape = len(state_ids)
    vector = np.zeros(shape=uni_dir_shape)
    vector[0] = 1
    return vector


def rate_assignment(assigned_rate_dict, rate_id_dict, rate_name_dict, transitions, source, destination, rate,
                    identification, name):
    """
    Adds transitions (as keys) and (inter alia) their rates (as values) to assigned_rate_dict, if source is part of the
    first state of the transition and destination is part of the second state of the transition. Additionally, all other
    contributors to the first and second state should be equal.

    Parameters
    ----------
    assigned_rate_dict : dict
        Destination of addable transitions and their rates. May be empty.
    rate_id_dict : dict
        Destination of addable transition value pairs and their ids. May be empty.
    rate_name_dict : dict
        Destination of addable transition value pairs and their names. May be empty.
    transitions : dict
        The return value of transition_pairs.
    source : str
        Search value of the first state.
    destination : str
        Search value of the second state.
    rate : float
        The rate of the target transition.
    identification : int
        The id of the target transition.
    name : str
        The name of the target transition.

    Returns
    -------
    assigned_rate_dict : dict
        The possibly altered input parameter.
    rate_id_dict : dict
        The possibly altered input parameter.
    rate_name_dict : dict
        The possibly altered input parameter.
    """
    for transition, value_pair in transitions.items():
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
                        assigned_rate_dict[transition] = rate
                        rate_id_dict[value_pair] = identification
                        rate_name_dict[value_pair] = name
    return assigned_rate_dict, rate_id_dict, rate_name_dict


def transition_rate_dict(rates, transitions):
    """
    Constructs dictionaries of transitions, inter alia transitions as keys and rates as values.

    Parameters
    ----------
    rates : dict
        The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and the
        value [k, name_of_transition] assigned to it.
    transitions : dict
        The return value of transition_pairs.

    Returns
    -------
    assigned_rate_dict : dict
        Contains all transitions as keys and their rates as values.
    rate_id_dict : dict
        Contains all transition value pairs as keys and their transition id as values.
    rate_name_dict : dict
        Contains all transition value pairs as keys and their names as values.
    transition_dict : dict
        Contains transition ids as keys and transition names as values.
    """
    assigned_rate_dict = dict()
    rate_id_dict = dict()
    rate_name_dict = dict()
    transition_dict = dict()

    for identification, (key, value) in enumerate(rates.items()):
        split = key.split("_")
        source, destination = split[1], split[2]
        rate = value[0]
        name = value[1]
        transition_dict[identification] = name
        assigned_rate_dict, rate_id_dict, rate_name_dict = rate_assignment(assigned_rate_dict, rate_id_dict,
                                                                           rate_name_dict, transitions, source,
                                                                           destination, rate, identification, name)
    return assigned_rate_dict, rate_id_dict, rate_name_dict, transition_dict


def absorbing_states(rate_name_dict, state_ids):
    """
    Collects all absorbing states, i.e., the states that have no outgoing transitions and are therefore terminations
    of the Markov chain.

    Parameters
    ----------
    rate_name_dict : dict
        The third return value of transition_rate_dict.
    state_ids : Collection
        Contains all state's identification numbers.

    Returns
    -------
    absorb_states : Collection
        Contains all absorbing states.
    """
    absorb_states = state_ids[:]
    for key in rate_name_dict:
        present_state = key[0]
        if present_state in absorb_states:
            absorb_states.remove(present_state)

    return absorb_states


def transition_matrices(rates, transitions):
    """
    Constructs a matrix of shape (sqrt(len(transitions)), sqrt(len(transitions))) with zeros in all positions except if
    the index [a, b] is equal to a unique value pair of transitions (see transition_pairs) which is listed in rates.

    Parameters
    ----------
    rates : dict
        The first return value of transition_rate_dict.
    transitions : dict
        The return value of transition_pairs.

    Returns
    -------
    transition_rate_matrix : np.ndarray
        Contains the rate constants for each transition at the corresponding index pair.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of each row of transition_rate_matrix, i.e., the sum of all transition rates of a state.
    """
    uni_dir_shape = int(np.sqrt(len(transitions)))
    transition_rate_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))

    for transition, rate in rates.items():
        transition_rate_matrix[transitions[transition]] = rate

    transition_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))
    row_sums = transition_rate_matrix.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            transition_matrix[i] = transition_rate_matrix[i] / row_sum

    return transition_rate_matrix, transition_matrix, row_sums


def network(rates):
    """
    Constructs network based on the states (nodes) and their transitions (edges) given.

    Parameters
    ----------
    rates : dict
        The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and the
        value [k, name_of_transition] assigned to it.

    Returns
    -------
    G : nx.DiGraph
        Contains nodes and edges of the Markov chain.
    """
    graph = nx.MultiDiGraph()  # each edge has entry ('node1', 'node2', id of this node combination)
    edges = []
    for key, value in rates.items():
        split = key.split('_')
        source, destination = split[1], split[2]
        edge = (source, destination, {'w': value[1]})
        edges.append(edge)
    graph.add_edges_from(edges)

    return graph


def occupation_time_prediction(single_states, rates):

    mean_lifetimes = np.zeros(len(single_states))
    lifetimes = []
    for i, state in enumerate(single_states):
        total_rate = 0
        for key, value in rates.items():
            split = key.split("_")
            source = split[1]
            if source == state:
                total_rate += value

        mean_lifetime = 1/total_rate
        mean_lifetimes[i] = mean_lifetime
        lifetime_pdf = expon(scale=mean_lifetime)
        lifetimes.append(lifetime_pdf)

    a = np.zeros(shape=(len(single_states), len(single_states)))
    b = np.zeros(len(single_states))
    result = np.linalg.solve(a, b)
    occurrences = result
