from enum import Enum
import numpy as np


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


def state_pairs(number, states=("S0", "S1", "T1", "R", "B")):
    """
    Combines all given states with themselves number of times.

    Parameters
    ----------
    number : int
        The amount of combinations.
    states : iterable object
        Contains elements of type str.

    Returns
    -------
    joined_states : enum.EnumMeta
        The elements e are combined with an underscore between each element. Each combination (i.e., joined_state)
        receives a unique value. The set of combinations can therefore be considered an enumeration.
    """
    state_pair_generator = recursion(number, number, states)

    strings = []
    for state_pair in state_pair_generator:
        string = ""
        for state in state_pair:
            if len(string) > 0:
                string += f"_{state}"
            else:
                string += f"{state}"

        strings.append(string)

    joined_states = Enum("Joined_States", strings, start=0)

    return joined_states


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


def initial_row_vector(transitions):
    """
    Returns an array of zeros of shape (sqrt(len(transitions)),) with a 1 at position 0.

    Parameters
    ----------
    transitions : dict
        The return value of transition_pairs.

    Returns
    -------
    vector : np.ndarray
        Is of shape (sqrt(len(transitions)),) with a 1 at position 0 (else 0).
    """
    uni_dir_shape = int(np.sqrt(len(transitions)))
    vector = np.zeros(shape=uni_dir_shape)
    vector[0] = 1
    return vector


def rate_assignment(assigned_rate_dict, rate_name_dict, transitions, source, destination, rate, name):
    """
    Adds transitions (as keys) and their rates (as values) to assigned_rate_dict, if source is part of the first state
    of the transition and destination is part of the second state of the transition. Additionally, all other
    contributors to the first and second state should be equal.
    Also returns value pairs of these transitions (as keys) and their names (as values) as a dict.

    Parameters
    ----------
    assigned_rate_dict : dict
        Destination of addable transitions and their rates. May be empty.
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
    name : str
        The name of the target transition.

    Returns
    -------
    assigned_rate_dict : dict
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
                        rate_name_dict[value_pair] = name
    return assigned_rate_dict, rate_name_dict


def transition_rate_dict(rates, transitions):
    """
    Constructs a dictionary with transitions as keys and their rates as values and a dictionary with transition value
    pairs as keys and their names as values.

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
        Contains all transitions and their rates if the rate is > 0.
    rate_name_dict : dict
        Contains all transition value pairs and their names if the rate is > 0.
    """
    assigned_rate_dict = dict()
    rate_name_dict = dict()

    for key, value in rates.items():
        split = key.split("_")
        source, destination = split[1], split[2]
        rate = value[0]
        name = value[1]
        assigned_rate_dict, rate_name_dict = rate_assignment(assigned_rate_dict, rate_name_dict, transitions, source,
                                                             destination, rate, name)
    return assigned_rate_dict, rate_name_dict


def absorbing_states(rate_name_dict, state_ids):
    """
    Collects all absorbing states, i.e., the states that have no outgoing transitions and are therefore terminations
    of the Markov chain.

    Parameters
    ----------
    rate_name_dict : dict
        The second return value of transition_rate_dict.
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
    the index [a, b] is equal to a unique value pair of transitions (see transition_pairs) if the transition is listed
    in rates.

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


def predefining(number, states, rates):
    """
    Can be used to initialize a system once instead of running all initializing functions with the same parameters
    multiple times. Return value can be used as input value of parameter 'predefined' of class FluorophoreSystem.

    Parameters
    ----------
    number : int
        Number of fluorophores of the system.
    states : iterable object
        Contains elements of type str.
    rates : dict
        The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and the
        value k assigned to it.

    Returns
    -------
    predefined_args : list
        Contains all crucial return values of initializing functions.
    """
    joined_states = state_pairs(number, states)
    state_names = []
    state_ids = []
    for joined_state in joined_states:
        state_names.append(joined_state.name)
        state_ids.append(joined_state.value)
    transitions = transition_pairs(joined_states)
    assigned_rate_dict, rate_name_dict = transition_rate_dict(rates, transitions)
    vector = initial_row_vector(transitions)
    _, transition_matrix, row_sums = transition_matrices(assigned_rate_dict, transitions)

    predefined_args = [joined_states, state_names, transitions, assigned_rate_dict, rate_name_dict, vector,
                       transition_matrix, row_sums]

    return predefined_args
