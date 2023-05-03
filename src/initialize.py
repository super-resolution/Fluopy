import pandas as pd
import numpy as np
import networkx as nx
import src.formulas as fo


def determine_rate_constants(path, irradiance, wavelength, fluorophore, dstorm_parameters):
    """
    Determines rate constants and constructs list of transitions based on fluorophore-specific excel files.

    Parameters
    ----------
    path : str
        Path of directory containing excel files.
    irradiance : float
        The irradiance in kW/cm².
    wavelength : float
        In nm.
    fluorophore : str
        'cy5' or else.
    dstorm_parameters : dict
        May contain the following keys: reducing_agent, concentration, k_pet, ph, same.

    Returns
    -------
    transitions : list
        Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
        (str), abbreviation (str), fluorescence (bool)].
    """
    transitions = []

    wavenumber, wavelength, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
    photon_flux = fo.calculate_photon_flux(irradiance, frequency)

    path_properties = path + fluorophore + '_properties.xlsx'
    path_absorption = path + fluorophore + '_absorption.xlsx'
    dataframe_properties = pd.read_excel(path_properties, sheet_name=0, index_col='property')
    dataframe_absorption = pd.read_excel(path_absorption, sheet_name=0, index_col='wavelength [nm]')

    relative_extinction = dataframe_absorption.loc[int(wavelength), 'relative extinction']
    maximum_extinction_coefficient = dataframe_properties.loc['maximum extinction coefficient [1/(cm M)]', 'value']
    quantum_yield = dataframe_properties.loc['quantum yield', 'value']
    fluorescence_lifetime = dataframe_properties.loc['fluorescence lifetime [s]', 'value']

    effective_extinction = relative_extinction * maximum_extinction_coefficient
    excitation_rate = fo.calculate_excitation_rate(photon_flux=photon_flux, extinction_coefficient=effective_extinction)
    transitions.append(['k_S0_S1', excitation_rate, 'excitation', 'EXC', False])

    emission_rate = fo.calculate_emission_rate(quantum_yield, fluorescence_lifetime)
    transitions.append(['k_S1_S0', emission_rate, 'fluorescent emission', 'FLU', True])

    isc_st_rate = dataframe_properties.loc['intersystem crossing ST rate [1/s]', 'value']
    transitions.append(['k_S1_T1', isc_st_rate, 'intersystem crossing ST', 'ISCST', False])

    if fluorophore == 'cy5':
        isomerization_rate = dataframe_properties.loc['isomerization [1/s]', 'value']
        back_isomerization_cross_section = dataframe_properties.loc['back-isomerization cross section [cm²]', 'value']
        back_isomerization_rate = fo.calculate_back_isomerization_rate(photon_flux, back_isomerization_cross_section)
        internal_conversion_rate = fo.calculate_internal_conversion_rate(quantum_yield, emission_rate,
                                                                         isomerization_rate, isc_st_rate)
        transitions.append(['k_S1_S0', internal_conversion_rate, 'internal conversion S', 'ICS', False])
        transitions.append(['k_S1_Cis', isomerization_rate, 'isomerization', 'ISO', False])
        transitions.append(['k_Cis_S0', back_isomerization_rate, 'backisomerization', 'BISO', False])
    else:
        internal_conversion_rate = fo.calculate_internal_conversion_rate(quantum_yield, emission_rate, isc_st_rate)
        transitions.append(['k_S1_S0', internal_conversion_rate, 'internal conversion S', 'ICS', False])

    isc_ts_rate = dataframe_properties.loc['intersystem crossing TS rate [1/s]', 'value']
    transitions.append(['k_T1_S0', isc_ts_rate, 'intersystem crossing TS', 'ISCTS', False])

    dstorm_reduction_rate = dataframe_properties.loc['dstorm reduction rate [1/(s M)]', 'value']
    dstorm_reduction_rate = fo.calculate_reduction_rate(k_pet=dstorm_reduction_rate, **dstorm_parameters)
    transitions.append(['k_T1_OFF', dstorm_reduction_rate, 'reduction', 'RED', False])

    dstorm_oxidation_rate = dataframe_properties.loc['dstorm oxidation rate [1/s]', 'value']
    transitions.append(['k_OFF_S0', dstorm_oxidation_rate, 'oxidation', 'OX', False])

    photobleaching_rate = dataframe_properties.loc['photobleaching rate [1/s]', 'value']
    transitions.append(['k_T1_B', photobleaching_rate, 'photobleaching', 'BLE', False])

    return transitions


def extract_single_states(transitions):
    """
    Extracts the single states based on the description of each transition (like k_S1_S0).

    Parameters
    ----------
    transitions : list
            Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
            (str), abbreviation (str), fluorescence (bool)].

    Returns
    -------
    single_states : list
        Contains elements of type str describing each single state a single fluorophore can occupy.
    """
    single_states = []
    for transition in transitions:
        description = transition[0]
        _, source, destination = description.split('_')
        if source not in single_states:
            single_states.append(source)
        if destination not in single_states:
            single_states.append(destination)

    return single_states


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
    Combines all given single states with themselves, separated by an underscore, number of times.

    Parameters
    ----------
    number : int
        The amount of combinations.
    single_states : list
        Contains elements of type str describing each single state a single fluorophore can occupy.

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
    Adds a transition in list form (see construct_transition_rate_list) to the transition_rate_list, if source is part
    of the first joined state of the transition and destination is part of the second joined state of the transition.
    All other contributors to the first and second joined state should be constant.

    Parameters
    ----------
    transition_rate_list : iterable
        Destination of addable transitions and their rates. May be empty.
    joined_transitions : dict
        The return value of transition_pairs.
        Contains all combinations of joined_states (combined with double underscore) as keys and their id pairs as
        values.
    source : str
        Search value of the first joined state.
    destination : str
        Search value of the second joined state.
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
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.
    joined_transitions : dict
        The return value of transition_pairs.
        Contains all combinations of joined_states (combined with double underscore) as keys and their id pairs as
        values.

    Returns
    -------
    transition_rate_list : list
        Contains lists of each possible transition. These lists contain the following:
            - name of joined transition (str), e.g. 'S0_S1__S0_S0'
            - joined state id pair (tuple), e.g. (1, 0)
            - single transition id (int), e.g. 1
            - rate of transition (float), e.g. 7e9
            - trivial name of transition (str), e.g. 'fluorescent emission'
            - fluorescence (bool), e.g. True
    """
    transition_rate_list = list()

    for identification, row in single_transitions.iterrows():
        name = row['name']
        split = name.split("_")
        source, destination = split[1], split[2]

        rate = row['rate']
        trivial_name = row['trivial_name']
        fluorescence = row['fluorescence']
        # noinspection PyTypeChecker
        transition_rate_list = rate_assignment(transition_rate_list, joined_transitions, source, destination, rate,
                                               identification, trivial_name, fluorescence)

    return transition_rate_list


def add_absorbing_states(joined_states, joined_transitions):
    """
    Adds a column to joined_states with information whether the joined state is absorbing, i.e., the joined state has no
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


def construct_transition_matrices(joined_transitions, joined_states):
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
        joined state.
    """
    uni_dir_shape = joined_states.index.size
    transition_rate_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))

    for rate, index in zip(joined_transitions['rate'], joined_transitions['joined_states_id']):
        transition_rate_matrix[index] += rate  # all transitions effecting the very same
        # source/destination joined states are added up

    transition_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))
    row_sums = transition_rate_matrix.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            transition_matrix[i] = transition_rate_matrix[i] / row_sum

    return transition_matrix, row_sums


def construct_network(single_transitions):
    """
    Constructs network based on the single states (nodes) and their transitions (edges) given.

    Parameters
    ----------
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.

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
        edge = (source, destination, {'w': row['abbreviation']})
        edges.append(edge)
    network.add_edges_from(edges)

    return network
