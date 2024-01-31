"""
Module transitions
"""
from enum import Enum
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field, asdict
from itertools import product
import src.network as net
import src.formulas as fo
import warnings
import src.fluorophores as fl


class SingleState(Enum):
    """
    Assigns a unique identifier (value) to each possible photophysical state.
    """
    S0 = 0
    S1 = 1
    S2 = 2
    T1 = 3
    T2 = 4
    Cis = 5
    OFF = 6
    B = 7
    OFF2 = 8


class PairedState(Enum):
    """
    Assigns a combination of SingleState to each energy transfer related paired state. E.g., the classical Förster
    resonance energy transfer needs one fluorophore to be in S1 and another fluorophore closeby to be in S0. After the
    transition, the first fluorophore will be in S0 and the other in S1.
    """
    S1_S0 = [SingleState.S1, SingleState.S0]
    S0_S1 = [SingleState.S0, SingleState.S1]
    S1_T1 = [SingleState.S1, SingleState.T1]
    S1_Cis = [SingleState.S1, SingleState.Cis]
    S0_Cis = [SingleState.S0, SingleState.Cis]
    S1_OFF = [SingleState.S1, SingleState.OFF]
    S0_S0 = [SingleState.S0, SingleState.S0]
    S0_T2 = [SingleState.S0, SingleState.T2]
    S1_S1 = [SingleState.S1, SingleState.S1]
    S0_T1 = [SingleState.S0, SingleState.T1]
    S0_OFF2 = [SingleState.S0, SingleState.OFF2]

    @property
    def single_state_values(self):
        """
        Returns a tuple of SingleState values.
        """
        return self.value[0].value, self.value[1].value

    @property
    def acceptor(self):
        """
        Returns the acceptor (second value).
        """
        return self.value[1]

    @property
    def donor(self):
        """
        Returns the donor (first value).
        """
        return self.value[0]


@dataclass
class TransitionAttributes:
    """
    Contains constant attributes of photophysical transitions.

    Attributes
    ----------
    abbreviation : str
        Abbreviation of the transition.
    initial_state : SingleState, PairedState
        Initial state of the transition.
    final_state : SingleState, PairedState
        Final state of the transition.
    photon : bool
        Whether the transition emits a photon.
    """
    abbreviation: str
    initial_state: SingleState | PairedState
    final_state: SingleState | PairedState
    photon: bool


class TransitionType(Enum):
    """
    Assigns constant attributes to each possible photophysical transition.
    """
    # general
    EXCITATION = TransitionAttributes('EXC', SingleState.S0, SingleState.S1, False)
    FLUORESCENT_EMISSION = TransitionAttributes('FLU', SingleState.S1, SingleState.S0, True)
    INTERSYSTEM_CROSSING_ST = TransitionAttributes('ISCST', SingleState.S1, SingleState.T1, False)
    INTERSYSTEM_CROSSING_TS = TransitionAttributes('ISCTS', SingleState.T1, SingleState.S0, False)
    INTERNAL_CONVERSION_S = TransitionAttributes('ICS', SingleState.S1, SingleState.S0, False)
    SINGLET_QUENCHING = TransitionAttributes('SQ', SingleState.S1, SingleState.S0, False)
    PHOTOBLEACHING_1 = TransitionAttributes('BLE1', SingleState.T1, SingleState.B, False)
    PHOTOBLEACHING_2 = TransitionAttributes('BLE2', SingleState.T2, SingleState.B, False)
    REVERSE_INTERSYSTEM_CROSSING = TransitionAttributes('RISC', SingleState.T2, SingleState.S1, False)

    # dstorm
    ET_CYCLE_T = TransitionAttributes('ETT', SingleState.T1, SingleState.S0, False)
    ET_CYCLE_S = TransitionAttributes('ETS', SingleState.S1, SingleState.S0, False)
    REDUCTION_T = TransitionAttributes('REDT', SingleState.T1, SingleState.OFF, False)
    REDUCTION_S = TransitionAttributes('REDS', SingleState.S1, SingleState.OFF, False)
    OXIDATION = TransitionAttributes('OXI', SingleState.OFF, SingleState.S0, False)
    OXIDATION_2 = TransitionAttributes('OXI2', SingleState.OFF2, SingleState.S0, False)

    # cis trans isomerization
    ISOMERIZATION = TransitionAttributes('ISO', SingleState.S1, SingleState.Cis, False)
    BACKISOMERIZATION = TransitionAttributes('BISO', SingleState.Cis, SingleState.S0, False)

    # energy transfers
    HOMO_FRET = TransitionAttributes('HFRET', PairedState.S1_S0, PairedState.S0_S1, False)
    CIS_FRET_1 = TransitionAttributes('CFRET', PairedState.S1_Cis, PairedState.S0_Cis, False)
    CIS_FRET_2 = TransitionAttributes('CFRET2', PairedState.S1_Cis, PairedState.S0_S0, False)
    OFF_FRET = TransitionAttributes('OFRET', PairedState.S1_OFF, PairedState.S0_S0, False)
    OFF_FRET_2 = TransitionAttributes('OFRET2', PairedState.S1_OFF, PairedState.S0_OFF2, False)
    S_S_ANNIHILATION = TransitionAttributes('SSA', PairedState.S1_S1, PairedState.S0_S1, False)
    S_T_ANNIHILATION = TransitionAttributes('STA', PairedState.S1_T1, PairedState.S0_T1, False)

    # rhodamines
    H2O_ATTACK_S = TransitionAttributes('H2OS', SingleState.S1, SingleState.OFF, False)
    BACK_REACTION = TransitionAttributes('BR', SingleState.OFF, SingleState.S0, False)
    H2O_ATTACK_T = TransitionAttributes('H2OT', SingleState.T1, SingleState.OFF, False)

    @property
    def abbreviation(self):
        """
        Returns the abbreviation of type str.
        """
        return self.value.abbreviation

    @property
    def initial_state(self):
        """
        Returns the initial state of type SingleState or PairedState.
        """
        return self.value.initial_state

    @property
    def final_state(self):
        """
        Returns the final state of type SingleState or PairedState.
        """
        return self.value.final_state

    @property
    def photon(self):
        """
        Returns bool indicating whether the transition emits a photon.
        """
        return self.value.photon


@dataclass(slots=True)  # frozen=True if code will not be modified (autoreload complications otherwise)
class Transition:
    """
    Contains constant and variable attributes of photophysical transitions.

    Attributes
    ----------
    id : int
        The id of the transition. Not None if transition is part of a TransitionSet.
    transition_type : TransitionType
        The photophysical type of the transitions with its constant attributes.
    abbreviation : str
        The abbreviation of the transition.
    initial_state : SingleState, PairedState
        The initial state of the transition.
    final_state : SingleState, PairedState
        The final state of the transition.
    rate : float
        The rate of the transition.
    photon : bool
        Whether the transition emits a photon.
    energy_transfer : bool
        Whether the transition is an energy transfer.
    distance : None, float
        Not None if the transition is an energy transfer. The distance of the two involved fluorophores.
    """
    id: int = field(init=False)
    transition_type: TransitionType = field()
    abbreviation: str = field(init=False)
    initial_state: SingleState | PairedState = field(init=False)
    final_state: SingleState | PairedState = field(init=False)
    rate: float = field()
    photon: bool = field(init=False)
    energy_transfer: bool = field(init=False)
    distance: Optional[float] = None

    def __post_init__(self):
        # __setattr__ needed if frozen=True
        object.__setattr__(self, 'abbreviation', self.transition_type.abbreviation)
        object.__setattr__(self, 'initial_state', self.transition_type.initial_state)
        object.__setattr__(self, 'final_state', self.transition_type.final_state)
        object.__setattr__(self, 'photon', self.transition_type.photon)
        object.__setattr__(self, 'id', None)
        if isinstance(self.initial_state, PairedState):
            if self.distance is None:
                raise AttributeError('distance has to be defined if transition is energy transfer.')
            object.__setattr__(self, 'distance', np.round(self.distance, 3))
            object.__setattr__(self, 'energy_transfer', True)
            object.__setattr__(self, 'abbreviation', self.abbreviation + f'({self.distance:.1f})')

        else:
            object.__setattr__(self, 'energy_transfer', False)


class TransitionSet:
    """
    Collection of all relevant transitions and related attributes. Allows optional post-init-modification and
    (subsequent) finalization.

    Attributes
    ----------
    transitions : Collection
        Contains all given transitions of type Transition with non-zero rate.
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as index and their other attributes as
        columns.
    single_states : 1-D array_like
        Contains the values of all relevant SingleStates.
    fluorophore_system : src.fluorophores.FluorophoreSystem
        Container for attributes of multiple, interrelated fluorophores.
    combined_state_transitions_df : pd.DataFrame
        Contains realizable combined_state_transitions with their id as index and their other attributes as columns.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possbile
        combined_state_transitions.
    """
    def __init__(self, transitions, fluorophore_system):
        """
        Parameters
        ----------
        transitions : Collection
            Contains transitions of type Transition.
        fluorophore_system : src.fluorophores.FluorophoreSystem
            Container for attributes of multiple, interrelated fluorophores.
        """
        self.fluorophore_system = fluorophore_system
        self.transitions = [transition for transition in transitions if transition.rate != 0]
        for i, transition in enumerate(self.transitions):
            transition.id = i
        self.transition_df = pd.DataFrame([asdict(transition) for transition in self.transitions])
        self.transition_df.set_index('id', inplace=True)
        
        ets = self.transition_df[self.transition_df['energy_transfer']].groupby(['transition_type'], sort=False)

        for _, et in ets:
            unusable_transitions = list(set(et['distance']) - set(fluorophore_system.distances.values()))
            unusable_distances = list(set(fluorophore_system.distances.values()) - set(et['distance']))
            if len(unusable_transitions) > 0:
                warnings.warn(f'{et["transition_type"].iloc[0].name} distance(s) {unusable_transitions} ' +
                              'do not correspond to the fluorophore distance(s).', stacklevel=2)
            if len(unusable_distances) > 0: 
                warnings.warn(f'{et["transition_type"].iloc[0].name} does not include the fluorophore ' +
                              f'distance(s) {unusable_distances}.', stacklevel=2)
            
        self.single_states = get_single_states(self.transitions, self.transition_df)

        self.combined_state_transitions_df = None
        self.transition_matrix = None
        self.row_sums = None

    def filter_by_abbreviation(self, remove_list=None):
        """
        Removes transitions by their abbreviation.

        Parameters
        ----------
        remove_list : Collection
            Contains abbreviations of type str. Transitions with abbreviations contained in this list are removed. If
            the abbreviation of an energy transfer is not specified by its distance, all energy transfers of this type
            will be removed.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        if remove_list is None:
            remove_list = []
        filtered_transitions = []
        for transition in self.transitions:
            if not transition.energy_transfer and transition.abbreviation not in remove_list:
                filtered_transitions.append(transition)
            elif not transition.energy_transfer:
                pass
            else:
                if transition.abbreviation[:transition.abbreviation.find('(')] not in remove_list:
                    if transition.abbreviation not in remove_list:
                        filtered_transitions.append(transition)

        return TransitionSet(transitions=filtered_transitions, fluorophore_system=self.fluorophore_system)

    def adjust_rates(self, change_dict=None):
        """
        Modify rates of transitions.

        Parameters
        ----------
        change_dict : dict
            Contains abbreviations of transitions as key and rates as values.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        if change_dict is None:
            change_dict = {}
        for transition in self.transitions:
            if transition.abbreviation in change_dict:
                transition.rate = change_dict[transition.abbreviation]

        return TransitionSet(transitions=self.transitions, fluorophore_system=self.fluorophore_system)

    def finalize(self):
        """
        Construct combined_state_transitions_df, transition_matrix and row_sums.

        Returns
        -------
        self
        """
        state_combinations = get_state_combinations(states=self.single_states, repeat=self.fluorophore_system.count)
        combined_state_transitions = get_combined_state_transitions(state_combinations=state_combinations)
        combined_state_transitions_with_rates = \
            construct_transition_rate_list(transition_df=self.transition_df,
                                           combined_state_transitions=combined_state_transitions,
                                           distance_lookup=self.fluorophore_system.distances)

        self.combined_state_transitions_df = pd.DataFrame(combined_state_transitions_with_rates,
                                                          columns=['initial_state', 'final_state', 'abbreviation',
                                                                   'transition_id', 'rate', 'photon'])
        self.combined_state_transitions_df.index.name = 'id'

        self.transition_matrix, self.row_sums = \
            construct_transition_matrix(combined_state_transitions_df=self.combined_state_transitions_df)

        return self
   

    def reduce(self):
        """
        Return another TransitionSet that contains no transitions to absorbing states.
        If energy transfers are not possible, the system is additionally reduced to one fluorophore.
        
        Returns
        -------
        TransitionSet
            Modified instance.
        """
        transitions = self.transitions.copy()
        remove_list = []
        for i, single_state in enumerate(self.single_states):
            single_state = SingleState(single_state)
            if single_state not in self.transition_df['initial_state'].values:
                remove_list.append(self.transition_df[self.transition_df['final_state'] == single_state].index)
        transitions = [transition for i, transition in enumerate(transitions) if i not in remove_list]
        if not (self.transition_df['energy_transfer'] == True).any() and self.fluorophore_system.count > 1:
            fluorophore_system = fl.FluorophoreSystem(fluorophores=[self.fluorophore_system.fluorophores[0]])
        else:
            fluorophore_system = self.fluorophore_system
        reduced_set = TransitionSet(transitions=transitions, fluorophore_system=fluorophore_system)
        reduced_set.finalize()

        return reduced_set



    def plot(self, graph_type='shell', colors=None, scale=1):
        """
        Plot photophysical system as network/graph.

        Parameters
        ----------
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
        graph = net.construct_graph_states(transition_df=self.transition_df, numerical=False)
        ax = net.plot_graph(G=graph, graph_type=graph_type, colors=colors, scale=scale)

        return ax


def get_single_states(transitions, transition_df):
    """
    Gets the values of SingleStates that occur in transitions.

    Parameters
    ----------
    transitions : Collection
        Contains transitions of type Transition with non-zero rate.
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as index and their other attributes as
        columns.

    Returns
    -------
    single_states : np.ndarray
        Contains the values of all relevant SingleStates.
    """
    single_states = []
    for transition in transitions:
        initial_state = transition.initial_state
        final_state = transition.final_state
        if isinstance(initial_state, SingleState):
            if initial_state.value not in single_states:
                single_states.append(initial_state.value)
            if final_state.value not in single_states:
                single_states.append(final_state.value)
        else:
            for initial_st, final_st in zip(initial_state.value, final_state.value):
                if initial_st.value not in single_states:
                    single_states.append(initial_st.value)
                if final_st.value not in single_states:
                    single_states.append(final_st.value)
    single_states = np.array(single_states)
    single_state_df = pd.DataFrame(single_states, columns=['single_states'])
    single_state_df['absorbing'] = False
    for i, single_state in single_state_df['single_states'].items():
        if single_state not in transition_df['initial_state'].apply(lambda x: x.value).values:
            single_state_df.at[i, 'absorbing'] = True
    final_states = transition_df['final_state'].apply(lambda x: x.value).values
    absorbing_states = single_state_df['single_states'][single_state_df['absorbing']]
    transition_df['absorbing'] = False
    if not absorbing_states.empty:
        indices = np.where(final_states == absorbing_states.values)[0]
        transition_df.loc[indices, 'absorbing'] = True
    
    return single_states


def get_state_combinations(states, repeat=2):
    """
    Combines all given states with themselves repeat times. Cartesian product, see itertools.product().

    Parameters
    ----------
    states : np.ndarray
        States to be combined.
    repeat : int
        Number of repetitions.

    Returns
    -------
    list
        Contains state combinations of type tuple.
    """
    return list(product(states, repeat=repeat))


def get_combined_state_transitions(state_combinations):
    """
    Combines all given state_combinations with themselves 2 times. Cartesian product, see itertools.product().
    Each combination resembles a combined_state_transition.

    Parameters
    ----------
    state_combinations : Collection
        state_combinations to be combined.

    Returns
    -------
    list
        Contains combinations of state_combinations of type tuple.
    """
    return list(product(state_combinations, repeat=2))


def rate_assignment_standard(transition, transition_rate_list, combined_state_transitions):
    """
    Adds a realizable combined_state_transition that is no energy transfer as a list to the transition_rate_list.
    Here, a combined_state_transition is realizable, if its first state_combination (i.e., current_state) and its second
    state_combination (i.e., future_state) have and only have a change in the state of one fluorophore that can also be
    found in the photophysical transition.

    Parameters
    ----------
    transition : Tuple
        Contains the index and the Transition as a pd.Series.
    transition_rate_list : Collection
        Destination of realizable combined_state_transitions.
    combined_state_transitions : Collection
        Contains combinations of state_combinations of type tuple.

    Returns
    -------
    transition_rate_list : list
        The altered input parameter.
    """
    transition_id = transition[0]
    transition = transition[1]
    source = transition['initial_state'].value
    destination = transition['final_state'].value

    for (current_state, future_state) in combined_state_transitions:
        if source in current_state:
            indices_current = [i for i, e in enumerate(current_state) if e == source]
            for i in indices_current:
                if destination == future_state[i]:
                    future_state_part = future_state[:i] + future_state[i+1:]
                    current_state_part = current_state[:i] + current_state[i+1:]
                    if not future_state_part == current_state_part:
                        break
                    else:
                        transition_rate_list.append([current_state, future_state, transition['abbreviation'],
                                                     transition_id, transition['rate'], transition['photon']])

    return transition_rate_list


def rate_assignment_energy_transfer(transition, transition_rate_list, combined_state_transitions, distance_lookup):
    """
    Adds a realizable combined_state_transition that is also an energy transfer as a list to the transition_rate_list.
    Here, a combined_state_transition is realizable, if its first state_combination (i.e., current_state) and its second
    state_combination (i.e., future_state) have and only have a change in the state of two fluorophores that can also be
    found in the photophysical transition. Also, the distance property of the transition has to match the distance of
    the two respective fluorophores.

    Parameters
    ----------
    transition : Tuple
        Contains the index and the Transition as a pd.Series.
    transition_rate_list : Collection
        Destination of realizable combined_state_transitions.
    combined_state_transitions : Collection
        Contains combinations of state_combinations of type tuple.
    distance_lookup : dict
        Contains tuples of 2 fluorophore ids as keys and their distance as values.

    Returns
    -------
    transition_rate_list : list
        The altered input parameter.
    """
    transition_id = transition[0]
    transition = transition[1]
    source_donor, source_acceptor = transition['initial_state'].single_state_values
    destination_donor, destination_acceptor = transition['final_state'].single_state_values
    distance = transition['distance']
    for (current_state, future_state) in combined_state_transitions:
        if source_donor in current_state and source_acceptor in current_state and destination_donor in future_state \
                and destination_acceptor in future_state:
            indices_current_1 = [i for i, e in enumerate(current_state) if e == source_donor]
            indices_current_2 = [i for i, e in enumerate(current_state) if e == source_acceptor]
            for i in indices_current_1:
                if destination_donor == future_state[i]:
                    for j in indices_current_2:
                        if destination_acceptor == future_state[j] and distance == distance_lookup[(i, j)]:
                            if i > j:
                                future_state_part = future_state[:j] + future_state[j+1:i] + \
                                                    future_state[i+1:]
                                current_state_part = current_state[:j] + current_state[j+1:i] + \
                                    current_state[i+1:]
                            else:
                                future_state_part = future_state[:i] + future_state[i+1:j] + \
                                                    future_state[j+1:]
                                current_state_part = current_state[:i] + current_state[i+1:j] + \
                                    current_state[j+1:]
                            if not future_state_part == current_state_part:
                                break
                            else:
                                transition_rate_list.append([current_state, future_state, transition['abbreviation'],
                                                             transition_id, transition['rate'], transition['photon']])

    return transition_rate_list


def construct_transition_rate_list(transition_df, combined_state_transitions, distance_lookup):
    """
    Constructs a list that contains lists of each realizable combined_state_transition. The inner lists contain
    initial state_combination, final state_combination, abbreviation, transition id, rate and whether a photon is
    emitted.

    Parameters
    ----------
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as index and their other attributes as
        columns.
    combined_state_transitions : Collection
        Contains combinations of state_combinations of type tuple.
    distance_lookup : dict
        Contains tuples of 2 fluorophore ids as keys and their distance as values.

    Returns
    -------
    transition_rate_list : list
        Contains lists of each realizable combined_state_transition.
    """
    transition_rate_list = list()
    for transition in transition_df.iterrows():
        if isinstance(transition[1]['initial_state'], SingleState):
            transition_rate_list = \
                rate_assignment_standard(transition=transition, transition_rate_list=transition_rate_list,
                                         combined_state_transitions=combined_state_transitions)
        else:
            transition_rate_list = \
                rate_assignment_energy_transfer(transition=transition, transition_rate_list=transition_rate_list,
                                                combined_state_transitions=combined_state_transitions,
                                                distance_lookup=distance_lookup)
    return transition_rate_list


def construct_transition_matrix(combined_state_transitions_df):
    """
    Constructs a matrix of shape (combined_state_transitions_df.index.size, combined_state_transitions_df.index.size).
    The matrix is non-zero at a position, if the first index is a final_state in combined_state_transition_df which
    is the initial_state of the second index. In other words, the matrix is non-zero, if a transition (first index or
    row) can be followed by another transition (second index or column).

    Parameters
    ----------
    combined_state_transitions_df : pd.DataFrame
        Contains realizable combined_state_transitions with their id as index and their other attributes as columns.

    Returns
    -------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possbile
        combined_state_transitions.
    """
    transition_count = combined_state_transitions_df.index.size
    transition_rate_matrix = np.zeros(shape=(transition_count, transition_count))

    for i, row in combined_state_transitions_df.iterrows():
        final_state = row['final_state']
        indices = combined_state_transitions_df[combined_state_transitions_df['initial_state'] == final_state].index
        transition_rate_matrix[i][indices] = combined_state_transitions_df['rate'][indices]

    row_sums = transition_rate_matrix.sum(axis=1)
    row_sums_exp = np.tile(np.expand_dims(row_sums, axis=1), row_sums.size)
    mask = np.ma.make_mask(row_sums_exp)
    transition_matrix = np.divide(transition_rate_matrix, row_sums_exp,
                                  out=np.zeros_like(transition_rate_matrix), where=mask)

    return transition_matrix, row_sums


def get_energy_transfer_transitions(transition_type, distances, rates=None, calculate_rates=None):
    """
    Gets an energy transfer transition of transition_type for each unique distance present in distances.

    Parameters
    ----------
    transition_type : TransitionType
        Type of the energy transfer transition.
    distances : dict
        Contains tuples of 2 fluorophore ids as keys and their distance as values.
    rates : None, array_like
        Contains (unsorted) rate values.
    calculate_rates : None, dict
        Contains the keys emission_rate, spectral_overlap_integral, dipole_orientation_factor.

    Returns
    -------
    energy_transfer_transitions : Collection
        Contains energy transfer transitions of type Transition.
    """
    if rates is None and calculate_rates is None:
        raise AttributeError('either rates or calculate_rates must be defined.')
    elif rates is not None and calculate_rates is not None:
        raise AttributeError('either rates or calculate_rates must be None.')

    distances = np.unique(list(distances.values()))
    if rates is not None:
        rates_sorted = np.sort(rates)
        rates_sorted_flip = np.flip(rates_sorted)
        distances_sorted = np.sort(distances)
        energy_transfer_transitions = [Transition(rate=rate, transition_type=transition_type, distance=distance)
                                       for rate, distance in zip(rates_sorted_flip, distances_sorted)]
    else:
        energy_transfer_transitions = []
        for distance in distances:
            rate = fo.calculate_fret_rate(distance=distance, **calculate_rates)
            energy_transfer_transitions.append(Transition(rate=rate, transition_type=transition_type,
                                                          distance=distance))

    return energy_transfer_transitions
