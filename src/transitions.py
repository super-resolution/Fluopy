from enum import Enum
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field, asdict
from itertools import product
import src.network as net
import src.formulas as fo
from pathlib import Path


class SingleState(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    T1 = 3
    T2 = 4
    Cis = 5
    OFF = 6
    B = 7


class PairedState(Enum):
    S1_S0 = [SingleState.S1, SingleState.S0]
    S0_S1 = [SingleState.S0, SingleState.S1]
    S1_T1 = [SingleState.S1, SingleState.T1]
    S1_Cis = [SingleState.S1, SingleState.Cis]
    S1_OFF = [SingleState.S1, SingleState.OFF]
    S0_S0 = [SingleState.S0, SingleState.S0]
    S0_T2 = [SingleState.S0, SingleState.T2]

    @property
    def single_state_values(self):
        return self.value[0].value, self.value[1].value


@dataclass
class TransitionAttributes:
    abbreviation: str
    initial_state: SingleState | PairedState
    final_state: SingleState | PairedState
    photon: bool


class TransitionType(Enum):
    EXCITATION = TransitionAttributes('EXC', SingleState.S0, SingleState.S1, False)
    FLUORESCENT_EMISSION = TransitionAttributes('FLU', SingleState.S1, SingleState.S0, True)
    INTERSYSTEM_CROSSING_ST = TransitionAttributes('ISCST', SingleState.S1, SingleState.T1, False)
    INTERSYSTEM_CROSSING_TS = TransitionAttributes('ISCTS', SingleState.T1, SingleState.S0, False)
    INTERNAL_CONVERSION_S = TransitionAttributes('ICS', SingleState.S1, SingleState.S0, False)
    PHOTOBLEACHING_1 = TransitionAttributes('BLE1', SingleState.T1, SingleState.B, False)
    PHOTOBLEACHING_2 = TransitionAttributes('BLE2', SingleState.T2, SingleState.B, False)

    REDUCTION = TransitionAttributes('RED', SingleState.T1, SingleState.OFF, False)
    OXIDATION = TransitionAttributes('OXI', SingleState.OFF, SingleState.S0, False)

    ISOMERIZATION = TransitionAttributes('ISO', SingleState.S1, SingleState.Cis, False)
    BACKISOMERIZATION = TransitionAttributes('BISO', SingleState.Cis, SingleState.S0, False)

    HOMO_FRET = TransitionAttributes('HFRET', PairedState.S1_S0, PairedState.S0_S1, False)
    TRIPLET_FRET = TransitionAttributes('TFRET', PairedState.S1_T1, PairedState.S0_T2, False)
    CIS_FRET = TransitionAttributes('CFRET', PairedState.S1_Cis, PairedState.S0_S0, False)
    OFF_FRET = TransitionAttributes('OFRET', PairedState.S1_OFF, PairedState.S0_S0, False)

    @property
    def abbreviation(self):
        return self.value.abbreviation

    @property
    def initial_state(self):
        return self.value.initial_state

    @property
    def final_state(self):
        return self.value.final_state

    @property
    def photon(self):
        return self.value.photon


@dataclass(slots=True)  # frozen=True if code will not be modified (autoreload complications otherwise)
class Transition:
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
        if isinstance(self.initial_state, PairedState):
            if self.distance is None:
                raise AttributeError('distance has to be defined if transition is energy transfer.')
            object.__setattr__(self, 'energy_transfer', True)
            object.__setattr__(self, 'abbreviation', self.abbreviation + f'({self.distance:.1f})')

        else:
            object.__setattr__(self, 'energy_transfer', False)


class TransitionSet:

    def __init__(self, transitions, fluorophore_system):
        self.transitions = [transition for transition in transitions if transition.rate != 0]
        for i, transition in enumerate(self.transitions):
            transition.id = i
        self.transition_df = pd.DataFrame([asdict(transition) for transition in self.transitions])
        self.transition_df.set_index('id', inplace=True)
        self.single_states = get_single_states(self.transitions)

        self.fluorophore_system = fluorophore_system

        self.combined_state_transitions_df = None
        self.transition_matrix = None
        self.row_sums = None

        self.predicted_lifetime_distributions = None
        self.predicted_transition_time_distributions = None
        self.predicted_state_occurrences = None
        self.predicted_transition_occurrences = None

    def filter_by_abbreviation(self, remove_list):
        filtered_transitions = []
        for transition in self.transitions:
            if not transition.energy_transfer and transition.abbreviation not in remove_list:
                filtered_transitions.append(transition)
            else:
                if transition.abbreviation[:transition.abbreviation.find('(')] not in remove_list:
                    filtered_transitions.append(transition)

        return TransitionSet(transitions=filtered_transitions, fluorophore_system=self.fluorophore_system)

    def adjust_rates(self, change_dict):
        for transition in self.transitions:
            if transition.abbreviation in change_dict:
                transition.rate = change_dict[transition.abbreviation]

    def finalize(self):
        state_combinations = get_state_combinations(self.single_states, self.fluorophore_system.count)
        combined_state_transitions = get_combined_state_transitions(state_combinations)
        combined_state_transitions_with_rates = construct_transition_rate_list(self.transitions,
                                                                               combined_state_transitions,
                                                                               self.fluorophore_system.distances)

        self.combined_state_transitions_df = pd.DataFrame(combined_state_transitions_with_rates,
                                                          columns=['initial_state', 'final_state', 'abbreviation',
                                                                   'transition_id', 'rate', 'photon'])
        self.combined_state_transitions_df.index.name = 'id'

        self.transition_matrix, self.row_sums = construct_transition_matrix(self.combined_state_transitions_df)

        return self

    def plot(self, graph_type='shell', colors=None):
        network = net.construct_network(self.transition_df)
        fig, ax = net.plot_network(network, graph_type, colors)

        return fig, ax


def get_state_combinations(states, fluorophore_count):
    return list(product(states, repeat=fluorophore_count))


def get_combined_state_transitions(state_combinations):
    return list(product(state_combinations, repeat=2))


def rate_assignment_standard(transition, transition_rate_list, combined_state_transitions):
    source = transition.initial_state.value
    destination = transition.final_state.value

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
                        name = transition.abbreviation
                        rate = transition.rate
                        photon = transition.photon
                        transition_id = transition.id
                        transition_rate_list.append([current_state, future_state, name, transition_id, rate, photon])

    return transition_rate_list


def rate_assignment_energy_transfer(transition, transition_rate_list, combined_state_transitions, distance_lookup):
    source_1, source_2 = transition.initial_state.single_state_values
    destination_1, destination_2 = transition.final_state.single_state_values
    distance = transition.distance
    for (current_state, future_state) in combined_state_transitions:
        if source_1 in current_state and source_2 in current_state and destination_1 in future_state and destination_2 \
                in future_state:
            indices_current_1 = [i for i, e in enumerate(current_state) if e == source_1]
            indices_current_2 = [i for i, e in enumerate(current_state) if e == source_2]
            for i in indices_current_1:
                if destination_1 == future_state[i]:
                    for j in indices_current_2:
                        if destination_2 == future_state[j] and distance == distance_lookup[(i, j)]:
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
                                name = transition.abbreviation
                                rate = transition.rate
                                photon = transition.photon
                                transition_id = transition.id
                                transition_rate_list.append([current_state, future_state, name, transition_id, rate,
                                                             photon])

    return transition_rate_list


def construct_transition_rate_list(transitions, combined_state_transitions, distance_lookup):
    transition_rate_list = list()
    for transition in transitions:
        if isinstance(transition.initial_state, SingleState):
            transition_rate_list = rate_assignment_standard(transition, transition_rate_list,
                                                            combined_state_transitions)
        else:
            transition_rate_list = rate_assignment_energy_transfer(transition, transition_rate_list,
                                                                   combined_state_transitions,
                                                                   distance_lookup)
    return transition_rate_list


def get_single_states(transitions):
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

    return single_states


def construct_transition_matrix(combined_state_transitions_df):

    transition_count = combined_state_transitions_df.index.size
    transition_rate_matrix = np.zeros(shape=(transition_count, transition_count))

    for i, row in combined_state_transitions_df.iterrows():
        final_state = row['final_state']
        indices = combined_state_transitions_df[combined_state_transitions_df['initial_state'] == final_state].index
        transition_rate_matrix[i][indices] = combined_state_transitions_df['rate'][indices]

    row_sums = transition_rate_matrix.sum(axis=1)

    transition_matrix = np.divide(transition_rate_matrix, np.expand_dims(row_sums, axis=1),
                                  out=np.zeros_like(transition_rate_matrix), where=row_sums != 0)

    return transition_matrix, row_sums


def get_energy_transfer_transitions(transition_type, distances, rates=None, calculate_rates=None):
    if rates is None and calculate_rates is None:
        return AttributeError('either rates or calculate_rates must be defined.')
    elif rates is not None and calculate_rates is not None:
        return AttributeError('either rates or calculate_rates must be None.')
    distances = np.unique(list(distances.values()))
    if rates is not None:
        energy_transfer_transitions = [Transition(rate=rate, transition_type=transition_type, distance=distance)
                                       for rate, distance in zip(rates, distances)]
    else:
        energy_transfer_transitions = []
        for distance in distances:
            rate = fo.calculate_fret_rate(distance=distance, **calculate_rates)
            energy_transfer_transitions.append(Transition(rate=rate, transition_type=transition_type,
                                                          distance=distance))

    return energy_transfer_transitions


def load_transitions(fluorophore_system, irradiance, wavelength, **dstorm_parameters):
    if not hasattr(fluorophore_system.fluorophores[0], 'attributes'):
        raise ValueError('load_transitions() not available for this kind of fluorophore.')
    data = fluorophore_system.fluorophores[0].attributes
    distances = fluorophore_system.distances

    path_absorption = Path(__file__).parent / r'fluorophores\cy5_absorption.xlsx'
    dataframe_absorption = pd.read_excel(path_absorption, sheet_name=0, index_col='wavelength [nm]')

    wavenumber, wavelength, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
    photon_flux = fo.calculate_photon_flux(irradiance, frequency)
    relative_extinction = dataframe_absorption.loc[int(wavelength), 'relative extinction']
    effective_extinction = relative_extinction * data.maximum_extinction_coefficient

    excitation_rate = fo.calculate_excitation_rate(photon_flux=photon_flux, extinction_coefficient=effective_extinction)
    excitation = Transition(rate=excitation_rate, transition_type=TransitionType.EXCITATION)

    emission_rate = fo.calculate_emission_rate(data.quantum_yield, data.fluorescence_lifetime)
    emission = Transition(rate=emission_rate, transition_type=TransitionType.FLUORESCENT_EMISSION)

    isc_st = Transition(rate=data.isc_st_rate, transition_type=TransitionType.INTERSYSTEM_CROSSING_ST)

    isc_ts = Transition(rate=data.isc_ts_rate, transition_type=TransitionType.INTERSYSTEM_CROSSING_TS)

    isomerization = Transition(rate=data.iso_rate, transition_type=TransitionType.ISOMERIZATION)

    biso_rate = fo.calculate_back_isomerization_rate(photon_flux, data.biso_cross_section)
    bisomerization = Transition(rate=biso_rate, transition_type=TransitionType.BACKISOMERIZATION)

    internal_conversion_rate = fo.calculate_internal_conversion_rate(data.quantum_yield, emission_rate,
                                                                     data.iso_rate, data.isc_st_rate)
    internal_conversion = Transition(rate=internal_conversion_rate,
                                        transition_type=TransitionType.INTERNAL_CONVERSION_S)

    dstorm_red_rate = fo.calculate_reduction_rate(k_pet=data.dstorm_red_rate_mol, **dstorm_parameters)
    dstorm_red = Transition(rate=dstorm_red_rate, transition_type=TransitionType.REDUCTION)

    dstorm_oxi = Transition(rate=data.dstorm_oxi_rate, transition_type=TransitionType.OXIDATION)

    calculate_homo_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': data.j_homo_fret,
                           'dipole_orientation_factor': data.fret_kappa_sq}
    homo_frets = get_energy_transfer_transitions(transition_type=TransitionType.HOMO_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_homo_fret)

    calculate_cis_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': data.j_cis_fret,
                           'dipole_orientation_factor': data.fret_kappa_sq}
    cis_frets = get_energy_transfer_transitions(transition_type=TransitionType.CIS_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_cis_fret)

    calculate_triplet_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': data.j_triplet_fret,
                           'dipole_orientation_factor': data.fret_kappa_sq}
    triplet_frets = get_energy_transfer_transitions(transition_type=TransitionType.TRIPLET_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_triplet_fret)

    calculate_off_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': data.j_off_fret,
                           'dipole_orientation_factor': data.fret_kappa_sq}
    off_frets = get_energy_transfer_transitions(transition_type=TransitionType.OFF_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_off_fret)

    transitions = [excitation, emission, isc_st, isc_ts, isomerization, bisomerization, internal_conversion, dstorm_red,
                   dstorm_oxi] + homo_frets + cis_frets + triplet_frets + off_frets

    return transitions
