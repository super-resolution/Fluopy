import numpy as np
import src.simulation as si
import src.custom_plot as cp
from src.transitions import SingleState


class TCSPC:
    def __init__(self, transitions):
        if transitions.transition_matrix is None:
            raise ValueError('tcspc not available if transitions not finalized.')
        self.transitions = transitions

        self.modified_transition_matrix, self.modified_row_sums = self.modify_transition_matrix()

        self.transition_series = None
        self.time_series = None

        self.observed_lifetimes = None

    def modify_transition_matrix(self):
        df = self.transitions.combined_state_transitions_df
        excitations = df[df['abbreviation'] == 'EXC']
        indices_to_modify = excitations.index.values[self.transitions.fluorophore_system.count:]
        transition_rate_matrix = self.transitions.transition_matrix * self.transitions.row_sums
        transition_rate_matrix[:, indices_to_modify] = 0
        modified_row_sums = transition_rate_matrix.sum(axis=1)
        modified_transition_matrix = np.divide(transition_rate_matrix, np.expand_dims(modified_row_sums, axis=1),
                                               out=np.zeros_like(transition_rate_matrix), where=modified_row_sums != 0)
        return modified_transition_matrix, modified_row_sums

    def run(self, start_at, n_steps, seed):
        self.time_series, _, self.transition_series = \
            si.direct_method_steps(self.modified_transition_matrix, self.modified_row_sums,
                                   self.transitions.combined_state_transitions_df, start_at, n_steps, seed)

    def get_observed_lifetimes(self):
        if self.transition_series is None:
            raise ValueError('get_observed_lifetimes() not available if run() has not been called.')
        df = self.transitions.combined_state_transitions_df
        excitation_values = df[df['abbreviation'] == 'EXC'].index.values
        fluorescence_values = df[df['abbreviation'] == 'FLU'].index.values
        excitation_indices = np.in1d(self.transition_series, excitation_values).nonzero()[0]
        emission_indices = np.in1d(self.transition_series, fluorescence_values).nonzero()[0]
        excitation_times = self.time_series[excitation_indices]
        emission_times = self.time_series[emission_indices]

        corresponding_excitation_time_indices = np.searchsorted(excitation_times, emission_times, side='right') - 1
        corresponding_excitation_times = excitation_times[corresponding_excitation_time_indices]

        self.observed_lifetimes = emission_times - corresponding_excitation_times

        return self

    def plot(self, **kwargs):
        kwargs.setdefault('type_', 'hist')
        kwargs.setdefault('xlabel', 'observed lifetime [s]')
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)

        fig, ax = cp.universal_figure(data=self.observed_lifetimes, **kwargs)

        return fig, ax

    @staticmethod
    def predict(transition_df, fluorophore_count, accuracy, size, seed):
        # since only one fluorophore gets excited, all the other fluorophores can be seen as S0. Only if homoFRET
        # happens, one fluorophore stays in S1, otherwise S1 is gone. Each energy transfer that starts from S1 has
        # to be considered; if two fluorophores, they have to be considered once. If three fluorophores in equilateral
        # triangle, twice. If four fluorophores in square, the smaller distance rate twice and the larger distance rate
        # once. This has to be considered for homo FRET probability, fluorescence lifetime and fluorescence probability.
        fluorescence_lifetime, hfret_probability, fluorescence_probability = get_transition_probabilities(transition_df)
        rng = np.random.default_rng(seed)


        return None


def get_transition_probabilities(transition_df):
    non_energy_transfer_S1_transitions = transition_df[transition_df['initial_state'] == SingleState.S1]
    fluorescence_lifetime = None
    hfret_probability = None
    fluorescence_probability = None

    return fluorescence_lifetime, hfret_probability, fluorescence_probability
