import numpy as np
import src.NEW_simulation as si
import src.custom_plot as cp


class TCSPC:
    def __init__(self):
        self.modified_transition_matrix = None
        self.modified_row_sums = None

        self.transition_series = None
        self.time_series = None

        self.observed_lifetimes = None

    def modify_transition_matrix(self, transition_matrix, row_sums, combined_state_transitions_df, number_fluorophores):
        excitations = combined_state_transitions_df[combined_state_transitions_df['abbreviation'] == 'EXC']
        indices_to_modify = excitations.index.values[number_fluorophores:]
        transition_rate_matrix = transition_matrix * row_sums
        transition_rate_matrix[:, indices_to_modify] = 0
        self.modified_row_sums = transition_rate_matrix.sum(axis=1)
        self.modified_transition_matrix = np.divide(transition_rate_matrix,
                                                    np.expand_dims(self.modified_row_sums, axis=1),
                                                    out=np.zeros_like(transition_rate_matrix),
                                                    where=self.modified_row_sums != 0)
        return self.modified_transition_matrix, self.modified_row_sums

    def run_simulation(self, combined_state_transitions_df, start_at, n_steps, seed):
        self.time_series, _, self.transition_series = si.direct_method_steps(self.modified_transition_matrix,
                                                                             self.modified_row_sums,
                                                                             combined_state_transitions_df,
                                                                             start_at, n_steps, seed)

    def get_observed_lifetimes(self, combined_state_transitions_df):
        excitation_values = combined_state_transitions_df[combined_state_transitions_df['abbreviation'] ==
                                                       'EXC'].index.values
        fluorescence_values = combined_state_transitions_df[combined_state_transitions_df['abbreviation'] ==
                                                         'FLU'].index.values
        excitation_indices = np.in1d(self.transition_series, excitation_values).nonzero()[0]
        emission_indices = np.in1d(self.transition_series, fluorescence_values).nonzero()[0]
        excitation_times = self.time_series[excitation_indices]
        emission_times = self.time_series[emission_indices]

        corresponding_excitation_time_indices = np.searchsorted(excitation_times, emission_times, side='right') - 1
        corresponding_excitation_times = excitation_times[corresponding_excitation_time_indices]

        self.observed_lifetimes = emission_times - corresponding_excitation_times

        return self.observed_lifetimes

    def plot(self, **kwargs):
        kwargs.setdefault('type_', 'hist')
        kwargs.setdefault('xlabel', 'observed lifetime [s]')
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)

        fig, ax = cp.universal_figure(data=self.observed_lifetimes, **kwargs)

        return fig, ax
