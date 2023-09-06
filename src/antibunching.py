"""
Module antibunching
Differences to the TCSPC module are: The excitation laser pulse has a duration. The 'stop' signal (i.e., the time when
fluorescence occurs) is delayed by delay_time.
Note: some sources mention that antibunching experiments can be performed
using continuous excitation light - this case is essentially the output produced by the unaltered version of the
simulation followed by FCS-analysis (the antibunching curve in the very beginning).
"""
import numpy as np
import src.simulation as si
import src.figure as fi


class Antibunching:
    """
    Container of Antibunching-related attributes and methods.

    Attributes
    ----------
    transitions : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    modified_transition_matrix : np.ndarray
            Contains the normalized rate constants (i.e., point probabilities) for each possible
            combined_state_transition at the corresponding index pair. Modified (see modify_transition_matrix()).
    modified_row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
        combined_state_transitions. Modified (see modify_transition_matrix()).
    delayed_lifetimes : 1-D array_like
        Simulated times between photon-driven excitation and fluorescent emission not considering whether the emission
        comes from the originally excited fluorophore.
    """
    def __init__(self, transitions):
        """
        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        """
        if transitions.transition_matrix is None:
            raise ValueError('antibunching not available if transitions not finalized.')
        self.transitions = transitions
        self.modified_transition_matrix, self.modified_row_sums = self.modify_transition_matrix()
        self.delayed_lifetimes = None

    def modify_transition_matrix(self):
        """
        Modifies the transition matrix in a way that prevents fluorophores to be excited. This is achieved by setting
        all excitation rates to zero.

        Returns
        -------
        modified_transition_matrix : np.ndarray
            Contains the normalized rate constants (i.e., point probabilities) for each possible
            combined_state_transition at the corresponding index pair. Modified (see modify_transition_matrix()).
        modified_row_sums : np.ndarray
            Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
            combined_state_transitions. Modified (see modify_transition_matrix()).
        """
        df = self.transitions.combined_state_transitions_df
        excitations = df[df['abbreviation'] == 'EXC']
        indices_to_modify = excitations.index.values
        transition_rate_matrix = self.transitions.transition_matrix * np.expand_dims(self.transitions.row_sums, axis=1)
        transition_rate_matrix[:, indices_to_modify] = 0
        modified_row_sums = transition_rate_matrix.sum(axis=1)
        modified_transition_matrix = np.divide(transition_rate_matrix, np.expand_dims(modified_row_sums, axis=1),
                                               out=np.zeros_like(transition_rate_matrix), where=modified_row_sums != 0)
        return modified_transition_matrix, modified_row_sums

    def run(self, pulse_length=1e-10, pulse_frequency=80, pulse_number=100, start_at=None, seed=None, size=10):
        """
        Runs a simulation based on the direct method of the gillespie algorithm (i.e., stochastic simulation algorithm).

        Parameters
        ----------
        pulse_length : float
            Duration of laser pulse in s. During this time interval of each pulse, excitations can happen.
        pulse_frequency : float
            Frequency of pulse in MHz.
        pulse_number : int
            Number of pulses.
        start_at : None, tuple
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible SingleState values. See
            src.transitions.TransitionSet.single_states.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        self
        """
        if start_at is None:
            start_at = tuple(np.zeros(shape=self.transitions.fluorophore_system.count, dtype=int))
        df = self.transitions.combined_state_transitions_df
        start_index = df[df['final_state'] == start_at].index[0]
        self.time_series, self.transition_series = \
            direct_method_pulse(transition_matrix=self.transitions.transition_matrix,
                                modified_transition_matrix=self.modified_transition_matrix,
                                row_sums=self.transitions.row_sums,
                                modified_row_sums=self.modified_row_sums,
                                start_index=start_index, seed=seed, size=size)

        return self


def direct_method_pulse(transition_matrix, modified_transition_matrix, row_sums, modified_row_sums, start_index, seed,
                        size, pulse_length=1e-10, pulse_frequency=80, pulse_number=100):
    rng = np.random.default_rng(seed)

    current_state_index = start_index

    time_series = [0]
    transition_series = []

    random_numbers = rng.uniform(low=0, high=1, size=(size, 2))

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    m_transition_matrix_sorted_indices = np.argsort(modified_transition_matrix, axis=1)
    sorted_m_transition_matrix = np.take_along_axis(modified_transition_matrix,
                                                    m_transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_m_trm = np.cumsum(sorted_m_transition_matrix, axis=1)

    waiting_time = 1 / (pulse_frequency * 1000 * 1000)
    print(waiting_time)

    i = 0
    j = 1
    for _ in range(pulse_number):
        print('iteration')
        current_time = time_series[-1]

        pulse_end_time = current_time + pulse_length
        next_pulse = current_time + waiting_time

        while time_series[-1] < next_pulse:
            print(time_series[-1])
            if time_series[-1] < pulse_end_time:
                use_row_sums = row_sums
                use_trm_sorted_indices = transition_matrix_sorted_indices
                use_cumsum_sorted_trm = cumsum_sorted_trm
            else:
                use_row_sums = modified_row_sums
                use_trm_sorted_indices = m_transition_matrix_sorted_indices
                use_cumsum_sorted_trm = cumsum_sorted_m_trm

            current_state_lambda = use_row_sums[current_state_index]

            if current_state_lambda == 0:
                break

            transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i - (j - 1) * size + 1, 0])

            sorted_index = np.searchsorted(use_cumsum_sorted_trm[current_state_index],
                                           random_numbers[i - (j - 1) * size + 1, 1])

            next_transition = use_trm_sorted_indices[current_state_index, sorted_index]

            transition_series.append(next_transition)
            time_series.append(time_series[-1] + transition_time)

            i += 1
            if i == j * size - 1:
                random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
                j += 1
            current_state_index = next_transition
        else:
            time_series[-1] = next_pulse
        continue
    return time_series, transition_series

