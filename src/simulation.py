"""
Module simulation
"""
import numpy as np


class Simulation:
    """
    Container of simulation-associated attributes and methods.

    Attributes
    ----------
    transitions : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to state_series[i] and transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    state_series : np.ndarray
        Contains 1-D array_like for each fluorophore representing its state at index i corresponding to time_series[i].
    """
    def __init__(self, transitions):
        """
        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        """
        if transitions.transition_matrix is None:
            raise ValueError('simulation not available if transitions not finalized.')
        self.transitions = transitions
        self.time_series = None
        self.transition_series = None
        self.state_series = None

    def run(self, start_at=None, size=int(1e5), end_time=None, seed=None):
        """
        Runs a simulation based on the direct method of the gillespie algorithm (i.e., stochastic simulation algorithm).
        Can either be based on maxmimum number of steps or maximum total time.

        Parameters
        ----------
        start_at : None, tuple
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible SingleState values. See
            transitions.single_states.
        size : int
            If end_time is None, serves as maximum number of simulation steps.
            If end_time is not None, serves as size of random_numbers drawn at once.
        end_time : None, float
            If not None, time at which simulation ends.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if start_at is None:
            start_at = tuple(np.zeros(shape=self.transitions.fluorophore_system.count, dtype=int))
        df = self.transitions.combined_state_transitions_df
        start_index = df[df['final_state'] == start_at].index[0]
        if end_time is None:
            self.time_series, self.transition_series = \
                direct_method_steps(transition_matrix=self.transitions.transition_matrix,
                                    row_sums=self.transitions.row_sums,
                                    start_index=start_index, size=size, seed=seed)
        else:
            self.time_series, self.transition_series = \
                direct_method_time(transition_matrix=self.transitions.transition_matrix,
                                   row_sums=self.transitions.row_sums,
                                   start_index=start_index, size=size, end_time=end_time, seed=seed)

        final_states = self.transitions.combined_state_transitions_df['final_state']

        self.state_series = np.empty(shape=(len(final_states[0]), self.time_series.size), dtype=np.int64)
        self.state_series[:, 0] = start_at

        for i, _ in enumerate(final_states[0]):
            final_states_fluorophore = final_states.map(lambda x: x[i]).to_numpy()
            self.state_series[i][1:] = final_states_fluorophore[self.transition_series]


def direct_method_steps(transition_matrix, row_sums, start_index=0, size=10, seed=None):
    """
    The direct method of the gillespie algorithm (i.e., stoschastic simulation algorithm). Here, the propensities are
    equal to the rate constants because the population is always 1. Additionally, the state change vector is redundant
    because each transition leads the a shift in populations by 1.
    This version is based on a maximum number of steps.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possbile
        combined_state_transitions.
    start_index : int
        Starting index. The final state of the transition indexed is the starting state configuration.
    size : int
        Maximum number of simulation steps.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    time_step_series = np.empty(size + 1)
    time_step_series[0] = 0
    transition_series = np.empty(size, dtype=np.int64)

    # a random index (in this case the first index) at which the final state of a transition equals start_at
    current_state_index = start_index

    random_numbers = rng.uniform(low=0, high=1, size=(size, 2))

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    for i in range(size):
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:  # there is no outgoing transition
            # the Markov chain has encountered an absorbing state
            time_step_series = time_step_series[:i + 1]
            transition_series = transition_series[:i + 1]
            break
        # inverse transform sampling using the quantile function of the exponential distribution
        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index], random_numbers[i, 1])

        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = current_state_index = next_transition

        time_step_series[i + 1] = transition_time

    time_series = np.cumsum(time_step_series)

    return time_series, transition_series


def direct_method_time(transition_matrix, row_sums, start_index=0, size=10, end_time=10, seed=None):
    """
    The direct method of the gillespie algorithm (i.e., stoschastic simulation algorithm). Here, the propensities are
    equal to the rate constants because the population is always 1. Additionally, the state change vector is redundant
    because each transition leads the a shift in populations by 1.
    This version is based on a maxmimum total time.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possbile
        combined_state_transitions.
    start_index : int
        Starting index. The final state of the transition indexed is the starting state configuration.
    size : int
        Size of random_numbers drawn at once.
    end_time : float
        Time at which simulation ends.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    current_state_index = start_index

    time_series = [0]
    transition_series = []

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    random_numbers = rng.uniform(low=0, high=1, size=(size, 2))

    if end_time is not None:
        i = 0
        j = 1
        while time_series[-1] < end_time:
            current_state_lambda = row_sums[current_state_index]

            if current_state_lambda == 0:
                break

            transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i - (j - 1) * size + 1, 0])

            sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index],
                                           random_numbers[i - (j - 1) * size + 1, 1])

            next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

            transition_series.append(next_transition)
            time_series.append(time_series[-1] + transition_time)

            i += 1
            if i == j * size - 1:
                random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
                j += 1
            current_state_index = next_transition

    transition_series = np.array(transition_series[:-1])
    time_series = np.array(time_series[:-1])

    return time_series, transition_series
