"""
Module simulation
"""
import gc
import os
import warnings
import numpy as np
import src.network as net
import iteround as it
import src.transitions as tr


class Simulation:
    """
    Container of simulation-associated attributes and methods.

    Attributes
    ----------
    transitions : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to state_series[i] and transition_series[i - 1].
        If end_time was not None, includes end_time at index -1 that does not correspond to any of state_series or
        transition_series.
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
        self.memmap_path = None

    def run(self, start_at=None, size=1e5, end_time=None, seed=None, use_memmap=None):
        """
        Runs a simulation based on the direct method of the gillespie algorithm (i.e., stochastic simulation algorithm).
        Can either be based on maximum number of steps or maximum total time.

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
            If not None, time at which simulation ends in s.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        use_memmap : None, str
            Determines the path where memmaps shall be stored. If empty str, saved in current working directory.

        Returns
        -------
        None
        """
        if start_at is None:
            start_at = tuple(np.zeros(shape=self.transitions.fluorophore_system.count, dtype=int))
        elif len(start_at) != self.transitions.fluorophore_system.count:
            raise ValueError("The number of starting states doesn't match the number of fluorophores.")
        size = int(size)
        df = self.transitions.combined_state_transitions_df
        start_index = df[df['final_state'] == start_at].index[0]
        if end_time is None:
            self.time_series, self.transition_series = \
                direct_method_steps(transition_matrix=self.transitions.transition_matrix,
                                    row_sums=self.transitions.row_sums,
                                    start_index=start_index, size=size, seed=seed, use_memmap=use_memmap)
        else:
            self.time_series, self.transition_series = \
                direct_method_time(transition_matrix=self.transitions.transition_matrix,
                                   row_sums=self.transitions.row_sums,
                                   start_index=start_index, size=size, end_time=end_time, seed=seed,
                                   use_memmap=use_memmap)

        final_states = self.transitions.combined_state_transitions_df['final_state']

        if use_memmap is not None:
            self.memmap_path = use_memmap
            self.state_series = np.memmap(os.path.join(use_memmap, 'state_series'), dtype=np.int8, mode='w+',
                                          shape=(len(final_states[0]), self.transition_series.size + 1))
            self.state_series[:] = np.empty(shape=(len(final_states[0]), self.transition_series.size + 1),
                                            dtype=np.int8)
        else:
            self.state_series = np.empty(shape=(len(final_states[0]), self.transition_series.size + 1), dtype=np.int8)
        self.state_series[:, 0] = start_at

        for i, _ in enumerate(final_states[0]):
            final_states_fluorophore = final_states.map(lambda x: x[i]).to_numpy(dtype=np.int8)
            self.state_series[i][1:] = final_states_fluorophore[self.transition_series]
        if use_memmap is not None:
            self.state_series.flush()
    
    def approximate(self, prediction=None, strategy='individual', size=1e5, seed=None):
        """
        Approximate the series using the stationary distribution of the Markov chain. Absorbing Markov chains
        have a stationary distribution nonzero only in their absorbing states, hence are not suited for the
        algorithm. 
        Cannot be used for energy transfers - only one fluorophore independent of any other fluorophore is 
        simulated. Reversible reactions/transitions and simple cycles are not suited for this algorithm unless 
        it comprises the most commonly occurring transition/state.

        Parameters
        ----------
        prediction : src.statistics.Prediction 
            Prediction: Container of lifetimes, state and transition occurrences obtained by computation.
        strategy : str
            Defines the strategy to mimic a step-by-step simulation, by default 'individual'.
            'individual' is to consecutively filling up the array with the topologically sorted transitions.
            'cycles' is to put all possible consecutive transition cycles into an array and then shuffling it.
        size : int
            Roughly the size of the output arrays (strategy = 'individual') or the total number of cycles
            (strategy = 'cycles'), by default 1e5
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.
        """
        size = int(size)
        if strategy == 'individual':
            self.time_series, self.transition_series = fill_individual_transitions(prediction=prediction, transitions=self.transitions, size=size, seed=seed)
        elif strategy == 'cycles':
            self.time_series, self.transition_series = fill_simple_cycles(prediction=prediction, transitions=self.transitions, size=size, seed=seed)
        else:
            raise ValueError('strategy unknown.')
        final_states = self.transitions.transition_df['final_state'].apply(lambda x: x.value)
        self.state_series = np.empty(shape=(self.transition_series.size + 1, ), dtype=np.int8)
        self.state_series[0] = (self.transitions.transition_df['initial_state'][self.transition_series[0]]).value   
        self.state_series[1:] = final_states[self.transition_series]
        self.state_series = np.expand_dims(self.state_series, axis=0)

    def apply_absorbing(self, transition_set):
        """
        If there is a single transition that leads to an absorbing state, it can be applied to the different series
        in retrospect. It is intended to extend 'approximate' to also cover absorbing Markov chains, but it can be
        used on the 'run' result, too.
        It works using the fundamental matrix of the absorbing Markov chain. 

        Parameters
        ----------
        transition_set : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes. Allows optional post-init-modification and
            (subsequent) finalization.
        """
        graph = net.construct_graph_transitions(transition_set.transition_df, numerical=True)

        original_transition_matrix = transition_set.transition_matrix.copy()
        for i, single_state in enumerate(transition_set.single_states):
            single_state_obj = tr.SingleState(single_state)
            if single_state_obj not in transition_set.transition_df['initial_state'].values:
                drop_transition = transition_set.transition_df[transition_set.transition_df['final_state'] == single_state_obj].index.values[0]
                predecessors = np.array(list(graph.predecessors(drop_transition)))               
        # Q takes the original transition matrix into account, because within Q the state that leads
        # to the absorbing state has to take on the probability GIVEN the possibility of the transition
        # to the absorbing state.
        Q = np.delete(original_transition_matrix, drop_transition, axis=0)
        Q = np.delete(Q, drop_transition, axis=1)
        I_t = np.identity(Q.shape[0])
        N = np.linalg.inv(I_t - Q)
        starting_transition = self.transition_series[0]
        limiting_transition = np.argmax(N[starting_transition, predecessors])
        count = int(N[starting_transition, predecessors][limiting_transition])
        final_transition = predecessors[limiting_transition]
        transition_occurrences_at = np.where(self.transition_series == final_transition)[0]
        if transition_occurrences_at.size > count:
            cut_at = transition_occurrences_at[count]
            self.transition_series = self.transition_series[:cut_at + 1]
            self.transition_series = np.append(self.transition_series, final_transition)
            self.state_series = self.state_series[:cut_at + 2]
            self.state_series = np.append(self.state_series, transition_set.transition_df['final_state'][drop_transition].value)
            # approximation - the possibility of an absorbing state should reduce the lifetimes of the involved states
            # this is neglected here because it has a low impact given the low probability of it happening
            self.time_series = self.time_series[:cut_at + 3]
        else:
            warnings.warn('Time series too short for bleaching to statistically have occurred.')


    def delete_memmaps(self):
        """
        Delete the memmap variables and files. Note: if the memmaps are attempted to be accessed after deletion, python
        crashes.
        Source: https://stackoverflow.com/questions/39953501/i-cant-remove-file-created-by-memmap

        Returns
        -------
        None
        """
        self.transition_series._mmap.close()
        self.time_series._mmap.close()
        self.state_series._mmap.close()
        del self.transition_series
        del self.time_series
        del self.state_series
        os.remove(os.path.join(self.memmap_path, 'transition_series'))
        os.remove(os.path.join(self.memmap_path, 'time_series'))
        os.remove(os.path.join(self.memmap_path, 'state_series'))


def direct_method_steps(transition_matrix, row_sums, start_index=0, size=10, seed=None,
                        use_memmap=None):
    """
    The direct method of the gillespie algorithm (i.e., stochastic simulation algorithm). Here, the propensities are
    equal to the rate constants because the population is always 1. Additionally, the state change vector is redundant
    because each transition leads to a shift in populations by 1.
    This version is based on a maximum number of steps.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
        combined_state_transitions.
    start_index : int
        Starting index. The final state of the transition indexed is the starting state configuration.
    size : int
        Maximum number of simulation steps.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.
    use_memmap : None, str
        Determines the path where memmaps shall be stored. If empty str, saved in current working directory.

    Returns
    -------
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    if use_memmap is not None:
        time_step_series = np.memmap(os.path.join(use_memmap, 'time_step_series'),
                                     dtype=np.float32, mode='w+', shape=(size+1, ))
        transition_series = np.memmap(os.path.join(use_memmap, 'transition_series'),
                                      dtype=np.uint32, mode='w+', shape=(size, ))
        time_series = np.memmap(os.path.join(use_memmap, 'time_series'),
                                dtype=np.float64, mode='w+', shape=(size+1, ))
        random_numbers = np.memmap(os.path.join(use_memmap, 'random_numbers'),
                                   dtype=np.float64, mode='w+', shape=(size, 2))
        random_numbers[:] = rng.uniform(low=0, high=1, size=(size, 2))
    else:
        time_step_series = np.empty(size + 1, dtype=np.float32)
        transition_series = np.empty(size, dtype=np.uint32)
        time_series = np.empty(size + 1, dtype=np.float64)
        random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
    time_step_series[0] = 0

    # a random index (in this case the first index) at which the final state of a transition equals start_at
    current_state_index = start_index

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)
    absorbing_state_reached = False

    for i in range(size):
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:  # there is no outgoing transition
            # the Markov chain has encountered an absorbing state
            absorbing_state_reached = True
            break
        # inverse transform sampling using the quantile function of the exponential distribution
        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index], random_numbers[i, 1])

        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = current_state_index = next_transition

        time_step_series[i + 1] = transition_time

    if absorbing_state_reached:
        if use_memmap is not None:
            time_step_series.flush()
            transition_series.flush()
            time_series.flush()
            time_step_series = np.memmap(os.path.join(use_memmap, 'time_step_series'), mode='r+',
                                         dtype=np.float32, shape=(i + 1,))
            transition_series = np.memmap(os.path.join(use_memmap, 'transition_series'), mode='r+',
                                          dtype=np.uint32, shape=(i,))
            time_series = np.memmap(os.path.join(use_memmap, 'time_series'),
                                    dtype=np.float64, mode='r+', shape=(i + 1,))
        else:
            time_step_series.resize((i + 1, ))
            transition_series.resize((i, ))
            time_series.resize((i + 1, ))

    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    if use_memmap is not None:
        del time_step_series
        del random_numbers
        gc.collect()
        os.remove(os.path.join(use_memmap, 'time_step_series'))
        os.remove(os.path.join(use_memmap, 'random_numbers'))
        transition_series.flush()
        time_series.flush()

    return time_series, transition_series


def direct_method_time(transition_matrix, row_sums, start_index=0, size=10, end_time=10, seed=None,
                       use_memmap=None):
    """
    The direct method of the gillespie algorithm (i.e., stochastic simulation algorithm). Here, the propensities are
    equal to the rate constants because the population is always 1. Additionally, the state change vector is redundant
    because each transition leads to a shift in populations by 1.
    This version is based on a maximum total time.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
        combined_state_transitions.
    start_index : int
        Starting index. The final state of the transition indexed is the starting state configuration.
    size : int
        Size of random_numbers drawn at once.
    end_time : float
        Time at which simulation ends in s.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.
    use_memmap : None, str
        Determines the path where memmaps shall be stored. If empty str, saved in current working directory.

    Returns
    -------
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
        Includes end_time at index -1 that does not correspond to any of transition_series.
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    current_state_index = start_index

    if use_memmap is not None:
        transition_series = np.memmap(os.path.join(use_memmap, 'transition_series'),
                                      dtype=np.uint32, mode='w+', shape=(size, ))
        time_series = np.memmap(os.path.join(use_memmap, 'time_series'),
                                dtype=np.float64, mode='w+', shape=(size+1, ))
        random_numbers = np.memmap(os.path.join(use_memmap, 'random_numbers'),
                                   dtype=np.float64, mode='w+', shape=(size, 2))
        random_numbers[:] = rng.uniform(low=0, high=1, size=(size, 2))
    else:
        transition_series = np.empty(size, dtype=np.uint32)
        time_series = np.empty(size + 1, dtype=np.float64)
        random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
        # time_series size + 1 and not size + 2 because random_numbers (and transition_series) has size
        # transition_series then 'loses' one transition, so its final size will be not higher than size - 1

    time_series[0] = 0
    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    abso = 0
    i = 0
    j = 1
    while time_series[i] < end_time:
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:
            abso = 1
            break
        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i - (j - 1) * size, 0])

        sorted_index = np.searchsorted(cumsum_sorted_trm[current_state_index],
                                       random_numbers[i - (j - 1) * size, 1])

        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = next_transition
        time_series[i+1] = time_series[i] + transition_time
        current_state_index = next_transition

        i += 1
        if i == j * size:
            j += 1
            random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
            if use_memmap is not None:
                time_series.flush()
                transition_series.flush()
                time_series = np.memmap(os.path.join(use_memmap, 'time_series'), mode='r+',
                                        dtype=np.float64, shape=(j*size+1, ))
                transition_series = np.memmap(os.path.join(use_memmap, 'transition_series'), mode='r+',
                                              dtype=np.uint32, shape=(j*size, ))
            else:
                time_series.resize((j*size+1, ))
                transition_series.resize((j*size, ))

    time_series[i + abso] = end_time

    if use_memmap is not None:
        del random_numbers
        gc.collect()
        os.remove(os.path.join(use_memmap, 'random_numbers'))
        time_series.flush()
        transition_series.flush()
        time_series.base.resize(8*(i+1+abso))  # 8 because it is 64/8 byte per number and resize works with byte
        transition_series.base.resize(4*(i-1+abso))
        time_series.flush()
        transition_series.flush()
        time_series = np.memmap(os.path.join(use_memmap, 'time_series'), mode='r+', dtype=np.float64,
                                shape=(i+1+abso, ))
        transition_series = np.memmap(os.path.join(use_memmap, 'transition_series'), mode='r+', dtype=np.uint32,
                                      shape=(i-1+abso,))
    else:
        transition_series = transition_series[:i-1+abso]
        time_series = time_series[:i+1+abso]

    return time_series, transition_series


def fill_individual_transitions(prediction, transitions, size, seed):
    """
    Constructs an approximation of simulated stochastic data based on a predicted stationary distribution of 
    a Markov chain. 
    The transitions are ordered via a topological sort and processed accordingly. Successor transitions are 
    placed behind their predecessors. The topological sort is possible via a temporary conversion of the graph
    to a directed acyclic graph (DAG).

    Parameters
    ----------
    prediction : src.statistics.Prediction
        Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
        methods.
    size : int
        Maximum number of simulation steps. Due to rounding, actual size might vary.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    transition_occurrences = prediction.stationary_distribution_transitions * size
    transition_occurrences = transition_occurrences.astype(np.int64)
    maximum_transition_index = np.argmax(transition_occurrences)
    
    starting_transition = transitions.transition_df.index[maximum_transition_index]
    
    graph = net.construct_graph_transitions(transitions.transition_df, numerical=True)
    graph_suited, _ = net.check_graph_suitable(G=graph, starting_node=starting_transition)
    if not graph_suited:
        raise ValueError('graph is not suited for the algorithm. Check for loops that do not contain the most occurring state.')
    transition_order = net.determine_node_order(G=graph, starting_node=starting_transition)
    
    rng = np.random.default_rng(seed)    
    transition_series = np.full(transition_occurrences[maximum_transition_index], starting_transition)

    for i, transition in enumerate(transition_order):
        transition_indices = np.where(transition_series == transition)[0]
        occurrences = transition_indices.size
        rng.shuffle(transition_indices)
        follow_up_transitions = np.array(list(graph.successors(transition)))
        rates = transitions.transition_df['rate'][follow_up_transitions].to_numpy()
        repeats = rates * occurrences / rates.sum()
        rounded_repeats = it.saferound(repeats, places=0, topline=occurrences)  # topline such that each transition will get a followup transition
        rounded_repeats = np.array(rounded_repeats, dtype=np.int64)
        if starting_transition in follow_up_transitions:
            indices_starting_transition = np.where(follow_up_transitions == starting_transition)[0]
            follow_up_transitions = np.delete(follow_up_transitions, indices_starting_transition)
            number_of_deletions = rounded_repeats[indices_starting_transition].sum()
            rounded_repeats = np.delete(rounded_repeats, indices_starting_transition)
            transition_indices = transition_indices[:-number_of_deletions]
        follow_up_transitions = np.repeat(follow_up_transitions, repeats=rounded_repeats)
        insert_at = transition_indices + 1
        transition_series = np.insert(transition_series, insert_at, follow_up_transitions)
    
    time_step_series = np.empty(transition_series.size + 1, dtype=np.float64)
    time_step_series[0] = 0
    for i, transition_time_distribution in enumerate(prediction.transition_time_distributions):
        indices = np.where(transition_series == transitions.transition_df.index[i])[0]
        drawn_lifetimes = transition_time_distribution.rvs(indices.size, random_state=rng)
        time_step_series[indices + 1] = drawn_lifetimes
    
    time_series = np.empty_like(time_step_series, dtype=np.float64)
    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    return time_series, transition_series


def fill_simple_cycles(prediction, transitions, size, seed):
    """
    Constructs an approximation of simulated stochastic data based on a predicted stationary distribution of 
    a Markov chain. 
    The graph is splitted into its simple cycles, their total occurrences are computed, followed by shuffling.

    Parameters
    ----------
    prediction : src.statistics.Prediction
        Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
        methods.
    size : int
        Total number of simulated cycles.
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

    transition_occurrences = prediction.stationary_distribution_transitions * size
    transition_occurrences = transition_occurrences.astype(np.int64)
    maximum_transition_index = np.argmax(transition_occurrences)
    starting_transition = transitions.transition_df.index[maximum_transition_index]
    
    graph = net.construct_graph_transitions(prediction.transition_df, numerical=True)
    graph_suited, cycles = net.check_graph_suitable(G=graph, starting_node=starting_transition)
    if not graph_suited:
        raise ValueError('graph is not suited for the algorithm. Check for loops that do not contain the most occurring state.')

    relative_probabilities = []
    longest_cycle = 0
    for cycle in cycles:
        if len(cycle) > longest_cycle:
            longest_cycle = len(cycle)
        relative_probability_cycle = 1
        for i in range(len(cycle)):
            if i == len(cycle[1:]):
                j = 0
            else:
                j = i + 1
            current_transition = cycle[i]
            future_transition = cycle[j]
            rate = transitions.transition_df['rate'][current_transition]
            initial_state = transitions.transition_df['initial_state'][current_transition]
            all_rates = transitions.transition_df['rate'][transitions.transition_df['initial_state'] == initial_state].to_numpy()
            relative_probability = rate / all_rates.sum()
            relative_probability_cycle *= relative_probability

        relative_probabilities.append(relative_probability_cycle)
    occurrences = (np.array(relative_probabilities) * size).astype(int)
    
    redundant_value = transitions.transition_df.index.max() + 1
    transition_series = np.full((occurrences.sum(), longest_cycle), redundant_value)

    all_repeats = []
    for cycle, rounded_repeat in zip(cycles, occurrences):
        cycle_to_repeat = np.full(longest_cycle, redundant_value)
        cycle_to_repeat[:len(cycle)] = cycle
        repeats = np.repeat(cycle_to_repeat[np.newaxis,...], rounded_repeat, axis=0)
        all_repeats.append(repeats)

    all_repeats = np.concatenate(all_repeats)
    indices = np.arange(occurrences.sum())
    rng.shuffle(indices)

    transition_series[indices] = all_repeats
    transition_series = transition_series.flatten()
    transition_series = np.delete(transition_series, np.where(transition_series == redundant_value)[0])
    
    time_step_series = np.empty(transition_series.size + 1, dtype=np.float64)
    time_step_series[0] = 0
    for i, transition_time_distribution in enumerate(prediction.transition_time_distributions):
        indices = np.where(transition_series == transitions.transition_df.index[i])[0]
        drawn_lifetimes = transition_time_distribution.rvs(indices.size)
        time_step_series[indices + 1] = drawn_lifetimes
    
    time_series = np.empty_like(time_step_series, dtype=np.float64)
    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    return time_series, transition_series
