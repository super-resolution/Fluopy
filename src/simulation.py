"""
Module simulation
"""
import gc
import os
import numpy as np
import iteround as it
import src.transitions as tr
import src.network as net
import itertools
import random


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

    def run(self, start_at=None, size=int(1e5), end_time=None, seed=None, use_memmap=None):
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


def simulate_prediction_I(prediction, steps, seed):
    """
    Constructs simulated stochastic data based on a predicted stationary distribution of states of a Markov chain. Known
    mean lifetimes are used to draw the respective number of lifetimes of a state.
    I: Determines order of states via topological sort and inserts follow up states behind
    each.
    Markov chains with absorbing states have stationary distributions that are nonzero only in absorbing states, hence
    are not suited for this algorithm.
    Cannot be used for energy transfers - only one fluorophore independent of any other fluorophore is simulated.
    Reversible reactions/transitions and loops are not suited for this algorithm unless it comprises the most commonly occurring
    state.

    Parameters
    ----------
    prediction : src.statistics.Prediction
        Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
        methods.
    steps : int
        Maximum number of simulation steps. Due to rounding, actual size might vary.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_step_series : 1-D array_like
        The simulated time steps. At index i, they correspond to state_series[i].
    state_series : 1-D array_like
        The simulated states. At index i, they correspond to time_step_series[i] (i.e., their lifetimes).
    """
    state_occurrences = prediction.stationary_distribution_states * steps
    state_occurrences = state_occurrences.astype(np.int64)
    maximum_state_index = np.argmax(state_occurrences)
    transition_df = prediction.transition_df
    states = prediction.single_states
    starting_state = states[maximum_state_index]
    
    graph = net.construct_network(prediction.transition_df, numerical=True)
    graph_suited, _ = net.check_graph_suitable(G=graph, starting_node=starting_state)
    if not graph_suited:
        raise ValueError('graph is not suited for the algorithm. Check for loops that do not contain the most occurring state.')
    state_order = net.determine_node_order(G=graph, starting_node=starting_state)
    
    rng = np.random.default_rng(seed)
    # special case for fluorophores: S0 always leads to S1
    following_initial_state = transition_df['final_state'][transition_df['initial_state'] == tr.SingleState(starting_state)]
    if following_initial_state.size == 1:
        follow_up = following_initial_state.iloc[0].value
        state_series = np.tile([starting_state, follow_up], state_occurrences[maximum_state_index])
        skip_first = True
    else:
        state_series = np.full(state_occurrences[maximum_state_index], starting_state)

    for i, state in enumerate(state_order):
        if i == 0 and skip_first:
            continue
        state_indices = np.where(state_series == state)[0]
        occurrences = state_indices.size
        rng.shuffle(state_indices)
        follow_up_states = transition_df['final_state'][transition_df['initial_state'] == tr.SingleState(state)]
        rates = transition_df['rate'][transition_df['initial_state'] == tr.SingleState(state)].to_numpy()
        repeats = rates * occurrences / rates.sum()
        rounded_repeats = it.saferound(repeats, places=0, topline=occurrences)
        rounded_repeats = np.array(rounded_repeats, dtype=np.int64)
        follow_ups_to_insert = np.array([follow_up.value for follow_up in follow_up_states])
        if starting_state in follow_ups_to_insert:
            indices_starting_state = np.where(follow_ups_to_insert == starting_state)[0]
            follow_ups_to_insert = np.delete(follow_ups_to_insert, indices_starting_state)
            number_of_deletions = rounded_repeats[indices_starting_state].sum()
            rounded_repeats = np.delete(rounded_repeats, indices_starting_state)
            state_indices = state_indices[:-number_of_deletions]
        follow_ups_to_insert = np.repeat(follow_ups_to_insert, repeats=rounded_repeats)
        insert_at = state_indices + 1
        state_series = np.insert(state_series, insert_at, follow_ups_to_insert)
    
    time_step_series = np.zeros_like(state_series, dtype=np.float64)
    lifetime_distributions = prediction.lifetime_distributions
    for i, lifetime_distribution in enumerate(lifetime_distributions):
        indices = np.where(state_series == states[i])[0]
        drawn_lifetimes = lifetime_distribution.rvs(indices.size)
        time_step_series[indices] = drawn_lifetimes

    return time_step_series, state_series


def simulate_prediction_II(prediction, steps):
    """
    Constructs simulated stochastic data based on a predicted stationary distribution of states of a Markov chain. Known
    mean lifetimes are used to draw the respective number of lifetimes of a state.
    II: Gets all simple cycles, calculates their probability of occurrence and shuffles them.
    Markov chains with absorbing states have stationary distributions that are nonzero only in absorbing states, hence
    are not suited for this algorithm.
    Cannot be used for energy transfers - only one fluorophore independent of any other fluorophore is simulated.
    Reversible reactions/transitions are not suited for this algorithm unless it comprises the most commonly occurring
    state.

    Parameters
    ----------
    prediction : src.statistics.Prediction
        Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
        methods.
    steps : int
        Maximum number of simulation steps. The number represents cycle occurrences, not state occurrences.

    Returns
    -------
    time_step_series : 1-D array_like
        The simulated time steps. At index i, they correspond to state_series[i].
    state_series : 1-D array_like
        The simulated states. At index i, they correspond to time_step_series[i] (i.e., their lifetimes).
    """
    state_occurrences = prediction.stationary_distribution_states * steps
    state_occurrences = state_occurrences.astype(np.int64)
    maximum_state_index = np.argmax(state_occurrences)
    transition_df = prediction.transition_df
    states = prediction.single_states
    starting_state = states[maximum_state_index]
    
    graph = net.construct_network(prediction.transition_df, numerical=True)
    graph_suited, cycles = net.check_graph_suitable(G=graph, starting_node=starting_state)
    if not graph_suited:
        raise ValueError('graph is not suited for the algorithm. Check for loops that do not contain the most occurring state.')

    relative_probabilities = []
    for cycle in cycles:
        relative_probability_cycle = 1
        for i in range(len(cycle)):
            if i == len(cycle[1:]):
                j = 0
            else:
                j = i + 1
            current_state = cycle[i]
            future_state = cycle[j]
            rates = transition_df['rate'][(transition_df['initial_state'] == tr.SingleState(current_state)) & (transition_df['final_state'] == tr.SingleState(future_state))].to_numpy()
            all_rates = transition_df['rate'][transition_df['initial_state'] == tr.SingleState(current_state)].to_numpy()
            relative_probability = rates.sum() / all_rates.sum()
            relative_probability_cycle *= relative_probability

        relative_probabilities.append(relative_probability_cycle)
    occurrences = (np.array(relative_probabilities) * steps).astype(int)
    all_repeats = []
    for cycle, rounded_repeat in zip(cycles, occurrences.astype(int)):
        repeats = np.repeat(np.array(cycle)[np.newaxis,...], repeats=rounded_repeat, axis=0)
        all_repeats = all_repeats + repeats.tolist()

    random.shuffle(all_repeats)
    state_series = np.array(list(itertools.chain.from_iterable(all_repeats)))    
    time_step_series = np.zeros_like(state_series, dtype=np.float64)
    lifetime_distributions = prediction.lifetime_distributions
    for i, lifetime_distribution in enumerate(lifetime_distributions):
        indices = np.where(state_series == states[i])[0]
        drawn_lifetimes = lifetime_distribution.rvs(indices.size)
        time_step_series[indices] = drawn_lifetimes

    return time_step_series, state_series


def simulate_prediction_III(prediction, steps, seed):
    """
    Constructs simulated stochastic data based on a predicted stationary distribution of states of a Markov chain. Known
    mean lifetimes are used to draw the respective number of lifetimes of a state.
    III: Like II but shuffling and flattening happens in numpy. Requires all cycles to be stretched to a common size. May
    be faster but more memory consuming.
    Markov chains with absorbing states have stationary distributions that are nonzero only in absorbing states, hence
    are not suited for this algorithm.
    Cannot be used for energy transfers - only one fluorophore independent of any other fluorophore is simulated.
    Reversible reactions/transitions are not suited for this algorithm unless it comprises the most commonly occurring
    state.

    Parameters
    ----------
    prediction : src.statistics.Prediction
        Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
        methods.
    steps : int
        Maximum number of simulation steps. The number represents cycle occurrences, not state occurrences.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_step_series : 1-D array_like
        The simulated time steps. At index i, they correspond to state_series[i].
    state_series : 1-D array_like
        The simulated states. At index i, they correspond to time_step_series[i] (i.e., their lifetimes).
    """
    state_occurrences = prediction.stationary_distribution_states * steps
    state_occurrences = state_occurrences.astype(np.int64)
    maximum_state_index = np.argmax(state_occurrences)
    transition_df = prediction.transition_df
    states = prediction.single_states
    starting_state = states[maximum_state_index]
    
    rng = np.random.default_rng(seed)

    graph = net.construct_network(prediction.transition_df, numerical=True)
    graph_suited, cycles = net.check_graph_suitable(G=graph, starting_node=starting_state)
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
            current_state = cycle[i]
            future_state = cycle[j]
            
            rates = transition_df['rate'][(transition_df['initial_state'] == tr.SingleState(current_state)) & (transition_df['final_state'] == tr.SingleState(future_state))].to_numpy()
            all_rates = transition_df['rate'][transition_df['initial_state'] == tr.SingleState(current_state)].to_numpy()
            relative_probability = rates.sum() / all_rates.sum()
            relative_probability_cycle *= relative_probability

        relative_probabilities.append(relative_probability_cycle)
    occurrences = (np.array(relative_probabilities) * steps).astype(int)

    redundant_value = max(states) + 1
    state_series = np.full((occurrences.sum(), longest_cycle), redundant_value)

    all_repeats = []
    for cycle, rounded_repeat in zip(cycles, occurrences.astype(int)):
        cycle_to_repeat = np.full(longest_cycle, redundant_value)
        cycle_to_repeat[:len(cycle)] = cycle
        repeats = np.repeat(cycle_to_repeat[np.newaxis,...], rounded_repeat, axis=0)
        all_repeats.append(repeats)

    all_repeats = np.concatenate(all_repeats)
    indices = np.arange(occurrences.sum())
    rng.shuffle(indices)

    state_series[indices] = all_repeats
    state_series = state_series.flatten()
    state_series = np.delete(state_series, np.where(state_series == redundant_value)[0])
       
    time_step_series = np.zeros_like(state_series, dtype=np.float64)
    lifetime_distributions = prediction.lifetime_distributions
    for i, lifetime_distribution in enumerate(lifetime_distributions):
        indices = np.where(state_series == states[i])[0]
        drawn_lifetimes = lifetime_distribution.rvs(indices.size)
        time_step_series[indices] = drawn_lifetimes

    return time_step_series, state_series
