"""
Module simulation_GPU
"""
import cupy as cp
import numpy as np
from torch import tensor, cuda, repeat_interleave
import src.network as net


def list_devices():
    print(f'pytorch device count: {cuda.device_count()}')
    print(f'pytorch device: {cuda.current_device()}')
    print(f'cupy device count: {cp.cuda.runtime.getDeviceCount()}')
    print(f'cupy device: {cp.cuda.runtime.getDevice()}')


def round_retain_sum(array, topline):
    keep_track = cp.arange(0, array.size)
    orig_sum = cp.round(topline, decimals=0)
    new_round = cp.round(array, decimals=0)
    local_sum = cp.round(cp.sum(new_round), decimals=0)

    while local_sum != orig_sum:
        diff = cp.round(orig_sum - local_sum, decimals=0)
        if diff < 0.:
            increment = -1
            reverse = False
        else:
            increment = 1
            reverse = True
        tweaks = cp.abs(diff).astype(int)
        new_round, keep_track = sort_by_diff(array, new_round, keep_track, reverse)
        iterations = cp.arange(0, cp.amin(cp.append(tweaks, array.size)))
        for ith in iterations:
            new_round[ith] += increment
        local_sum = cp.round(cp.sum(new_round), decimals=0)

    reverse_sorting = np.argsort(keep_track)
    rounded_array = new_round[reverse_sorting]
    return rounded_array


def sort_by_diff(array, rounded_array, keep_track, reverse):
    diff = array - rounded_array
    if reverse:
        sorted = cp.argsort(diff)[::-1]
    else:
        sorted = cp.argsort(diff)
    return rounded_array[sorted], keep_track[sorted]


def fill_individual_transitions(prediction, transitions, size, seed):

    transition_occurrences = prediction.stationary_distribution_transitions * size
    transition_occurrences = transition_occurrences.astype(cp.int64)
    maximum_transition_index = np.argmax(transition_occurrences)
    
    starting_transition = transitions.transition_df.index[maximum_transition_index]
    
    graph = net.construct_graph_transitions(transitions.transition_df, numerical=True)
    graph_suited, _ = net.check_graph_suitable(G=graph, starting_node=starting_transition)
    if not graph_suited:
        raise ValueError('graph is not suited for the algorithm. Check for loops that do not contain the most occurring state.')
    transition_order = net.determine_node_order(G=graph, starting_node=starting_transition)
    
    all_rates = cp.array(transitions.transition_df['rate'])
    rng = cp.random.default_rng(seed)    
    transition_series = cp.full(transition_occurrences[maximum_transition_index], starting_transition)
    for i, transition in enumerate(transition_order):
        transition_indices = cp.where(transition_series == transition)[0]
        occurrences = transition_indices.size
        cp.random.shuffle(transition_indices)
        follow_up_transitions = cp.array(list(graph.successors(transition)))
        rates = all_rates[follow_up_transitions]
        repeats = rates * occurrences / rates.sum()
        rounded_repeats = round_retain_sum(repeats, topline=occurrences)
        rounded_repeats = cp.array(rounded_repeats, dtype=cp.int64)
        if starting_transition in follow_up_transitions:
            if follow_up_transitions.size == 1:
                continue
            # no np.delete in cupy or pytorch
            indices_not_starting_transition = cp.where(follow_up_transitions != starting_transition)[0]
            follow_up_transitions = follow_up_transitions[indices_not_starting_transition]
            number_of_deletions = rounded_repeats.sum() - rounded_repeats[indices_not_starting_transition].sum()
            rounded_repeats = rounded_repeats[indices_not_starting_transition]
            transition_indices = transition_indices[:-number_of_deletions]
        
        # no np.repeat in cupy
        # but why
        #follow_up_transitions = follow_up_transitions.get()
        follow_up_transitions_torch = tensor(follow_up_transitions, device='cuda')
        #rounded_repeats = rounded_repeats.get()
        rounded_repeats = tensor(rounded_repeats, device='cuda')
        follow_up_transitions_torch = repeat_interleave(follow_up_transitions_torch, rounded_repeats)
        follow_up_transitions = cp.asarray(follow_up_transitions_torch)
        
        # no np.insert in cupy or pytorch
        insert_at = transition_indices + 1
        transition_series_new = cp.full(shape=(transition_series.size + insert_at.size), fill_value=-1)
        sorted_indices = cp.argsort(insert_at)
        add_to_indices = cp.arange(0, insert_at.size)
        insert_at[sorted_indices] += add_to_indices
        transition_series_new[insert_at] = follow_up_transitions
        transition_series_new[cp.where(transition_series_new == -1)[0]] = transition_series
        transition_series = transition_series_new

    
    time_step_series = cp.empty(transition_series.size + 1, dtype=cp.float64)
    time_step_series[0] = 0
    for i, transition_time_distribution in enumerate(prediction.transition_time_distributions):
        indices = cp.where(transition_series == transitions.transition_df.index[i])[0]
        drawn_lifetimes = transition_time_distribution.rvs(indices.size)
        time_step_series[indices + 1] = drawn_lifetimes
    
    time_series = cp.empty_like(time_step_series, dtype=cp.float64)
    time_series[:] = cp.cumsum(time_step_series, dtype=cp.float64)

    return time_series, transition_series


def prep_direct_method_time(transition_matrix):

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(transition_matrix, transition_matrix_sorted_indices, axis=1)
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    return transition_matrix_sorted_indices, cumsum_sorted_trm


def direct_method_time(transition_matrix_sorted_indices, cumsum_sorted_trm, row_sums, start_index=0, size=10,
                       end_time=10, seed=None):
    rng = cp.random.default_rng(seed)
    current_state_index = start_index

    time_series = cp.empty(size + 1, dtype=cp.float64)
    transition_series = cp.empty(size, dtype=cp.uint32)
    random_numbers = rng.uniform(low=0, high=1, size=(size, 2))

    time_series[0] = 0

    abso = 0
    i = 0
    j = 1
    while time_series[i] < end_time:
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:
            abso = 1
            break
        transition_time = 1 / current_state_lambda * cp.log(1 / random_numbers[i - (j - 1) * size, 0])
        sorted_index = cp.searchsorted(cumsum_sorted_trm[current_state_index],
                                       random_numbers[i - (j - 1) * size, 1])
        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = next_transition
        time_series[i+1] = time_series[i] + transition_time
        current_state_index = next_transition

        i += 1
        if i == j * size:
            j += 1
            random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
            time_series = cp.resize(time_series, (j*size+1, ))
            transition_series = cp.resize(transition_series, (j*size,))
    
    time_series[i + abso] = end_time

    transition_series = transition_series[:i-1+abso]
    time_series = time_series[:i+1+abso]

    return time_series, transition_series


def direct_method_time_large_array(transition_matrix_sorted_indices, cumsum_sorted_trm, row_sums, start_index=0, size=10,
                       end_time=10, seed=None, number_fluorophores=1):
    rng = cp.random.default_rng(seed)
    current_state_index = start_index

    time_series = cp.empty((number_fluorophores, size + 1), dtype=cp.float64)
    transition_series = cp.empty((number_fluorophores, size), dtype=cp.uint32)
    random_numbers = rng.uniform(low=0, high=1, size=(size, number_fluorophores, 2))

    time_series[:, 0] = 0


    # fill the fluorophores already at end_time with zeros
    
    abso = 0
    i = 0
    j = 1
    while time_series[i] < end_time:
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:
            abso = 1
            break
        transition_time = 1 / current_state_lambda * cp.log(1 / random_numbers[i - (j - 1) * size, 0])
        sorted_index = cp.searchsorted(cumsum_sorted_trm[current_state_index],
                                       random_numbers[i - (j - 1) * size, 1])
        next_transition = transition_matrix_sorted_indices[current_state_index, sorted_index]

        transition_series[i] = next_transition
        time_series[i+1] = time_series[i] + transition_time
        current_state_index = next_transition

        i += 1
        if i == j * size:
            j += 1
            random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
            time_series = cp.resize(time_series, (j*size+1, ))
            transition_series = cp.resize(transition_series, (j*size,))
    
    time_series[i + abso] = end_time

    transition_series = transition_series[:i-1+abso]
    time_series = time_series[:i+1+abso]

    return time_series, transition_series
