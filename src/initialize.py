from enum import Enum
import numpy as np


def recursion(number, original_number, iterable, collector=None):
    if collector is None:
        collector = []
    if number >= 1:
        for i in iterable:
            collector.append(i)
            yield from recursion(number - 1, original_number, iterable, collector)

    else:
        if len(collector) > original_number:
            diff = len(collector) - original_number
            for pos in range(diff):
                del collector[original_number - 1 - (pos+1)]
        yield collector
        collector.pop()


def state_pairs(number, states=("S0", "S1", "T1", "R", "B")):

    state_pair_generator = recursion(number, number, states)

    strings = []
    for state_pair in state_pair_generator:
        string = ""
        for state in state_pair:
            if len(string) > 0:
                string += f"_{state}"
            else:
                string += f"{state}"

        strings.append(string)

    joined_states = Enum("Joined_States", strings, start=0)

    return joined_states


def transition_pairs(joined_states):

    trans_pairs = [(joined_state_1.name, joined_state_2.name) for joined_state_1 in joined_states
                   for joined_state_2 in joined_states]
    transitions = {f"{joined_state_1}__{joined_state_2}": (i // len(joined_states), i % len(joined_states))
                   for i, (joined_state_1, joined_state_2) in enumerate(trans_pairs)}

    return transitions


def initial_row_vector(transitions):
    uni_dir_shape = int(np.sqrt(len(transitions)))
    vector = np.zeros(shape=uni_dir_shape)
    vector[0] = 1
    return vector


def rate_assignment(assigned_rate_dict, transitions, source, destination, rate):
    for transition in transitions:
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
                        assigned_rate_dict[transition] = rate
    return assigned_rate_dict


def transition_rate_dict(rates, transitions):
    assigned_rate_dict = dict()

    for name, rate in rates.items():
        split = name.split("_")
        source, destination = split[1], split[2]
        assigned_rate_dict = rate_assignment(assigned_rate_dict, transitions, source, destination, rate)
    return assigned_rate_dict


def induction(rate_dict, transitions, induction_rate, states):
    if states == ("S0", "S1", "T1", "R", "B"):
        for transition in transitions:
            current_state, future_state = transition.split("__")
            current_state_split = current_state.split("_")
            future_state_split = future_state.split("_")
            if {"S1", "R"}.issubset(set(current_state_split)):
                indices_one = [i for i, e in enumerate(current_state_split) if e == "S1"]
                indices_two = [i for i, e in enumerate(current_state_split) if e == "R"]
                for i_1 in indices_one:
                    for i_2 in indices_two:
                        if "S0" in future_state_split[i_1] and "S0" in future_state_split[i_2]:
                            future_state_part = future_state_split[:]
                            current_state_part = current_state_split[:]
                            for h in sorted([i_1, i_2], reverse=True):
                                del(future_state_part[h])
                                del(current_state_part[h])
                            if not future_state_part == current_state_part:
                                break
                            else:
                                rate_dict[transition] = induction_rate
        return rate_dict
    elif states == ("ON", "OFF", "B"):
        for transition in transitions:
            current_state, future_state = transition.split("__")
            current_state_split = current_state.split("_")
            future_state_split = future_state.split("_")
            if {"ON", "OFF"}.issubset(set(current_state_split)):
                indices_one = [i for i, e in enumerate(current_state_split) if e == "ON"]
                indices_two = [i for i, e in enumerate(current_state_split) if e == "OFF"]
                for i_1 in indices_one:
                    for i_2 in indices_two:
                        if "ON" in future_state_split[i_1] and "ON" in future_state_split[i_2]:
                            future_state_part = future_state_split[:]
                            current_state_part = current_state_split[:]
                            for h in sorted([i_1, i_2], reverse=True):
                                del(future_state_part[h])
                                del(current_state_part[h])
                            if not future_state_part == current_state_part:
                                break
                            else:
                                rate_dict[transition] += induction_rate  # here the rate is added since the transition
                                # occurs occasionally as well.


def transition_matrices(rates, transitions):
    uni_dir_shape = int(np.sqrt(len(transitions)))
    transition_rate_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))

    for transition, rate in rates.items():
        transition_rate_matrix[transitions[transition]] = rate

    transition_matrix = np.zeros(shape=(uni_dir_shape, uni_dir_shape))
    row_sums = transition_rate_matrix.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            transition_matrix[i] = transition_rate_matrix[i] / row_sum

    return transition_matrix, transition_rate_matrix, row_sums


def predefining(number, states, rates, induction_rate=None):
    joined_states = state_pairs(number, states)
    state_names = []
    for joined_state in joined_states:
        state_names.append(joined_state.name)
    transitions = transition_pairs(joined_states)
    assigned_rate_dict = transition_rate_dict(rates, transitions)
    if induction_rate:
        assigned_rate_dict = induction(assigned_rate_dict, transitions, induction_rate, states)
    vector = initial_row_vector(number)
    transition_matrix, _, row_sums = transition_matrices(assigned_rate_dict, transitions)

    predefined_args = [joined_states, state_names, transitions, assigned_rate_dict, induction, vector,
                       transition_matrix, row_sums]

    return predefined_args
