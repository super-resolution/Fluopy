import warnings
import numpy as np


# the rate constant of off_state_rescue is expected to be dependent on distance and number of fluorophores
def off_state_rescue_rate(distance, number, constant):
    k = number * (1/distance) * constant

    return k


def off_state_rescue(distance, number, constant, assigned_rate_dict, rate_name_dict, transitions, states, k_in=None):
    """
    Adds the concept of off state recovery of one fluorophore induced by the non-emitting transition from S1 to S0 of
    a second fluorophore to the assigned_rate_dict and rate_name_dict.

    Parameters
    ----------
    distance : float
        Distance of the fluorophores to each other.
    number : int
        Number of fluorophores in proximity.
    constant : float
        Some number [...].
    assigned_rate_dict : dict
        The first return value of initialize.transition_rate_dict.
    rate_name_dict : dict
        The second return value of initialize.transition_rate_dict.
    transitions : dict
        The return value of initialize.transition_pairs.
    states : iterable object
        Contains elements of type str.
    k_in : float
        Predefined rate of off_state_rescue. Makes distance, number and constant unnecessary.

    Returns
    -------
    assigned_rate_dict : dict
        The possibly altered input parameter.
    rate_name_dict : dict
        The possibly altered input parameter.
    """
    if not k_in:
        k = off_state_rescue_rate(distance, number, constant)
    else:
        k = k_in
    if states == ("tS0", "tS1", "tT1", "cS0", "cS1", "OFF", "B"):
        for transition, value_pair in transitions.items():
            current_state, future_state = transition.split("__")
            current_state_split = current_state.split("_")
            future_state_split = future_state.split("_")
            if {"tS1", "OFF"}.issubset(set(current_state_split)):
                indices_one = [i for i, e in enumerate(current_state_split) if e == "tS1"]
                indices_two = [i for i, e in enumerate(current_state_split) if e == "OFF"]
                for i_1 in indices_one:
                    for i_2 in indices_two:
                        if "tS0" in future_state_split[i_1] and "tS0" in future_state_split[i_2]:
                            future_state_part = future_state_split[:]
                            current_state_part = current_state_split[:]
                            for h in sorted([i_1, i_2], reverse=True):
                                del(future_state_part[h])
                                del(current_state_part[h])
                            if not future_state_part == current_state_part:
                                break
                            else:
                                assigned_rate_dict[transition] = k
                                rate_name_dict[value_pair] = "off_state_rescue"
        return assigned_rate_dict, rate_name_dict
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
                                assigned_rate_dict[transition] += k  # here the rate is added since the
                                # transition occurs occasionally as well.
                                # The name of the off-on transition is not overwritten with induction
        return assigned_rate_dict, rate_name_dict
    else:
        warnings.warn("The concept of off-state recovery could not be added due to mismatch of state names!")
        return assigned_rate_dict, rate_name_dict


def fret_rate(distance, emission_rate, spectral_overlap_integral, dipole_orientation_factor, constant):
    k = constant * ((dipole_orientation_factor**2 * emission_rate) / distance**6) * spectral_overlap_integral

    return k


def fret(distance, emission_rate, spectral_overlap_integral, dipole_orientation_factor, constant,
         assigned_rate_dict, rate_name_dict, transitions, states, k_in=None):
    if not k_in:
        k = fret_rate(distance, emission_rate, spectral_overlap_integral, dipole_orientation_factor, constant)
    else:
        k = k_in

    if states == ("S0", "S1", "T1", "R", "B"):
        for transition, value_pair in transitions.items():
            current_state, future_state = transition.split("__")
            current_state_split = current_state.split("_")
            future_state_split = future_state.split("_")
            if {"S1", "S0"}.issubset(set(current_state_split)):
                indices_one = [i for i, e in enumerate(current_state_split) if e == "S1"]
                indices_two = [i for i, e in enumerate(current_state_split) if e == "S0"]
                for i_1 in indices_one:
                    for i_2 in indices_two:
                        if "S0" in future_state_split[i_1] and "S1" in future_state_split[i_2]:
                            future_state_part = future_state_split[:]
                            current_state_part = current_state_split[:]
                            for h in sorted([i_1, i_2], reverse=True):
                                del(future_state_part[h])
                                del(current_state_part[h])
                            if not future_state_part == current_state_part:
                                break
                            else:
                                assigned_rate_dict[transition] = k
                                rate_name_dict[value_pair] = "fret"
        return assigned_rate_dict, rate_name_dict


def dexter_rate(distance, spectral_overlap_integral, van_der_waals_radius_sum, constant):
    k = constant * spectral_overlap_integral * np.exp(-(2*distance)/van_der_waals_radius_sum)

    return k


def dexter_energy_transfer(distance, spectral_overlap_integral, van_der_waals_radius_sum, constant,
                           assigned_rate_dict, rate_name_dict, transitions, states, k_in=None):
    """


    Parameters
    ----------
    distance
    spectral_overlap_integral
    van_der_waals_radius_sum
    constant
    assigned_rate_dict
    rate_name_dict
    transitions
    states
    k_in

    Returns
    -------

    """
    if not k_in:
        k = dexter_rate(distance, spectral_overlap_integral, van_der_waals_radius_sum, constant)
    else:
        k = k_in

    if states == ("S0", "S1", "T1", "R", "B"):
        for transition, value_pair in transitions.items():
            current_state, future_state = transition.split("__")
            current_state_split = current_state.split("_")
            future_state_split = future_state.split("_")
            if {"S1", "S0"}.issubset(set(current_state_split)):
                indices_one = [i for i, e in enumerate(current_state_split) if e == "S1"]
                indices_two = [i for i, e in enumerate(current_state_split) if e == "S0"]
                for i_1 in indices_one:
                    for i_2 in indices_two:
                        if "S0" in future_state_split[i_1] and "S1" in future_state_split[i_2]:
                            future_state_part = future_state_split[:]
                            current_state_part = current_state_split[:]
                            for h in sorted([i_1, i_2], reverse=True):
                                del(future_state_part[h])
                                del(current_state_part[h])
                            if not future_state_part == current_state_part:
                                break
                            else:
                                assigned_rate_dict[transition] = k
                                rate_name_dict[value_pair] = "dexter"
        return assigned_rate_dict, rate_name_dict


def trivial_rate(distance, spectral_overlap_integral, constant)
