import numpy as np
import warnings
import pandas as pd
import src.simulation as si


def simulate_TCSPC(
    transition_set,
    emitting_transition_ids,
    et_transition_ids=None,
    number_pulses=1e5,
    pulse_duration=5e-11,
    time_between_pulses=25e-9,
    excitation_rates={},
    frame_time="1ms",
    size=1e5,
    store_time_points=False,
    seed=None,
):
    """
    Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence lifetime
    measurement). Methodically the direct method of the gillespie algorithm.
    The simulation is bound to start at the state configuration where all fluorophores
    are in the ground state.
    Note: the fluorescence lifetimes are the time differences of photon emission
    to last laser pulse.


    Parameters
    ----------
    transition_set : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes
    emitting_transition_ids : dict
        Contains the combined_state_transition indices as keys and their probability of
        passing a bandpass filter as values.
    et_transition_ids : None, list
        Contains the combined_state_transition indices that are emissions when energy
        transfer is available.
    number_pulses : int
        Number of pulses simulated.
    pulse_duration : float
        The duration of a laser pulse in s. This time is used to calculate the
        probability of excitation, other than that it is neglected.
    time_between_pulses : float
        The time between 2 laser pulses in s.
    excitation_rates : dict
        Contains the fluorophore names as keys and the excitation rates as values.
        Assumes uniform irradiance over the pulse duration.
    frame_time : str
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
    size : int
        Size of random numbers drawn at once.
    store_time_points : bool
        Whether to store the time points at which emissions are detected.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and
        the number of events (i.e., detected emissions) as values.
    event_time_points : 1-D array_like
        The time points at which emissions are detected.
    lifetimes_DA : 1-D array_like
        Contains the fluorescence lifetimes of detected emissions when energy transfer
        available.
    lifetimes_D : 1-D array_like
        Contains the fluorescence lifetimes of detected emissions when energy transfer
        not available.
    lifetimes_all : 1-D array_like
        Contains the fluorescence lifetimes of all detected emissions.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    transition_matrix_non_exc = transition_set.transition_matrix
    row_sums_non_exc = transition_set.row_sums

    transition_matrix_sorted_indices_non_exc = np.argsort(
        transition_matrix_non_exc, axis=1
    )
    sorted_transition_matrix_non_exc = np.take_along_axis(
        transition_matrix_non_exc, transition_matrix_sorted_indices_non_exc, axis=1
    )
    cumsum_sorted_trm_non_exc = np.cumsum(sorted_transition_matrix_non_exc, axis=1)
    number_fluorophores = transition_set.fluorophore_system.count
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            f.identity
            for f in transition_set.fluorophore_system.fluorophores
            if f.name == fluorophore
        ]
        excitation_rate = excitation_rates[fluorophore]
        excitation_probability = 1 - np.exp(
            -excitation_rate * pulse_duration
        )  # CDF of exponential distribution
        excitation_probabilities[fluorophore_ids] = excitation_probability
    excitable_indices = np.where(excitation_probabilities != 0)[0]
    rng = np.random.default_rng(seed)
    frame_time = pd.Timedelta(frame_time) / np.timedelta64(1, "s")
    frames = number_pulses * time_between_pulses / frame_time
    time_stamps = np.linspace(0, np.ceil(frames) * frame_time, int(np.ceil(frames)) + 1)
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        warnings.warn(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(frame_time/time_between_pulses)):.1e} pulses)."
        )
    warnings.warn(
        f"the last frame (of index {time_stamps[-1]}) has {frames-int(frames):.2e} "
        "times the pulses of other frames."
    )
    photon_collector = np.zeros(time_stamps.size)
    df = transition_set.combined_state_transitions_df
    current_state_index = df.loc[
        df["final_state"] == tuple(np.zeros(number_fluorophores))
    ].index[0]
    lifetimes_D = []
    lifetimes_DA = []
    lifetimes_all = []
    remember = [np.inf, np.inf]
    if et_transition_ids is None:
        et_transition_ids = []
    i = 0
    j = 1
    k = 0
    l = 1
    S0 = 0
    S1 = 1
    skip = False
    checked = False
    not_broken = True
    random_numbers_exc = rng.uniform(low=0, high=1, size=(size, excitable_indices.size))
    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    random_indices = np.array(
        [
            rng.integers(low=0, high=num + 1, size=size)
            for num in range(excitable_indices.size)
        ]
    )
    if store_time_points:
        time_points = []
    else:
        event_time_points = None

    while i < number_pulses:
        i += 1
        if i >= j * size:
            j = int(np.floor(i / size)) + 1
            random_numbers_exc = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
            random_indices = np.array(
                [
                    rng.integers(low=0, high=num + 1, size=size)
                    for num in range(excitable_indices.size)
                ]
            )
        time = (i - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(df["final_state"][current_state_index])
        S0s = np.where(current_states[excitable_indices] == S0)[0]
        # if there are any S0 states, excitations can occur
        if S0s.size != 0:
            not_broken = True
            excitations = np.where(
                random_numbers_exc[i - (j - 1) * size, S0s]
                < excitation_probabilities[excitable_indices[S0s]]
            )[0]
            # if excitations occur, the fluorophores are set to S1
            if excitations.size != 0:
                current_states[excitable_indices[S0s[excitations]]] = S1
            # if no excitation happened and if a transition that is remembered could
            # take place, the simulation continues from there
            elif remember[1] < next_pulse_time:
                skip = True
                next_transition = remember[0]
                time = remember[1]
            # if no excitation happened and if a potentielly remembered transition
            # could not take place, and if the number of pulses until an excitation
            # event have not been checked immidiately before, the number of pulses
            # until the next excitation event is calculated
            elif not checked:
                number_pulses_until_next_excitation = rng.geometric(
                    p=excitation_probabilities[excitable_indices[S0s]]
                )
                indices_minimum = np.where(
                    number_pulses_until_next_excitation
                    == np.min(number_pulses_until_next_excitation)
                )[0]
                random_index = indices_minimum[
                    random_indices[indices_minimum.size - 1][i - (j - 1) * size]
                ]
                potential_i = i + number_pulses_until_next_excitation[random_index] - 1
                # if the next excitation event is before a remembered transition takes
                # place, the excitation is realized at the calculated pulse (the pulse
                # still happens in case multiple fluorophores are excited, thats why
                # continue is used)
                if potential_i * time_between_pulses < remember[1]:
                    current_states[excitable_indices[S0s[random_index]]] = S1
                    current_state_index = df[
                        df["final_state"] == tuple(current_states)
                    ].index[0]
                    i = potential_i
                    checked = True
                    continue
                # if the next excitation event is after a remembered transition takes
                # place, the simulation continues from there
                else:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    i = int(np.floor(time / time_between_pulses))
                    next_pulse_time = (i + 1) * time_between_pulses
        # if there are no S0 states, no excitations can occur. If something is
        # remembered, the simulation continues from there
        else:
            # if not_broken is False, this means that in the previous pulse it was
            # already at this point (all non-excitable) and did not get past the
            # current_state_lambda above 0. This means that the Markov chain has
            # encountered an absorbing state that is not excitable S0.
            # Note that not_broken could not be set to False if the elif below is
            # carried out (due to the skip) which immidiately is followed by
            # current_state_lambda == 0. This could be the case if one fluorophore
            # is bleached and the other excitable S0, but then it will not get to this
            # point again immidiately after.
            if not not_broken:
                warnings.warn(
                    "All fluorophores underwent photobleaching or entered "
                    "another Markov chain absorbing state."
                )
                event_time_series = pd.Series(
                    photon_collector, index=time_stamps, dtype=np.int64
                )
                if store_time_points:
                    event_time_points = np.array(time_points)
                lifetimes_DA = np.array(lifetimes_DA)
                lifetimes_D = np.array(lifetimes_D)
                lifetimes_all = np.array(lifetimes_all)
                return (
                    event_time_series,
                    event_time_points,
                    lifetimes_DA,
                    lifetimes_D,
                    lifetimes_all,
                )
            elif remember[1] != np.inf and not checked:
                skip = True
                next_transition = remember[0]
                time = remember[1]
                i = int(np.floor(time / time_between_pulses))
                next_pulse_time = (i + 1) * time_between_pulses
            not_broken = False

        checked = False
        current_state_index = df[df["final_state"] == tuple(current_states)].index[0]
        while time < next_pulse_time:
            if not skip:
                k += 1
                if k == l * size:
                    l += 1
                    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
                current_state_lambda = row_sums_non_exc[current_state_index]
                if current_state_lambda == 0:
                    remember = [np.inf, np.inf]
                    break
                not_broken = True
                transition_time = (1 / current_state_lambda) * np.log(
                    1 / random_numbers[k - (l - 1) * size, 0]
                )
                time += transition_time
                sorted_index = np.searchsorted(
                    cumsum_sorted_trm_non_exc[current_state_index],
                    random_numbers[k - (l - 1) * size, 1],
                )

                next_transition = transition_matrix_sorted_indices_non_exc[
                    current_state_index, sorted_index
                ]
                if time > next_pulse_time:
                    remember = [next_transition, time]
                    break
            skip = False
            current_state_index = next_transition
            if next_transition in emitting_transition_ids:
                if (
                    random_numbers[k - (l - 1) * size, 2]
                    < emitting_transition_ids[next_transition]
                ):
                    if next_transition in et_transition_ids:
                        lifetimes_DA.append(time - last_pulse_time)
                    else:
                        lifetimes_D.append(time - last_pulse_time)
                    lifetimes_all.append(time - last_pulse_time)
                    frame = int(np.ceil(time / frame_time))
                    try:
                        photon_collector[frame] += 1
                        if store_time_points:
                            time_points.append(time)
                    except:
                        pass

    event_time_series = pd.Series(photon_collector, index=time_stamps, dtype=np.int64)
    if store_time_points:
        event_time_points = np.array(time_points)
    lifetimes_DA = np.array(lifetimes_DA)
    lifetimes_D = np.array(lifetimes_D)
    lifetimes_all = np.array(lifetimes_all)

    return (
        event_time_series,
        event_time_points,
        lifetimes_DA,
        lifetimes_D,
        lifetimes_all,
    )


def simulate_TCSPC_detailed(
    transition_set,
    emitting_transition_ids,
    et_transition_ids=None,
    number_pulses=1e5,
    pulse_duration=5e-11,
    time_between_pulses=25e-9,
    excitation_rates={},
    frame_time="1ms",
    size=1e5,
    store_time_points=False,
    seed=None,
):
    """
    Simulates experimental TCSPC data (i.e., pulsed excitation for fluorescence lifetime
    measurement). Methodically the direct method of the gillespie algorithm.
    The simulation is bound to start at the state configuration where all fluorophores
    are in the ground state.
    Note: the fluorescence lifetimes are the time differences of photon emission
    to last laser pulse.


    Parameters
    ----------
    transition_set : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes
    emitting_transition_ids : dict
        Contains the combined_state_transition indices as keys and their probability of
        passing a bandpass filter as values.
    et_transition_ids : None, list
        Contains the combined_state_transition indices that are emissions when energy
        transfer is available.
    number_pulses : int
        Number of pulses simulated.
    pulse_duration : float
        The duration of a laser pulse in s. This time is used to calculate the
        probability of excitation, other than that it is neglected.
    time_between_pulses : float
        The time between 2 laser pulses in s.
    excitation_rates : dict
        Contains the fluorophore names as keys and the excitation rates as values.
        Assumes uniform irradiance over the pulse duration.
    frame_time : str
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
    size : int
        Size of random numbers drawn at once.
    store_time_points : bool
        Whether to store the time points at which emissions are detected.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.

    Returns
    -------
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and
        the number of events (i.e., detected emissions) as values.
    event_time_points : 1-D array_like
        The time points at which emissions are detected.
    lifetimes_DA : 1-D array_like
        Contains the fluorescence lifetimes of detected emissions when energy transfer
        available.
    lifetimes_D : 1-D array_like
        Contains the fluorescence lifetimes of detected emissions when energy transfer
        not available.
    lifetimes_all : 1-D array_like
        Contains the fluorescence lifetimes of all detected emissions.
    """
    number_pulses = int(number_pulses)
    size = int(size)
    transition_matrix_non_exc = transition_set.transition_matrix
    row_sums_non_exc = transition_set.row_sums

    transition_matrix_sorted_indices_non_exc = np.argsort(
        transition_matrix_non_exc, axis=1
    )
    sorted_transition_matrix_non_exc = np.take_along_axis(
        transition_matrix_non_exc, transition_matrix_sorted_indices_non_exc, axis=1
    )
    cumsum_sorted_trm_non_exc = np.cumsum(sorted_transition_matrix_non_exc, axis=1)
    number_fluorophores = transition_set.fluorophore_system.count
    for f in transition_set.fluorophore_system.fluorophores:
        if f.name not in excitation_rates:
            raise ValueError("excitation rates must be provided for all fluorophores.")
    excitation_probabilities = np.zeros(number_fluorophores)
    for fluorophore in excitation_rates:
        fluorophore_ids = [
            f.identity
            for f in transition_set.fluorophore_system.fluorophores
            if f.name == fluorophore
        ]
        excitation_rate = excitation_rates[fluorophore]
        excitation_probability = 1 - np.exp(
            -excitation_rate * pulse_duration
        )  # CDF of exponential distribution
        excitation_probabilities[fluorophore_ids] = excitation_probability
    excitable_indices = np.where(excitation_probabilities != 0)[0]
    rng = np.random.default_rng(seed)
    frame_time = pd.Timedelta(frame_time) / np.timedelta64(1, "s")
    frames = number_pulses * time_between_pulses / frame_time
    time_stamps = np.linspace(0, np.ceil(frames) * frame_time, int(np.ceil(frames)) + 1)
    time_stamps = np.round(time_stamps, decimals=12)
    if frames < 1:
        warnings.warn(
            f"Not enough laser pulses to completely simulate a single frame (requires "
            f"at least {int(np.ceil(frame_time/time_between_pulses)):.1e} pulses)."
        )
    warnings.warn(
        f"the last frame (of index {time_stamps[-1]}) has {frames-int(frames):.2e} "
        "times the pulses of other frames."
    )
    photon_collector = np.zeros(time_stamps.size)
    df = transition_set.combined_state_transitions_df
    current_state_index = df.loc[
        df["final_state"] == tuple(np.zeros(number_fluorophores))
    ].index[0]
    time_series = []
    transition_series = []
    excitation_series = []
    lifetimes_D = []
    lifetimes_DA = []
    lifetimes_all = []
    remember = [np.inf, np.inf]
    if et_transition_ids is None:
        et_transition_ids = []
    i = 0
    j = 1
    k = 0
    l = 1
    S0 = 0
    S1 = 1
    skip = False
    checked = False
    not_broken = True
    random_numbers_exc = rng.uniform(low=0, high=1, size=(size, excitable_indices.size))
    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
    random_indices = np.array(
        [
            rng.integers(low=0, high=num + 1, size=size)
            for num in range(excitable_indices.size)
        ]
    )
    if store_time_points:
        time_points = []
    else:
        event_time_points = None

    while i < number_pulses:
        i += 1
        if i >= j * size:
            j = int(np.floor(i / size)) + 1
            random_numbers_exc = rng.uniform(
                low=0, high=1, size=(size, excitable_indices.size)
            )
            random_indices = np.array(
                [
                    rng.integers(low=0, high=num + 1, size=size)
                    for num in range(excitable_indices.size)
                ]
            )
        time = (i - 1) * time_between_pulses
        last_pulse_time = time
        next_pulse_time = time + time_between_pulses
        current_states = np.array(df["final_state"][current_state_index])
        S0s = np.where(current_states[excitable_indices] == S0)[0]
        # if there are any S0 states, excitations can occur
        if S0s.size != 0:
            not_broken = True
            # multiple excitations can occur within one pulse
            excitations = np.where(
                random_numbers_exc[i - (j - 1) * size, S0s]
                < excitation_probabilities[excitable_indices[S0s]] 
            )[0]
            # if excitations occur, the fluorophores are set to S1
            if excitations.size != 0:
                current_states[excitable_indices[S0s[excitations]]] = S1
                excitation_series.extend(excitable_indices[S0s[excitations]])
                time_series.extend([time] * excitations.size)
            # if no excitation happened and if a transition that is remembered could
            # take place, the simulation continues from there
            elif remember[1] < next_pulse_time:
                skip = True
                next_transition = remember[0]
                time = remember[1]
            # if no excitation happened and if a potentielly remembered transition
            # could not take place, and if the number of pulses until an excitation
            # event have not been checked immidiately before, the number of pulses
            # until the next excitation event is calculated
            elif not checked:
                number_pulses_until_next_excitation = rng.geometric(
                    p=excitation_probabilities[excitable_indices[S0s]]
                )
                indices_minimum = np.where(
                    number_pulses_until_next_excitation
                    == np.min(number_pulses_until_next_excitation)
                )[0]
                random_index = indices_minimum[
                    random_indices[indices_minimum.size - 1][i - (j - 1) * size]
                ]
                potential_i = i + number_pulses_until_next_excitation[random_index] - 1
                # if the next excitation event is before a remembered transition takes
                # place, the excitation is realized at the calculated pulse (the pulse
                # still happens in case multiple fluorophores are excited, thats why
                # continue is used)
                if potential_i * time_between_pulses < remember[1]:
                    current_states[excitable_indices[S0s[random_index]]] = S1
                    excitation_series.append(excitable_indices[S0s[random_index]])
                    time_series.append(potential_i * time_between_pulses)
                    current_state_index = df[
                        df["final_state"] == tuple(current_states)
                    ].index[0]
                    i = potential_i
                    checked = True
                    continue
                # if the next excitation event is after a remembered transition takes
                # place, the simulation continues from there
                else:
                    skip = True
                    next_transition = remember[0]
                    time = remember[1]
                    i = int(np.floor(time / time_between_pulses))
                    next_pulse_time = (i + 1) * time_between_pulses
        # if there are no S0 states, no excitations can occur. If something is
        # remembered, the simulation continues from there
        else:
            # if not_broken is False, this means that in the previous pulse it was
            # already at this point (all non-excitable) and did not get past the
            # current_state_lambda above 0. This means that the Markov chain has
            # encountered an absorbing state that is not excitable S0.
            # Note that not_broken could not be set to False if the elif below is
            # carried out (due to the skip) which immidiately is followed by
            # current_state_lambda == 0. This could be the case if one fluorophore
            # is bleached and the other excitable S0, but then it will not get to this
            # point again immidiately after.
            if not not_broken:
                warnings.warn(
                    "All fluorophores underwent photobleaching or entered "
                    "another Markov chain absorbing state."
                )
                event_time_series = pd.Series(
                    photon_collector, index=time_stamps, dtype=np.int64
                )
                if store_time_points:
                    event_time_points = np.array(time_points)
                lifetimes_DA = np.array(lifetimes_DA)
                lifetimes_D = np.array(lifetimes_D)
                lifetimes_all = np.array(lifetimes_all)
                time_series = np.array(time_series)
                if time_series[0] != 0:
                    time_series = np.insert(time_series, 0, 0)
                transition_series = np.array(transition_series)
                excitation_series = np.array(excitation_series)
                return (
                    event_time_series,
                    event_time_points,
                    lifetimes_DA,
                    lifetimes_D,
                    lifetimes_all,
                    time_series,
                    transition_series,
                    excitation_series,
                )
            elif remember[1] != np.inf and not checked:
                skip = True
                next_transition = remember[0]
                time = remember[1]
                i = int(np.floor(time / time_between_pulses))
                next_pulse_time = (i + 1) * time_between_pulses
            not_broken = False

        checked = False
        current_state_index = df[df["final_state"] == tuple(current_states)].index[0]
        while time < next_pulse_time:
            if not skip:
                k += 1
                if k == l * size:
                    l += 1
                    random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
                current_state_lambda = row_sums_non_exc[current_state_index]
                if current_state_lambda == 0:
                    remember = [np.inf, np.inf]
                    break
                not_broken = True
                transition_time = (1 / current_state_lambda) * np.log(
                    1 / random_numbers[k - (l - 1) * size, 0]
                )
                time += transition_time
                sorted_index = np.searchsorted(
                    cumsum_sorted_trm_non_exc[current_state_index],
                    random_numbers[k - (l - 1) * size, 1],
                )

                next_transition = transition_matrix_sorted_indices_non_exc[
                    current_state_index, sorted_index
                ]
                if time > next_pulse_time:
                    remember = [next_transition, time]
                    break
            skip = False
            current_state_index = next_transition
            transition_series.append(next_transition)
            excitation_series.append(-1)
            time_series.append(time)
            if next_transition in emitting_transition_ids:
                if (
                    random_numbers[k - (l - 1) * size, 2]
                    < emitting_transition_ids[next_transition]
                ):
                    if next_transition in et_transition_ids:
                        lifetimes_DA.append(time - last_pulse_time)
                    else:
                        lifetimes_D.append(time - last_pulse_time)
                    lifetimes_all.append(time - last_pulse_time)
                    frame = int(np.ceil(time / frame_time))
                    try:
                        photon_collector[frame] += 1
                        if store_time_points:
                            time_points.append(time)
                    except:
                        pass
    
    # if checked, the last simulated pulse is past the last existing pulse
    if checked:
        time_series = time_series[:-1]
        excitation_series = excitation_series[:-1]

    event_time_series = pd.Series(photon_collector, index=time_stamps, dtype=np.int64)
    if store_time_points:
        event_time_points = np.array(time_points)
    lifetimes_DA = np.array(lifetimes_DA)
    lifetimes_D = np.array(lifetimes_D)
    lifetimes_all = np.array(lifetimes_all)
    time_series = np.array(time_series)
    if time_series[0] != 0:
        time_series = np.insert(time_series, 0, 0)
    transition_series = np.array(transition_series)
    excitation_series = np.array(excitation_series)

    return (
        event_time_series,
        event_time_points,
        lifetimes_DA,
        lifetimes_D,
        lifetimes_all,
        time_series,
        transition_series,
        excitation_series,
    )


def space_multiple_excitations(time_series):
    indices = np.unique(time_series, return_index=True)[1]
    mask = np.ones(time_series.shape, dtype=bool)
    mask[indices] = False
    time_series[mask] = np.nextafter(time_series[mask], float('inf'))
    return time_series


def insert_excitations(transition_series, transition_set, excitation_series):
    transition_series_ad = transition_series.copy()
    df = transition_set.combined_state_transitions_df
    excitations = df[df['abbreviation'] == 'EXC']
    indices_transitions = np.where(excitation_series == -1)[0]
    diffs = np.diff(indices_transitions)
    diffs = np.insert(diffs, 0, 2)
    number_fluorophores = 2
    for i, diff in enumerate(range(number_fluorophores)):
        diff += 2
        processing = np.where(diffs >= diff)[0]
        corresponding_excitations = excitation_series[indices_transitions[processing] - 1 - i]
        if i != 0:
            insert_at = np.searchsorted(already_processed, processing, side='left')
            already_processed = np.insert(already_processed, insert_at, processing)
            processing += insert_at
        else:
            already_processed = processing
        transitions = transition_series_ad[processing]
        final_states = np.array(df['initial_state'].iloc[transitions].values.tolist())
        initial_states = final_states.copy()
        initial_states[np.arange(initial_states.shape[0]), corresponding_excitations] = 0
        df2 = pd.DataFrame({'initial_state': map(tuple, initial_states.tolist()), 'final_state': map(tuple, final_states.tolist())})
        df2.reset_index(names='original_index', inplace=True)
        merged = (excitations.reset_index(names='id').merge(df2, on=['initial_state', 'final_state'], how='inner'))
        indices = np.array(merged['id'].tolist())
        original_indices = np.array(merged['original_index'].tolist())
        transition_series_ad = np.insert(transition_series_ad, processing[original_indices], indices)

        return transition_series_ad


def get_state_series(transition_set, transition_series):
    start_at = tuple(
            np.zeros(shape=transition_set.fluorophore_system.count, dtype=int)
        )
    final_states = transition_set.combined_state_transitions_df["final_state"]

    state_series = np.empty(
            shape=(transition_set.fluorophore_system.count, transition_series.size + 1),
            dtype=np.int8,
        )
    state_series[:, 0] = start_at

    for i, _ in enumerate(final_states[0]):
        final_states_fluorophore = final_states.map(lambda x: x[i]).to_numpy(
            dtype=np.int8
        )
        state_series[i][1:] = final_states_fluorophore[transition_series]
    
    return state_series


def prepare_return_values(photon_collector, time_stamps, store_time_points, time_points,
                          lifetimes_DA, lifetimes_D, lifetimes_all, time_series, transition_series,
                          excitation_series, transition_set):
    event_time_series = pd.Series(photon_collector, index=time_stamps, dtype=np.int64)
    if store_time_points:
        event_time_points = np.array(time_points)
    lifetimes_DA = np.array(lifetimes_DA)
    lifetimes_D = np.array(lifetimes_D)
    lifetimes_all = np.array(lifetimes_all)
    time_series = np.array(time_series)
    if time_series[0] != 0:
        time_series = np.insert(time_series, 0, 0)
    transition_series = np.array(transition_series)
    excitation_series = np.array(excitation_series)

    time_series = space_multiple_excitations(time_series)
    transition_series = insert_excitations(transition_series, transition_set, excitation_series)
    state_series = get_state_series(transition_set, transition_series)

    simulation_object = si.Simulation(transition_set)
    simulation_object.time_series = time_series
    simulation_object.transition_series = transition_series
    simulation_object.state_series = state_series

    return event_time_series, event_time_points, lifetimes_DA, lifetimes_D, lifetimes_all, simulation_object
