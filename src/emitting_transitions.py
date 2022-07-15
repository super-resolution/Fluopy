import numpy as np
import pandas as pd


def identify_emitting_transitions(transitions):
    emitting_transitions = []
    emitting_transitions_indices = []
    for transition in transitions:
        current_state, future_state = transition.split("__")
        current_state_split = current_state.split("_")
        future_state_split = future_state.split("_")
        if "S1" in current_state_split:
            indices_current = [i for i, e in enumerate(current_state_split) if e == "S1"]
            for i in indices_current:
                if "S0" in future_state_split[i]:
                    future_state_part = future_state_split[:i] + future_state_split[i+1:]
                    current_state_part = current_state_split[:i] + current_state_split[i+1:]
                    if not future_state_part == current_state_part:
                        break
                    else:
                        emitting_transitions.append(transition)
                        emitting_transitions_indices.append(transitions[transition])

    return emitting_transitions, emitting_transitions_indices


def search_sequence(arr, seq):
    """
    Returns boolean array of length arr which contains True at indices i, where arr[i] starts (and continues) with seq.

    Parameters
    ----------
    arr
    seq

    Returns
    -------
    mask

    """
    arr_size, seq_size = arr.size, seq.size
    r_seq = np.arange(seq_size)  # since all sequences are of size 2, r_seq is always [0, 1]
    index_array_2d = np.arange(arr_size - seq_size + 1)[:, None]  # creates a 2D array of all indices of arr
    # note that +1 due to arange stops one step ahead and -seq_size due to the later addition of r_seq
    check_match_at = index_array_2d + r_seq  # the first index of arr [0] is converted to [0, 1] and so on
    check_match = arr[check_match_at] == seq  # checks if arr at index check_match_at equals (partially) the input seq
    mask = check_match.all(axis=1)  # returns True if all of check_match along axis 1 are True

    return mask


def emitter_mask(state_series, emitting_transitions_indices):
    """
    Returns boolean array of length state_series which contains True at indices i, where state_series[i] is the state
    immediately after the emission has happened.

    Parameters
    ----------
    state_series
    emitting_transitions_indices

    Returns
    -------

    """
    mask_1 = np.zeros(shape=len(state_series), dtype=np.int64)
    for emitting_transition_index_pair in emitting_transitions_indices:

        emitting_transition_index_pair = np.array(emitting_transition_index_pair)
        mask_2 = search_sequence(state_series, emitting_transition_index_pair)
        mask_1[np.where(mask_2)[0] + 1] = int(1)
        # +1 since the emission of a photon happens during the transition from S1 to S0, the emitting signal
        # will coincide with the appearance of S0 and will be set at the end of the time interval of S1 and
        # at the beginning of the time interval of S0
    mask = mask_1.astype(bool)

    return mask


def pandas_event_time_series(events_at, unit, resample):
    events_at_zero = np.insert(events_at, 0, 0)  # add time zero to the events (there will be no event)
    timedeltas = pd.to_timedelta(events_at_zero, unit=unit)
    events = np.ones(shape=(len(events_at_zero)))
    events[0] = 0
    series = pd.Series(events, index=timedeltas)
    series = series.resample(resample).sum()
    timedeltas = series.index
    in_seconds = timedeltas / np.timedelta64(1, "s")
    series.index = in_seconds

    return series


def blinking(pandas_series, threshold):
    df = pd.DataFrame({'frame': np.arange(0, len(pandas_series)), 'intensity': pandas_series.values})
    df = df[df.intensity > threshold]

    frames = df.frame.values
    differences = np.diff(frames)

    mask = np.where(differences > 1)
    off_periods = differences[mask] - 1
    if frames[0] != 0:
        off_periods = np.insert(off_periods, 0, frames[0] + 1)

    on_periods = np.diff(mask[0])
    on_periods = np.insert(on_periods, 0, mask[0][0] + 1)
    on_periods = np.append(on_periods, frames.shape[0] - 1 - mask[0][-1])

    return on_periods, off_periods


def blink_statistics(pandas_series, threshold, memory=0, remove_heading_off_period=True):
    df = pd.DataFrame({'frame': np.arange(0, len(pandas_series)), 'intensity': pandas_series.values})
    df = df[df.intensity > threshold]

    frames = df.frame.values

    differences = np.diff(frames)
    off_periods_indices = np.where(differences > 1 + memory)[0]

    off_periods_frames = frames[off_periods_indices] + 1
    on_periods_indices = np.split(np.arange(0, frames.shape[0]), off_periods_indices + 1)

    if off_periods_indices.size == 0:
        off_periods = np.array([], dtype=int)
        if frames[0] > memory and not remove_heading_off_period:
            off_periods = np.insert(off_periods, 0, frames[0])
            on_periods = np.array([frames[-1] - frames[0] + 1])
            off_periods_frames = np.array([0])
            on_periods_frames = np.array(frames[0])
            if remove_heading_off_period:
                off_periods = np.delete(off_periods, 0)
        elif not remove_heading_off_period:
            on_periods = np.array([frames[-1] + 1])
            on_periods_frames = np.array([0])
        else:
            on_periods = np.array([frames[-1] - frames[0] + 1])
            on_periods_frames = np.array(frames[0])

    else:
        off_periods_zero = np.where(differences > 1 + memory, 0, differences)
        # if off period, convert to 0
        interrupt_by_one_frame = np.where(off_periods_indices[1:] - off_periods_indices[:-1] == 1)[0] + 1
        # store indices where off periods are interrupted by only 1 on frame
        on_mask_init = np.ones(off_periods_indices.shape, bool)
        # initialize a boolean mask
        on_mask_init[interrupt_by_one_frame] = False
        # the mask is False where only 1 frame is on
        if off_periods_zero[0] == 0:
            on_mask_init[0] = False
        # interrupt_by_one_frame cannot store the information if the first frame is on and
        # followed by off frames

        cumsum = np.cumsum(off_periods_zero)
        # cumulative sum of off_periods_zero; if 0 (off period) the value stays the same
        uniques, counts = np.unique(cumsum, return_counts=True)
        duplices = uniques[np.where(counts > 1)]
        duplices[1:] -= duplices[:-1]
        # get the values of cumsum which appear several times indicating an off period
        # the last step is to back calculate the cumulative sum

        on_periods = np.ones(off_periods_indices.shape, dtype=int)
        # initialize array of on_periods (ones for every entry)
        if duplices[0] == 0:
            duplices = duplices[1:]
        # if, in the beginning, more than one off period are consecutively interrupted by only
        # one frame, the cumsum will give several 0 (meaning that 0 will be one of duplices)
        # even though these on periods should stay 1
        on_periods[on_mask_init] = duplices + 1
        # change the values which are not 1 to their true values

        max_index_off = np.max(off_periods_indices)
        # the index of the last off period
        last_on_period = np.sum(off_periods_zero[max_index_off + 1:]) + 1
        # the last on period is the sum of differences starting after the last off period
        on_periods = np.append(on_periods, last_on_period)
        # add the last on period
        off_periods = differences[off_periods_indices] - 1
        # the off periods

        if frames[0] > memory:
            off_periods = np.insert(off_periods, 0, frames[0])
            # add an initial off period if the series doesn't start with on period
            off_periods_frames = np.insert(off_periods_frames, 0, 0)
            on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
            if remove_heading_off_period:
                off_periods = np.delete(off_periods, 0)
                off_periods_frames = np.delete(off_periods_frames, 0)
        elif frames[0] != 0:
            if not remove_heading_off_period:
                on_periods[0] += frames[0]
                # add the initial on frames if they are rescued by memory
                on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
                on_periods_frames = np.insert(on_periods_frames, 0, 0)
            else:
                on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
                on_periods_frames = np.insert(on_periods_frames, 0, frames[0])
        else:
            on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
            on_periods_frames = np.insert(on_periods_frames, 0, 0)

    return on_periods, off_periods, on_periods_frames, off_periods_frames, on_periods_indices


def frac_int_time(pandas_series, fraction):
    end_time = pandas_series.index[-1]
    cumsum = pandas_series.cumsum()
    cumsum_norm = cumsum.multiply(1 / pandas_series.max())

    arrival_time = cumsum_norm.gt(fraction).idxmax()
    arrival_time_rel = arrival_time / end_time

    return arrival_time_rel


########################################################################################################################


def on_states(state_names):
    on_counts = np.zeros(len(state_names))
    for i, state_name in enumerate(state_names):
        states = state_name.split("_")
        counter = states.count("ON")
        on_counts[i] = counter

    return on_counts


def emission_count(s0s1_rate, s1s0_rate, on_counts, state_series, time_step_series, resample=5e-3, seed=100):
    rng = np.random.default_rng(seed)

    on_counts_series = on_counts[state_series]

    repeats = time_step_series[1:] / resample
    repeats = np.round(repeats)
    repeats[np.where(repeats == 0)] = 1
    repeats = repeats.astype(int)

    stretched = np.repeat(on_counts_series[:-1], repeats)

    mean_emissions_per_s = s0s1_rate * s1s0_rate / (s0s1_rate + s1s0_rate)  # this holds true if the two state markov
    # chain can only be of values 0 and 1. Can be shown by simulation.
    emissions_per_resample = rng.poisson(lam=mean_emissions_per_s*resample, size=stretched.shape)

    emissions = stretched * emissions_per_resample

    emission_time_series = np.arange(0, len(stretched)*resample, resample)
    return emissions, emission_time_series
