import numpy as np
import pandas as pd


def identify_emitting_transitions(transitions):
    """
    Identifies the transitions and their index-pairs that resemble spontaneous emissions. The photon is expected to
    be of an energy level such that it reflects the molecule's fluorescence rather than phosphorescence.

    Parameters
    ----------
    transitions : dict
        Contains transitions as keys and index-pairs as values.

    Returns
    -------
    emitting_transitions : list
        Contains emitting transitions.
    emitting_transitions_indices : list
        Contains emitting transition indices.
    """
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
    Returns boolean array of length (arr - seq + 1)  which contains True at indices i, where arr[i] starts
    (and continues) with seq.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    seq : np.ndarray
        Search sequence.

    Returns
    -------
    mask : np.ndarray
        Boolean array of length (arr - seq + 1), True at i if arr[i] starts and continues with seq.

    """
    arr_size, seq_size = arr.size, seq.size
    r_seq = np.arange(seq_size)  # since all sequences are of size 2, r_seq is always [0, 1]
    index_array_2d = np.arange(arr_size - seq_size + 1)[:, None]  # creates a 2D array of all potential indices of arr
    # note that index at arr_size - seq_size + 1 is the last possible index at which seq can start (and continue)
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
    state_series : np.ndarray
        Contains the sequence of state indices of the Markov chain.
    emitting_transitions_indices : list
        Contains emitting transition indices (output of identify_emitting_transitions)

    Returns
    -------
    mask : np.ndarray
        Boolean array of length state_series, True at i if state_series[i] is a state following an emitting transition.
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


def detected_emissions(emitting_mask, photon_collection, seed):
    """
    Returns a possibly altered emitting_mask, dictated by photon_collection. A random subset of photon_collection
    times the total True count of emitting mask is kept as True, rest is converted to False.

    Parameters
    ----------
    emitting_mask : np.ndarray
        Output of emitter_mask.
    photon_collection : float
        Number between 0 and 1, dictates the fraction of kept True values of emitting_mask.
    seed : int
        Seed to initialize a BitGenerator.

    Returns
    -------
    collection_mask : np.ndarray
        Possibly altered emitting_mask.
    """
    rng = np.random.default_rng(seed)
    emission_indices = np.nonzero(emitting_mask)[0]
    not_collected_total = round((1 - photon_collection) * len(emission_indices))
    not_collected_emission_indices = rng.choice(emission_indices, size=not_collected_total, replace=False)
    collection_mask = emitting_mask.copy()
    collection_mask[not_collected_emission_indices] = False

    return collection_mask


def pandas_event_time_series(events_at, unit, resample):
    """
    Resampling of events_at assuming each entry representing one event at this time point (measured in unit). Binning
    or resampling is specified by resample.

    Parameters
    ----------
    events_at : np.ndarray
        Contains the time points at which an event occurs.
    unit : str
        Unit of events_at. One of "W", "D", "h", "m", "S", "ms", "us", "ns".
    resample : str
        See pandas time series user's guide offset aliases for possible input values.

    Returns
    -------
    series : pd.Series
        Contains the time step in seconds as index and the number of events as values.
    """
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


# def blinking(pandas_series, threshold):
#     df = pd.DataFrame({'frame': np.arange(0, len(pandas_series)), 'intensity': pandas_series.values})
#     df = df[df.intensity > threshold]
#
#     frames = df.frame.values
#     differences = np.diff(frames)
#
#     mask = np.where(differences > 1)
#     off_periods = differences[mask] - 1
#     if frames[0] != 0:
#         off_periods = np.insert(off_periods, 0, frames[0] + 1)
#
#     on_periods = np.diff(mask[0])
#     on_periods = np.insert(on_periods, 0, mask[0][0] + 1)
#     on_periods = np.append(on_periods, frames.shape[0] - 1 - mask[0][-1])
#
#     return on_periods, off_periods


def blink_statistics(pandas_series, threshold=0, memory=0, remove_heading_off_period=True):
    """
    Returns on and off times of a pandas.Series given that each entry represents the collected photons of one frame.
    The threshold parameter is a minimum value of photons per frame to be considered an on-frame. The memory parameter
    provides a number of off-frames to be skipped/neglected.

    Parameters
    ----------
    pandas_series : pd.Series
        Contains the time step (of a frame) in seconds as index and the number of events (photons) as values.
    threshold : int
        Minimum value of photons per frame to be considered an on-frame.
    memory : int
        Number of off-frames to be neglected. They are included in the on times.
    remove_heading_off_period : bool
        If True and the series starts with an off-frame, the leading off-frame is discarded.

    Returns
    -------
    on_periods : np.ndarray
        Contains the lengths of each on-period.
    off_periods : np.ndarray
        Contains the lengths of each off-period.
    on_periods_frames : np.ndarray
        Contains the first frame of each on-period.
    off_periods_frames : np.ndarray
        Contains the first frame of each off-period.
    """
    df = pd.DataFrame({'frame': np.arange(0, len(pandas_series)), 'intensity': pandas_series.values})
    df = df[df.intensity > threshold]

    frames = df.frame.values

    differences = np.diff(frames)
    off_periods_indices = np.where(differences > 1 + memory)[0]

    off_periods_frames = frames[off_periods_indices] + 1

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
        if duplices.size != 0:
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

    return on_periods, off_periods, on_periods_frames, off_periods_frames


def frac_int_time(pandas_series, fraction):
    """
    Returns the relative time at which the specified fraction of the total collected photons is reached.

    Parameters
    ----------
    pandas_series : pd.Series
        Contains the time step (of a frame) in seconds as index and the number of events (photons) as values.
    fraction : float
        Number between 0 and 1.

    Returns
    -------
    arrival_time_rel : float
        The relative time at which the fraction of the total collected photons is reached.
    """
    end_time = pandas_series.index[-1]

    cumsum = pandas_series.cumsum()
    cumsum_norm = cumsum.multiply(1 / cumsum.max())
    arrival_time = cumsum_norm.gt(fraction).idxmax()
    arrival_time_rel = arrival_time / end_time

    return arrival_time_rel


########################################################################################################################


def on_states(state_names):
    """
    Counts the number of on states of each state (i.e., joined_state).

    Parameters
    ----------
    state_names : Collection
        Contains all state names.

    Returns
    -------
    on_counts : np.ndarray
        Contains the number of on states of each state.
    """
    on_counts = np.zeros(len(state_names))
    for i, state_name in enumerate(state_names):
        states = state_name.split("_")
        counter = states.count("ON")
        on_counts[i] = counter

    return on_counts


def emission_count(s0s1_rate, s1s0_rate, on_counts, state_series, time_step_series, resample=5e-3, seed=100):
    """
    Samples the on counts over a delta time (resample) and converts them into photon counts. This involves stretching
    the data since simulated time steps are expected to be larger than resample, meaning that the resulting time steps
    and their corresponding photon counts are an approximation.

    Parameters
    ----------
    s0s1_rate : float
        Rate constant of the transition from S0 to S1.
    s1s0_rate : float
        Rate constant of the transition from S1 to S0.
    on_counts : np.ndarray
        The return value of on_states.
    state_series : np.ndarray
        Contains the sequence of state indices of the Markov chain.
    time_step_series : np.ndarray
        Contains the time step until the corresponding state occurs (starting from the previous state).
    resample : float
        The delta time over which the number of photon emissions shall be sampled.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    emissions : np.ndarray
        Contains the photon counts per time step (i.e., resample).
    emission_time_series : np.ndarray
        Contains the time points at which the corresponding photon count occurs.
    """
    rng = np.random.default_rng(seed)

    on_counts_series = on_counts[state_series]  # converts the state_series into a series of on counts

    repeats = time_step_series[1:] / resample
    repeats = np.round(repeats)
    repeats[np.where(repeats == 0)] = 1
    repeats = repeats.astype(int)  # an entry in repeats is how often the resample value fits into the time step

    stretched = np.repeat(on_counts_series[:-1], repeats)  # stretch each entry of the on_counts_series by the
    # corresponding entry of repeats

    mean_emissions_per_s = s0s1_rate * s1s0_rate / (s0s1_rate + s1s0_rate)  # this holds true if the two state markov
    # chain can only be of values 0 and 1. Can be shown by simulation.
    emissions_per_resample = rng.poisson(lam=mean_emissions_per_s*resample, size=stretched.shape)

    emissions = stretched * emissions_per_resample

    emission_time_series = np.arange(0, len(stretched)*resample, resample)

    return emissions, emission_time_series
