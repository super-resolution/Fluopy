import numpy as np
import pandas as pd
from scipy.stats import geom


def get_emissions(unique_transitions, transition_series):
    """
    Construct an array that contains indices for time_series which yield the time points at which emissions have
    happened.

    Parameters
    ----------
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.
    transition_series : np.ndarray
        The return value of processing.generate_transition_series.
        Contains the NEXT transition id for each corresponding simulated joined state (except the last).

    Returns
    -------
    emissions : np.ndarray
        Contains indices that correspond to time points at which emissions have happened. Using it to index
        state_series or transition_series will result in the outcome AFTER the emission event (hence, the joined state
        or transition the follows the emission event).
    """
    emitting_transition_ids = unique_transitions.loc[unique_transitions['fluorescence'] == True].index.to_numpy()
    emissions = np.where(transition_series == emitting_transition_ids)[0] + 1
    # + 1 since the emission of a photon happens during the transition and hence the signal will coincide with the
    # appearance of the follow-up joined state. The transition series contains the NEXT transition, so its original
    # index corresponds to the initial joined state, not the follow-up joined state.
    return emissions


def get_events(emissions, photon_collection_rate, seed=100):
    """
    Alters the emissions, keeping only a relative number of photon_collection_rate indices that are randomly selected.

    Parameters
    ----------
    emissions : np.ndarray
        The return value of get_emissions.
        Contains indices that correspond to time points at which emissions have happened.
    photon_collection_rate : float
        Number between 0 and 1, dictates the fraction of kept indices of emissions.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    events : np.ndarray
        Contains indices that correspond to time points at which emissions have happened and were detected.
    """
    rng = np.random.default_rng(seed)
    # select not-detected photons instead of detected photons to keep sorted indices
    amount_not_detected = round((1 - photon_collection_rate) * emissions.shape[0])
    no_events_indices = rng.choice(np.arange(0, emissions.shape[0]), size=amount_not_detected, replace=False)
    events = np.delete(emissions, no_events_indices)

    return events


def construct_event_time_series(time_points_events, resample, emccd_gain=None, seed=100):
    """
    Resamples time_points_events assuming that each entry represents one event at this time point measured in seconds.
    The time step size is defined by resample and its cumulative sum is the index in seconds. The accuracy is
    nanoseconds (dtype=timedelta64[ns]).

    Parameters
    ----------
    time_points_events : np.ndarray
        Contains the time points at which an event occurs.
    resample : str
        For possible input values, see pandas time series user's guid offset aliases.
        Resembles frame integration time.
    emccd_gain : int, None
        The gain of an EMCCD camera.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    event_time_series : pd.Series
        Contains the time points in seconds as index and the number of events (i.e., detected emissions) as values.
    """
    time_points_events = np.insert(time_points_events, 0, 0)  # add time zero to the time points
    # the 'false' event will be dealt with below (events[0] = 0)
    time_deltas = pd.to_timedelta(time_points_events, unit="s")

    if emccd_gain is not None:
        # emccd gain can be modelled with a gamma distribution
        # https://doi.org/10.1117/12.2004621
        # (and https://doi.org/10.1038/s41592-021-01236-x, https://doi.org/10.1038/nmeth.1447)
        # if number of photons is 1, it is exponential (since shape = number of photons), i.e. geometric
        events = geom.rvs(p=1/emccd_gain, size=time_points_events.shape[0], random_state=seed)
        events = events.astype(float)
    else:
        events = np.ones(shape=time_points_events.shape[0])
    events[0] = 0

    event_time_series = pd.Series(events, index=time_deltas)
    event_time_series = event_time_series.resample(resample).sum()
    time_deltas = event_time_series.index
    in_seconds = time_deltas / np.timedelta64(1, "s")
    event_time_series.index = in_seconds

    return event_time_series


def blink_statistics(event_time_series, threshold=0, memory=0, remove_heading_off_period=True):
    """
    Determines ON and OFF times of event_time_series given that each entry represents the collected photons of one
    frame.

    Parameters
    ----------
    event_time_series : pd.Series
        The return value of construct_event_time_series.
        Contains the time points in seconds as index (time steps in between resemble frames) and the number of events
        (i.e., detected photons) as values.
    threshold : int
        Maximum value of photons per frame to be considered an OFF frame.
    memory : int
        Number of OFF frames to be neglected. They are included in the ON times.
    remove_heading_off_period : bool
        If True and the series starts with an OFF frame, the leading OFF frame is discarded.

    Returns
    -------
    on_periods : np.ndarray
        Contains the durations of each ON period.
    off_periods : np.ndarray
        Contains the durations of each OFF period.
    on_periods_frames : np.ndarray
        Contains the first frame of each ON period.
    off_periods_frames : np.ndarray
        Contains the first frame of each OFF period.
    """
    df = pd.DataFrame({'frame': np.arange(0, event_time_series.size), 'intensity': event_time_series.values})

    df = df[df.intensity > threshold]

    frames = df.frame.values

    if frames.size != 0:

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
    else:
        on_periods = np.array([])
        off_periods = np.array([])
        on_periods_frames = np.array([])
        off_periods_frames = np.array([])

    return on_periods, off_periods, on_periods_frames, off_periods_frames
