"""
Module blinking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import figure as fi
from . import transitions as tr

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.emissions import Emissions
    from fluopy.simulation import Simulation


class Blinking:
    """
    Container for blinking-associated attributes.

    Attributes
    ----------
    emissions : fluopy.emissions.Emissions
        Container for emission-associated attributes.
    on_periods : npt.NDArray[np.int64]
        Contains the durations of each ON period (in frames).
    off_periods : npt.NDArray[np.int64]
        Contains the durations of each OFF period (in frames).
    on_periods_frames : npt.NDArray[np.int64]
        Contains the first frame of each ON period.
    off_periods_frames : npt.NDArray[np.int64]
        Contains the first frame of each OFF period.
    """

    def __init__(
        self, emissions: Emissions, threshold: int = 0, memory: int = 0
    ) -> None:
        """
        Parameters
        ----------
        emissions
            Container for emission-associated attributes.
        threshold
            Maximum value of photons per frame to be considered an OFF frame.
        memory
            Number of OFF frames to be neglected. They are included in the ON times.
        """
        self.emissions = emissions
        (
            self.on_periods,
            self.off_periods,
            self.on_periods_frames,
            self.off_periods_frames,
        ) = get_blinking_statistics(
            event_time_series=self.emissions.event_time_series,
            threshold=threshold,
            memory=memory,
        )

    def plot(
        self,
        mode: Literal[
            "on_histogram", "off_histogram", "on_frame_series", "off_frame_series"
        ] = "off_histogram",
        **kwargs: Any,
    ) -> npt.NDArray[mplAxes]:
        """
        Plot histogram or frame series of ON or OFF periods.

        Parameters
        ----------
        mode : str
            One of 'on_histogram', 'off_histogram', 'on_frame_series',
            'off_frame_series'.
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        sec_per_frame = (
            self.emissions.event_time_series.index[1]
            - self.emissions.event_time_series.index[0]
        )
        if mode == "on_histogram":
            data = self.on_periods
            axes = plot_histogram(
                data=data, mode="ON", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "on_boxplot":
            data = self.on_periods
            axes = plot_boxplot(
                data=data, mode="ON", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "off_histogram":
            data = self.off_periods
            axes = plot_histogram(
                data=data, mode="OFF", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "off_boxplot":
            data = self.off_periods
            axes = plot_boxplot(
                data=data, mode="OFF", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "on_frame_series":
            data = np.array([np.arange(0, self.on_periods.size), self.on_periods])
            axes = plot_frame_series(data=data, mode="ON", **kwargs)
        elif mode == "off_frame_series":
            data = np.array([np.arange(0, self.off_periods.size), self.off_periods])
            axes = plot_frame_series(data=data, mode="OFF", **kwargs)
        else:
            raise ValueError(f"mode {mode} unknown.")

        return axes


def get_blinking_statistics(
    event_time_series: pd.Series, threshold: int = 0, memory: int = 0
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """
    Determines ON and OFF times of event_time_series given that each entry represents
    the collected photons of one frame. The ending period (doesn't matter whether it is
    ON or OFF) is discarded. If event_time_series starts with an OFF period, it is
    discarded.

    Parameters
    ----------
    event_time_series
        The return value of construct_event
        Contains the time points in seconds as index (time steps in between resemble
        frames) and the number of events (i.e., detected photons) as values.
    threshold
        Maximum value of photons per frame to be considered an OFF frame.
    memory
        Number of OFF frames to be neglected. They are included in the ON times.

    Returns
    -------
    on_periods : npt.NDArray[np.int64]
        Contains the durations of each ON period (in frames).
    off_periods : npt.NDArray[np.int64]
        Contains the durations of each OFF period (in frames).
    on_periods_frames : npt.NDArray[np.int64]
        Contains the first frame of each ON period.
    off_periods_frames : npt.NDArray[np.int64]
        Contains the first frame of each OFF period.
    """
    df = pd.DataFrame(
        {
            "frame": np.arange(0, event_time_series.size),
            "intensity": event_time_series.values,
        }
    )
    df = df[df.intensity > threshold]
    frames = df.frame.values
    remove_last_on_period = False
    if frames.size != 0:
        if frames[-1] + 1 == event_time_series.size:
            remove_last_on_period = True
        differences = np.diff(frames)
        off_periods_indices = np.where(differences > 1 + memory)[0]

        off_periods_frames = frames[off_periods_indices] + 1

        if off_periods_indices.size == 0:
            off_periods = np.array([], dtype=int)
            if not remove_last_on_period:
                on_periods = np.array([frames[-1] - frames[0] + 1])
                on_periods_frames = np.array([frames[0]])
            else:
                on_periods = np.array([])
                on_periods_frames = np.array([])
        else:
            off_periods_zero = np.where(differences > 1 + memory, 0, differences)
            # if off period, convert to 0
            interrupt_by_one_frame = (
                np.where(off_periods_indices[1:] - off_periods_indices[:-1] == 1)[0] + 1
            )
            # store indices where off periods are interrupted by only 1 on frame
            on_mask_init = np.ones(off_periods_indices.shape, bool)
            # initialize a boolean mask
            on_mask_init[interrupt_by_one_frame] = False
            # the mask is False where only 1 frame is on
            if off_periods_zero[0] == 0:
                on_mask_init[0] = False
            # interrupt_by_one_frame cannot store the information if the first frame is
            # on and followed by off frames

            cumsum = np.cumsum(off_periods_zero)
            # cumulative sum of off_periods_zero; if 0 (off period) the value stays the
            # same
            uniques, counts = np.unique(cumsum, return_counts=True)
            duplices = uniques[np.where(counts > 1)]
            duplices[1:] -= duplices[:-1]
            # get the values of cumsum which appear several times indicating an off
            # period the last step is to back calculate the cumulative sum

            on_periods = np.ones(off_periods_indices.shape, dtype=int)
            # initialize array of on_periods (ones for every entry)
            if duplices.size != 0:
                if duplices[0] == 0:
                    duplices = duplices[1:]
            # if, in the beginning, more than one off period are consecutively
            # interrupted by only one frame, the cumsum will give several 0 (meaning
            # that 0 will be one of duplices) even though these on periods should stay 1
            on_periods[on_mask_init] = duplices + 1
            # change the values which are not 1 to their true values

            max_index_off = np.max(off_periods_indices)
            # the index of the last off period
            if not remove_last_on_period:
                last_on_period = np.sum(off_periods_zero[max_index_off + 1 :]) + 1
                # the last on period is the sum of differences starting after the last
                # off period
                on_periods = np.append(on_periods, last_on_period)
                # add the last on period
            off_periods = differences[off_periods_indices] - 1
            # the off periods
            if frames[0] > memory:
                off_periods = np.insert(off_periods, 0, frames[0])
                # add an initial off period if the series doesn't start with on period
                off_periods_frames = np.insert(off_periods_frames, 0, 0)
                on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
                off_periods = np.delete(off_periods, 0)
                off_periods_frames = np.delete(off_periods_frames, 0)
            elif frames[0] != 0:
                on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
                on_periods_frames = np.insert(on_periods_frames, 0, frames[0])
            else:
                on_periods_frames = np.sum([off_periods_frames, off_periods], axis=0)
                on_periods_frames = np.insert(on_periods_frames, 0, 0)
            on_periods_frames = on_periods_frames[: on_periods.size]

    else:
        on_periods = np.array([])
        off_periods = np.array([])
        on_periods_frames = np.array([])
        off_periods_frames = np.array([])

    return on_periods, off_periods, on_periods_frames, off_periods_frames


def get_off_statistics(
    simulation: Simulation, index: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Determines ON and OFF intervals of a single fluorophore, where the OFF interval is
    defined as the fluorophore's time spend in off state, whilst the ON interval is the
    times in between. This differs from blinking, since it does not consider whether a
    photon is detected during the ON time (hence, all OFF are of photophysical nature)
    and it only considers one fluorophore.

    Parameters
    ----------
    simulation
        Container of simulation-associated attributes and methods.
    index
        Determines the fluorophore to be looked at.

    Returns
    -------
    on_off_times : npt.NDArray[np.int64]
        Contains time points at which ON and OFF intervals start and end.
    on_off_values : npt.NDArray[np.int64]
        Values that correspond to on_off_times. 0 if time is associated with OFF, 1
        otherwise.
    """
    if index + 1 > simulation.state_series.shape[0]:
        raise ValueError(
            f"index assumes {index + 1} fluorophores but "
            f"{simulation.state_series.shape[0]} are present."
        )
    off_index = np.where(simulation.state_series[index] == tr.SingleState.OFF.value)[0]
    off2_index = np.where(simulation.state_series[index] == tr.SingleState.OFF2.value)[
        0
    ]
    off_indices = np.sort(np.concatenate([off_index, off2_index]))
    if off_indices.size == 0:
        raise ValueError("no photophysical OFF states found.")
    high_diffs = np.where(np.diff(off_indices) > 1)[0]
    starting_indices = off_indices[high_diffs + 1]
    starting_indices = np.insert(starting_indices, 0, off_indices[0])
    ending_indices = off_indices[high_diffs] + 1  # ends when new state (s0) is reached
    off_start = simulation.time_series[starting_indices]
    off_end = simulation.time_series[ending_indices]
    on_start = np.copy(off_end)
    on_start = np.insert(on_start, 0, 0)
    on_end = np.copy(off_start)
    merged = np.concatenate((off_start, off_end, on_start, on_end))
    on_off_times = np.sort(merged)
    on_off_values = np.ones(int(on_off_times.size / 2))
    on_off_values[1::2] = 0
    on_off_values = np.vstack((on_off_values, on_off_values)).ravel("F")
    if on_off_values.size != on_off_times.size:
        on_off_values = np.append(on_off_values, 0)

    return on_off_times, on_off_values


def get_analytical_off_statistics(
    off_frames: npt.ArrayLike,
    off_periods: npt.ArrayLike,
    on_frames: npt.ArrayLike,
    frame_time: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Intended to be used for visualizing analytical ON and OFF periods in time.

    Parameters
    ----------
    off_frames
        Contains the first frame of each OFF period.
    off_periods
        Contains the durations of each OFF period (in frames).
    on_frames
        Contains the first frame of each ON period.
    frame_time
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.

    Returns
    -------
    on_off_times : npt.NDArray[np.int64]
        Contains time points at which ON and OFF intervals start and end.
    on_off_values : npt.NDArray[np.int64]
        Corresponding values of on_off_frames. 1 if ON, 0 if OFF.
    """
    off_frames = np.asarray(off_frames)
    off_periods = np.asarray(off_periods)
    on_frames = np.asarray(on_frames)

    if on_frames.size != 0:
        if on_frames[0] != 0:
            off_frames = np.insert(off_frames, 0, 0)
            off_periods = np.insert(off_periods, 0, on_frames[0])
    off_frames_start_end = np.ravel([off_frames, off_frames + off_periods], order="F")
    on_off_frames = np.vstack((off_frames_start_end, off_frames_start_end)).ravel("F")
    on_off_frames = np.insert(on_off_frames, 0, 0)
    on_off_values = np.ones(int(on_off_frames.size / 2))
    on_off_values[1::2] = 0
    on_off_values = np.vstack((on_off_values, on_off_values)).ravel("F")
    if on_off_values.size != on_off_frames.size:
        on_off_values = np.append(on_off_values, 0)

    on_off_times = on_off_frames * (
        pd.to_timedelta(frame_time) / np.timedelta64(1, "s")
    )

    return on_off_times, on_off_values


def plot_off_statistics(
    on_off_times: npt.ArrayLike, on_off_values: npt.ArrayLike, **kwargs: Any
) -> npt.NDArray[mplAxes]:
    """
    Plot the photophysical OFF/ON of one fluorophore.

    Parameters
    ----------
    on_off_times
        Contains time points at which ON and OFF intervals start and end.
    on_off_values
        Values that correspond to all_times. 0 if time is associated with OFF, 1
        otherwise.
    kwargs
        kwargs for fluopy.figure.universal_figure arguments

    Returns
    -------
    npt.NDArray[mplAxes]
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault("type_", "line")
    kwargs.setdefault("fontsize", 16)
    kwargs.setdefault("xlabel", "time [s]")
    kwargs.setdefault("yticklabels", {"labels": ["OFF", "ON"]})
    kwargs.setdefault("yticks", [0, 1])
    kwargs.setdefault("ylabel", "")
    axes = fi.universal_figure(data=[on_off_times, on_off_values], **kwargs)

    return axes


def plot_histogram(
    data: npt.ArrayLike,
    mode: Literal["ON", "OFF"] = "OFF",
    density: bool = True,
    display_mean: bool = True,
    as_time: str | None = None,
    sec_per_frame: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[mplAxes]:
    """
    Plot histogram of ON or OFF periods.

    Parameters
    ----------
    data
        The data
    mode
        One of 'ON' or 'OFF'.
    density
        Whether to display the histogram as probability densities. Else, probabilities.
    display_mean
        Whether to display the mean inside the plot. The unit corresponds to the unit
        of the x-axis.
    as_time
        If not None, display the x-axis as time in unit as_time.
    sec_per_frame
        Duration of a frame in seconds.
    kwargs
        kwargs for fluopy.figure.universal_figure arguments

    Returns
    -------
    npt.NDArray[mplAxes]
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault("type_", "hist")
    if density:
        kwargs.setdefault("ylabel", "Prob. density")
        kwargs.setdefault("density", True)
    else:
        kwargs.setdefault("ylabel", "Probability")
        kwargs.setdefault("weights", np.ones_like(data) / data.size)
    if as_time is not None:
        kwargs.setdefault("xlabel", f"{mode} period ({as_time})")
        if as_time == "ms":
            data = data * sec_per_frame * 1000
        elif as_time == "s":
            data = data * sec_per_frame
        else:
            raise ValueError("given unit not implemented.")
    else:
        kwargs.setdefault("xlabel", "Consecutive frames")

    axes = fi.universal_figure(data=data, **kwargs)

    mean_color = "black"
    if "ylabelcolor" in kwargs:
        mean_color = kwargs["ylabelcolor"]
    if display_mean:
        mean = np.mean(data)
        axes[0][0].text(
            x=0.3,
            y=0.85,
            s=rf"$\mu = {mean:.2f}$",
            transform=axes[0][0].transAxes,
            fontsize=16,
            color=mean_color,
        )

    return axes


def plot_boxplot(
    data: npt.ArrayLike,
    mode: Literal["ON", "OFF"] = "OFF",
    as_time: str | None = None,
    sec_per_frame: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[mplAxes]:
    """
    Plot boxplot of ON or OFF periods.

    Parameters
    ----------
    data
        The data
    mode
        One of 'ON' or 'OFF'.
    as_time
        If not None, display the y-axis as time in unit as_time.
    sec_per_frame
        Duration of a frame in seconds.
    kwargs
        kwargs for fluopy.figure.universal_figure arguments

    Returns
    -------
    npt.NDArray[mplAxes]
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault("type_", "boxplot")
    kwargs.setdefault("fontsize", 16)
    if as_time is not None:
        kwargs.setdefault("ylabel", f"{mode} period ({as_time})")
        if as_time == "ms":
            data = data * sec_per_frame * 1000
        elif as_time == "s":
            data = data * sec_per_frame
        else:
            raise ValueError("given unit not implemented.")
    else:
        kwargs.setdefault("ylabel", "consecutive frames")

    axes = fi.universal_figure(data=data, **kwargs)

    return axes


def plot_frame_series(
    data: npt.ArrayLike, mode: Literal["ON", "OFF"] = "OFF", **kwargs: Any
) -> npt.NDArray[mplAxes]:
    """
    Plot frame series of ON or OFF periods.

    Parameters
    ----------
    data
        Contains x and y data (2D).
    mode
        One of 'ON' or 'OFF'.
    kwargs
        kwargs for fluopy.figure.universal_figure arguments

    Returns
    -------
    npt.NDArray[mplAxes]
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault("type_", "line")
    kwargs.setdefault("xlabel", "identity")
    kwargs.setdefault("ylabel", f"consecutive {mode} frames")

    axes = fi.universal_figure(data=data, **kwargs)

    return axes
