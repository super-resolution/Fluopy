"""
Extract fluorescence intermittency (blinking).
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


__all__: list[str] = ["Blinking"]


class Blinking:
    """
    Container for blinking-associated attributes.

    Attributes
    ----------
    emissions : fluopy.emissions.Emissions
        Container for emission-associated attributes.
    channel : str
        Detection channel used to determine blinking periods.
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
        self,
        emissions: Emissions,
        threshold: int = 0,
        memory: int = 0,
        channel: str | None = None,
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
        channel
            Detection channel used to determine blinking periods. If None, the only
            configured channel is used.
        """
        self.emissions = emissions
        self.channel = emissions.resolve_channel(channel)
        if self.emissions.event_time_series is None:
            raise ValueError("blinking statistics require extracted emissions.")
        event_time_series = self.emissions.select_event_time_series(self.channel)
        (
            self.on_periods,
            self.off_periods,
            self.on_periods_frames,
            self.off_periods_frames,
        ) = get_blinking_statistics(
            event_time_series=event_time_series,
            threshold=threshold,
            memory=memory,
        )

    def plot(
        self,
        mode: Literal[
            "on_histogram",
            "off_histogram",
            "on_frame_series",
            "off_frame_series",
            "on_boxplot",
            "off_boxplot",
        ] = "off_histogram",
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot histogram, boxplot or frame series of ON or OFF periods.

        Parameters
        ----------
        mode
            One of 'on_histogram', 'off_histogram', 'on_frame_series',
            'off_frame_series', 'on_boxplot', 'off_boxplot'.
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.emissions.event_time_series is None:
            raise ValueError("plotting requires extracted emissions.")
        event_time_series = self.emissions.select_event_time_series(self.channel)
        sec_per_frame = float(event_time_series.index[1] - event_time_series.index[0])
        if mode == "on_histogram":
            data = self.on_periods
            ax = plot_histogram(
                data=data, mode="ON", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "on_boxplot":
            data = self.on_periods
            ax = plot_boxplot(
                data=data, mode="ON", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "off_histogram":
            data = self.off_periods
            ax = plot_histogram(
                data=data, mode="OFF", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "off_boxplot":
            data = self.off_periods
            ax = plot_boxplot(
                data=data, mode="OFF", sec_per_frame=sec_per_frame, **kwargs
            )
        elif mode == "on_frame_series":
            data = np.array([np.arange(0, self.on_periods.size), self.on_periods])
            ax = plot_frame_series(data=data, mode="ON", **kwargs)
        elif mode == "off_frame_series":
            data = np.array([np.arange(0, self.off_periods.size), self.off_periods])
            ax = plot_frame_series(data=data, mode="OFF", **kwargs)
        else:
            raise ValueError(f"mode {mode} unknown.")

        return ax


def get_blinking_statistics(
    event_time_series: pd.Series[Any], threshold: int = 0, memory: int = 0
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
    intensities = event_time_series.to_numpy(dtype=np.int64)
    frames = np.flatnonzero(intensities > threshold).astype(np.int64)
    if frames.size == 0:
        return frames.copy(), frames.copy(), frames.copy(), frames.copy()

    splits = np.flatnonzero(np.diff(frames) > memory + 1)

    on_periods_frames = frames[np.r_[0, splits + 1]]
    on_last_frames = frames[np.r_[splits, frames.size - 1]]
    on_periods = on_last_frames - on_periods_frames + 1

    off_periods_frames = on_last_frames[:-1] + 1
    off_periods = on_periods_frames[1:] - off_periods_frames

    if frames[-1] == event_time_series.size - 1:
        on_periods = on_periods[:-1]
        on_periods_frames = on_periods_frames[:-1]

    return on_periods, off_periods, on_periods_frames, off_periods_frames


def get_off_statistics(
    simulation: Simulation, index: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
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
    on_off_times : npt.NDArray[np.float64]
        Contains time points at which ON and OFF intervals start and end.
    on_off_values : npt.NDArray[np.int8]
        Values that correspond to on_off_times. 0 if time is associated with OFF, 1
        otherwise.
    """
    state_series = simulation.state_series
    time_series = simulation.time_series
    if state_series is None or time_series is None:
        raise ValueError("OFF statistics require a completed simulation.")
    if index + 1 > state_series.shape[0]:
        raise ValueError(
            f"index assumes {index + 1} fluorophores but "
            f"{state_series.shape[0]} are present."
        )
    states = state_series[index]
    is_off = (states == tr.SingleState.OFF.value) | (
        states == tr.SingleState.OFF2.value
    )
    if not np.any(is_off):
        raise ValueError("no photophysical OFF states found.")

    changes = np.flatnonzero(is_off[1:] != is_off[:-1]) + 1
    starts = np.r_[0, changes]
    edges = np.r_[time_series[starts], time_series[-1]]

    on_off_times = np.repeat(edges, 2)[1:-1]
    on_off_values = np.repeat((~is_off[starts]).astype(np.int8), 2)

    return on_off_times, on_off_values


def get_analytical_off_statistics(
    off_frames: npt.ArrayLike,
    off_periods: npt.ArrayLike,
    on_frames: npt.ArrayLike,
    frame_time: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
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
    on_off_times : npt.NDArray[np.float64]
        Contains time points at which ON and OFF intervals start and end.
    on_off_values : npt.NDArray[np.int8]
        Corresponding values of on_off_frames. 1 if ON, 0 if OFF.
    """
    off_frames = np.asarray(off_frames)
    off_periods = np.asarray(off_periods)
    on_frames = np.asarray(on_frames)

    if on_frames.size != 0:
        if on_frames[0] != 0:
            off_frames = np.insert(arr=off_frames, obj=0, values=0)
            off_periods = np.insert(arr=off_periods, obj=0, values=on_frames[0])
    off_frames_start_end = np.ravel([off_frames, off_frames + off_periods], order="F")
    on_off_frames = np.vstack((off_frames_start_end, off_frames_start_end)).ravel("F")
    on_off_frames = np.insert(arr=on_off_frames, obj=0, values=0)
    on_off_values = np.ones(int(on_off_frames.size / 2), dtype=np.int8)
    on_off_values[1::2] = 0
    on_off_values = np.vstack((on_off_values, on_off_values)).ravel("F")
    if on_off_values.size != on_off_frames.size:
        on_off_values = np.append(on_off_values, np.array([0], dtype=np.int8))

    seconds_per_frame = float(pd.to_timedelta(frame_time) / np.timedelta64(1, "s"))
    on_off_times = np.asarray(on_off_frames * seconds_per_frame, dtype=np.float64)

    return on_off_times, on_off_values


def plot_off_statistics(
    on_off_times: npt.ArrayLike, on_off_values: npt.ArrayLike, **kwargs: Any
) -> mplAxes:
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
    matplotlib.axes.Axes
        The modified axis.
    """
    kwargs.setdefault("type_", "line")
    kwargs.setdefault("fontsize", 16)
    kwargs.setdefault("xlabel", "Time (s)")
    kwargs.setdefault("yticklabels", {"labels": ["OFF", "ON"]})
    kwargs.setdefault("yticks", [0, 1])
    kwargs.setdefault("ylabel", "")
    ax = fi.universal_figure(data=[on_off_times, on_off_values], **kwargs)

    return ax


def plot_histogram(
    data: npt.ArrayLike,
    mode: Literal["ON", "OFF"] = "OFF",
    density: bool = True,
    display_mean: bool = True,
    as_time: str | None = None,
    sec_per_frame: float | None = None,
    **kwargs: Any,
) -> mplAxes:
    """
    Plot histogram of ON or OFF periods.

    Parameters
    ----------
    data
        The data.
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
    matplotlib.axes.Axes
        The modified axis.
    """
    data_array = np.asarray(data, dtype=np.float64)
    kwargs.setdefault("type_", "hist")
    if density:
        kwargs.setdefault("ylabel", "Prob. density")
        kwargs.setdefault("density", True)
    else:
        kwargs.setdefault("ylabel", "Probability")
        kwargs.setdefault("weights", np.ones_like(data_array) / data_array.size)
    if as_time is not None:
        if sec_per_frame is None:
            raise ValueError("sec_per_frame is required when as_time is specified.")
        kwargs.setdefault("xlabel", f"{mode} period ({as_time})")
        if as_time == "ms":
            data_array = data_array * sec_per_frame * 1000
        elif as_time == "s":
            data_array = data_array * sec_per_frame
        else:
            raise ValueError("given unit not implemented.")
    else:
        kwargs.setdefault("xlabel", "Consecutive frames")

    ax = fi.universal_figure(data=data_array, **kwargs)

    mean_color = kwargs.get("ylabelcolor", "black")
    fontsize = kwargs.get("fontsize", 16)
    if display_mean:
        mean = np.mean(data_array)
        ax.text(
            x=0.3,
            y=0.85,
            s=rf"$\mu = {mean:.2f}$",
            transform=ax.transAxes,
            fontsize=fontsize,
            color=mean_color,
        )

    return ax


def plot_boxplot(
    data: npt.ArrayLike,
    mode: Literal["ON", "OFF"] = "OFF",
    as_time: str | None = None,
    sec_per_frame: float | None = None,
    **kwargs: Any,
) -> mplAxes:
    """
    Plot boxplot of ON or OFF periods.

    Parameters
    ----------
    data
        The data.
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
    matplotlib.axes.Axes
        The modified axis.
    """
    data_array = np.asarray(data, dtype=np.float64)
    kwargs.setdefault("type_", "boxplot")
    kwargs.setdefault("fontsize", 16)
    if as_time is not None:
        if sec_per_frame is None:
            raise ValueError("sec_per_frame is required when as_time is specified.")
        kwargs.setdefault("ylabel", f"{mode} period ({as_time})")
        if as_time == "ms":
            data_array = data_array * sec_per_frame * 1000
        elif as_time == "s":
            data_array = data_array * sec_per_frame
        else:
            raise ValueError("given unit not implemented.")
    else:
        kwargs.setdefault("ylabel", "consecutive frames")

    ax = fi.universal_figure(data=data_array, **kwargs)

    return ax


def plot_frame_series(
    data: npt.ArrayLike, mode: Literal["ON", "OFF"] = "OFF", **kwargs: Any
) -> mplAxes:
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
    matplotlib.axes.Axes
        The modified axis.
    """
    kwargs.setdefault("type_", "line")
    kwargs.setdefault("xlabel", "identity")
    kwargs.setdefault("ylabel", f"consecutive {mode} frames")

    ax = fi.universal_figure(data=data, **kwargs)

    return ax
