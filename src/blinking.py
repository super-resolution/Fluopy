import numpy as np
import pandas as pd
import src.custom_plot as cp


class Blinking:
    def __init__(self, emissions, threshold=0, memory=0, remove_heading_off_period=True):
        self.emissions = emissions
        self.on_periods, self.off_periods, self.on_periods_frames, self.off_periods_frames = \
            get_blinking_statistics(self.emissions.event_time_series, )

    def plot(self, mode, **kwargs):
        if mode == 'on_histogram':
            data = self.on_periods
            kwargs.setdefault('title', 'ON periods')
            fig, ax = plot_histogram(data=data, **kwargs)
        elif mode == 'off_histogram':
            data = self.off_periods
            kwargs.setdefault('title', 'OFF periods')
            fig, ax = plot_histogram(data=data, **kwargs)
        elif mode == 'on_frame_series':
            data = [np.arange(0, self.on_periods.size), self.on_periods]
            kwargs.setdefault('title', 'ON periods')
            fig, ax = plot_frame_series(data=data, **kwargs)
        elif mode == 'off_frame_series':
            data = [np.arange(0, self.off_periods.size), self.off_periods]
            kwargs.setdefault('title', 'OFF periods')
            fig, ax = plot_frame_series(data=data, **kwargs)
        else:
            raise AttributeError(f'mode {mode} unknown.')

        return fig, ax


def get_blinking_statistics(event_time_series, threshold=0, memory=0, remove_heading_off_period=True):
    df = pd.DataFrame({'frame': np.arange(0, event_time_series.size),
                       'intensity': event_time_series.values})

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


def plot_histogram(data, **kwargs):
    kwargs.setdefault('type_', 'hist')
    kwargs.setdefault('xlabel', 'consecutive frames')
    kwargs.setdefault('ylabel', 'PD')
    kwargs.setdefault('density', True)

    mean = np.mean(data)

    fig, ax = cp.universal_figure(data=data, **kwargs)
    ax[0][0].text(x=0.7, y=0.85, s=f"mean: {mean:.2f}", transform=ax[0][0].transAxes, fontsize=16)

    return fig, ax


def plot_frame_series(data, **kwargs):
    kwargs.setdefault('type_', 'line')
    kwargs.setdefault('xlabel', 'frame number')
    kwargs.setdefault('ylabel', 'consecutive frames')

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax
