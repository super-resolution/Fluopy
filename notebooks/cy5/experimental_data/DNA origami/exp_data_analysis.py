import os
import numpy as np
import pandas as pd
import src.blinking as bl
import src.emissions as em
import src.figure as fi
import src.miscellaneous as mi
import matplotlib.pyplot as plt


def get_files_in_path(directory):
    files = []
    # Walk through the directory
    for root, _, filenames in os.walk(directory):
        # Append full path of each file to the list
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def go_through_files(path, memory, threshold, skip_id=None):
    if skip_id is None:
        skip_id = []
    files = get_files_in_path(path)
    nrows = (len(files)-len(skip_id))*4
    axes = fi.universal_figure(nrows=nrows, ncols=2, fig_height=nrows*4, fig_width=12, scale=0.6)
    figure = mi.get_figure(axes)

    i=0
    for j, filepath in enumerate(files):
        if j in skip_id:
            continue
        data = pd.read_csv(filepath, sep='\t', header=1)
        if 'Intens.[kCnts]' in data.columns:
            data.rename(columns={'Intens.[kCnts]': 'Intens.[Cnts]'}, inplace=True)
            data['Intens.[Cnts]'] = data['Intens.[Cnts]'] * 1000
        data['Intens.[Cnts]'] = data['Intens.[Cnts]'].apply(lambda x: 0 if x < threshold else x)
        emissions = em.Emissions()
        emissions.event_time_series = pd.Series(data['Intens.[Cnts]'].values, data['Time[s]'])
        blinks = bl.Blinking(emissions, memory=memory)
        emissions.plot_time_series(axes=axes[i*4+0, 0])
        axes[i*4+0, 0].text(x=-0.2, y=1.1, s=f'index = {j}', 
                            transform=axes[i*4+0, 0].transAxes)
        emissions.plot_histogram(bins=30, axes=axes[i*4+0, 1], display_mean=True)
        axes[i*4+0, 1].text(x=0.8, y=0.9, s=f'# = {emissions.event_time_series[emissions.event_time_series > 0].size}',
                    transform=axes[i*4+0, 1].transAxes)
        blinks.plot(mode='off_histogram', axes=axes[i*4+1, 0], as_time='s')
        blinks.plot(mode='off_boxplot', axes=axes[i*4+1, 1], yscale='log', as_time='s')
        axes[i*4+1, 1].text(x=0.8, y=0.9, s=f'# (<10) = {blinks.off_periods[blinks.off_periods < 10].size}',
                            transform=axes[i*4+1, 1].transAxes)
        axes[i*4+1, 1].text(x=0.8, y=0.8, s=f'# (>10) = {blinks.off_periods[blinks.off_periods > 10].size}',
                            transform=axes[i*4+1, 1].transAxes)
        blinks.plot(mode='on_histogram', axes=axes[i*4+2, 0], as_time='ms')
        blinks.plot(mode='on_boxplot', axes=axes[i*4+2, 1], as_time='ms')
        axes[i*4+2, 1].text(x=0.8, y=0.9, s=f'# = {blinks.on_periods.size}',
                            transform=axes[i*4+2, 1].transAxes)
        axes[i*4+3, 0].axis('off')
        axes[i*4+3, 1].axis('off')
        i+= 1
    
    figure.tight_layout()


def sum_all_files(path, memory, threshold, skip_id=None):
    if skip_id is None:
        skip_id = []
    files = get_files_in_path(path)
    off_periods = []
    off_periods_small_10_num = []
    off_periods_large_10_num = []
    on_periods = []
    on_periods_num = []
    inten_means = []
    inten_num = []
    colors = plt.cm.viridis(range(len(files)))
    axes = fi.universal_figure(nrows=5, ncols=2, fig_height=16, fig_width=12, scale=0.6)
    figure = mi.get_figure(axes)

    for j, filepath in enumerate(files):
        if j in skip_id:
            continue
        data = pd.read_csv(filepath, sep='\t', header=1)
        if 'Intens.[kCnts]' in data.columns:
            data.rename(columns={'Intens.[kCnts]': 'Intens.[Cnts]'}, inplace=True)
            data['Intens.[Cnts]'] = data['Intens.[Cnts]'] * 1000
        data['Intens.[Cnts]'] = data['Intens.[Cnts]'].apply(lambda x: 0 if x < threshold else x)
        emissions = em.Emissions()
        emissions.event_time_series = pd.Series(data['Intens.[Cnts]'].values, data['Time[s]'])
        sec_per_frame = (
            emissions.event_time_series.index[1]
            - emissions.event_time_series.index[0]
        )
        blinks = bl.Blinking(emissions, memory=memory)
        off_periods.append(blinks.off_periods * sec_per_frame)
        off_periods_small_10 = blinks.off_periods[blinks.off_periods < 10]
        off_periods_large_10 = blinks.off_periods[blinks.off_periods > 10]
        off_periods_small_10_num.append(off_periods_small_10.size)
        off_periods_large_10_num.append(off_periods_large_10.size)
        on_periods.append(blinks.on_periods * sec_per_frame * 1000)
        on_periods_num.append(blinks.on_periods.size)
        inten_means.append(emissions.event_time_series[emissions.event_time_series > 0].mean())
        inten_num.append(emissions.event_time_series[emissions.event_time_series > 0].size)
        fi.universal_figure(type_='hist', data=emissions.event_time_series[emissions.event_time_series > 0], 
                            bins=30, axes=axes[4, 0], color=colors[j], alpha=0.2,
                            title='intensities', xlabel='intensity', ylabel='count')
        fi.universal_figure(type_='hist', data=emissions.event_time_series[emissions.event_time_series > 0], 
                            bins=30, axes=axes[4, 1], color=colors[j], alpha=0.2, 
                            density=True, title='intensities', xlabel='intensity', 
                            ylabel='PD')
    fi.universal_figure(type_='boxplot', data=off_periods, axes=axes[0, 0], yscale='log',
                        xlabel='files', ylabel='off period (s)')
    fi.universal_figure(type_='boxplot', data=on_periods, axes=axes[2, 0], yscale='log',
                        xlabel='files', ylabel='on period (ms)')
    fi.universal_figure(type_='line', data=[np.arange(0, len(inten_means)), inten_means], axes=axes[3, 0],
                        title='intensities', xlabel='files', ylabel='mean')
    axes[3, 0].set_ylim(0, )
    fi.universal_figure(type_='line', data=[np.arange(0, len(inten_num)), inten_num], axes=axes[3, 1],
                        title='intensities', xlabel='files', ylabel='count')
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_small_10_num)), off_periods_small_10_num], 
                        axes=axes[0,1],
                        title='off periods < 10s', xlabel='files', ylabel='count')
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_large_10_num)), off_periods_large_10_num],
                        axes=axes[1, 0],
                        title='off periods > 10s', xlabel='files', ylabel='count')
    fi.universal_figure(type_='line', data=[np.arange(0, len(on_periods_num)), on_periods_num], 
                        axes=axes[2,1],
                        title='on periods', xlabel='files', ylabel='count')
    figure.tight_layout()
    