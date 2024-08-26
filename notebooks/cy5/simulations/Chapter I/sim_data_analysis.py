import os
import numpy as np
import pandas as pd
import src.blinking as bl
import src.emissions as em
import src.figure as fi
import src.simulation as si
import src.miscellaneous as mi
import src.formulas as fo
import matplotlib.pyplot as plt
import src.analysis as an
import src.fcs as fcs_package


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mean_without_zeros(subseries):
    nonzero_values = subseries[subseries != 0]
    if nonzero_values.empty: 
        return 0
    return nonzero_values.mean()


def simulate_data(number_of_simulations, memory, threshold, transition_set, 
                  long_sim=False, memmap=False):
    nrows = number_of_simulations*6
    axes = fi.universal_figure(nrows=nrows, ncols=2, fig_height=nrows*5, fig_width=12, scale=0.6)
    rng = np.random.default_rng(10)
    off_periods = []
    off_periods_small_10_num = []
    off_periods_large_10_num = []
    on_periods = []
    on_periods_num = []
    inten_means = []
    inten_num = []
    colors = plt.cm.viridis(range(number_of_simulations))
    axes2 = fi.universal_figure(nrows=6, ncols=2, fig_height=20, fig_width=12, scale=0.6)
    figure2 = mi.get_figure(axes2)
    all_events_summed = pd.Series(np.zeros(300001), np.linspace(0, 300, 300001))

    for i in range(number_of_simulations):
        if long_sim:
            emis = em.Emissions(frame_time='1ms', bandpass=[665, 731], seed=rng)
            fluorescence_lifetimes = emis.simulate(
                transition_set=transition_set, seed=rng, store_time_points=True, 
                frames=300001, return_fl_lifetimes=True)
        else:
            if isinstance(memmap, str):
                memmap_path = memmap
            elif memmap:
                memmap_path = ''
            else: 
                memmap_path = None
            simulation = si.Simulation(transition_set=transition_set)
            simulation.run(size=1e7, end_time=300, seed=rng, use_memmap=memmap_path)
            emis = em.Emissions(frame_time='1ms', bandpass=[665, 731], seed=rng)
            analysis = an.Analysis(simulation)
            fluorescence_lifetimes = analysis.get_fluorescence_lifetimes()
            emis.extract(simulation)

        avrg_lifetimes = moving_average(fluorescence_lifetimes, n=5000)
        deciles = np.array_split(fluorescence_lifetimes, 10)
        decile_means = [np.mean(decile) for decile in deciles]
        photon_collection_rate = fo.calculate_photon_collection_rate(NA=1.45, n1=1.51)
        emis.add_photon_collection_objective(p=photon_collection_rate, seed=rng) 
        emis.add_transmittance(p=0.9, seed=rng)  # mirror 90/100
        emis.add_transmittance(p=0.99, seed=rng) # lens 1
        emis.add_transmittance(p=0.99, seed=rng) # lens 2
        emis.add_quantum_efficiency(p=0.85, seed=rng)
        emis.add_poisson_noise(rate=0.6, seed=rng)
        emis.apply_threshold(threshold=threshold)

        all_events_summed = all_events_summed + emis.event_time_series.values

        emis.plot_time_series(axes=axes[i*6+0, 0])
        emis.plot_histogram(bins=30, axes=axes[i*6+0, 1], display_mean=True)
        axes[i*6+0, 1].text(x=0.8, y=0.9, s=f'# = {emis.event_time_series[emis.event_time_series > 0].size}',
                    transform=axes[i*6+0, 1].transAxes)
        
        blinks = bl.Blinking(emis, memory=memory)
        blinks.plot(mode='off_histogram', axes=axes[i*6+1, 0], as_time='s')
        blinks.plot(mode='off_boxplot', axes=axes[i*6+1, 1], yscale='log', as_time='s')
        axes[i*6+1, 1].text(x=0.8, y=0.9, s=f'# (<10) = {blinks.off_periods[blinks.off_periods < 10].size}',
                            transform=axes[i*6+1, 1].transAxes)
        axes[i*6+1, 1].text(x=0.8, y=0.8, s=f'# (>10) = {blinks.off_periods[blinks.off_periods > 10].size}',
                            transform=axes[i*6+1, 1].transAxes)
        blinks.plot(mode='on_histogram', axes=axes[i*6+2, 0], as_time='ms')
        blinks.plot(mode='on_boxplot', axes=axes[i*6+2, 1], as_time='ms')
        axes[i*6+2, 1].text(x=0.8, y=0.9, s=f'# = {blinks.on_periods.size}',
                            transform=axes[i*6+2, 1].transAxes)
        blinks.plot(mode='on_frame_series', axes=axes[i*6+3, 0], yscale='log')
        blinks.plot(mode='off_frame_series', axes=axes[i*6+3, 1], yscale='log')
        rol_avrg_emissions = moving_average(emis.event_time_series.values, n=5000)
        means = emis.event_time_series.groupby(emis.event_time_series.index // 1.0).apply(mean_without_zeros)
        fi.universal_figure(type_='line', data=[emis.event_time_series.index[4999:], rol_avrg_emissions], axes=axes[i*6+4, 1],
                            title='intensities (rol avg)', xlabel='time + 5s (s)', ylabel='intensity')
        fi.universal_figure(type_='line', data=[np.arange(means.size), means], 
                            axes=axes[i*6+4, 0], title='nonzero intensities (every 1s)', xlabel='time (s)', ylabel='intensity')

        fi.universal_figure(type_='line', data=[np.arange(avrg_lifetimes.size), avrg_lifetimes], axes=axes[i*6+5, 0], 
                            color='blue', title='mean lifetimes', label='rol avrg')
        fi.universal_figure(type_='line', data=[np.linspace(0, avrg_lifetimes.size, 10), decile_means], axes=axes[i*6+5, 0], 
                            color='red', label='decile', legend=True,
                            xlabel='identity', ylabel='lifetime (s)')
        axes[i*6+5, 1].axis('off')

        sec_per_frame = (
            emis.event_time_series.index[1]
            - emis.event_time_series.index[0]
        )
        off_periods.append(blinks.off_periods * sec_per_frame)
        off_periods_small_10 = blinks.off_periods[blinks.off_periods < 10]
        off_periods_small_10_num.append(off_periods_small_10.size)
        off_periods_large_10 = blinks.off_periods[blinks.off_periods > 10]
        off_periods_large_10_num.append(off_periods_large_10.size)
        on_periods.append(blinks.on_periods * sec_per_frame * 1000)
        on_periods_num.append(blinks.on_periods.size)
        inten_means.append(emis.event_time_series[emis.event_time_series > 0].mean())
        inten_num.append(emis.event_time_series[emis.event_time_series > 0].size)

        fi.universal_figure(type_='hist', data=emis.event_time_series[emis.event_time_series > 0], 
                            bins=30, axes=axes2[4, 0], color=colors[i], alpha=0.2,
                            title='intensities', xlabel='intensity', ylabel='count')
        fi.universal_figure(type_='hist', data=emis.event_time_series[emis.event_time_series > 0], 
                            bins=30, axes=axes2[4, 1], color=colors[i], alpha=0.2, 
                            density=True, title='intensities', xlabel='intensity', 
                            ylabel='PD')
    
    rel_all_events = all_events_summed.cumsum() / all_events_summed.sum()

    figure = mi.get_figure(axes)
    figure.tight_layout()

    fi.universal_figure(type_='boxplot', data=off_periods, axes=axes2[0, 0], yscale='log',
                        xlabel='files', ylabel='off period (s)')
    axes2[0, 0].text(s='µ = {:.2e} s'.format(np.mean(np.concatenate(off_periods))), x=0.8, y=0.9,
                    transform=axes2[0, 0].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_small_10_num)), off_periods_small_10_num], 
                        axes=axes2[0, 1], title='off periods < 10 frames', xlabel='files', ylabel='count')
    axes2[0, 1].text(s='µ = {:.2e}'.format(np.mean(off_periods_small_10_num)), x=0.8, y=0.9,
                    transform=axes2[0, 1].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_large_10_num)), off_periods_large_10_num], 
                        axes=axes2[1, 0], title='off periods > 10 frames', xlabel='files', ylabel='count')
    axes2[1, 0].text(s='µ = {:.2e}'.format(np.mean(off_periods_large_10_num)), x=0.8, y=0.9,
                    transform=axes2[1, 0].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    fi.universal_figure(type_='boxplot', data=on_periods, axes=axes2[2, 0], yscale='log',
                        xlabel='files', ylabel='on period (ms)')
    axes2[1, 1].axis('off')
    axes2[2, 0].text(s='µ = {:.2e} ms'.format(np.mean(np.concatenate(on_periods))), x=0.8, y=0.9,
                    transform=axes2[2, 0].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    fi.universal_figure(type_='line', data=[np.arange(0, len(inten_means)), inten_means], axes=axes2[3, 0],
                        title='intensities', xlabel='files', ylabel='mean')
    axes2[3, 0].text(s='µ = {:.2e}'.format(np.mean(inten_means)), x=0.8, y=0.9,
                    transform=axes2[3, 0].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    axes2[3, 0].set_ylim(0, 100)
    fi.universal_figure(type_='line', data=[np.arange(0, len(inten_num)), inten_num], axes=axes2[3, 1],
                        title='intensities', xlabel='files', ylabel='count')
    axes2[3, 1].text(s='µ = {:.2e}'.format(np.mean(inten_num)), x=0.8, y=0.9,
                    transform=axes2[3, 1].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_small_10_num)), off_periods_small_10_num], 
                        axes=axes2[0,1],
                        title='off periods < 10 frames', xlabel='files', ylabel='count')
    fi.universal_figure(type_='line', data=[np.arange(0, len(off_periods_large_10_num)), off_periods_large_10_num],
                        axes=axes2[1, 0],
                        title='off periods > 10 frames', xlabel='files', ylabel='count')
    fi.universal_figure(type_='line', data=[np.arange(0, len(on_periods_num)), on_periods_num], 
                        axes=axes2[2,1],
                        title='on periods', xlabel='files', ylabel='count')
    axes2[2, 1].text(s='µ = {:.2e}'.format(np.mean(on_periods_num)), x=0.8, y=0.9,
                    transform=axes2[2, 1].transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    
    fi.universal_figure(data=[all_events_summed.index, rel_all_events], axes=axes2[5, 0], 
                        title='cumulative sum of events', ylabel='% emission', xlabel='time (s)')
    axes2[5, 1].axis('off')
    figure2.tight_layout()
