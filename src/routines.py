"""
Module routines
"""

import numpy as np
import pandas as pd
import src.emissions as em
import src.simulation as si
import src.formulas as fo
import src.fitting as fit


def get_bleaching_times(simulation):
    """
    Get the times where photobleaching occurred - for each fluorophore, one number will
    be extracted. If no bleaching occurred, the entry will be np.nan. The elements will
    be sorted, np.nan will be at the end.

    Parameters
    ----------
    simulation : src.simulation.Simulation
        Container for simulation-associated attributes

    Returns
    -------
    bleaching_times : 1-D array_like
        Times where photobleaching occurred.
    """
    df = simulation.transition_set.transition_df
    bleached_states = df[df['absorbing'] == True]['final_state']
    bleached_states = [x.value for x in bleached_states]
    if len(bleached_states) == 1:
        bleached_state = bleached_states[0]
    elif len(bleached_states) == 0:
        return np.full(simulation.state_series.shape[0], np.nan)
    else:
        raise NotImplementedError("Multiple bleaching states not yet implemented in " +
                                  "this function.")

    bleaching_times = []
    for state_series in simulation.state_series:
        if state_series[-1] == bleached_state:
            first_occurence = np.where(state_series == bleached_state)[0][0]
            time = simulation.time_series[first_occurence]
        else:
            time = np.nan
        bleaching_times.append(time)
    bleaching_times = np.sort(np.array(bleaching_times))
    
    return bleaching_times


def get_global_bleaching_rates(bleaching_times):
    """
    Get the global bleaching rates for each fluorophore. The global bleaching rate is 
    the inverse of the lifetime of a fluorophore starting from the last bleaching event. 

    Parameters
    ----------
    bleaching_times : 2-D array_like
        Times where photobleaching occurred. Each run is a row, each fluorophore a 
        column). Each row is sorted, np.nan will be at the end.

    Returns
    -------
    global_bleaching_rates : 2-D array_like
        The two global bleaching rates and the mixing factor for each fluorophore.
    """
    global_bleaching_rates = []
    delta_bleaching_times_all = []
    previous_times = np.zeros_like(bleaching_times.shape[0])
    for fluorophore in range(bleaching_times.shape[1]):
        bleaching_times_fluo = bleaching_times[:, fluorophore]
        delta_bleaching_times = bleaching_times_fluo - previous_times
        delta_bleaching_times = delta_bleaching_times[~np.isnan(delta_bleaching_times)]
        delta_bleaching_times_all.append(delta_bleaching_times)
        previous_times = bleaching_times_fluo
        p1, lambda_1, lambda_2 = fit.estimate_mixture_parameters(
            data=delta_bleaching_times,
            initial_guess=[0.1, 0.01, 0.5],
            bounds=[(0, 1), (0, None), (0, None)],
            truncation_low=0,
            truncation_up=300,
        )
        model_parameters = np.array([lambda_1, lambda_2, p1])
        global_bleaching_rates.append(model_parameters)
    global_bleaching_rates = np.array(global_bleaching_rates)

    return global_bleaching_rates, delta_bleaching_times_all


def fingerprint_analysis(
    transition_set,
    batch_size, 
    batches,
    filepath,
    filename,
    seed,
    use_memmap=None,
    ): 
    """
    Routine to perform fingerprint analysis. Returns the fingerprint data and the times
    where photobleaching occurred. Each run is stored as a parquet file. The bleaching 
    times are stored as a numpy file. 
    
    Parameters
    ----------
    transition_set : src.transition.TransitionSet
        Collection of all relevant transitions and related attributes.
    batch_size : int
        Size of each batch.
    batches : int
        Number of batches.
    filepath : str
        Path to save the fingerprint data.
    filename : str
        The name of the file. In the case of single_run data, the name is extended with
        the batch number.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.
    use_memmap : None, str
        If None, the data will be stored in memory. If a string, the data will be stored
        in a memmap file. Default is None.
    
    Returns
    -------
    fingerprint_data : 1-D array_like
        Fingerprint data - normalized cumulative emissions.
    bleaching_times : 2-D array_like
        Times where photobleaching occurred. Each run is a row, each fluorophore a 
        column). Each row is sorted, np.nan will be at the end.
    """
    rng = np.random.default_rng(seed)
    fingerprint_data = pd.Series(np.zeros(300001), 
                                 np.round(np.linspace(0, 300, 300001), decimals=12), 
                                 dtype=np.int32)
    output_file_bleach = fr"{filepath}\bleaching_times_{filename}.npy"
    bleaching_times_all_runs = []
    delta_times_photons_between_bleaching = [[] for _ in range(transition_set.fluorophore_system.count)]
    for i in range(batches):
        output_file_run = fr"{filepath}\single_runs_{filename}_batch_{i}.parquet"
        df = None
        for j in range(batch_size):
            simulation = si.Simulation(transition_set)
            simulation.run(size=1e6, seed=rng, end_time=300, use_memmap=use_memmap)
            bleaching_times = get_bleaching_times(simulation)
            bleaching_times_all_runs.append(bleaching_times)
            emis = em.Emissions(frame_time='1ms', bandpass=[665, 731], seed=rng)
            emis.extract(simulation)


            for n in range(transition_set.fluorophore_system.count):
                if n > 0:
                    start = bleaching_times[n-1]
                else:
                    start = 0
                start_index = np.searchsorted(emis.event_time_points, start)
                if bleaching_times.size > n:
                    end_index = np.searchsorted(emis.event_time_points, bleaching_times[n])
                    delta_times_photons_between_bleaching[n].append(emis.event_time_points[start_index:end_index] - start)
                else:
                    delta_times_photons_between_bleaching[n].append(emis.event_time_points[start_index:] - start)  # the delta, not the actual times
                    break
                    

            photon_collection_rate = fo.calculate_photon_collection_rate(
                NA=1.45, n1=1.51
            )
            emis.add_photon_collection_objective(p=photon_collection_rate, seed=rng) 
            emis.add_transmittance(p=0.9, seed=rng)  # mirror 90/100
            emis.add_transmittance(p=0.99, seed=rng) # lens 1
            emis.add_transmittance(p=0.99, seed=rng) # lens 2
            emis.add_quantum_efficiency(p=0.85, seed=rng)
            emis.add_poisson_noise(rate=0.6, seed=rng)
            emis.apply_threshold(threshold=10)
            emis.event_time_series.name = i*batch_size + j
            if df is None:
                df = emis.event_time_series
            else:
                df = pd.concat([df, emis.event_time_series], axis=1, ignore_index=False)
            fingerprint_data = fingerprint_data + emis.event_time_series
        df.to_parquet(output_file_run)
    bleaching_times_all_runs = np.array(bleaching_times_all_runs)
    np.save(output_file_bleach, bleaching_times_all_runs)
    fingerprint_data = fingerprint_data.cumsum() / fingerprint_data.sum()

    return fingerprint_data, bleaching_times_all_runs, delta_times_photons_between_bleaching    
    