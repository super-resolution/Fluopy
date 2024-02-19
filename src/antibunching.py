"""
Module antibunching
Basically the same as autocorrelation but methodically closer to the Hanbury Brown and Twiss
experiment.
"""
import numpy as np
import pycorrelate as pc
import src.figure as fi
import pandas as pd


def hanbury_brown_twiss(simulation, seed, exp_min, exp_max, points_per_base, base, normalize):
    """
    Creates two photon detection channels and correlates them.
        
    Parameters
    ----------
    simulation : src.simulation.Simulation
        Container of simulation-associated attributes and methods.
    seed : None, int, BitGenerator, Generator
        A seed to initialize the BitGenerator.
    exp_min : int
        Exponent of the minimum value.
    exp_max : int
        Exponent of the maximum value.
    points_per_base : int
        Number of points per base.
    base : int
        The base of the exponentiation.
    normalize : bool
        Whether to normalize the autocorrelation.

    Returns
    -------
    correlation : 1-D array_like
        Correlation values
    tau : 1-D array_like
        Time differences
    """
    rng = np.random.default_rng(seed)
    emission_indices = get_emission_indices()
    rng.shuffle(emission_indices)
    
    first_channel = np.sort(simulation.time_series[emission_indices[:int(emission_indices.size/2)]])
    second_channel = np.sort(simulation.time_series[emission_indices[int(emission_indices.size/2):]])

    if base ** exp_max > first_channel[-1]:
        raise ValueError('Base to the power of exp_max cannot be larger than the last time point.')
    bins = pc.make_loglags(exp_min=exp_min, exp_max=exp_max, points_per_base=points_per_base, base=base)
    correlation = pc.pcorrelate(t=first_channel, u=second_channel, bins=bins, normalize=normalize)
    tau = np.mean([bins[1:], bins[:-1]], 0)

    return correlation, tau


def plot(tau, correlation, normalize_to=None, unit='s', **kwargs):
    """
    Plot correlation data.

    Parameters
    ----------
    tau : 1-D array_like
        Time differences
    correlation : 1-D array_like
        Correlation values
    normalize_to : None, int
        Index of datapoint to which the data is normalized.
    unit : str
        One of 's', 'ms', 'us'. Influences the unit of the x-axis.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    tau_data, correl_data = np.copy(tau), np.copy(correlation)
    if normalize_to is not None:
        correl_data /= correl_data[normalize_to]

    adjust_unit = pd.to_timedelta(1, unit=unit).total_seconds()
    tau_data = tau_data / adjust_unit
    kwargs.setdefault('title', rf"$\tau_{{min}} = {tau_data[0]:.2e}$ {unit}")
    kwargs.setdefault('type_', "line")
    kwargs.setdefault('xscale', "log")
    kwargs.setdefault('xlabel', fr"$\tau \ [{unit}]$")
    kwargs.setdefault('ylabel', r"$G(\tau)$")

    axes = fi.universal_figure(data=[tau_data, correl_data], **kwargs)

    return axes


def get_emission_indices(simulation):
    """
    Get indices to apply to simulation.transition_series to yield emitting transitions.

    Parameters
    ----------
    simulation : src.simulation.Simulation
        Container of simulation-associated attributes and methods.

    Returns
    -------
    emission_indices : 1-D array_like
        Indices of emitting transitions to apply to simulation.transition_series.
    """
    df = simulation.transition_set.combined_state_transitions_df
    emitting_transition_ids = df.loc[df['photon'] == True].index.to_numpy()
    emission_indices = np.in1d(simulation.transition_series, emitting_transition_ids).nonzero()[0]

    return emission_indices
