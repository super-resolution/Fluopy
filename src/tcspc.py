"""Contains functions related to time-correlated single photon counting."""
import numpy as np
import src.processing as pr
from scipy.stats import erlang
import src.fluorophore_systems as fs


def tcspc_simulation(transitions, n_pulses, seed):
    """
    Simulates a TCSPC experiment, meaning that only the time points of fluorescence and the time points of the laser
    pulses are available data. Restricted to 2 fluorophores. Neglects instrument response function (IRF).

    Parameters
    ----------
    transitions : list
        Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
        (str), abbreviation (str), fluorescence (bool)]. In the case of energy transfers, the first entry is
        k_singlestate1_singlestate2__singlestate1_singlestate2, where the first part represents one fluorophore and
        the second part the other fluorophore.
    n_pulses : int
        Number of laser pulses being simulated.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    times : np.ndarray
        Contains the time differences between laser pulse and fluorescence.
    """
    system = fs.FluorophoreSystem(number_fluorophores=2, distances=1, transitions=transitions)
    fluorescence_id = system.unique_transitions.index[system.unique_transitions['fluorescence'] == True]
    start_id = system.joined_states.index[system.joined_states['name'] == 'S1_S0']
    rng = np.random.default_rng(seed)
    times = []
    for pulse in range(n_pulses):
        system.simulate(n_steps=100000, start_id=start_id, seed=rng)
        # n_steps sufficiently large to ensure the system encounters absorbing state
        transition_cum_sum, transition_sorted_indices = pr.multiple_transitions(system.joined_transitions,
                                                                                system.joined_states,
                                                                                system.unique_transitions)
        transition_series = pr.generate_transition_series(system.state_series, transition_cum_sum,
                                                          transition_sorted_indices, seed=rng)
        if transition_series[-1] == fluorescence_id:
            time = system.time_series[-1]
            times.append(time)

    times = np.array(times)

    return times


def fluorescence_lifetime_distribution_hfret_rvs(fluo_prob, fret_prob, fluo_lifetime, accuracy, size, seed):
    """
    Generates random variates of OBSERVED fluorescence lifetimes in a TCSPC experiment that involves homoFRET (i.e.,
    energy migration FRET, emFRET).

    Parameters
    ----------
    fluo_prob : float
        The probability of fluorescence in one step (e.g., directly after excitation).
    fret_prob : float
        The probability of fret in one step (e.g., directly after excitation).
    fluo_lifetime : float
        The theoretical fluorescence lifetime in presence of an acceptor.
    accuracy : int
        The accuracy of the resulting random variates. The higher, the more accurate.
    size : int
        The number of random variates generated.
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    times : np.ndarray
        Contains random variates of TCSPC-OBSERVED fluorescence lifetimes involving homoFRET.
    """
    rng = np.random.default_rng(seed)

    probabilities = []
    distributions = []
    for i in range(accuracy):
        probability = fluo_prob * fret_prob**i
        probabilities.append(probability)
        a = i + 1
        scale = fluo_lifetime  # the mean of the erlang distribution is a*scale
        distr = erlang(a=a, scale=scale)
        distributions.append(distr)

    probabilities = probabilities / np.sum(probabilities)
    random_numbers = np.random.uniform(0, 1, size)
    cumulative_probabilities = np.cumsum(probabilities)
    times = np.ones(size)
    for i in range(size):
        index = np.searchsorted(cumulative_probabilities, random_numbers[i])
        times[i] = distributions[index].rvs(size=1, random_state=rng)

    return times
