"""
Module tcspc
Assumes an infinitesimally short laser pulse that assures only one fluorophore to be excited per pulse. Another pulse
starts only if all fluorophores are in their ground state.
This simulation is suited to demonstrate observed fluorescence lifetimes of homo-FRET experiments.
This simulation is NOT suited to demonstrate effects of energy transfers that are not homo-FRET.
"""
import numpy as np
import src.simulation as si
import src.figure as fi
from src.transitions import SingleState, PairedState
from scipy.stats import erlang


class TCSPC:
    """
    Container of TCSPC-associated attributes and methods.

    Attributes
    ----------
    transitions : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    modified_transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each possible combined_state_transition
        at the corresponding index pair. Modified (see modify_transition_matrix()).
    modified_row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
        combined_state_transitions. Modified (see modify_transition_matrix()).
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to transition_series[i - 1].
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    observed_lifetimes : 1-D array_like
        Simulated times between photon-driven excitation and fluorescent emission not considering whether the emission
        comes from the originally excited fluorophore.
    predicted_obs_lifetimes : 1-D array_like
        Computed times between photon-driven excitation and fluorescent emission not considering whether the emission
        comes from the originally excited fluorophore.
    true_fluorescence_lifetime : float
        The true (not observed) fluorescence lifetime extracted from simulated data.
    predicted_true_fluorescence_lifetime : float
        The computed true (not observed) fluorescence lifetime.
    """
    def __init__(self, transitions):
        """
        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        """
        if transitions.transition_matrix is None:
            raise ValueError('tcspc not available if transitions not finalized.')
        self.transitions = transitions
        self.modified_transition_matrix, self.modified_row_sums = self.modify_transition_matrix()
        self.time_series = None
        self.transition_series = None
        self.observed_lifetimes = None
        self.predicted_obs_lifetimes = None
        self.true_fluorescence_lifetime = None
        self.predicted_true_fluorescence_lifetime = None

    def modify_transition_matrix(self):
        """
        Modifies the transition matrix in a way that prevents multiple fluorophores to be in a different photophysical
        state than the ground state simultaneously. This is achieved by setting all excitation rates that do not effect
        an all ground state fluorophore configuration to zero.

        Returns
        -------
        modified_transition_matrix : np.ndarray
            Contains the normalized rate constants (i.e., point probabilities) for each possible
            combined_state_transition at the corresponding index pair. Modified (see modify_transition_matrix()).
        modified_row_sums : np.ndarray
            Contains the sum of each row of non-normalized transition rates, i.e., the sum of rates of all possible
            combined_state_transitions. Modified (see modify_transition_matrix()).
        """
        df = self.transitions.combined_state_transitions_df
        excitations = df[df['abbreviation'] == 'EXC']
        indices_to_modify = excitations.index.values[self.transitions.fluorophore_system.count:]
        transition_rate_matrix = self.transitions.transition_matrix * np.expand_dims(self.transitions.row_sums, axis=1)
        transition_rate_matrix[:, indices_to_modify] = 0
        modified_row_sums = transition_rate_matrix.sum(axis=1)
        modified_transition_matrix = np.divide(transition_rate_matrix, np.expand_dims(modified_row_sums, axis=1),
                                               out=np.zeros_like(transition_rate_matrix), where=modified_row_sums != 0)
        return modified_transition_matrix, modified_row_sums

    def run(self, start_at=None, size=100, seed=None):
        """
        Runs a simulation based on the direct method of the gillespie algorithm (i.e., stochastic simulation algorithm).

        Parameters
        ----------
        start_at : None, tuple
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible SingleState values. See
            src.transitions.TransitionSet.single_states.
        size : int
            Maximum number of simulation steps.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        self
        """
        if start_at is None:
            start_at = tuple(np.zeros(shape=self.transitions.fluorophore_system.count, dtype=int))
        df = self.transitions.combined_state_transitions_df
        start_index = df[df['final_state'] == start_at].index[0]
        self.time_series, self.transition_series = \
            si.direct_method_steps(transition_matrix=self.modified_transition_matrix,
                                   row_sums=self.modified_row_sums,
                                   start_index=start_index, size=size, seed=seed)

        return self

    def get_observed_lifetimes(self):
        """
        Get times between photon-driven excitation and fluorescent emission not considering whether the emission comes
        from the originally excited fluorophore.

        Returns
        -------
        None
        """
        if self.transition_series is None:
            raise ValueError('get_observed_lifetimes() not available if run() has not been called.')
        df = self.transitions.combined_state_transitions_df
        excitation_values = df[df['abbreviation'] == 'EXC'].index.values
        fluorescence_values = df[df['abbreviation'] == 'FLU'].index.values
        excitation_indices = np.in1d(self.transition_series, excitation_values).nonzero()[0]
        emission_indices = np.in1d(self.transition_series, fluorescence_values).nonzero()[0]
        excitation_times = self.time_series[excitation_indices + 1]
        emission_times = self.time_series[emission_indices + 1]
        pre_emission_times = self.time_series[emission_indices]
        self.true_fluorescence_lifetime = np.mean(emission_times - pre_emission_times)
        corresponding_excitation_time_indices = np.searchsorted(excitation_times, emission_times, side='right') - 1
        corresponding_excitation_times = excitation_times[corresponding_excitation_time_indices]

        self.observed_lifetimes = emission_times - corresponding_excitation_times

    def plot(self, mode='simulation', **kwargs):
        """
        Plot histogram of simulated or predicted observed lifetimes.

        Parameters
        ----------
        mode : str
            One of 'simulation', 'prediction'.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        kwargs.setdefault('type_', 'hist')
        kwargs.setdefault('ylabel', 'PD')
        kwargs.setdefault('density', True)
        kwargs.setdefault('xlabel', 'observed lifetime [s]')
        if mode == 'simulation':
            data = self.observed_lifetimes
            kwargs.setdefault('title', 'simulation')
        elif mode == 'prediction':
            data = self.predicted_obs_lifetimes
            kwargs.setdefault('title', 'prediction')
        else:
            raise AttributeError('mode has to be one of "simulation" or "prediction"')
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def predict(self, accuracy=10, size=10, seed=None):
        """
        Compute times between photon-driven excitation and fluorescent emission not considering whether the emission
        comes from the originally excited fluorophore.

        Parameters
        ----------
        accuracy : int
            The accuracy of the prediction. Higher energy transfer rates need higher accuracy to accomplish the same
            quality of prediction.
        size : int
            The number of random variates generated.
        seed : None, int, BitGenerator, Generator
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        self.predicted_true_fluorescence_lifetime, hfret_probability, fluorescence_probability = \
            get_transition_probabilities(self.transitions.transition_df, self.transitions.fluorophore_system)
        rng = np.random.default_rng(seed)
        probabilities = []
        distributions = []
        for i in range(accuracy):
            probability = fluorescence_probability * hfret_probability**i  # x**0 = 1
            probabilities.append(probability)
            distribution = erlang(a=i+1, scale=self.predicted_true_fluorescence_lifetime)
            distributions.append(distribution)
        weights = probabilities / np.sum(probabilities)  # otherwise, probabilites do not add up to 1
        random_numbers = np.random.uniform(0, 1, size)
        cumulative_probabilities = np.cumsum(weights)
        self.predicted_obs_lifetimes = np.ones(size)
        for i in range(size):
            index = np.searchsorted(cumulative_probabilities, random_numbers[i])
            self.predicted_obs_lifetimes[i] = distributions[index].rvs(size=1, random_state=rng)


def get_transition_probabilities(transition_df, fluorophore_system):
    """
    Gets the probabilities of homoFRET and fluorescent deexcitation and the fluorescence lifetime.
    Note: the values returned reflect a setup where only one fluorophore is excited at a time. This means that if a
    fluorophore is in S1, all other fluorophores are, at all times, in S0. Hence, if the rate of homoFRET is non-zero,
    it has to be considered constantly and alters the fluorescence lifetime and probability of fluorescence.
    Additionally, if more than 2 fluorophores are present, a total FRET rate is calculated depending on the distances.

    Parameters
    ----------
    transition_df : pd.DataFrame
        Dataframe of transitions containing their id as index and their other attributes as columns
        (see src.transitions.Transition).
    fluorophore_system : src.fluorophores.FluorophoreSystem
        Container for attributes of multiple, interrelated fluorophores.

    Returns
    -------
    fluorescence_lifetime : float
        The fluorescence lifetime (may differ from the observed lifetime, see note above).
    hfret_probability : float
        The probability of homoFRET (see note above).
    fluorescence_probability : float
        The probability of fluorescent deexcitation (see note above).
    """
    distances, occurrences = np.unique(list(fluorophore_system.distances.values()), return_counts=True)
    occurrences = occurrences / fluorophore_system.count

    df = transition_df
    fluorescence_rate = df[df['abbreviation'] == 'FLU']['rate'].values[0]
    non_et_s1_rates = df[df['initial_state'] == SingleState.S1]['rate']
    et_s1_s0_transitions = df[df['initial_state'] == PairedState.S1_S0]
    indices = np.where(et_s1_s0_transitions['distance'].values == distances)[0]
    et_s1_s0_rate_sum = (et_s1_s0_transitions['rate'] * occurrences[indices]).sum()
    fluorescence_lifetime = 1 / (non_et_s1_rates.sum() + et_s1_s0_rate_sum)
    hfret_probability = et_s1_s0_rate_sum * fluorescence_lifetime
    fluorescence_probability = fluorescence_rate * fluorescence_lifetime

    return fluorescence_lifetime, hfret_probability, fluorescence_probability
