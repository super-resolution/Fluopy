# import os
# import sys
# import inspect
#
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)


import numpy as np
import pandas as pd
import src.fcs as fcs
import src.processing as pr
import src.initialize as init
import src.gillespie_algorithm as ga
import src.emitting_transitions as et


class FluorophoreSystem:
    """
    Base class of fluorophore systems and their serial quantum state behavior based on continuous time Markov chain
    (CTMC) modelling.

    Attributes
    ----------
    - Defined during instantiation of class object -
    number_fluorophores : int
        Number of fluorophores of the system.
    distances : float, Collection
        Distances of the fluorophores to each other.
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        state.
    single_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str) and fluorescence (bool) of each transition, where their
        id is the index.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), single_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of all transition rates (rate constants) of each state.
    graph : nx.DiGraph
        Contains nodes and edges of the Markov chain.

    - Defined during method simulate() call -
    time_series : None, np.ndarray
        The time points at which the corresponding state occurs.
    time_step_series : None, np.ndarray
        The time step until the corresponding state occurs (starting from the previous state).
    state_series : None, np.ndarray
        The consecutive state's unique values.

    - Defined during method process() call -
    single_state_series : None, np.ndarray
        State series for each individual fluorophore (hence, single_states).
    transition_series : None, list
        Contains the next transition id for each corresponding state (except the last).
    single_state_lifetimes : None, dict
        Keys are single_state_occurrences, single_state_occurrences_all, lifetimes_fluorophore, lifetimes_single_states,
        lifetimes_single_states_all, mean_lifetimes, mean_lifetimes_all, total_lifetimes, total_lifetimes_all.
    transition_lifetimes : None, dict
        Keys are transition_occurrences, transition_occurrences_all, transition_times, transition_times_all,
        mean_transition_times, mean_transition_times_all.
    """
    def __init__(self, number_fluorophores, distances, single_states, rates):
        """
        Parameters
        ----------
        number_fluorophores : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        single_states : iterable object
            Contains elements of type str describing each state a single fluorophore can occupy.
        rates : list
            The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
            the value [k, name_of_transition] assigned to it.
        """
        self.number_fluorophores = number_fluorophores
        self.distances = distances

        self.single_states = {i: state for i, state in enumerate(single_states)}
        ###############################################################################################################
        self.single_transitions = pd.DataFrame(rates,  columns=['name', 'rate', 'trivial_name', 'fluorescence'])
        self.single_transitions.index.name = 'id'
        ###############################################################################################################
        joined_states = init.state_pairs(self.number_fluorophores, single_states)
        self.joined_states = pd.DataFrame.from_dict(joined_states, orient='index', columns=['', '']).reset_index()
        self.joined_states.columns = ['name', 'single_states', 'single_state_counter']
        self.joined_states.index.name = 'id'
        ###############################################################################################################
        joined_transitions = init.transition_pairs(self.joined_states)
        transition_rate_list = init.construct_transition_rate_list(self.single_transitions, joined_transitions)
        self.joined_transitions = pd.DataFrame(transition_rate_list, columns=['name', 'joined_states_id',
                                                                              'single_transition_id', 'rate',
                                                                              'trivial_name', 'fluorescence'])
        self.joined_transitions.index.name = 'id'
        ###############################################################################################################
        self.transition_matrix, self.row_sums = init.construct_transition_matrices(self.joined_transitions,
                                                                                   self.joined_states)
        self.joined_states = init.add_absorbing_states(self.joined_states, self.joined_transitions)
        self.graph = init.construct_network(self.single_transitions)

        self.time_series = None
        self.time_step_series = None
        self.state_series = None

        self.single_state_series = None
        self.single_state_lifetimes = None
        self.transition_lifetimes = None
        self.transition_series = None

    def simulate(self, n_steps=100, seed=100):
        """
        Simulates the CTMC using the direct method of the Gillespie algorithm.

        Parameters
        ----------
        n_steps : int
            Maximum number of simulation steps. If the Markov chain reaches an absorbing state, the simulation stops
            early.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        self.time_series, self.time_step_series, self.state_series = ga.direct_method(self.transition_matrix,
                                                                                      self.row_sums, n_steps, seed)

    def process(self, seed):
        """
        Collection of processing functions identifying unique states and occupation times.

        Parameters
        ----------
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """

        self.single_state_series = pr.convert_single_state_series(self.number_fluorophores, self.state_series,
                                                                  self.joined_states)
        transition_cum_sum, transition_sorted_indices = pr.multiple_transitions(self.joined_transitions,
                                                                                self.joined_states,
                                                                                self.single_transitions)
        self.transition_series = pr.generate_transition_series(self.state_series, transition_cum_sum,
                                                               transition_sorted_indices, seed)
        self.single_state_lifetimes, self.transition_lifetimes = \
            pr.time_occurrence_statistics(self.number_fluorophores, self.single_states, self.single_transitions,
                                          self.time_series, self.transition_series, self.single_state_series)


class GeneralModel(FluorophoreSystem):
    """
    Derived class from FluorophoreSystem. States follow the classic Jablonski model.

    Attributes
    ----------
    - Defined during method emitters() call -
    emitting_transitions : None, list
        Contains emitting transitions.
    emitting_transitions_indices : None, list
        Contains emitting transition indices.
    emitting_mask : None, np.ndarray
        Boolean array of length state_series, True at i if state_series[i] is a state following an emitting transition.
    detected_emission_mask : None, np.ndarray
        Copy of emitting_mask but some True entries may have changed to False dictated by parameter photon_collection.
    events_at : np.ndarray
        Contains the time points at which an event occurs.
    last_parameters : dict
        Contains the last parameters used to construct pandas_series.
    pandas_series : None, pd.Series
        Contains the time step (resample) in seconds as index and the number of events as values.
    on_periods : None, np.ndarray
        Contains the lengths of each on-period.
    off_periods : None, np.ndarray
        Contains the lengths of each off-period.
    statistics : None, dict
        Contains several statistical parameters associated with emission events.

    - Defined during method fcs() call -
    autocorrelation : None, tuple
        Contains two arrays, the first is the time differences that correspond to the autocorrelation values,
        the second is the autocorrelation values.
    """
    def __init__(self, number_fluorophores, distances, rates):
        """
        Parameters
        ----------
        number_fluorophores : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        rates : dict
            The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
            the value [k, name_of_transition] assigned to it.
        """
        single_states = ("S0", "S1", "T1", "B")
        super().__init__(number_fluorophores, distances, single_states, rates)

        self.emissions = None
        self.detected_emissions = None
        self.time_points_events = None
        self.last_parameters = None
        self.event_time_series = None
        self.on_periods = None
        self.off_periods = None
        self.emission_statistics = None

        self.autocorrelation = None

    def emitters(self, photon_collection_rate=1, resample="5ms", emccd_gain=None, threshold=0, memory=0,
                 remove_heading_off_period=False, seed=100):
        """
        Collection of functions identifying emitting transitions and other related properties.

        Parameters
        ----------
        photon_collection_rate : float
            Number between 0 and 1, dictates the fraction of kept True values of emitting_mask.
        resample : str
            Resamples events. See pandas time series user's guide offset aliases for possible input values.
        emccd_gain : int
            The gain of an emccd.
        threshold : int
            Maximum value of photons per frame (i.e., resample) to be considered an off-frame.
        memory : int
            Number of off-frames to be neglected. They are included in the on times.
        remove_heading_off_period : bool
            If True and the series starts wit an off-frame, the leading off-frame is discarded.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        self.emissions = et.get_emissions(self.single_transitions, self.transition_series)
        self.detected_emissions = et.get_detected_emissions(self.emissions, photon_collection_rate, seed)
        self.time_points_events = self.time_series[self.detected_emissions]
        self.event_time_series = et.construct_event_time_series(self.time_points_events, resample, emccd_gain, seed)
        self.last_parameters = dict(resample=resample, emccd_gain=emccd_gain, seed=seed)
        self.on_periods, self.off_periods, _, _ = et.blink_statistics(self.event_time_series, threshold, memory,
                                                                      remove_heading_off_period)
        self.emission_statistics = {'total_events': self.emissions.size,
                                    'total_detected_events': self.event_time_series.sum(),
                                    'mean_time_between_events': np.mean(np.diff(self.time_points_events))}

    def fcs(self, normalize=True, log=True, m=2, deltat=5e-3):
        """
        Autocorrelation as exploited in typical fluorescence correlation spectroscopy (FCS) setups.

        Parameters
        ----------
        normalize : bool
            Whether to normalize the correlation.
        log : bool
            Whether to compute the autocorrelation on a logarithmic scale.
        m : int
            Defines the number of points on one level (i.e., 1, 2, 4, 8, etc.),
            E.g., m=4 leads to 1,2,3,4; 2,4,6,8; 4,8,12,16; ... .
        deltat : float
            The time difference between each entry of pandas_series in seconds. If the required parameter does not
            match the current one, a new pandas_series is constructed (but it is not stored as an attribute).

        Returns
        -------
        autocorrelation : tuple
            Contains a np.ndarray of time differences and a np.ndarray of autocorrelation values.
        """
        if self.event_time_series is None:
            raise TypeError('pandas_series is None')
        else:
            current_deltat = self.last_parameters["resample"]
            current_deltat = pd.to_timedelta(current_deltat)
            required_deltat = pd.to_timedelta(deltat, unit="s")
            if required_deltat != current_deltat:
                new_pandas_series = et.pandas_event_time_series(self.time_points_events, resample=required_deltat,
                                                                emccd_gain=self.last_parameters["emccd_gain"],
                                                                seed=self.last_parameters["seed"])
                use_pandas_series = new_pandas_series
            else:
                use_pandas_series = self.event_time_series
            self.autocorrelation = fcs.autocorrelate(use_pandas_series, normalize, log, m, deltat)

            return self.autocorrelation


class Cy5CisModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the simplified cis isomer of the fluorophore Cy5.
    """
    def __init__(self, number_fluorophores, distances, rates):
        single_states = ("tS0", "tS1", "tT1", "Cis", "B")
        super(GeneralModel, self).__init__(number_fluorophores, distances, single_states, rates)
        # sets the method resolution order __mro__ such that it starts from GeneralModel


class Cy5CisdSTORMModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the dSTORM OFF state.
    """
    def __init__(self, number_fluorophores, distances, rates):
        single_states = ("tS0", "tS1", "tT1", "Cis", "OFF", "B")
        super(GeneralModel, self).__init__(number_fluorophores, distances, single_states, rates)


class dSTORMModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the dSTORM OFF state.
    """
    def __init__(self, number_fluorophores, distances, rates):
        single_states = ("S0", "S1", "T1", "OFF", "B")
        super(GeneralModel, self).__init__(number_fluorophores, distances, single_states, rates)
