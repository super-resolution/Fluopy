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
import src.animations as an
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
    number : int
        Number of fluorophores of the system.
    distances : float, Collection
        Distances of the fluorophores to each other.
    rates : dict
        The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and the
        value [k (float), name_of_transition (str)] assigned to it.
    single_states : iterable object
        Contains elements of type str describing each state a single fluorophore can occupy. A combination of
        single_states will be denoted as state.
    single_state_counter : dict
        Contains the joined_states as keys and np.ndarray as values. The values resemble the number of single_states
        (at the corresponding index) contained in joined_states.
    single_state_id : dict
        Contains the joined_states as keys and np.ndarray as values. The values contain the single_state indices
        in the correct order. E.g., the key 'S0_S1_S0' will have the value [0, 1, 0].
    Joined_States : enum.EnumMeta
        All possible singe_state combinations (given the number of fluorophores). A joined state will be denoted as
        state.
    state_names : Collection
        Contains all state names.
    state_ids : Collection
        Contains all state's identification numbers.
    state_identifier : dict
        Contains all state names as keys and state identification numbers as values.
    transitions : dict
        Contains all combinations of Joined_States as keys and their unique value pair as values.
    assigned_rate_dict : dict
        Contains all transitions (combinations of Joined_States) as keys and their rates as values.
    rate_id_dict : dict
        Contains all transition value pairs as keys and their transition id as values.
    rate_name_dict : dict
        Contains all transition value pairs as keys and their names as values.
    transition_dict : dict
        Contains transition ids as keys and transition names as values.
    initial_row_vector : np.ndarray
        Each state has an entry (hence, shape (len(state_ids),)) of 0 except a 1 at position 0.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of all transition rates (rate constants) of each state.
    absorbing_states : Collection
        Contains all absorbing states, i.e., states without outgoing transitions.
    graph : nx.DiGraph
        Contains nodes and edges of the Markov chain.

    - Defined during method simulate() call -
    time_series : None, np.ndarray
        The time points at which the corresponding state occurs.
    time_step_series : None, np.ndarray
        The time step until the corresponding state occurs (starting from the previous state).
    state_series : None, np.ndarray
        The consecutive state's unique values.
    transition_series : None, list
        Contains the next transition id for each corresponding state (except the last).

    - Defined during method process() call -
    duplices : None, list
        Contains lists of two elements, where the first element is the unique value of a state and the second element
        is the unique value of another state that is ordered equal to the first state.
    unique_series : None, np.ndarray
        Copy of state_series but with every second element of a list of duplices replaced by its first element.
    unique_states : None, np.ndarray
        Every state that occurs in unique_series.
    unique_joined_states : None, list
        Contains elements of Joined_States if their unique value occurs in unique_states. It is ordered by unique value.
    unique_names : None, list
        Contains all unique state names (if their unique value occurs in unique_states).
    unique_transitions : None, dict
        Contains all combinations of Joined_States as keys and their unique value pair as values, if they occur in
        unique_states.
    unique_series_converted : None, np.ndarray
        Copy of unique_series but each value is replaced by its corresponding ranking number in ascending order.
    single_state_series : None, np.ndarray
        State series for each individual fluorophore (hence, single_states).
    single_state_lifetimes : None, dict
        Keys are single_state_occurrences, single_state_occurrences_all, lifetimes_fluorophore, lifetimes_single_states,
        lifetimes_single_states_all, mean_lifetimes, mean_lifetimes_all, total_lifetimes, total_lifetimes_all.
    transition_lifetimes : None, dict
        Keys are transition_occurrences, transition_occurrences_all, transition_times, transition_times_all,
        mean_transition_times, mean_transition_times_all.
    """
    def __init__(self, number, distances, single_states, rates):
        """
        Parameters
        ----------
        number : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        single_states : iterable object
            Contains elements of type str describing each state a single fluorophore can occupy.
        rates : dict
            The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
            the value [k, name_of_transition] assigned to it.
        """
        self.number = number
        self.distances = distances
        self.rates = rates
        self.single_states = single_states

        self.Joined_States, self.single_state_counter, self.single_state_id = init.state_pairs(self.number,
                                                                                               self.single_states)
        self.state_names = []
        self.state_ids = []
        self.state_identifier = {}
        for joined_state in self.Joined_States:
            self.state_names.append(joined_state.name)
            self.state_ids.append(joined_state.value)
            self.state_identifier[joined_state.name] = joined_state.value
        self.transitions = init.transition_pairs(self.Joined_States)
        self.assigned_rate_dict, self.rate_id_dict, self.rate_name_dict, self.transition_dict = \
            init.transition_rate_dict(self.rates, self.transitions)
        self.initial_row_vector = init.initial_row_vector(self.state_ids)
        _, self.transition_matrix, self.row_sums = init.transition_matrices(self.assigned_rate_dict,
                                                                            self.transitions)
        self.absorbing_states = init.absorbing_states(self.rate_name_dict, self.state_ids)
        self.graph = init.network(self.rates)

        self.time_series = None
        self.time_step_series = None
        self.state_series = None
        self.transition_series = None

        self.duplices = None
        self.unique_series = None
        self.unique_states = None
        self.unique_joined_states = None
        self.unique_names = None
        self.unique_transitions = None
        self.unique_series_converted = None
        self.single_state_series = None
        self.single_state_lifetimes = None
        self.transition_lifetimes = None

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
        self.time_series, self.time_step_series, self.state_series, self.transition_series = \
            ga.direct_method_py(self.row_sums, self.initial_row_vector, self.transition_matrix, self.rate_id_dict,
                                n_steps, seed)

    def process(self):
        """
        Collection of processing functions identifying unique states and occupation times.
        """
        self.duplices = pr.identify_duplices(self.state_names)

        self.unique_series, self.unique_states, self.unique_joined_states, self.unique_names = \
            pr.uniques(self.duplices, self.state_series, self.Joined_States)

        self.unique_transitions = init.transition_pairs(self.unique_joined_states)

        self.unique_series_converted = pr.convert_unique_states(self.unique_series, self.unique_states)

        self.single_state_series = pr.convert_single_state_series(self.number, self.state_series,
                                                                  self.state_ids, self.single_state_id)
        self.single_state_lifetimes, self.transition_lifetimes = \
            pr.occupation_time_single_states(self.number, self.rates, self.time_series, self.transition_series,
                                             self.single_state_series, self.single_states)


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
    def __init__(self, number, distances, rates):
        """
        Parameters
        ----------
        number : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        rates : dict
            The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
            the value [k, name_of_transition] assigned to it.
        """
        single_states = ("S0", "S1", "T1", "B")
        super().__init__(number, distances, single_states, rates)

        self.emitting_transitions = None
        self.emitting_transitions_indices = None
        self.emitting_mask = None
        self.detected_emission_mask = None
        self.events_at = None
        self.last_parameters = None
        self.pandas_series = None
        self.on_periods = None
        self.off_periods = None
        self.statistics = None

        self.autocorrelation = None

    def animate(self, index_min=0, index_range=100, fps=10, saveas="writer_test.mp4"):
        """
        Animate (part of) the state_series displayed in a Jablonski diagram.

        Parameters
        ----------
        index_min : int
            Starting index for state_series (and time_series, time_step_series).
        index_range : int
            Number of steps to animate.
        fps : int
            Animation frame rate.
        saveas : str
            Defines the save location of the outfile.
        """
        an.jablonski_diagram(self.time_series, self.time_step_series, self.state_series, self.transition_series,
                             self.transition_dict,
                             self.number, self.state_names, self.single_states, index_min, index_range, fps, saveas)

    def emitters(self, photon_collection=1, resample="5ms", emccd_gain=None, threshold=0, memory=0,
                 use_unique=True, remove_heading_off_period=False, seed=100):
        """
        Collection of functions identifying emitting transitions and other related properties.

        Parameters
        ----------
        photon_collection : float
            Number between 0 and 1, dictates the fraction of kept True values of emitting_mask.
        resample : str
            Resamples events. See pandas time series user's guide offset aliases for possible input values.
        emccd_gain : int
            The gain of an emccd.
        threshold : int
            Maximum value of photons per frame (i.e., resample) to be considered an off-frame.
        memory : int
            Number of off-frames to be neglected. They are included in the on times.
        use_unique : bool
            Whether to use the unique states for the estimations (instead of the original states).
        remove_heading_off_period : bool
            If True and the series starts wit an off-frame, the leading off-frame is discarded.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        if use_unique:
            if not self.unique_transitions:
                super().process()
            elif len(self.time_series) != len(self.unique_series_converted):
                super().process()
            use_transitions = self.unique_transitions
            use_state_series = self.unique_series_converted  # the conversion takes place during init.transition_pairs
            # (origin of self.unique_transitions), too, because self.unique_joined_states is ordered
        else:
            use_transitions = self.transitions
            use_state_series = self.state_series

        self.emitting_transitions, self.emitting_transitions_indices = \
            et.identify_emitting_transitions(use_transitions, self.single_states)

        self.emitting_mask = et.emitter_mask(use_state_series, self.emitting_transitions_indices)
        # time-consuming
        self.detected_emission_mask = et.detected_emissions(self.emitting_mask, photon_collection, seed)
        # important: the photon detection rate of the microscope must not impact the actual emission, only their
        # detection!
        self.events_at = self.time_series[self.detected_emission_mask]
        self.last_parameters = dict(resample=resample, emccd_gain=emccd_gain, seed=seed)
        self.pandas_series = et.pandas_event_time_series(self.events_at, resample=resample, emccd_gain=emccd_gain,
                                                         seed=seed)
        self.on_periods, self.off_periods, _, _ = et.blink_statistics(self.pandas_series, threshold, memory,
                                                                      remove_heading_off_period)
        self.statistics = {'total_events': np.count_nonzero(self.emitting_mask),
                           'mean_time_between_events': np.mean(np.diff(self.events_at)),
                           'total_detected_events': np.count_nonzero(self.detected_emission_mask)}

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
        if self.pandas_series is None:
            raise TypeError('pandas_series is None')
        else:
            current_deltat = self.last_parameters["resample"]
            current_deltat = pd.to_timedelta(current_deltat)
            required_deltat = pd.to_timedelta(deltat, unit="s")
            if required_deltat != current_deltat:
                new_pandas_series = et.pandas_event_time_series(self.events_at, resample=required_deltat,
                                                                emccd_gain=self.last_parameters["emccd_gain"],
                                                                seed=self.last_parameters["seed"])
                use_pandas_series = new_pandas_series
            else:
                use_pandas_series = self.pandas_series
            self.autocorrelation = fcs.autocorrelate(use_pandas_series, normalize, log, m, deltat)

            return self.autocorrelation


class Cy5CisModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the simplified cis isomer of the fluorophore Cy5.
    """
    def __init__(self, number, distances, rates):
        single_states = ("tS0", "tS1", "tT1", "Cis", "B")
        super(GeneralModel, self).__init__(number, distances, single_states, rates)
        # sets the method resolution order __mro__ such that it starts from GeneralModel


class Cy5CisdSTORMModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the dSTORM OFF state.
    """
    def __init__(self, number, distances, rates):
        single_states = ("tS0", "tS1", "tT1", "Cis", "OFF", "B")
        super(GeneralModel, self).__init__(number, distances, single_states, rates)


class dSTORMModel(GeneralModel):
    """
    Derived class from GeneralModel. States include the dSTORM OFF state.
    """
    def __init__(self, number, distances, rates):
        single_states = ("S0", "S1", "T1", "OFF", "B")
        super(GeneralModel, self).__init__(number, distances, single_states, rates)
