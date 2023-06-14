"""Classes to represent fluorophores in photophysical simulations."""
import numpy as np
import pandas as pd
import src.fcs as fcs
import src.figures as fi
import src.processing as pr
import src.initialize as init
import src.gillespie_algorithm as ga
import src.emitting_transitions as et


pd.options.display.float_format = '{:.2e}'.format


class FluorophoreSystem:
    """
    Base class of fluorophore systems and their serial quantum state behavior based on continuous time Markov chain
    (CTMC) modelling.

    Attributes
    ----------
    - Defined during instantiation of class object -
    parameter_collection : pd.DataFrame
        Collection of parameters given to any method of FluorophoreSystem and its derived classes. Multiindex level 0
        describes the class method, level 1 describes the parameter passed to that method.
    single_states : dict
        Contains (key, value) pairs of type (int, str), where the key denotes the id and the value the name of the
        single state.
    unique_transitions : pd.DataFrame
        Contains name (str), rate (float), trivial_name (str), abbreviation (str) and fluorescence (bool) of each
        transition, where their id is the index.
    joined_states : pd.DataFrame
        Contains name (str), single_states (array), single_state_counter (array) and absorbing (bool) of each joined
        state, where their id is the index.
    joined_transitions : pd.DataFrame
        Contains name (str), joined_states_id (tuple), unique_transition_id (int), rate (float), trivial_name (str) and
        fluorescence (bool) of each joined state transition, where their id is the index.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corresponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum of all transition rates of a
        joined state.
    graph : nx.DiGraph
        Contains nodes and edges of the Markov chain.
    plot : src.figures.FigureCollection
        Base class of plots.

    - Defined during method simulate() call -
    time_series : np.ndarray
        The simulated time points at which the corresponding joined states occur.
    time_step_series : np.ndarray
        The simulated time steps until the corresponding joined states occur (starting from the previous joined state).
        Therefore, the lifetime of a joined state of index i is the time step of time_step_series[i+1].
    state_series : np.ndarray
        The simulated consecutive joined states ids.

    - Defined during method process() call -
    single_state_series : np.ndarray
        State series for each individual fluorophore (hence, simulated consecutive single states ids).
    transition_series : np.ndarray
        Contains the NEXT transition id for each corresponding simulated joined state (except the last).
    single_state_lifetimes : dict
        Keys are 'single_state_occurrences', 'single_state_occurrences_all', 'single_state_occurrences_pred',
        'lifetimes_fluorophore', 'lifetimes_single_states', 'lifetimes_single_states_all',
        'lifetimes_single_states_pred', 'mean_lifetimes', 'mean_lifetimes_all', 'mean_lifetimes_pred',
        'total_lifetimes', 'total_lifetimes_all', 'total_lifetimes_pred'.
    transition_lifetimes : dict
        Keys are 'transition_occurrences', 'transition_occurrences_all', 'transition_occurrences_pred',
        'transition_times', 'transition_times_all', 'transition_times_pred', 'mean_transition_times',
        'mean_transition_times_all', 'mean_transition_times_pred'.

    - Defined during method emitters() call -
    emissions : np.ndarray
        Contains indices that correspond to time points at which emissions have happened. Using it to index
        state_series or transition_series will result in the outcome AFTER the emission event (hence, the joined state
        or transition the follows the emission event).
    events : np.ndarray
        Contains indices that correspond to time points at which emissions have happened and were detected.
    time_points_events : np.ndarray
        Contains the time points at which an event occurs.
    event_time_series : pd.Series
        Contains the time points in seconds as index and the number of events as values.
    on_periods : np.ndarray
        Contains the durations of each ON period.
    off_periods : np.ndarray
        Contains the durations of each OFF period.
    emission_statistics : dict
        Keys are 'total_emissions', 'total_events', 'mean_time_between_events'.

    - Defined during method fcs() call -
    autocorrelation : tuple
        Contains two arrays, the first is the time differences (i.e., tau or lag time) that correspond to the
        autocorrelation values, the second is the autocorrelation values.
    """
    def __init__(self, number_fluorophores, distances, transitions, remove=None):
        """
        Parameters
        ----------
        number_fluorophores : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        transitions : list
            Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
            (str), abbreviation (str), fluorescence (bool)]. In the case of energy transfers, the first entry is
            k_singlestate1_singlestate2__singlestate1_singlestate2, where the first part represents one fluorophore and
            the second part the other fluorophore.
        remove : None, list
            Contains abbreviations (str) of transitions to be removed.
        """
        transitions = init.filter_transitions(transitions=transitions, remove=remove)
        single_states = init.extract_single_states(transitions=transitions)
        self.parameter_collection = pd.DataFrame([number_fluorophores, distances, single_states, transitions, remove],
                                                 index=[['init', 'init', 'init', 'init', 'init'],
                                                        ['number_fluorophores', 'distances', 'single_states',
                                                         'transitions', 'remove']])

        self.single_states = {i: state for i, state in enumerate(single_states)}
        ###############################################################################################################
        self.unique_transitions = pd.DataFrame(transitions,  columns=['name', 'rate', 'trivial_name', 'abbreviation',
                                                                      'fluorescence'])
        self.unique_transitions.index.name = 'id'
        ###############################################################################################################
        joined_states = init.state_pairs(number=self.parameter_collection.loc[('init', 'number_fluorophores'), 0],
                                         single_states=single_states)
        self.joined_states = pd.DataFrame.from_dict(joined_states, orient='index', columns=['', '']).reset_index()
        self.joined_states.columns = ['name', 'single_states', 'single_state_counter']
        self.joined_states.index.name = 'id'
        ###############################################################################################################
        joined_transitions = init.transition_pairs(joined_states=self.joined_states)
        transition_rate_list = init.construct_transition_rate_list(unique_transitions=self.unique_transitions,
                                                                   joined_transitions=joined_transitions)
        self.joined_transitions = pd.DataFrame(transition_rate_list, columns=['name', 'joined_states_id',
                                                                              'unique_transition_id', 'rate',
                                                                              'trivial_name', 'fluorescence'])
        self.joined_transitions.index.name = 'id'
        ###############################################################################################################
        self.transition_matrix, self.row_sums = \
            init.construct_transition_matrices(joined_transitions=self.joined_transitions,
                                               joined_states=self.joined_states)
        self.joined_states = init.add_absorbing_states(joined_states=self.joined_states,
                                                       joined_transitions=self.joined_transitions)
        self.graph = init.construct_network(unique_transitions=self.unique_transitions)

        self.plot = fi.FigureCollection(system=self)

        self.time_series = None
        self.time_step_series = None
        self.state_series = None

        self.single_state_series = None
        self.transition_series = None
        self.single_state_lifetimes = None
        self.transition_lifetimes = None

        self.emissions = None
        self.events = None
        self.time_points_events = None
        self.event_time_series = None
        self.on_periods = None
        self.off_periods = None
        self.emission_statistics = None

        self.autocorrelation = None

    def update_transitions(self, update):
        """
        Update the unique_transition dataframe, where update contains the abbreviations and their new rate values.

        Parameters
        ----------
        update : dict
            Keys are abbreviations of transitions, values are rates.
        """
        for transition in self.parameter_collection.loc[('init', 'transitions'), 0]:
            if transition[3] in update:
                transition[1] = update[transition[3]]

        kwargs = self.parameter_collection.loc['init', 0].to_dict()
        del kwargs['single_states']
        FluorophoreSystem.__init__(self, **kwargs)

    def simulate(self, n_steps=100, start_id=0, end_time=None, seed=100):
        """
        Simulates the CTMC using the direct method of the Gillespie algorithm.

        Parameters
        ----------
        n_steps : int
            Maximum number of simulation steps. If the Markov chain reaches an absorbing joined state, the simulation
            stops early.
            If end_time is not None, n_steps serves as size parameter instead of simulation step parameter.
        start_id : int
            Id of the starting joined state.
        end_time : float
            Time at which simulation ends. If not None, simulation will be carried out until time limit is met instead
            of n_steps. n_steps is then used as size parameter.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        if 'simulate' in self.parameter_collection.index.unique(level=0):
            self.parameter_collection.drop('simulate', level=0, inplace=True)
        new_dataframe = pd.DataFrame([n_steps, start_id, seed], index=[['simulate', 'simulate', 'simulate'],
                                                                       ['n_steps', 'start_id', 'seed']])
        self.parameter_collection = pd.concat([self.parameter_collection, new_dataframe])

        if end_time is None:
            self.time_series, self.time_step_series, self.state_series = \
                ga.direct_method_steps(transition_matrix=self.transition_matrix, row_sums=self.row_sums,
                                       n_steps=n_steps, start_id=start_id, seed=seed)
        else:
            self.time_series, self.time_step_series, self.state_series = \
                ga.direct_method_time(transition_matrix=self.transition_matrix, row_sums=self.row_sums,
                                      size=n_steps, start_id=start_id, end_time=end_time, seed=seed)

    def process(self, seed=100):
        """
        Collection of processing functions to generate single state series, transition series and statistics based on
        the simulation results.

        Parameters
        ----------
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        if 'process' in self.parameter_collection.index.unique(level=0):
            self.parameter_collection.drop('process', level=0, inplace=True)
        new_dataframe = pd.DataFrame([seed], index=[['process'], ['seed']])
        self.parameter_collection = pd.concat([self.parameter_collection, new_dataframe])

        number_fluorophores = self.parameter_collection.loc[('init', 'number_fluorophores'), 0]
        self.single_state_series = \
            pr.convert_single_state_series(number_fluorophores=number_fluorophores, state_series=self.state_series,
                                           joined_states=self.joined_states)
        transition_cum_sum, transition_sorted_indices = \
            pr.multiple_transitions(joined_transitions=self.joined_transitions, joined_states=self.joined_states,
                                    unique_transitions=self.unique_transitions)
        self.transition_series = pr.generate_transition_series(state_series=self.state_series,
                                                               transition_cum_sum=transition_cum_sum,
                                                               transition_sorted_indices=transition_sorted_indices,
                                                               seed=seed)
        self.single_state_lifetimes, self.transition_lifetimes = \
            pr.time_occurrence_statistics(number_fluorophores=number_fluorophores, single_states=self.single_states,
                                          unique_transitions=self.unique_transitions, time_series=self.time_series,
                                          transition_series=self.transition_series,
                                          single_state_series=self.single_state_series)

    def emitters(self, photon_collection_rate=1, resample="5ms", emccd_gain=None, threshold=0, memory=0,
                 remove_heading_off_period=False, seed=100):
        """
        Collection of functions identifying emitting transitions and other related properties.

        Parameters
        ----------
        photon_collection_rate : float
            Number between 0 and 1, dictates the fraction of kept indices of emissions.
        resample : str
            See pandas time series user's guide offset aliases for possible input values.
            Resembles frame integration time.
        emccd_gain : int
            The gain of an EMCCD camera.
        threshold : int
            Maximum value of photons per frame to be considered an OFF frame.
        memory : int
            Number of OFF frames to be neglected. They are included in the ON times.
        remove_heading_off_period : bool
            If True and the series starts wit an OFF frame, the leading OFF frame is discarded.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        if 'emitters' in self.parameter_collection.index.unique(level=0):
            self.parameter_collection.drop('emitters', level=0, inplace=True)
        new_dataframe = pd.DataFrame([photon_collection_rate, resample, emccd_gain, threshold, memory,
                                      remove_heading_off_period, seed],
                                     index=[['emitters', 'emitters', 'emitters', 'emitters', 'emitters',
                                             'emitters', 'emitters'],
                                            ['photon_collection_rate', 'resample', 'emccd_gain', 'threshold', 'memory',
                                             'remove_heading_off_period', 'seed']])
        self.parameter_collection = pd.concat([self.parameter_collection, new_dataframe])

        self.emissions = et.get_emissions(unique_transitions=self.unique_transitions,
                                          transition_series=self.transition_series)
        self.events = et.get_events(emissions=self.emissions, photon_collection_rate=photon_collection_rate, seed=seed)
        self.time_points_events = self.time_series[self.events]
        self.event_time_series = et.construct_event_time_series(time_points_events=self.time_points_events,
                                                                resample=resample, emccd_gain=emccd_gain, seed=seed)
        self.on_periods, self.off_periods, _, _ = \
            et.blink_statistics(event_time_series=self.event_time_series, threshold=threshold, memory=memory,
                                remove_heading_off_period=remove_heading_off_period)
        self.emission_statistics = {'total_emissions': self.emissions.size,
                                    'total_events': self.events.size,
                                    'mean_time_between_events': np.mean(np.diff(self.time_points_events))}

    def fcs(self, normalize=True, log=True, m=2, deltat=None):
        """
        Autocorrelation as exploited in typical fluorescence correlation spectroscopy (FCS) setups.

        Parameters
        ----------
        normalize : bool
            Whether to normalize the correlation.
        log : bool
            Whether to compute the autocorrelation on a logarithmic scale.
        m : int
            Defines the number of points on one level (i.e., 1, 2, 4, 8, etc.), E.g., m=4 leads to 1,2,3,4; 2,4,6,8;
            4,8,12,16; ... .
            Only needed if log is True.
        deltat : str
            The minimum tau value. Defines the minimum lag time (time difference) > 0. If None, it is set equal to the
            resample value of method emitters(). See pandas time series user's guide offset aliases for other possible
            input values.
            Resembles frame integration time.
        """
        if deltat is None:
            deltat = self.parameter_collection.loc[('emitters', 'resample'), 0]

        if 'fcs' in self.parameter_collection.index.unique(level=0):
            self.parameter_collection.drop('fcs', level=0, inplace=True)
        new_dataframe = pd.DataFrame([normalize, log, m, deltat], index=[['fcs', 'fcs', 'fcs', 'fcs'],
                                                                         ['normalize', 'log', 'm', 'deltat']])
        self.parameter_collection = pd.concat([self.parameter_collection, new_dataframe])

        deltat_float = pd.to_timedelta(deltat) / np.timedelta64(1, 's')
        if deltat == self.parameter_collection.loc[('emitters', 'resample'), 0]:
            event_time_series = self.event_time_series
        else:
            event_time_series = et.construct_event_time_series(time_points_events=self.time_points_events,
                                                                   resample=deltat,
                                                                   emccd_gain=self.parameter_collection.loc[
                                                                       ('emitters', 'emccd_gain'), 0],
                                                                   seed=self.parameter_collection.loc[
                                                                       ('emitters', 'seed'), 0])
        try:
            self.autocorrelation = fcs.autocorrelate(event_time_series=event_time_series, normalize=normalize,
                                                 log=log, m=m, deltat=deltat_float)
        except ValueError:
            print('Logarithmic autocorrelation not possible. Event_time_series too short.')


class Cy5(FluorophoreSystem):
    """
    Derived class from FluorophoreSystem. Photophysical processes represent the fluorophore Cy5. A special property of
    this fluorophore is its isomerization processes that convert the fluorophore between a cis and a trans state.
    """
    def __init__(self, number_fluorophores, distances, transitions=None, user=None, irradiance=None, wavelength=None,
                 dstorm_parameters=None, remove=None):
        """
        Parameters
        ----------
        number_fluorophores : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        transitions: None, list
            Contains a list for each transition like [k_singlestate1_singlestate2 (str), rate (float), trivial name
            (str), abbreviation (str), fluorescence (bool)]. In the case of energy transfers, the first entry is
            k_singlestate1_singlestate2__singlestate1_singlestate2, where the first part represents one fluorophore and
            the second part the other fluorophore.
        user : str
            One of r'\vie43sq', r'\SagixOffice'.
        irradiance : float
            The irradiance in kW/cm².
        wavelength : float
            In nm.
        dstorm_parameters : dict
            May contain the following keys: reducing_agent, concentration, k_pet, ph, same.
        remove : None, list
            Contains abbreviations (str) of transitions to be removed.
        """
        if transitions is None:
            file_path = rf"C:\Users{user}\OneDrive - Universität Würzburg\GitHub\Photoswitching\src\fluorophores\\"
            transitions = init.determine_rate_constants(path=file_path, distances=distances, irradiance=irradiance,
                                                        wavelength=wavelength, fluorophore='cy5',
                                                        dstorm_parameters=dstorm_parameters)
        super().__init__(number_fluorophores=number_fluorophores, distances=distances, transitions=transitions,
                         remove=remove)
