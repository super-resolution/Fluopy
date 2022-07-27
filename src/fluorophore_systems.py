import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.gillespie_algorithm as ga
import src.processing as pr
import src.initialize as init
import src.animations as an
import src.ssa_cython as ga_cy
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
    states : iterable object
        Contains elements of type str describing each state a single fluorophore can occupy.
    rates : dict
        The transition rates (i.e., rate constants [1/s]).
    Joined_States : enum.EnumMeta
        All possible state combinations (given the number of fluorophores).
    state_names : Collection
        Contains all state names.
    transitions : dict
        Contains all combinations of Joined_States as keys and their unique value pair as values.
    assigned_rate_dict : dict
        Contains all transitions and their rates if the rate is > 0.
    initial_row_vector : np.ndarray
        Is of shape (sqrt(len(transitions)),) with a 1 at position 0 (else 0).
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., the point probabilities) for each transition at the corrsponding
        index pair.
    row_sums : np.ndarray
        Contains the sum of all transition rates (rate constants) of each state.

    - Defined during method simulate() call -
    time_series : None, np.ndarray
        The time points at which the corresponding state occurs.
    time_step_series : None, np.ndarray
        The time step until the corresponding state occurs (starting from the previous state).
    state_series : None, np.ndarray
        The consecutive state's unique values.

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
    unique_series_converted : None, np.ndarray
        Copy of unique_series but each value is replaced by its corresponding ranking number in ascending order.
    occupation_time_total : None, np.ndarray
        The total occupation time of each of unique_states.
    occupation_time_mean : None, np.ndarray
        The mean occupation time of each of unique_states.
    """
    def __init__(self, number, distances, states, rates, predefined=None, induction_rate=None):
        """
        Parameters
        ----------
        number : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        states : iterable object
            Contains elements of type str describing each state a single fluorophore can occupy.
        rates : dict
            The transition rates (i.e., rate constants [1/s]).
        predefined : None, list
            Contains all crucial return values of initializing functions.
        induction_rate : None, float
            If not None, add the concept of off state recovery of one fluorophore induced by the non-emitting transition
            from S1 to S0 of a second fluorophore with rate constant induction_rate.
        """
        self.number = number
        self.distances = distances
        self.states = states
        self.rates = rates
        if predefined:
            self.Joined_States, self.state_names, self.transitions, self.assigned_rate_dict, self.initial_row_vector, \
                self.transition_matrix, self.row_sums = predefined
        else:
            self.Joined_States = init.state_pairs(self.number, self.states)
            self.state_names = []
            for joined_state in self.Joined_States:
                self.state_names.append(joined_state.name)
            self.transitions = init.transition_pairs(self.Joined_States)
            self.assigned_rate_dict = init.transition_rate_dict(self.rates, self.transitions)
            if induction_rate:
                self.assigned_rate_dict = init.induction(self.assigned_rate_dict, self.transitions, induction_rate,
                                                         self.states)
            self.initial_row_vector = init.initial_row_vector(self.transitions)
            _, self.transition_matrix, self.row_sums = init.transition_matrices(self.assigned_rate_dict,
                                                                                self.transitions)

        self.time_series = None
        self.time_step_series = None
        self.state_series = None

        self.duplices = None
        self.unique_series = None
        self.unique_states = None
        self.unique_joined_states = None
        self.unique_names = None
        self.unique_transitions = None
        self.unique_series_converted = None
        self.occupation_time_total = None
        self.occupation_time_mean = None

    def simulate(self, n_steps, seed, base):
        """
        Simulates the CTMC using the direct method of the Gillespie algorithm.

        Parameters
        ----------
        n_steps : int
            Maximum number of simulation steps. If the Markov chain reaches an absorbing state, the simulation stops
            early.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        base : str
            One of "py", "cy" determining whether to use the cython of python implementation.
        """
        if base == "py":
            self.time_series, self.time_step_series, self.state_series = \
                ga.direct_method_py(self.row_sums, self.initial_row_vector, self.transition_matrix, n_steps, seed)
        else:
            self.time_series, self.time_step_series, self.state_series = \
                ga_cy.direct_method_cy(self.row_sums, self.initial_row_vector, self.transition_matrix, n_steps, seed)

    def process(self):
        """
        Collection of processing functions identifying unique states and occupation times.
        """
        self.duplices = pr.identify_duplices(self.state_names)

        self.unique_series, self.unique_states, self.unique_joined_states, self.unique_names = \
            pr.uniques(self.duplices, self.state_series, self.Joined_States)

        self.unique_transitions = init.transition_pairs(self.unique_joined_states)

        self.unique_series_converted = pr.convert_unique_states(self.unique_series, self.unique_states)

        self.occupation_time_total, self.occupation_time_mean = \
            pr.occupation_t(self.time_step_series, self.unique_series, self.unique_states)


class JablonskiModel(FluorophoreSystem):
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
        Copy of emitting_mask but some True entries may have changed to False dictated by photon_collection.
    pandas_series : None, pd.Series
        Contains the time step (resample) in seconds as index and the number of events as values.
    on_periods : None, np.ndarray
        Contains the lengths of each on-period.
    off_periods : None, np.ndarray
        Contains the lengths of each off-period.

    - Defined during method fcs() call -
    autocorrelation : None, tuple
        Contains two arrays, the first is the time points that correspond to the autocorrelation values, the second is
        the autocorrelation values.
    """
    def __init__(self, number, distances, rates, predefined=None, induction_rate=None):
        """
        Parameters
        ----------
        number : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        rates : dict
            The transition rates (i.e., rate constants [1/s]).
        predefined : None, list
            Contains all crucial return values of initializing functions.
        induction_rate : None, float
            If not None, add the concept of off state recovery of one fluorophore induced by the non-emitting transition
            from S1 to S0 of a second fluorophore with rate constant induction_rate.
        """
        states = ("S0", "S1", "T1", "R", "B")
        super().__init__(number, distances, states, rates, predefined, induction_rate)

        self.emitting_transitions = None
        self.emitting_transitions_indices = None
        self.emitting_mask = None
        self.detected_emission_mask = None
        self.pandas_series = None
        self.on_periods = None
        self.off_periods = None

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
        an.jablonski_diagram(self.time_series, self.time_step_series, self.state_series, self.number,
                             self.state_names, self.states, index_min, index_range, fps, saveas)

    def emitters(self, photon_collection=1, resample=None, unit=None, threshold=0, memory=0, use_unique=True,
                 remove_heading_off_period=False, seed=100):
        """
        Collection of functions identifying emitting transitions and other related properties.

        Parameters
        ----------
        photon_collection : float
            Number between 0 and 1, dictates the fraction of kept True values of emitting_mask.
        resample : str
            Resamples events. See pandas time series user's guide offset aliases for possible input values.
        unit : str
            Unit of time_series. One of "W", "D", "h", "m", "S", "ms", "us", "ns".
        threshold : int
            Minimum value of photons per frame (i.e., resample) to be considered an on-frame.
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
            use_transitions = self.unique_transitions
            use_state_series = self.unique_series_converted  # the conversion takes place during init.transition_pairs,
            # too, because  self.unique_joined_states is ordered
        else:
            use_transitions = self.transitions
            use_state_series = self.state_series

        self.emitting_transitions, self.emitting_transitions_indices = \
            et.identify_emitting_transitions(use_transitions)

        self.emitting_mask = et.emitter_mask(use_state_series, self.emitting_transitions_indices)
        # time-consuming

        self.detected_emission_mask = et.detected_emissions(self.emitting_mask, photon_collection, seed)
        # important: the photon detection rate of the microscope must not impact the actual emission, only their
        # detection!

        self.pandas_series = et.pandas_event_time_series(self.time_series[self.detected_emission_mask], unit, resample)

        self.on_periods, self.off_periods, _, _ = et.blink_statistics(self.pandas_series, threshold, memory,
                                                                      remove_heading_off_period)

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
            Defines the number of points on one level (i.e., 1, 2, 4, 8, etc.), E.g., m=4 leads to
        deltat : float
            The time between each entry of pandas_series.
        """
        self.autocorrelation = pr.autocorrelate(self.pandas_series, normalize, log, m, deltat)


class OnOffModel(FluorophoreSystem):
    """
    Derived class from FluorophoreSystem. States follow a simplified model of 'On', 'Off' and 'Bleached'.

    Attributes
    ----------
    - Defined during method emitters() call -
    on_counts : None, np.ndarray
        Contains the number of on states of each state.
    emissions : None, np.ndarray
        Contains the photon counts per time step (i.e., resample).
    emission_time_series : None, np.ndarray
        Contains the time points at which the corresponding photon count occurs.
    """
    def __init__(self, number, distances, rates, predefined=None, induction_rate=None):
        """
        Parameters
        ----------
        number : int
            Number of fluorophores of the system.
        distances : float, Collection
            Distances of the fluorophores to each other.
        rates : dict
            The transition rates (i.e., rate constants [1/s]).
        predefined : None, list
            Contains all crucial return values of initializing functions.
        induction_rate : None, float
            If not None, add the concept of off state recovery of one fluorophore induced by the non-emitting transition
            from S1 to S0 of a second fluorophore with rate constant induction_rate.
        """
        states = ("ON", "OFF", "B")
        super().__init__(number, distances, states, rates, predefined, induction_rate)

        self.on_counts = None
        self.emissions = None
        self.emission_time_series = None

    def animate(self, index_min=0, index_range=100, fps=10, saveas="writer_test.mp4"):
        """
        Animate (part of) the state_series displayed as On/Off/Bleached states.

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
        an.on_off_diagram(self.time_series, self.time_step_series, self.state_series, self.number,
                          self.state_names, self.states, index_min, index_range, fps, saveas)

    def emitters(self, s0s1_rate=None, s1s0_rate=None, resample=None, seed=None):
        """
        Counts the number of on states of each state, samples these counts over a delta time and converts them into
        photon counts.

        Parameters
        ----------
        s0s1_rate : float
            Rate constant of the transition from S0 to S1.
        s1s0_rate : float
            Rate constant of the transition from S1 to S0.
        resample : float
            The delta time over which the number of photon emissions shall be sampled.
        seed : None, int, BitGenerator, Generator
            Seed to initialize a BitGenerator.
        """
        self.on_counts = et.on_states(self.state_names)
        self.emissions, self.emission_time_series = et.emission_count(s0s1_rate, s1s0_rate, self.on_counts,
                                                                      self.state_series, self.time_step_series,
                                                                      resample, seed)
