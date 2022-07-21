import src.gillespie_algorithm as ga
import src.processing as pr
import src.initialize as init
import src.animations as an
import src.ssa_cython as ga_cy
import src.emitting_transitions as et


class FluorophoreSystem:
    def __init__(self, number, distances, states, rates, predefined=None, induction_rate=None):
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
        if base == "py":
            self.time_series, self.time_step_series, self.state_series = \
                ga.direct_method_py(self.row_sums, self.initial_row_vector, self.transition_matrix, n_steps, seed)
        else:
            self.time_series, self.time_step_series, self.state_series = \
                ga_cy.direct_method_cy(self.row_sums, self.initial_row_vector, self.transition_matrix, n_steps, seed)

        return self.time_series, self.time_step_series, self.state_series

    def process(self):
        self.duplices = pr.identify_duplices(self.state_names)

        self.unique_series, self.unique_states, self.unique_joined_states, self.unique_names = \
            pr.uniques(self.duplices, self.state_series, self.Joined_States)

        self.unique_transitions = init.transition_pairs(self.unique_joined_states)

        self.unique_series_converted = pr.convert_unique_states(self.unique_states, self.unique_series)

        self.occupation_time_total, self.occupation_time_mean = \
            pr.occupation_t(self.time_step_series, self.unique_series, self.unique_states)

        return self.unique_series_converted, self.unique_states, self.occupation_time_mean


class JablonskiModel(FluorophoreSystem):
    def __init__(self, number, distances, rates, predefined=None, induction_rate=None):
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
        an.jablonski_diagram(self.time_series, self.time_step_series, self.state_series, self.number,
                             self.state_names, self.states, index_min, index_range, fps, saveas)

    def emitters(self, photon_collection=1, resample=None, unit=None, threshold=0, memory=0, use_unique=True,
                 remove_heading_off_period=False, seed=100):
        if use_unique:
            if not self.unique_transitions:
                super().process()
            use_transitions = self.unique_transitions
            use_state_series = self.unique_series_converted  # the conversion takes place during init.transition_pairs,
            # too
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

        return self.emitting_mask, self.pandas_series, self.on_periods, self.off_periods

    def fcs(self, normalize=True, log=True, m=2, deltat=5e-3):
        self.autocorrelation = pr.autocorrelate(self.pandas_series, normalize, log, m, deltat)

        return self.autocorrelation


class OnOffModel(FluorophoreSystem):
    def __init__(self, number, distances, rates, predefined=None, induction_rate=None):
        states = ("ON", "OFF", "B")
        super().__init__(number, distances, states, rates, predefined, induction_rate)

        self.on_counts = None
        self.emissions = None
        self.emission_time_series = None

    def animate(self, index_min=0, index_range=100, fps=10, saveas="writer_test.mp4"):
        an.on_off_diagram(self.time_series, self.time_step_series, self.state_series, self.number,
                          self.state_names, self.states, index_min, index_range, fps, saveas)

    def emitters(self, s0s1_rate=None, s1s0_rate=None, resample=None, seed=None):
        self.on_counts = et.on_states(self.state_names)
        self.emissions, self.emission_time_series = et.emission_count(s0s1_rate, s1s0_rate, self.on_counts,
                                                                      self.state_series, self.time_step_series,
                                                                      resample, seed)
        return self.emissions, self.emission_time_series
