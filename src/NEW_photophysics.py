from src.NEW_emissions import Emissions
from src.NEW_simulation import Simulation
from src.NEW_statistics import Statistics
from src.NEW_fcs import FCS
from src.NEW_blinking import Blinking
from src.NEW_tcspc import TCSPC


class PhotophysicsMC:
    def __init__(self, fluorophore_system, transitions):
        self.fluorophore_system = fluorophore_system
        self.transitions = transitions
        self.statistics = Statistics()
        self.simulation = Simulation()
        self.emissions = None
        self.fcs = FCS()
        self.tcspc = TCSPC()
        self.blinking = Blinking()

    def finalize_transitions(self):
        # this is not invoked to still being able to call the filter_by_abbreviation method
        self.transitions.construct_combined_state_transitions(self.fluorophore_system)
        self.transitions.construct_transition_matrix()

    def predict(self):
        self.statistics.predict(self.transitions)

    def simulate(self, start_at, size, end_time, seed):
        if self.transitions.transition_matrix is None:
            raise ValueError('call method finalize_transitions() before calling simulate().')
        self.simulation.run(self.transitions, start_at, size, end_time, seed)

        if not (self.transitions.transition_df['energy_transfer'] == True).any() and self.statistics.prediction is None:
            self.statistics.predict(self.transitions)

        self.statistics.analyze(self.simulation, self.transitions)

    def fetch_emissions(self, photon_collection_rate, resample, emccd_gain, seed):
        if self.simulation.transition_series is None:
            raise ValueError('call method simulate() before calling emissions().')
        self.emissions = Emissions(self.transitions.combined_state_transitions_df, self.simulation.transition_series,
                                   self.simulation.time_series, photon_collection_rate, resample, emccd_gain,
                                   seed)

    def perform_fcs(self, exp_min=-8, exp_max=2, points_per_base=4, base=10, normalize=True):
        if self.emissions.event_time_points is None:
            raise ValueError('call method fetch_emissions() before calling perform_FCS().')
        self.fcs.autocorrelate_time_points(self.emissions.event_time_points, exp_min, exp_max, points_per_base, base,
                                           normalize)

    def perform_tcspc(self, start_at=(0, 0, 0), n_steps=1000, seed=100):
        self.tcspc.modify_transition_matrix(self.transitions.transition_matrix, self.transitions.row_sums,
                                            self.transitions.combined_state_transitions_df,
                                            self.fluorophore_system.count)
        self.tcspc.run_simulation(self.transitions.combined_state_transitions_df, start_at, n_steps, seed)
        self.tcspc.get_observed_lifetimes(self.transitions.combined_state_transitions_df)

    def evaluate_blinking(self):
        if self.emissions.event_time_series is None:
            raise ValueError('call method fetch_emissions() before calling evaluate_blinking().')
        self.blinking.get_blinking_statistics(self.emissions.event_time_series)
