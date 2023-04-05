# ########################################################################################################################
# # OnOffModel
# ########################################################################################################################
#
#
# def on_states(state_names):
#     """
#     Counts the number of on states of each state (i.e., joined_state).
#
#     Parameters
#     ----------
#     state_names : Collection
#         Contains all state names.
#
#     Returns
#     -------
#     on_counts : np.ndarray
#         Contains the number of on states of each state.
#     """
#     on_counts = np.zeros(len(state_names))
#     for i, state_name in enumerate(state_names):
#         states = state_name.split("_")
#         counter = states.count("ON")
#         on_counts[i] = counter
#
#     return on_counts
#
#
# def emission_count(s0s1_rate, s1s0_rate, on_counts, state_series, time_step_series, resample=5e-3, seed=100):
#     """
#     Samples the on counts over a delta time (resample) and converts them into photon counts. This involves stretching
#     the data since simulated time steps are expected to be larger than resample, meaning that the resulting time steps
#     and their corresponding photon counts are an approximation.
#
#     Parameters
#     ----------
#     s0s1_rate : float
#         Rate constant of the transition from S0 to S1.
#     s1s0_rate : float
#         Rate constant of the transition from S1 to S0.
#     on_counts : np.ndarray
#         The return value of on_states.
#     state_series : np.ndarray
#         Contains the sequence of state indices of the Markov chain.
#     time_step_series : np.ndarray
#         Contains the time step until the corresponding state occurs (starting from the previous state).
#     resample : float
#         The delta time over which the number of photon emissions shall be sampled.
#     seed : None, int, BitGenerator, Generator
#         Seed to initialize a BitGenerator.
#
#     Returns
#     -------
#     emissions : np.ndarray
#         Contains the photon counts per time step (i.e., resample).
#     emission_time_series : np.ndarray
#         Contains the time points at which the corresponding photon count occurs.
#     """
#     rng = np.random.default_rng(seed)
#
#     on_counts_series = on_counts[state_series]  # converts the state_series into a series of on counts
#
#     repeats = time_step_series[1:] / resample
#     repeats = np.round(repeats)
#     repeats[np.where(repeats == 0)] = 1
#     repeats = repeats.astype(int)  # an entry in repeats is how often the resample value fits into the time step
#
#     stretched = np.repeat(on_counts_series[:-1], repeats)  # stretch each entry of the on_counts_series by the
#     # corresponding entry of repeats
#
#     mean_emissions_per_s = s0s1_rate * s1s0_rate / (s0s1_rate + s1s0_rate)  # this holds true if the two state markov
#     # chain can only be of values 0 and 1. Can be shown by simulation.
#     emissions_per_resample = rng.poisson(lam=mean_emissions_per_s*resample, size=stretched.shape)
#
#     emissions = stretched * emissions_per_resample
#
#     emission_time_series = np.arange(0, len(stretched)*resample, resample)
#
#     return emissions, emission_time_series
#
#
# class OnOffModel(FluorophoreSystem):
#     """
#     Derived class from FluorophoreSystem. States follow a simplified model of 'On', 'Off' and 'Bleached'.
#
#     Attributes
#     ----------
#     - Defined during method emitters() call -
#     on_counts : None, np.ndarray
#         Contains the number of on states of each state.
#     emissions : None, np.ndarray
#         Contains the photon counts per time step (i.e., resample).
#     emission_time_series : None, np.ndarray
#         Contains the time points at which the corresponding photon count occurs.
#     """
#     def __init__(self, number, distances, rates):
#         """
#         Parameters
#         ----------
#         number : int
#             Number of fluorophores of the system.
#         distances : float, Collection
#             Distances of the fluorophores to each other.
#         rates : dict
#             The transition from state 1 to state 2 with rate constant k [1/s] should have the key k_state1_state2 and
#             the value [k, name_of_transition] assigned to it.
#         """
#         single_states = ("ON", "OFF", "B")
#         super().__init__(number, distances, single_states, rates)
#
#         self.on_counts = None
#         self.emissions = None
#         self.emission_time_series = None
#
#     def animate(self, index_min=0, index_range=100, fps=10, saveas="writer_test.mp4"):
#         """
#         Animate (part of) the state_series displayed as On/Off/Bleached states.
#
#         Parameters
#         ----------
#         index_min : int
#             Starting index for state_series (and time_series, time_step_series).
#         index_range : int
#             Number of steps to animate.
#         fps : int
#             Animation frame rate.
#         saveas : str
#             Defines the save location of the outfile.
#         """
#         an.on_off_diagram(self.time_series, self.time_step_series, self.state_series, self.number,
#                           self.state_names, self.single_states, index_min, index_range, fps, saveas)
#
#     def emitters(self, s0s1_rate=None, s1s0_rate=None, resample=None, seed=None):
#         """
#         Counts the number of on states of each state, samples these counts over a delta time and converts them into
#         photon counts.
#
#         Parameters
#         ----------
#         s0s1_rate : float
#             Rate constant of the transition from S0 to S1.
#         s1s0_rate : float
#             Rate constant of the transition from S1 to S0.
#         resample : float
#             The delta time over which the number of photon emissions shall be sampled.
#         seed : None, int, BitGenerator, Generator
#             Seed to initialize a BitGenerator.
#         """
#         self.on_counts = et.on_states(self.state_names)
#         self.emissions, self.emission_time_series = et.emission_count(s0s1_rate, s1s0_rate, self.on_counts,
#                                                                       self.state_series, self.time_step_series,
#                                                                       resample, seed)
############################################################
# def on_off_diagram(time_series, time_step_series, state_series, number, state_names, single_states, index_min=0,
#                    index_range=100, fps=10, saveas="writer_test.mp4"):
#     """
#     Animate (part of) the state_series displayed as On/Off/Bleached states.
#
#     Parameters
#     ----------
#     time_series : np.ndarray
#         Contains the time points at which the corresponding state occurs.
#     time_step_series : np.ndarray
#         Contains the time step until the corresponding state occurs (starting from the previous state).
#     state_series : np.ndarray
#         Contains the consecutive state's unique values.
#     number : int
#         Number of fluorophores of the system.
#     state_names : Collection
#         Contains all state names.
#     single_states : iterable object
#         Contains elements of type str.
#     index_min : int
#         Starting index for state_series (and time_series, time_step_series).
#     index_range : int
#         Number of steps to animate.
#     fps : int
#         Animation frame rate.
#     saveas : str
#         Defines the save location of the outfile.
#
#     Returns
#     -------
#     None
#     """
#     if single_states != ("ON", "OFF", "B"):
#         raise ValueError("states have to be equal to (""ON"", ""OFF"", ""B"")")
#     else:
#         ffmpegwriter = manimation.writers["ffmpeg"]
#         metadata = dict(title="On Off diagram", artist="Sagix",
#                         comment="Markov Chain visualized in the On Off diagram")
#         writer = ffmpegwriter(fps=fps, metadata=metadata)
#
#         fig = plt.figure(figsize=(14, 7))
#         plt.ylim(0, 2)
#         plt.xlim(0, 4)
#         x_diff = 4 / (number+1)
#         circle_positions_x = np.arange(x_diff, 4+x_diff, x_diff)
#
#         ax = plt.gca()
#         ax.get_xaxis().set_visible(False)
#         plt.yticks([], [])
#
#         colors = ["r", "grey", "w"]
#         handles = [mpatches.Patch(facecolor=color, edgecolor="k", label=state) for color, state in zip(colors,
#                                                                                                        single_states)]
#         ax.legend(handles=handles, prop={"size": 20}, loc="upper left")
#
#         exponents = np.floor(np.log10(time_step_series[1:]))
#         min_expo = np.min(exponents)
#
#         special_case = False
#
#         if len(state_series[index_min:])-1 <= index_range:
#             index_range = len(state_series[index_min:])
#             special_case = True
#
#         with writer.saving(fig, saveas, 100):
#             for i_1 in range(index_min, index_min + index_range):
#                 state_index = int(state_series[i_1])
#                 state = state_names[state_index]
#                 state = state.split("_")
#                 for i_2 in range(number):
#                     index = single_states.index(state[i_2])
#                     plt.plot([circle_positions_x[i_2]], [1], marker="o", markerfacecolor=colors[index],
#                              markeredgecolor="k", markersize=80)
#
#                 if special_case and i_1 == index_range-1:
#                     next_transition_in = np.inf
#                     frames = 1
#                     next_frame_in = np.inf
#                 else:
#                     next_transition_in = time_step_series[i_1 + 1]  # i_1 + 1 because each time interval of state i_1 is
#                     # the one until it was occupied, here the time it stays at state i (until next transition happens)
#                     # is desired
#
#                     exponent = exponents[i_1]  # i_1 since exponents is defined with array[1:]
#                     frames = int(1 + exponent - min_expo)
#                     next_frame_in = frames / fps
#
#                 total_time = time_series[i_1]
#
#                 row_labels = ["total", "next tr", "next fr"]
#                 column_label = ["time [s]"]
#                 cell_texts = [[f"{total_time:.2e}"], [f"{next_transition_in:.2e}"], [f"{next_frame_in}"]]
#                 table = plt.table(cellText=cell_texts, rowLabels=row_labels, colLabels=column_label, cellLoc="center",
#                                   colWidths=[0.1], rowLoc="center", loc="upper right")
#                 table.set_fontsize(13)
#                 table.scale(1.5, 1.5)
#
#                 for _ in range(frames):
#                     writer.grab_frame()
#
#         plt.close()
