# def simulation_tau_leaping(initial_row_vector, transition_rate_matrix, tau, n_steps=100, seed=100):
#     """
#     The tau-leaping method of the gillespie algorithm (i.e., the stochastic simulation algorithm). Note that the state
#     change vector is trivial, since each transition leads to a decrease in the current state by 1 and an increase in the
#     following state by 1.
#     This implementation does not avoid negative populations and does not employ the algorithm for efficient step size
#     selection.
#
#     Parameters
#     ----------
#     initial_row_vector : np.ndarray
#         The return value of initialize.initial_row_vector.
#     transition_rate_matrix : np.ndarray
#         The first return value of initialize.transition_matrices.
#     tau : float
#         Time step.
#     n_steps : int
#         Maximum number of simulation steps. If the Markov chain reaches an absorbing state, the simulation stops early.
#     seed : None, int, BitGenerator, Generator
#         Seed to initialize a BitGenerator.
#
#     Returns
#     -------
#     current_state : np.ndarray
#         The ending state composition.
#     state_series : list
#         Collection of all state compositions.
#     time_step_series : np.ndarray
#         The time step until the corresponding state occurs (starting from the previous state).
#     """
#     rng = np.random.default_rng(seed)
#
#     current_state = initial_row_vector
#
#     time_step_series = np.zeros(n_steps + 1)
#
#     state_series = [current_state]
#
#     for i in range(n_steps):
#         current_state_index = np.where(current_state > 0)
#         current_state_transition_rates = transition_rate_matrix[current_state_index]
#         current_state_players = current_state[current_state_index]
#         current_state_numbers = np.expand_dims(current_state_players, axis=1)
#         r_j = current_state_numbers * current_state_transition_rates
#         # since in this case (in contrast to direct method of the here applied circumstances) the population of
#         # states is often != 1, the propensities are no longer equal to the rate constants (a_j = k * X_i, where a_j
#         # is propensity of reaction/transition j, k is rate constant (unit s^-1), X_i is population of state i; true
#         # for first order reactions only!)
#         if np.sum(current_state_transition_rates) == 0:
#             break
#
#         k_j = rng.poisson(lam=r_j*tau, size=r_j.shape)
#         # the poisson distribution expresses the probability of a given number of events occurring in a fixed interval
#         # of time (or space) if these events occur with a known constant mean rate and independently of the time since
#         # the last event
#
#         origin_state_change = -np.sum(k_j, axis=1)
#         new_state_change = np.sum(k_j, axis=0)
#         # Note that state change vectors v_ij are not necessary in this case, since each event decreases the origin by 1
#         # and increases the destination by 1
#
#         total_state_change = new_state_change.copy()
#         total_state_change[current_state_index] += origin_state_change
#         current_state = np.sum([current_state, total_state_change], axis=0)  # the tau-leaping approximation
#         time_step_series[i + 1] = tau
#         state_series.append(current_state)
#
#     return current_state, state_series, time_step_series


# def step_size_selection(epsilon, r_j, current_state_players, current_state_index):
#     """
#     The step size tau of the tau leaping algorithm has to be selected carefully to fulfill the leap condition; it
#     states that each tau is small enough to not cause significant change in the value of the transition rates along the
#     subinterval [t, t + tau]. This function employs the algorithm of Cao et al.
#
#     :return:
#     """
#     rate_sums = np.sum(r_j, axis=1)
#     outer_index = np.arange(len(current_state_index[0]))
#     r_j[outer_index, current_state_index] = -rate_sums  # to include the origin state changes in µ_i
#     print("r_i")
#     print(r_j)
#     µ_i = np.sum(r_j, axis=0)
#     print("µ_i")
#     print(µ_i)
#     # sigma_i_squared is not necessary since state change vector squared is the same (all entries are 1 or -1),
#     # therefore the absolute value is the same
#
#     g_i = np.max(r_j, axis=1)
#     print("g_i")
#     print(g_i)
#
#     bounding = epsilon * current_state_players / g_i
#     print("bounding")
#     print(bounding)
#     condition = np.where(bounding > 1, bounding, 1)
#     print("condition")
#     print(condition)
#     all_boundings = np.ones(len(µ_i))
#     all_boundings[current_state_index] = condition
#     print("all boundings")
#     print(all_boundings)
#     relation = all_boundings / np.absolute(µ_i)
#     print("relation")
#     print(relation)
#     tau = np.min(relation)
#     print("tau")
#     print(tau)
#
#     rate_sums = np.sum(multiplied_rate, axis=1)  # auxiliary value µi
#
#     highest_rates = np.max(multiplied_rate, axis=1)
#     bounding = (epsilon * current_state_players) / highest_rates
#     condition = np.where(bounding > 1, bounding, 1)
#     relation = condition / rate_sums
#     tau = np.min(relation)
#     return tau

# import numpy as np
# import src.gillespie_algorithm as ga
# import src.processing as pr
