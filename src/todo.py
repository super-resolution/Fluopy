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
