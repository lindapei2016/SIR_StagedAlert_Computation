###############################################################################

# Based off of Dave 1:1 discussion on 03/04/2024 -- see
#   Notability notes for that date
# Look at 2-stage system with only 1 max allowed lockdown

# Option A (wiggle room)
# - Step 1: find lockdown start time such that peak infections are below capacity
# - Step 2: find lockdown end time such that second peak is not worse than first
# - Note: because the Fujiwara et al. 2022 formula I am using (Formula S8)
#       is for the case when "the maximum appears during the intervention,"
#       it means the second peak is not worse than the first --
#       so Step 2 is unnecessary / redundant
# - Step 3: infer kappas to make this viable
#
# Option B (symmetric staged alert)
# - From Option A, infer the threshold --
#       we will get a more restricted range of kappas

###############################################################################

import numpy as np
import glob
import pandas as pd
from scipy import optimize

import SIR_det_2stages as SIR
import SIR_det_2stages_params as SIR_params

import matplotlib.pyplot as plt

###############################################################################

# Fujiwara et al. 2022 Equation S8 (algebraically rearranged)
# If the maximum number infected occurs during one-shot lockdown,
#   the following equality holds
# Important note -- not all values of i_max, beta0, tau, kappa
#   are valid -- for example, i_max = 0.1, beta0 = 3/10, tau = 10,
#   kappa = 0.8 returns a negative proportion recovered, which is
#   impossible -- this is because kappa is too high and causes
#   the maximum number infected to happen AFTER the single lockdown
def recovered_lockdown_start(i_max, beta0, tau, kappa):
    '''
    :param i_max: float in [0,1] corresponding to
        maximum proportion infected (peak infected)
    :param beta0: transmission rate
    :param tau: average duration of infection
    :param kappa: float in [0,1] corresponding to
        transmission reduction under lockdown
    :return: float in [0,1] corresponding to
        proportion of recovered individuals at the
        start of lockdown i.e. r(t_on) in Fujiwara
        et al. 2022 notation
    '''

    return 1/(beta0*tau) * (1 + np.log(beta0*tau*(1-kappa))) - (1-i_max)*(1-kappa)


# Fujiwara et a. 2022 Equation S8 (original equation)
# # This is just to check the math/derivation for recovered_lockdown_start()
def recovered_lockdown_start_original_equation(r, beta0, tau, kappa):
    '''
    :param r: float in [0,1] corresponding to
        proportion of recovered individuals at the
        start of lockdown i.e. r(t_on)
    :param beta0: see recovered_lockdown_start()
    :param tau: see recovered_lockdown_start()
    :param kappa: see recovered_lockdown_start()
    :return: i_max, float in [0,1] corresponding to
        maximum proportion infected (peak infected)
    '''

    return 1 + r / (1 - kappa) - (1 + np.log(beta0 * tau * (1 - kappa))) / (beta0 * tau * (1 - kappa))


# One version of the equation has I0_val + S0_val simply replaced by 1
#   but this assumes that I0_val + S0_val = 1. This assumption
#   does not hold for our use-case of max infections.
def compute_max_infections(I0_val, S0_val, R0_val):
    return I0_val + S0_val - 1 / R0_val - np.log(R0_val * S0_val) / R0_val


# I_constraint = 0.1 here
# beta0 = 3/10.0
# tau = 10
# S_start = 1-int(1e-3)
I_constraint = SIR_params.I_constraint
beta0 = SIR_params.beta0
tau = SIR_params.tau
S_start = SIR_params.S_start
I_start = SIR_params.I_start
kappa = 0.5

# print(compute_max_infections(I_start, S_start, beta0*tau))

# Formulas corresponding to max infected, case (ii), (S8) in Fujiwara
#   et al. 2022
computation_set_A = True

if computation_set_A:
    r_on_array = []
    s_on_array = []
    i_on_array = []
    t_on_array = []

    i_max_array = []

    # Time at which there is herd immunity under
    #   intervention (reduced transmission)
    s_at_i_lockdown_peak_array = []
    r_at_i_lockdown_peak_array = []
    t_at_i_lockdown_peak_array = []

    # Time at which there is herd immunity under
    #   no intervention
    # i_at_herd_immunity_array = []
    # r_at_herd_immunity_array = []
    # t_at_herd_immunity_array = []

    # I had an initial idea that as an upper bound for t_off, compute
    #   state at which herd immunity (without intervention) is reached
    #   under intervention (decreased transmission)
    # But under intervention, it is possible for I to decrease to almost
    #   0 without reaching herd immunity (without intervention, without
    #   transmission reduction) -- so the simulation runs for way too long
    #   -- therefore, I'm omitting this computation for now

    kappa = 0.5
    for i in np.linspace(I_constraint, 0, num=100, endpoint=False):
        r = recovered_lockdown_start(i, beta0, tau, kappa)
        if r < 0:
            break
        r_on_array.append(r)
        i_max_array.append(i)


    # Simulate "base" system (no policy) to get S, I
    #   and also time t at value of r
    for r in r_on_array:
        problem = SIR.ProblemInstance()
        problem.kappa = 0.5
        problem.stopping_condition = "recovered"
        problem.stopping_condition_recovered_proportion = r
        problem.threshold_up = np.inf
        problem.threshold_down = np.inf
        problem.simulate_policy()
        s_on_array.append(problem.results.S[-1])
        i_on_array.append(problem.results.I[-1])
        t_on_array.append(len(problem.results.S)/SIR_params.ODE_steps)


    # Also, compute state at which herd immunity (WITH) intervention
    #   is reached under intervention, i.e. compute state at
    #   peak infections under intervention
    # Assume that intervention starts at a given i (or recovered,
    #   from above)
    # Here, threshold_down is 0 because we do not want to exit
    #   the intervention while obtaining these quantities
    for ix in np.arange(len(i_on_array)):
        i = i_on_array[ix]
        problem = SIR.ProblemInstance()
        problem.kappa = 0.5
        problem.stopping_condition = "susceptible"
        problem.stopping_condition_susceptible_proportion = \
            1/(beta0 * tau * (1-kappa))
        problem.threshold_up = i
        problem.threshold_down = 0
        problem.simulate_policy()

        s_at_i_lockdown_peak_array.append(problem.results.S[-1])
        r_at_i_lockdown_peak_array.append(problem.results.R[-1])
        t_at_i_lockdown_peak_array.append(len(problem.results.S) / SIR_params.ODE_steps)

        # problem.stopping_condition_susceptible_proportion = \
        #     1/(beta0 * tau)
        # problem.simulate_policy()
        #
        # i_at_herd_immunity_array.append(problem.results.I[-1])
        # r_at_herd_immunity_array.append(problem.results.R[-1])
        # t_at_herd_immunity_array.append(len(problem.results.I)/SIR_params.ODE_steps)


# So we have a set of (feasible) interventions that
#   start at a given time / given proportion infected
# Simulate all those interventions
# Save the sample paths in the following dictionary
# Then we can calculate what the maximum is after
#   stopping at a given timepoint
# Key is value in t_on_array
S_sample_paths_dict = {}
I_sample_paths_dict = {}

for ix in np.arange(len(i_on_array)):

    i = i_on_array[ix]

    problem = SIR.ProblemInstance()
    problem.kappa = 0.5
    # problem.stopping_condition = "susceptible"
    # problem.stopping_condition_susceptible_proportion = \
    #     1/(beta0 * tau * (1-problem.kappa))
    problem.threshold_up = i
    problem.threshold_down = 0
    problem.simulate_policy()

    S_sample_paths_dict[t_on_array[ix]] = problem.results.S[::int(SIR_params.ODE_steps)]
    I_sample_paths_dict[t_on_array[ix]] = problem.results.I[::int(SIR_params.ODE_steps)]


# Maximum number of infections after lockdown, with keys being
#   day of lockdown start
i_max_after_lockdown_dict = {}

for t in t_on_array:

    i_max_after_lockdown_array = []

    for day in np.linspace(50, 150, 10, endpoint=True):

        if len(I_sample_paths_dict[t]) > day:
            i_max_after_lockdown_array.append(
                compute_max_infections(I_sample_paths_dict[t][int(day)], S_sample_paths_dict[t][int(day)], beta0*tau))
        else:
            break

    i_max_after_lockdown_dict[t] = i_max_after_lockdown_array

breakpoint()

# plt.clf()
# plt.plot(t_on_array, np.array(i_max_array)/SIR_params.I_constraint, label="i_max / i_constraint")
# plt.plot(t_on_array, s_on_array, label="s at lockdown")
# plt.plot(t_on_array, r_on_array, label="r at lockdown")
# plt.plot(t_on_array, np.array(i_on_array)/SIR_params.I_constraint, label="i / i_constraint at lockdown")
# plt.ylabel("Proportion")
# plt.xlabel("Simulation time t of lockdown")
# plt.legend()
# plt.show()


breakpoint()