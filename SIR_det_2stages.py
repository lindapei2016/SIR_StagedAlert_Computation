###############################################################################

# This program "simulates" a deterministic SIR model with a 2-stage staged
#   alert policy by advancing a system of ODEs in discretized timeODE_steps.
#   Each stage is triggered when the proportion of infected individuals
#   reaches a pre-specified threshold, and each stage has an immediate
#   and deterministic transmission reduction that mimics a real-world
#   policy intervention.

# Note -- the discretization is a little bit goofy in the sense that
#   you can lockdown for a partial day (during one of the ODE timesteps during
#   a given "day") -- might be good to fix this
#  Also might be good to only output S and I on the actual days,
#   not for all of the ODE timesteps

###############################################################################

# Imports
import SIR_det_2stages_params as SIR_params

from mpi4py import MPI
import numpy as np
import itertools
import math

# Flush printing
import sys

sys.stdout.flush()

# For exploratory plotting -- turn this off for cluster
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")

import scipy as sp
import time

###############################################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

eps = 1e-6

###############################################################################


class Results:
    '''
    Container for results of simulating staged-alert policy.
    '''

    def __init__(self):
        '''
        :param cost: [scalar] cost of staged-alert policy over
            simulation timeframe
        :param num_lockdowns: [int] number of lockdowns
        :param x0: [array of 0-1s] jth element indicates whether
            system was in stage 0 ("normal") at simulation time j
        :param x1: [array of 0-1s] jth element indicates whether
            system was in stage 1 (lockdown) at simulation time j
        :param S: [array of reals] jth element is proportion
            of population in Susceptible compartment j
        :param I: [array of reals] jth element is proportion
            of population in Infected compartment at time j
        :return: [None]
        '''
        self.cost = np.inf
        self.num_lockdowns = 0
        self.x0 = np.array([])
        self.x1 = np.array([])
        self.S = np.array([])
        self.I = np.array([])
        self.R = np.array([])

    def __call__(self):
        '''
        Prints instance attributes
        :return: [None]
        '''
        print("Cost is " + str(self.cost))
        print("Number of lockdowns: " + str(self.num_lockdowns))
        print("x0: " + str(self.x0))
        print("x1: " + str(self.x1))
        print("S: " + str(self.S))
        print("I: " + str(self.I))
        print("R: " + str(self.R))

    def update(self, cost, num_lockdowns, x0, x1, S, I, R):
        '''
        See __init__ parameters for documentation.
        Updates all attributes according to passed values.
        :return: [None]
        '''
        self.cost = cost
        self.num_lockdowns = num_lockdowns
        self.x0 = x0
        self.x1 = x1
        self.S = S
        self.I = I
        self.R = R

    def clear(self):
        '''
        Resets all attributes to default values in place.
        :return: [None]
        '''
        self.cost = np.inf
        self.num_lockdowns = 0
        self.x0 = np.array([])
        self.x1 = np.array([])
        self.S = np.array([])
        self.I = np.array([])
        self.R = np.array([])


class ProblemInstance:
    '''
    Class for simulating and optimizing staged alert policies.
    '''

    def __init__(self):
        self.time_end = SIR_params.time_end
        self.ODE_steps = SIR_params.ODE_steps
        self.I_constraint = SIR_params.I_constraint
        self.S_start = SIR_params.S_start
        self.I_start = SIR_params.I_start
        self.beta0 = SIR_params.beta0
        self.tau = SIR_params.tau
        self.kappa = SIR_params.kappa
        self.cost0 = SIR_params.cost0
        self.cost1 = SIR_params.cost1
        self.threshold_up = SIR_params.threshold_up
        self.threshold_down = SIR_params.threshold_down
        self.grid_grain = SIR_params.grid_grain
        self.inertia = SIR_params.inertia
        self.stopping_condition = SIR_params.stopping_condition
        self.max_lockdowns_allowed = SIR_params.max_lockdowns_allowed
        self.full_output = SIR_params.full_output

        self.results = Results()

    @staticmethod
    def thresholds_generator(threshold_up_info, threshold_down_info, symmetric=True):
        """
        :param threshold_up_info: [3-tuple] with elements corresponding to
            start point, end point, and step size (all must be integers)
            for candidate values for threshold_up
        :param threshold_down_info: see threshold_down
        :param symmetric: [Boolean] determines whether threshold_up
            is enforced to be the same as threshold_down
        :return: [array] of distinct 2-tuples representing all possible
            combos generated from threshold_up_info and threshold_down_info, where
            each 2-tuple's second value is less than its first value.
        """

        # Create an array (grid) of potential values for each threshold
        threshold_up_options = np.arange(threshold_up_info[0],
                                         threshold_up_info[1],
                                         threshold_up_info[2])
        threshold_down_options = np.arange(threshold_down_info[0],
                                           threshold_down_info[1],
                                           threshold_down_info[2])

        if symmetric == True:
            threshold_candidates_feasible = []

            for combo in threshold_up_options:
                threshold_candidates_feasible.append((combo, combo))

        else:
            # Using Cartesian products, create a list of 2-tuple combos
            threshold_options = [threshold_up_options, threshold_down_options]
            threshold_candidates = []
            for combo in itertools.product(*threshold_options):
                threshold_candidates.append(combo)

            threshold_candidates_feasible = threshold_candidates

            # Eliminate 2-tuples that do not satisfy monotonicity constraint
            # However, ties in thresholds are allowed
            # threshold_up cannot be smaller than threshold_down
            # threshold_candidates_feasible = []
            # for combo in threshold_candidates:
            #     if np.all(np.diff(combo) <= 0):
            #        threshold_candidates_feasible.append(combo)

        return threshold_candidates_feasible

    def cost_func_linear(self, x0_vector, x1_vector):

        return self.cost0 * np.sum(x0_vector) + self.cost1 * np.sum(x1_vector)

    def simulate_policy(self):

        self.results.clear()

        num_lockdowns = 0

        feasible = True
        cost = np.inf

        S = np.zeros(self.time_end)
        I = np.zeros(self.time_end)
        R = np.zeros(self.time_end)
        beta = np.zeros(self.time_end)
        x0 = np.zeros(self.time_end)
        x1 = np.zeros(self.time_end)

        S[0] = self.S_start
        I[0] = self.I_start
        R[0] = 0

        # Start in stage 0
        x0[0] = 1
        x1[0] = 0

        I_constraint = self.I_constraint

        threshold_up = self.threshold_up
        threshold_down = self.threshold_down

        beta0 = self.beta0
        tau = self.tau

        kappa = self.kappa

        ODE_steps = self.ODE_steps

        inertia = self.inertia

        # We want the ability to make the first stage change whenever
        # e.g. if inertia = 1e3, do NOT want the inertia constraint to prevent
        #   moving out of stage 0 until simulation time 1e3
        # So only consider the inertia constraint after the first stage change
        # We only have to check inertia_constraint_active when in stage 0
        #   and moving to a different stage. Since we start in stage 0, if we
        #   are in stage 1 (or 2), we have already made the first stage change,
        #   so the inertia_constraint_active is True.
        inertia_constraint_active = False

        stopping_condition = self.stopping_condition

        time_since_last_change = 0

        t = 0

        # while S > 1/R0 (herd immunity not yet reached)
        for t in range(1, self.time_end):

            time_since_last_change += 1

            beta[t] = beta0 * (1 - kappa * x1[t - 1])

            S[t] = S[t - 1] - (beta[t] * S[t - 1] * I[t - 1]) * 1 / ODE_steps

            I[t] = I[t - 1] + (beta[t] * S[t - 1] * I[t - 1] - I[t - 1] / tau) * \
                   1 / ODE_steps

            R[t] = R[t-1] + (I[t - 1] / tau) * (1 / ODE_steps)

            # If previous stage was stage 0, check conditions for moving to stage 1
            if x0[t - 1] == 1:
                if I[t] >= threshold_up:
                    if inertia_constraint_active:
                        # If lockdown budget left, move to stage 1 (lockdown)
                        if num_lockdowns < self.max_lockdowns_allowed and time_since_last_change >= inertia:
                            num_lockdowns += 1
                            x1[t] = 1
                            time_since_last_change = 0
                        # If no lockdown budget left, remain in stage 0
                        # Or if changing stages not allowed due to inertia, remain in stage 0
                        else:
                            x0[t] = x0[t - 1]
                    else:
                        # We skip checking if num_lockdowns < self.max_lockdowns_allowed because
                        #   this must be True for the first stage change / lockdown,
                        #   assuming the non-trivial specification self.max_lockdowns >= 1
                        inertia_constraint_active = True
                        num_lockdowns += 1
                        x1[t] = 1
                        time_since_last_change = 0
                else:
                    x0[t] = x0[t - 1]
            # Else if previous stage was stage 1, check conditions for moving to stage 0
            elif x1[t - 1] == 1:
                if I[t] < threshold_down:
                    # If infections decreasing, move to stage 0
                    # This condition handles the case where
                    #   lockdown_up < lockdown_down -- in this case,
                    #   we only want to leave lockdown if infections
                    #   are decreasing -- otherwise, we would leave
                    #   lockdown as soon as we entered lockdown
                    if I[t] < I[t - 1] and time_since_last_change >= inertia:
                        x0[t] = 1
                        time_since_last_change = 0
                    # If infections still increasing or inertia constraint, remain in stage 1
                    else:
                        x1[t] = x1[t - 1]
                else:
                    x1[t] = x1[t - 1]

            if I[t] >= I_constraint:
                feasible = False

                if not self.full_output:
                    break

            # If we are out of lockdown and have reached herd immunity
            #   with respect to beta0 (unmitigated transmission rate),
            #   we will not return to lockdown so we will not accumulate
            #   additional costs -- can early terminate here for
            #   optimization purposes
            if stopping_condition == "herd_immunity":
                if x0[t] == 1 and S[t] < 1 / (self.beta0 * self.tau):
                    break

        x0 = x0[:t + 1]
        x1 = x1[:t + 1]
        S = S[:t + 1]
        I = I[:t + 1]
        R = R[:t + 1]

        if feasible:
            cost = self.cost_func_linear(x0, x1)

        self.results.update(cost, num_lockdowns, x0, x1, S, I, R)

    def simulate_many_policies(self, policies):
        '''
        Calls simulate_policy as subroutine for each policy in policies
        :param policies: [array] of 2-tuples, each of which have a first value
            less than or equal to its second value -- e.g., output from
            thresholds_generator
        :param full_output: [Boolean] whether to output x0, x1, S, I
            in addition to cost
        :return: cost_history: [array] of cost scalars -- ith value corresponds
                to cost of ith policy in policies
            total_x0_history: [array] of nonnegative scalars -- ith value
                corresponds to number of days ith policy in policies spends
                in stage 0
            total_x1_history: [array] of nonnegative scalars -- same as
                total_x0_history but for number of days in stage 1
            all arrays have length equal to length of policies parameter
        '''

        cost_history = []
        num_lockdowns_history = []
        total_x0_history = []
        total_x1_history = []

        num_policies = len(policies)

        for i in range(num_policies):
            thresholds = policies[i]
            self.threshold_up = thresholds[0]
            self.threshold_down = thresholds[1]

            self.simulate_policy()
            cost_history.append(self.results.cost)
            num_lockdowns_history.append(self.results.num_lockdowns)

            if self.full_output:
                total_x0_history.append(np.sum(self.results.x0))
                total_x1_history.append(np.sum(self.results.x1))

        if not self.full_output:
            return cost_history, num_lockdowns_history
        else:
            return cost_history, num_lockdowns_history, total_x0_history, total_x1_history

    def find_optimum(self, policies, filename_prefix):
        '''
        Calls simulate_many_policies as subroutine on set of policies
            and identifies best policy (with min cost)
        Can print this function to write optimization summary
            as line is .txt/.csv file during cluster runs
        :param policies:
        :param filename_prefix: [str] common prefix for output files
        :return: [str] common prefix for output files,
            [int] ID of best policy (with min cost)
            [int] cost of best policy
            [int] number of lockdowns in best policy
        '''

        if not self.full_output:
            cost_history, num_lockdowns_history = self.simulate_many_policies(policies)
            best = np.argmin(cost_history)
            np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
            np.savetxt(filename_prefix + "_num_lockdowns_history.csv", num_lockdowns_history, delimiter=",")
            return filename_prefix, policies[best], cost_history[best], num_lockdowns_history[best]

        else:

            cost_history, num_lockdowns_history, total_x0_history, total_x1_history = self.simulate_many_policies(
                policies)
            best = np.argmin(cost_history)
            np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
            np.savetxt(filename_prefix + "_num_lockdowns_history.csv", num_lockdowns_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x0_history.csv", total_x0_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x1_history.csv", total_x1_history, delimiter=",")
            return filename_prefix, policies[best], cost_history[best], \
                   num_lockdowns_history[best], total_x0_history[best], total_x1_history[best]


###############################################################################

# Check analytically to see if a threshold is feasible
#   (for two stages, a "normal" stage and a lockdown stage).
# Use a root-finding method to find the value of S

# One version of the equation has I0_val + S0_val simply replaced by 1
#   but this assumes that I0_val + S0_val = 1. This assumption
#   does not hold for our use-case of max infections.
def compute_max_infections(I0_val, S0_val, R0_val):
    return I0_val + S0_val - 1 / R0_val - np.log(R0_val * S0_val) / R0_val


def build_sol_curve_eq(I0_val, I_val, S0_val, R0_val):
    def sol_curve_eq(S_val):
        '''
        Solution curve is I = I0 + S0 - S + log(S/S0)/R0
            (see Hethcote 2000 "The Mathematics of
            Infectious Diseases").
        Solution curve equation: for a given proportion
            susceptible, returns difference between
            pre-specified proportion infected and
            actual proportion infected when proportion
            susceptible equals
        Want to find root of this equation to find
            S that corresponds with I
        :param S_val: [scalar in [0,1]] proportion susceptible.
        :return: [scalar] actual proportion infected when
            proportion susceptible equals S_val minus I_val
        '''
        return I0_val + S0_val - I_val - S_val + np.log(S_val / S0_val) / R0_val

    return sol_curve_eq


###############################################################################

# problem = ProblemInstance()
# problem.inertia = 0
# problem.kappa = 0.5
# problem.threshold_up = 0.067
# problem.threshold_down = 0.09
# problem.full_output = True
# problem.simulate_policy()
#
# breakpoint()
#
# sol_curve_eq = build_sol_curve_eq(problem.I_start,
#                                   problem.threshold_up,
#                                   problem.S_start,
#                                   problem.beta0 * problem.tau)
#
# S_val = np.max(sp.optimize.newton(sol_curve_eq, [eps, 1]))
#
# sol_curve_eq = build_sol_curve_eq(problem.threshold_up,
#                                   problem.threshold_down,
#                                   S_val,
#                                   problem.beta0 * (1 - problem.kappa) * problem.tau)
#
# S_val = np.min(sp.optimize.newton(sol_curve_eq, [eps, S_val, 1]))
#
# print(compute_max_infections(problem.threshold_down,
#                              S_val,
#                              problem.beta0 * problem.tau))
#
# problem.simulate_policy()
# I_descending = np.sort(problem.results.I)[::-1]
# print(I_descending)
#
# breakpoint()

# According to formula, need beta*tau <= 1.693 roughly to have peak infections
#   less than 0.1 -- for beta = beta0 * (1-kappa), beta0 = 0.3 and tau = 10,
#   this means we need kappa >= 0.436 roughly.

# For peak infections <= 0.2, need beta*tau <= 2.27 --> kappa >= 0.243
# For peak infections <= 0.4, need beta*tau <= 3.95 --> any kappa will do
