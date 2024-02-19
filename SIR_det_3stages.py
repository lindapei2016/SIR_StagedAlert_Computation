###############################################################################

# This program "simulates" a deterministic SIR model with a 3-stage staged
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

# For simplicity, we have symmetric thresholds for the 3-stage case.
#   The 2-stage code allows asymmetric thresholds.
# As a result of different structure, the method thresholds_generator() is
#   different for the 3-stage case compared to the 2-stage case.

# For simplicity, we have an unlimited number of lockdowns.
#   Having a maximum number of lockdowns for a 3-stage system
#   probably makes less sense.

# Stage 0 (x0) --> "low" -- no transmission reduction
# Stage 1 (x1) --> "medium"
# Stage 2 (x2) --> "high"

###############################################################################

# Imports
import SIR_det_3stages_params as SIR_params

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
        :param num_lockdowns: [int] number of lockdowns (referring
            to strictest stage, stage 2)
        :param x0: [array of 0-1s] jth element indicates whether
            system was in stage 0 ("normal") at simulation time j
        :param x1: [array of 0-1s] jth element indicates whether
            system was in stage 1 (medium stage) at simulation time j
        :param x2: [array of 0-1s] jth element indicates whether
            system was in stage 2 (lockdown) at simulation time j
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
        self.x2 = np.array([])
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
        print("x2: " + str(self.x2))
        print("S: " + str(self.S))
        print("I: " + str(self.I))
        print("R: " + str(self.R))

    def update(self, cost, num_lockdowns, x0, x1, x2, S, I, R):
        '''
        See __init__ parameters for documentation.
        Updates all attributes according to passed values.
        :return: [None]
        '''
        self.cost = cost
        self.num_lockdowns = num_lockdowns
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
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
        self.x2 = np.array([])
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
        self.medium_kappa = SIR_params.medium_kappa
        self.high_kappa = SIR_params.high_kappa
        self.low_stage_cost = SIR_params.low_stage_cost
        self.medium_stage_cost = SIR_params.medium_stage_cost
        self.high_stage_cost = SIR_params.high_stage_cost
        self.medium_threshold = SIR_params.medium_threshold
        self.high_threshold = SIR_params.high_threshold
        self.grid_grain = SIR_params.grid_grain
        self.inertia = SIR_params.inertia
        self.full_output = SIR_params.full_output
        self.stopping_condition = SIR_params.stopping_condition
        self.results = Results()

    @staticmethod
    def thresholds_generator(medium_threshold_info, high_threshold_info):
        """
        :param medium_threshold_info: [3-tuple] with elements corresponding to
            start point, end point, and step size (all must be integers)
            for candidate values for medium_threshold
        :param high_threshold_info: see medium_threshold_info
        :return: [array] of distinct 2-tuples representing all possible
            combos generated from medium_threshold_info and high_threshold_info, where
            each 2-tuple's second value is less than its first value.
        """

        # Create an array (grid) of potential values for each threshold
        medium_threshold_options = np.arange(medium_threshold_info[0],
                                         medium_threshold_info[1],
                                         medium_threshold_info[2])
        high_threshold_options = np.arange(high_threshold_info[0],
                                           high_threshold_info[1],
                                           high_threshold_info[2])

        # Using Cartesian products, create a list of 2-tuple combos
        threshold_options = (medium_threshold_options, high_threshold_options)
        threshold_candidates = []
        for combo in itertools.product(*threshold_options):
            if combo[0] <= combo[1]:
                threshold_candidates.append(combo)

        return threshold_candidates

    def cost_func_linear(self, x0_vector, x1_vector, x2_vector):

        return self.low_stage_cost * np.sum(x0_vector) + \
               self.medium_stage_cost * np.sum(x1_vector) + \
               self.high_stage_cost * np.sum(x2_vector)

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
        x2 = np.zeros(self.time_end)

        S[0] = self.S_start
        I[0] = self.I_start
        R[0] = 0

        # Start in stage 0
        x0[0] = 1
        x1[0] = 0
        x2[0] = 0

        I_constraint = self.I_constraint

        medium_threshold = self.medium_threshold
        high_threshold = self.high_threshold

        beta0 = self.beta0
        tau = self.tau

        medium_kappa = self.medium_kappa
        high_kappa = self.high_kappa

        ODE_steps = self.ODE_steps

        inertia = self.inertia

        # We want the ability to make the first stage change whenever
        # e.g. if inertia = 1e3, do NOT want the inertia constraint to prevent
        #   moving out of stage 0 until simulation time 1e3
        # So only consider the inertia constraint after the first stage change
        # Similar to logic from 2-stage system, if we are in the stage 0
        #   infections level, and at the previous timepoint we were not in stage 0,
        #   the first stage change has already occurred because we start the
        #   simulation in stage 0. Therefore, we do not need to check
        #   inertia_constraint_active, because it IS active.
        inertia_constraint_active = False

        stopping_condition = self.stopping_condition

        time_since_last_change = 0

        t = 0

        # If stopping_condition == "herd_immunity",
        #   this loops while S > 1/R0 (herd immunity not yet reached)
        for t in range(1, self.time_end):

            # Tracking for inertia requirement
            time_since_last_change += 1

            # ~~~~~~~~~~~~~~~~~~~~~~~~ #
            # Logic for SIR equations
            # ~~~~~~~~~~~~~~~~~~~~~~~~ #

            # Update transmission rate depending on stage
            beta[t] = beta0 * (1 - medium_kappa * x1[t - 1] - high_kappa * x2[t - 1])

            S[t] = S[t - 1] - (beta[t] * S[t - 1] * I[t - 1]) * 1 / ODE_steps

            I[t] = I[t - 1] + (beta[t] * S[t - 1] * I[t - 1] - I[t - 1] / tau) * \
                   (1 / ODE_steps)

            R[t] = R[t - 1] + (I[t - 1] / tau) * (1 / ODE_steps)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # Logic for 3-stage system:
            #   no max lockdowns, symmetric thresholds, inertia
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # Decision tree:
            # Check I[t] versus threshold (high, medium, low/none)
            # Check stage of previous timepoint
            # Check if inertia requirement allows changing of stages

            if I[t] >= high_threshold:
                # If stage of previous timepoint was also 2,
                #   stay in this stage
                if x2[t - 1] == 1:
                    x2[t] = 1
                else:
                    if inertia_constraint_active:
                        if time_since_last_change >= inertia:
                            x2[t] = 1
                            time_since_last_change = 0 # the stage has just been changed
                            num_lockdowns += 1
                        else: # inertia requires staying in the previous stage
                            x0[t] = x0[t - 1]
                            x1[t] = x1[t - 1]
                            x2[t] = x2[t - 1]
                    else:
                        inertia_constraint_active = True
                        x2[t] = 1
                        time_since_last_change = 0
                        num_lockdowns += 1
            elif I[t] >= medium_threshold:
                if x1[t - 1] == 1:
                    x1[t] = 1
                else:
                    if inertia_constraint_active:
                        if time_since_last_change >= inertia:
                            x1[t] = 1
                            time_since_last_change = 0
                        else:
                            x0[t] = x0[t - 1]
                            x1[t] = x1[t - 1]
                            x2[t] = x2[t - 1]
                    else:
                        inertia_constraint_active = True
                        x1[t] = 1
                        time_since_last_change = 0
                        num_lockdowns += 1
            else: # low stage
                if x0[t - 1] == 1:
                    x0[t] = 1
                else:
                    # See comment when initializing inertia_constraint_active
                    # At this point, inertia_constraint_active must be True,
                    #   so we do not need to check it.
                    if time_since_last_change >= inertia:
                        x0[t] = 1
                        time_since_last_change = 0
                    else:
                        x0[t] = x0[t - 1]
                        x1[t] = x1[t - 1]
                        x2[t] = x2[t - 1]

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
        x2 = x2[:t + 1]
        S = S[:t + 1]
        I = I[:t + 1]
        R = R[:t + 1]

        if feasible:
            cost = self.cost_func_linear(x0, x1, x2)

        self.results.update(cost, num_lockdowns, x0, x1, x2, S, I, R)

    def simulate_many_policies(self, policies):
        '''
        Calls simulate_policy as subroutine for each policy in policies
        :param policies: [array] of 2-tuples, each of which have a first value
            less than or equal to its second value -- e.g., output from
            thresholds_generator
        :param full_output: [Boolean] whether to output x0, x1, x2, S, I
            in addition to cost
        :return: cost_history: [array] of cost scalars -- ith value corresponds
                to cost of ith policy in policies
            total_x0_history: [array] of nonnegative scalars -- ith value
                corresponds to number of days ith policy in policies spends
                in stage 0
            total_x1_history: [array] of nonnegative scalars -- same as
                total_x0_history but for number of days in stage 1
            total_x2_history: [array] of nonnegative scalars -- same as
                total_x0_history but for number of days in stage 2
            all arrays have length equal to length of policies parameter
        '''

        cost_history = []
        num_lockdowns_history = []
        total_x0_history = []
        total_x1_history = []
        total_x2_history = []

        num_policies = len(policies)

        for i in range(num_policies):
            thresholds = policies[i]
            self.medium_threshold = thresholds[0]
            self.high_threshold = thresholds[1]

            self.simulate_policy()
            cost_history.append(self.results.cost)
            num_lockdowns_history.append(self.results.num_lockdowns)

            if self.full_output:
                total_x0_history.append(np.sum(self.results.x0))
                total_x1_history.append(np.sum(self.results.x1))
                total_x2_history.append(np.sum(self.results.x2))

        if not self.full_output:
            return cost_history, num_lockdowns_history
        else:
            return cost_history, num_lockdowns_history, total_x0_history, total_x1_history, total_x2_history

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

            cost_history, num_lockdowns_history, total_x0_history, total_x1_history, total_x2_history = \
                self.simulate_many_policies(policies)
            best = np.argmin(cost_history)
            np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
            np.savetxt(filename_prefix + "_num_lockdowns_history.csv", num_lockdowns_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x0_history.csv", total_x0_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x1_history.csv", total_x1_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x2_history.csv", total_x2_history, delimiter=",")
            return filename_prefix, policies[best], cost_history[best], \
                   num_lockdowns_history[best], total_x0_history[best], total_x1_history[best], total_x2_history[best]


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