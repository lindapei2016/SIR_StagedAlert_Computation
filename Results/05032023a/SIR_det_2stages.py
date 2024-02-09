###############################################################################

# This program "simulates" a deterministic SIR model with a 2-stage staged
#   alert policy by advancing a system of ODEs in discretized timeODE_steps.
#   Each stage is triggered when the proportion of infected individuals
#   reaches a pre-specified threshold, and each stage has an immediate
#   and deterministic transmission reduction that mimics a real-world
#   policy intervention.
# Linda Pei 2023

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

# import scipy as sp

import time

###############################################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

eps = 1e-6


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

    def update(self, cost, num_lockdowns, x0, x1, S, I):
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
        beta = np.zeros(self.time_end)
        x0 = np.zeros(self.time_end)
        x1 = np.zeros(self.time_end)

        S[0] = self.S_start
        I[0] = self.I_start

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

        for t in range(1, self.time_end):

            beta[t] = beta0 * (1 - kappa * x1[t - 1])

            S[t] = S[t - 1] - (beta[t] * S[t - 1] * I[t - 1]) * 1 / ODE_steps

            I[t] = I[t - 1] + (beta[t] * S[t - 1] * I[t - 1] - I[t - 1] / tau) * \
                   1 / ODE_steps

            # If previous stage was stage 0, check conditions for moving to stage 1
            if x0[t-1] == 1:
                if I[t] >= threshold_up:
                    # If lockdown budget left, move to stage 1 (lockdown)
                    if num_lockdowns < self.max_lockdowns_allowed:
                        num_lockdowns += 1
                        x1[t] = 1
                    # If no lockdown budget left, remain in stage 0
                    else:
                        x0[t] = x0[t-1]
                else:
                    x0[t] = x0[t-1]
            # Else if previous stage was stage 1, check conditions for moving to stage 0
            elif x1[t-1] == 1:
                if I[t] < threshold_down:
                    # If infections decreasing, move to stage 0
                    # This condition handles the case where
                    #   lockdown_up < lockdown_down -- in this case,
                    #   we only want to leave lockdown if infections
                    #   are decreasing -- otherwise, we would leave
                    #   lockdown as soon as we entered lockdown
                    if I[t] < I[t-1]:
                        x0[t] = 1
                    # If infections still increasing, remain in stage 1
                    else:
                        x1[t] = x1[t-1]
                else:
                    x1[t] = x1[t-1]

            if I[t] >= I_constraint:
                feasible = False

                if not self.full_output:
                    x0 = x0[:t + 1]
                    x1 = x1[:t + 1]
                    S = S[:t + 1]
                    I = I[:t + 1]
                    break

        if feasible:
            cost = self.cost_func_linear(x0, x1)

        self.results.update(cost, num_lockdowns, x0, x1, S, I)

    def simulate_many_policies(self, policies):

        '''
        :param policies: [array] of 2-tuples, each of which have a first value
            less than or equal to its second value -- e.g., output from
            thresholds_generator
        :param full_output: [Boolean] whether to output x0, x1, S, I
            in addition to cost
        :return: cost_history: [array] of cost scalars -- ith value corresponds
                to cost of ith policy in policies
            total_x0_history: [array] of nonnegative scalars -- ith value
                corresponds to number of days ith policy in policies spends
                in stage 1
            total_x1_history: [array] of nonnegative scalars -- same as
                total_x0_history but for number of days in stage 2
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

        if not self.full_output:
            cost_history, num_lockdowns_history = self.simulate_many_policies(policies)
            best = np.argmin(cost_history)
            np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
            np.savetxt(filename_prefix + "_num_lockdowns_history.csv", num_lockdowns_history, delimiter=",")
            return filename_prefix, policies[best], cost_history[best], num_lockdowns_history[best]

        else:
            cost_history, num_lockdowns_history, total_x0_history, total_x1_history = self.simulate_many_policies(policies)
            best = np.argmin(cost_history)
            np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
            np.savetxt(filename_prefix + "_num_lockdowns_history.csv", num_lockdowns_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x0_history.csv", total_x0_history, delimiter=",")
            np.savetxt(filename_prefix + "_total_x1_history.csv", total_x1_history, delimiter=",")
            return filename_prefix, policies[best], cost_history[best], num_lockdowns_history[best], total_x0_history[best], total_x1_history[best]


