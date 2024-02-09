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
# import matplotlib.pyplot as plt
# import matplotlib
import math

# matplotlib.use("TkAgg")

###############################################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

eps = 1e-6

class ProblemInstance:

    def __init__(self):
        self.time_end = SIR_params.time_end
        self.ODE_steps = SIR_params.ODE_steps
        self.i_constraint = SIR_params.i_constraint
        self.s_start = SIR_params.s_start
        self.i_start = SIR_params.i_start
        self.beta0 = SIR_params.beta0
        self.tau = SIR_params.tau
        self.kappa1 = SIR_params.kappa1
        self.kappa2 = SIR_params.kappa2
        self.cost1 = SIR_params.cost1
        self.cost2 = SIR_params.cost2
        self.threshold1 = SIR_params.cost1
        self.threshold2 = SIR_params.cost2
        self.grid_grain = SIR_params.grid_grain

    @staticmethod
    def thresholds_generator(stage1_info, stage2_info):

        """
        :param stage1_info: [3-tuple] with elements corresponding to
            start point, end point, and step size (all must be integers)
            for candidate values for stage 2
        :param stage2_info: see stage1_info
        :return: [array] of distinct 2-tuples representing all possible
            combos generated from stage1_info and stage2_info, where
            each 2-tuple's first value is less than its second value.
        """

        # Create an array (grid) of potential thresholds for each stage
        stage1_options = np.arange(stage1_info[0], stage1_info[1], stage1_info[2])
        stage2_options = np.arange(stage2_info[0], stage2_info[1], stage2_info[2])

        # Using Cartesian products, create a list of 5-tuple combos
        stage_options = [stage1_options, stage2_options]
        candidate_feasible_combos = []
        for combo in itertools.product(*stage_options):
            candidate_feasible_combos.append(combo)

        # Eliminate 5-tuples that do not satisfy monotonicity constraint
        # However, ties in thresholds are allowed
        feasible_combos = []
        for combo in candidate_feasible_combos:
            if np.all(np.diff(combo) >= 0):
                feasible_combos.append(combo)

        return feasible_combos

    def cost_func_linear(self, x1_vector, x2_vector):

        return self.cost1 * np.sum(x1_vector) + self.cost2 * np.sum(x2_vector)

    def simulate_policy(self, full_output=False):

        feasible = True
        cost = np.inf

        s = np.zeros(self.time_end)
        i = np.zeros(self.time_end)
        beta = np.zeros(self.time_end)
        x1 = np.zeros(self.time_end)
        x2 = np.zeros(self.time_end)

        s[0] = self.s_start
        i[0] = self.i_start

        x1[0] = 0
        x2[0] = 0

        i_constraint = self.i_constraint

        threshold1 = self.threshold1
        threshold2 = self.threshold2

        beta0 = self.beta0
        tau = self.tau

        kappa1 = self.kappa1
        kappa2 = self.kappa2

        ODE_steps = self.ODE_steps

        for t in range(1, self.time_end):

            beta[t] = beta0 * (1 - kappa1 * x1[t - 1] - kappa2 * x2[t - 1])

            s[t] = s[t - 1] - (beta[t] * s[t - 1] * i[t - 1]) * 1 / ODE_steps

            i[t] = i[t - 1] + (beta[t] * s[t - 1] * i[t - 1] - i[t - 1] / tau) * \
                   1 / ODE_steps

            if i[t] >= threshold2:
                x2[t] = 1
            elif threshold2 > i[t] >= threshold1:
                x1[t] = 1

            if i[t] >= i_constraint:
                feasible = False

        if feasible:
            cost = self.cost_func_linear(x1, x2)

        if full_output == False:
            return cost, x1, x2
        else:
            return cost, x1, x2, s, i

    def simulate_many_policies(self, policies):

        '''
        :param policies: [array] of 2-tuples, each of which have a first value
            less than or equal to its second value -- e.g., output from
            thresholds_generator
        :return: cost_history: [array] of cost scalars -- ith value corresponds
                to cost of ith policy in policies
            total_x1_history: [array] of nonnegative scalars -- ith value
                corresponds to number of days ith policy in policies spends
                in stage 1
            total_x2_history: [array] of nonnegative scalars -- same as
                total_x1_history but for number of days in stage 2
            all arrays have length equal to length of policies parameter
        '''

        cost_history = []
        total_x1_history = []
        total_x2_history = []

        num_policies = len(policies)

        for i in range(num_policies):
            thresholds = policies[i]
            self.threshold1 = thresholds[0]
            self.threshold2 = thresholds[1]

            cost, x1, x2 = self.simulate_policy()

            cost_history.append(cost)
            total_x1_history.append(np.sum(x1))
            total_x2_history.append(np.sum(x2))

        return cost_history, total_x1_history, total_x2_history

    def find_optimum(self, policies, filename_prefix):

        cost_history, total_x1_history, total_x2_history = self.simulate_many_policies(policies)
        best = np.argmin(cost_history)
        np.savetxt(filename_prefix + "_cost_history.csv", cost_history, delimiter=",")
        np.savetxt(filename_prefix + "_total_x1_history.csv", total_x1_history, delimiter=",")
        np.savetxt(filename_prefix + "_total_x2_history.csv", total_x2_history, delimiter=",")
        return policies[best], cost_history[best], total_x1_history[best], total_x2_history[best]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = ProblemInstance()
policies = ProblemInstance.thresholds_generator((0, 1, 1),
                                                (0, problem.i_constraint + eps, problem.grid_grain))

problem.kappa2 = 1/100 * (rank + 1)
print(problem.find_optimum(policies, str(int(problem.kappa2*100))))