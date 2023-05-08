###############################################################################

# Script B finds the optimal threshold value for an asymmetric threshold
#   given kappa (transmission reduction under stage 1, i.e., lockdown).
# We use 100 processors, and processor with rank r uses brute force
#   computation to find the optimal threshold_up and threshold_down values
#   for kappa = r/100.

###############################################################################

import SIR_det_2stages as SIR

from mpi4py import MPI
import numpy as np
import itertools
import math

# Flush printing
import sys

sys.stdout.flush()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem_type = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

eps = 1e-6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Put on cluster to find optimal threshold value given kappa,
#   where optimal means minimum number of days in stage 2

problem = SIR.ProblemInstance()
problem.kappa = 0.43 + 1/100 * ((rank + 1) % 58)

if problem_type == "A":
    symmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                  (0, problem.I_constraint + eps, problem.grid_grain),
                                                                  symmetric=True)
    asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                   (0, problem.I_constraint + eps, problem.grid_grain),
                                                                   symmetric=False)

    if rank < 58:
        problem.max_lockdowns_allowed = 2
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_max2"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_max2"))
    else:
        problem.max_lockdowns_allowed = 3
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_max3"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_max3"))

elif problem_type == "B":

    for val in (0.2, 0.25, 0.15):

        problem.I_constraint = val

        symmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                      (0, problem.I_constraint + eps, problem.grid_grain),
                                                                      symmetric=True)
        asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                       (0, problem.I_constraint + eps, problem.grid_grain),
                                                                       symmetric=False)

        problem.max_lockdowns_allowed = np.inf

        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_symmetric_nomax"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_asymmetric_nomax"))

        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_symmetric_max1"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_asymmetric_max1"))

elif problem_type == "C":

    problem.inertia = 14 * problem.ODE_steps

    symmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                  (0, problem.I_constraint + eps, problem.grid_grain),
                                                                  symmetric=True)
    asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                   (0, problem.I_constraint + eps, problem.grid_grain),
                                                                   symmetric=False)
    if rank < 58:
        problem.max_lockdowns_allowed = 3
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_max3"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_max3"))

        problem.max_lockdowns_allowed = 2
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_max2"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_max2"))

    else:
        problem.max_lockdowns_allowed = np.inf
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_nomax"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_nomax"))

        problem.max_lockdowns_allowed = 1
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_nomax"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_nomax"))
