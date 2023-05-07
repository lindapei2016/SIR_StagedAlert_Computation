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
problem.kappa = 1/100 * ((rank + 1) % 100)

if problem_type == "A":
    symmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                  (0, problem.I_constraint + eps, problem.grid_grain),
                                                                  symmetric=True)
    asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                   (0, problem.I_constraint + eps, problem.grid_grain),
                                                                   symmetric=False)

    if rank < 100:
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_max1"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_max1"))

    elif rank < 200:
        problem.max_lockdowns_allowed = np.inf
        print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_symmetric_nomax"))
        print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_asymmetric_nomax"))

elif problem_type == "B":

    if rank < 100:
        problem.I_constraint = 0.2
    elif rank < 200:
        problem.I_constraint = 0.4

    symmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                  (0, problem.I_constraint + eps, problem.grid_grain),
                                                                  symmetric=True)
    asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                                   (0, problem.I_constraint + eps, problem.grid_grain),
                                                                   symmetric=False)

    print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_symmetric_max1"))
    print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_asymmetric_max1"))

    problem.max_lockdowns_allowed = np.inf
    print(problem.find_optimum(symmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_symmetric_nomax"))
    print(problem.find_optimum(asymmetric_policies, str(int(problem.kappa * 100)) + "_constraint" + str(problem.I_constraint) + "_asymmetric_nomax"))
