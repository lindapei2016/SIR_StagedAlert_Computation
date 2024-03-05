###############################################################################

# Script A finds the optimal threshold value for a symmetric threshold
#   given kappa (transmission reduction under stage 1, i.e., lockdown).
# We use 100 processors, and processor with rank r uses brute force
#   computation to find the optimal symmetric threshold value for
#   kappa = r/100.

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

eps = 1e-6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Put on cluster to find optimal threshold value given kappa,
#   where optimal means minimum number of days in stage 2

# beta0 = 1/10, increment by 1/100 until 5/10

if rank < 50:
    for val in np.arange(1, 40)/100.0:
        problem = SIR.ProblemInstance()
        policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                            (0, problem.I_constraint + eps, problem.grid_grain),
                                                            symmetric=True)
        problem.inertia = 100 * 1000
        problem.beta0 = problem.beta0 + val
        problem.max_lockdowns_allowed = np.inf
        problem.kappa = (40 / 100) + (1/100) * (rank + 1)
        print(problem.find_optimum(policies, "nomax_inertia1e1_beta0_" + str(int(problem.beta0*100)) + "_kappa_" + str(int(problem.kappa * 100))))
elif rank < 100:
    for val in np.arange(1, 40)/100.0:
        problem = SIR.ProblemInstance()
        policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                            (0, problem.I_constraint + eps, problem.grid_grain),
                                                            symmetric=True)
        problem.inertia = 100 * 1000
        problem.beta0 = problem.beta0 + val
        problem.max_lockdowns_allowed = np.inf
        problem.kappa = (40 / 100) + (1/100) * (rank - 50 + 1)
        print(problem.find_optimum(policies, "nomax_inertia1e2_beta0_" + str(int(problem.beta0*100)) + "_kappa_" + str(int(problem.kappa * 100))))
else:
    for val in np.arange(1, 40)/100.0:
        problem = SIR.ProblemInstance()
        policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                            (0, problem.I_constraint + eps, problem.grid_grain),
                                                            symmetric=True)
        problem.inertia = 1000 * 1000
        problem.beta0 = problem.beta0 + val
        problem.max_lockdowns_allowed = np.inf
        problem.kappa = (40 / 100) + (1/100) * (rank - 100 + 1)
        print(problem.find_optimum(policies, "nomax_inertia1e3_beta0_" + str(int(problem.beta0*100)) + "_kappa_" + str(int(problem.kappa * 100))))