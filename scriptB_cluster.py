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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

eps = 1e-6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Put on cluster to find optimal threshold value given kappa,
#   where optimal means minimum number of days in stage 2

problem = SIR.ProblemInstance()
policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                    (0, 1 + eps, 0.05),
                                                    symmetric=False)

# 100 processors
problem.kappa = 1/100 * ((rank + 1) % 100)
print(problem.find_optimum(policies, str(int(problem.kappa * 100))))

# 400 processors
# 100 values of kappa to test from 1/100 to 1
# 4 chunks of policies to distribute

# problem.kappa = 1/100 * ((rank + 1) % 100)
# num_policies = len(policies)
# splits = np.array_split(np.array(policies), 4)

# print(problem.find_optimum(tuple(splits[(rank % 100) % 4]), str(int(problem.kappa * 100)) + "_" + str((rank % 100) % 4)))

