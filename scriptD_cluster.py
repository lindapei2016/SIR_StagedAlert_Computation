###############################################################################

# Script D finds the optimal threshold value for a symmetric threshold
#   given beta, constraint value, and kappa.
# Right now use 5x5 = 25 parallel processors (for 5 values of beta
#   and 5 values of constraint).
# Right now having 1 max lockdown!

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
                                                    (0, problem.I_constraint + eps, problem.grid_grain),
                                                    symmetric=True)

beta_constraint_combos = []

for combo in itertools.product((2 / 10, 3 / 10, 4 / 10, 5 / 10, 6 / 10),
                               (.1, .15, .20, .25, .30)):
    beta_constraint_combos.append(combo)

problem.beta0 = beta_constraint_combos[rank][0]
problem.I_constraint = beta_constraint_combos[rank][1]

for i in range(100, 20, -1):
    problem.kappa = 1 / 100 * (i + 1)
    if not problem.full_output:
    	filename_prefix, best_policy, best_cost, best_num_lockdowns = problem.find_optimum(policies, "R0" + str(int(problem.beta0 * problem.tau)) + "constraint" + str(problem.I_constraint) + "kappa" + str(problem.kappa))
    print(filename_prefix, best_policy, best_cost, best_num_lockdowns)

    if best_cost == np.inf:
    	break
