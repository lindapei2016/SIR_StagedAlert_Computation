###############################################################################

# Script C takes the optimal threshold values from Script A
#   and simulates them to obtain their x1_history to determine
#   how many times they were in lockdown.

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

problem.kappa = 1 / 100 * (rank + 1)
print(problem.find_optimum(policies, str(int(problem.kappa * 100))))

