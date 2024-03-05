###############################################################################

# Script A finds the optimal threshold value for a symmetric threshold
#   given kappa (transmission reduction under stage 1, i.e., lockdown).
# We use 100 processors, and processor with rank r uses brute force
#   computation to find the optimal symmetric threshold value for
#   kappa = r/100.

###############################################################################

import SIR_det_3stages as SIR

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

# 3 levels of inertia (10, 100, 1000 times discretization)
# 2x2 combinations
# Low, high basic reproduction number under lockdown
# Medium stage has half the reduction of high stage, or they have a reduction that is closer

# 12 processors 0, 1, 2, ..., 11
# First 4 processors have inertia_scaling = 10
# Next 4 have inertia_scaling = 100
# Last 4 have inertia_scaling = 1000

# Here, R_{0, on} < 0.7 (low basic reproduction number under lockdown)
# Specifically 3*.2 = 0.6

if rank < 4:
    inertia_scaling = 10
elif rank < 8:
    inertia_scaling = 100
else:
    inertia_scaling = 1000

if rank in (0, 4, 8):
    problem = SIR.ProblemInstance()
    policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                        (0, problem.I_constraint + eps, problem.grid_grain))
    problem.I_constraint = 0.1
    problem.inertia = inertia_scaling * 100
    problem.beta0 = 3/10.0
    problem.medium_kappa = 0.4
    problem.high_kappa = 0.8
    print(problem.find_optimum(policies, "3stages_lowR0on_half_inertia" + str(inertia_scaling)))

# Now have medium and high stage have closer reductions
elif rank in (1, 5, 9):
    problem = SIR.ProblemInstance()
    policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                        (0, problem.I_constraint + eps, problem.grid_grain))
    problem.I_constraint = 0.1
    problem.inertia = inertia_scaling * 100
    problem.beta0 = 3/10.0
    problem.medium_kappa = 0.6
    problem.high_kappa = 0.8
    print(problem.find_optimum(policies, "3stages_lowR0on_close_inertia" + str(inertia_scaling)))

# Here, R_{0, on} > 1.4 (high basic reproduction number under lockdown)
# Specifically 3*.5 = 1.5
elif rank in (2, 6, 10):
    problem = SIR.ProblemInstance()
    policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                        (0, problem.I_constraint + eps, problem.grid_grain))
    problem.I_constraint = 0.1
    problem.inertia = inertia_scaling * 100
    problem.beta0 = 3/10.0
    problem.medium_kappa = 0.25
    problem.high_kappa = 0.5
    print(problem.find_optimum(policies, "3stages_highR0on_half_inertia" + str(inertia_scaling)))

# Now have medium and high stage have closer reductions
else: # rank in (3, 7, 11)
    problem = SIR.ProblemInstance()
    policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                        (0, problem.I_constraint + eps, problem.grid_grain))
    problem.I_constraint = 0.1
    problem.inertia = inertia_scaling * 100
    problem.beta0 = 3/10.0
    problem.medium_kappa = 0.4
    problem.high_kappa = 0.5
    print(problem.find_optimum(policies, "3stages_highR0on_close_inertia" + str(inertia_scaling)))