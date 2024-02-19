###############################################################################

# Default parameters for deterministic SIR model with 3 stages.

###############################################################################

import numpy as np

# Number of ODE timesteps per time unit (discretization factor)
# 10 steps per tick is too rough for I_constraint = 0.1, beta0 = 3/10, tau = 10
#   (compared to continuous values)
# 100 steps seems like a good compromise
# 1000 is better, 10000 is almost continuous
ODE_steps = 100.0

# Total number of ODE timesteps per simulation
# A good combo:
#   ODE_steps = 100
#   time_end = 20000
time_end = 20000 * 1000

# Granularity for brute force search for optimization
grid_grain = 0.005

# Proxy for hospital capacity --
#   staged alert policies whose systems have I > I_constraint
#   are considered infeasible.
I_constraint = 0.1

# Starting conditions for SIR model
S_start = 1 - 0.001
I_start = 0.001

# Unmitigated transmission parameter and infectious period
# Using commonly used parameter values for start of covid
# Previously used beta0 = 3/10
beta0 = 1.0 / 10.0
tau = 10

# Transmission reduction under lockdown
# This must be in [0,1]
medium_kappa = 0.5
high_kappa = 0.7

# Cost of stage 0, 1, 2
# Need cost of stage 0 < cost of stage 1 < cost of stage 2
low_stage_cost = 0
medium_stage_cost = 10
high_stage_cost = 100

# Trigger at which to move to stage 1
#   and move out of stage 1, respectively --
#   refers to the proportion of infected (I) at which
#   to change stages.
# We require medium_threshold <= high_threshold
medium_threshold = np.inf
high_threshold = np.inf

# Set to False for performance runs.
# Change to True if also interested in values of x0, x1,
#   S, and I throughout the simulation.
# Note: when full_output is False, simulating
#   a staged alert policy terminates if it is infeasible,
#   i.e., when I hits I_constraint, simulating terminates.
# If the policy is infeasible and its I hits I_constraint
#   at time t, then x0, x1, S, and I are only recorded
#   through time t, and no further.
full_output = False

# Number of time units to remain in one stage
# e.g. Austin had 14-day rule, and to use
#   14-day rule set inertia to 14 * ODE_steps
inertia = 0

# Whether to stop at herd immunity or stop after first peak
# Options: {"herd_immunity", "first_peak"}
stopping_condition = "herd_immunity"
