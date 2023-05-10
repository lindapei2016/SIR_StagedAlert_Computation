###############################################################################

# Default parameters for deterministic SIR model with 2 stages.

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
time_end = 20000

# Granularity for brute force search for optimization
grid_grain = 0.001

# Proxy for hospital capacity --
#   staged alert policies whose systems have I > I_constraint
#   are considered infeasible.
I_constraint = 0.1

# Starting conditions for SIR model
S_start = 1 - 0.0025
I_start = 0.0025

# Unmitigated transmission parameter and infectious period
# Using commonly used parameter values for start of covid
beta0 = 3.0 / 10.0
tau = 10

# Transmission reduction under lockdown
# This must be in [0,1]
kappa = 0.8

# Cost of stage 0 and stage 1 (we need cost0 < cost1)
cost0 = 0
cost1 = 1

# Trigger at which to move to stage 1
#   and move out of stage 1, respectively --
#   refers to the proportion of infected (I) at which
#   to change stages.
# threshold_down actually does not have to be less than
#   or equal to threshold_up, because we only allow
#   moving out of stage 1 if infections are decreasing
#   relative to previous timepoint.
threshold_up = 1
threshold_down = 1

# Maximum number of times the system can move from
#   stage 0 to stage 1.
max_lockdowns_allowed = 1

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
