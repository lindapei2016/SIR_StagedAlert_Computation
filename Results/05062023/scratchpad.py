###############################################################################

# Scratchpad
# Contains various routines for plotting selected interesting systems
#   and obtaining additional analysis on staged-alert policies.

###############################################################################

import SIR_det_2stages as SIR

from mpi4py import MPI
import numpy as np
import itertools
import math

# Flush printing
import sys

sys.stdout.flush()

# For exploratory plotting -- turn this off for cluster
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

import scipy as sp

eps = 1e-6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()

problem.kappa = 0.57
problem.threshold_up = 0.096
problem.threshold_down = 0.096
problem.max_lockdowns_allowed = np.inf
problem.simulate_policy()
problem.results()

plt.plot(problem.results.I, color="goldenrod", label="No max lockdown")

problem.full_output = True
problem.kappa = 0.57
problem.threshold_up = 0.096
problem.threshold_down = 0.096
problem.max_lockdowns_allowed = 1
problem.simulate_policy()
problem.results()
plt.plot(problem.results.I, color="violet", label="1 max lockdown")
plt.title("Infected: 1 max lockdown vs no max lockdown, threshold_up 0.096, threshold_down 0.096")
plt.ylabel("Proportion infected")
plt.xlabel("Simulation time")
plt.axhline(y=0.1, color="red", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.show()

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Under 1 max lockdown, instance in which (0.091, 0.06) for
#   (threshold_up, threshold_down) respectively is optimal for
#   kappa = 0.55, but is infeasible under an increased kappa = 0.72
#   (counterintuitive).
# This occurs because only 1 max lockdown is allowed, and
#   a higher kappa value of 0.72 causes a quick departure from
#   lockdown. At this departure time, the remaining S is still
#   relatively large, and the system hits I > I_constraint
#   after lockdown is lifted.

problemA = SIR.ProblemInstance()
problemA.full_output = True
problemA.kappa = 0.72
problemA.threshold_up = 0.091
problemA.threshold_down = 0.06
problemA.simulate_policy()

problemB = SIR.ProblemInstance()
problemB.full_output = True
problemB.kappa = 0.55
problemB.threshold_up = 0.091
problemB.threshold_down = 0.06
problemB.simulate_policy()

plt.plot(problemA.results.S, color="yellowgreen", linestyle="--", label="S, kappa 0.72")
plt.plot(problemB.results.S, color="lightseagreen", linestyle=":", label="S, kappa 0.55")
plt.plot(problemA.results.I, color="darkorange", linestyle="--", label="I, kappa 0.72")
plt.plot(problemB.results.I, color="red", linestyle=":", label="I, kappa 0.55")
plt.title("S & I -- one lockdown allowed, threshold_up 0.091, threshold_down 0.06")
plt.ylabel("Population proportion (susceptible or infected)")
plt.xlabel("Simulation time units")
plt.axhline(y=0.1, color="red", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.show()

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()
problem.kappa = 0.5

plt.clf()

for t in (0.1,):
    problem.threshold_up = t
    problem.threshold_down = t
    problem.simulate_policy()
    # plt.plot(problem.results.I)
    # plt.plot(problem.tau * problem.beta0 * (1 - problem.kappa * problem.results.x1) * problem.results.S)
    print(t)
    # plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem.kappa = 0.5
problem.threshold_up = 0.04
problem.threshold_down = 0.04
problem.simulate_policy()

i = problem.results.I
s = problem.results.S

plt.plot(i)
plt.plot(s)
rate_of_increase = i * (1 - problem.kappa) * problem.beta0 * s - i / problem.tau
rate_of_increase[i < problem.threshold_up] = i[i < problem.threshold_up] * problem.beta0 * s[i < problem.threshold_up] - i[
    i < problem.threshold_up] / problem.tau
# plt.plot(rate_of_increase, label="Lower threshold")
problem.threshold_up = 0.08
problem.threshold_down = 0.08
problem.simulate_policy()

i = problem.results.I
s = problem.results.S

plt.plot(i)
plt.plot(s)

rate_of_increase = i * (1 - problem.kappa) * problem.beta0 * s - i / problem.tau
rate_of_increase[i < problem.threshold_up] = i[i < problem.threshold_up] * problem.beta0 * s[i < problem.threshold_up] - i[
    i < problem.threshold_up] / problem.tau
# plt.plot(rate_of_increase, label="Higher threshold")
plt.legend()
plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~