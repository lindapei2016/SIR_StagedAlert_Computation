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

plt.rcParams.update({"font.size": 14})
plt.rcParams.update({"lines.linewidth":2.0})

import scipy as sp

eps = 1e-6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()
problem.kappa = 0.6
problem.threshold_up = 0.05
problem.threshold_down = 0
problem.full_output = True
problem.simulate_policy()

marker_grid = np.arange(1000, len(problem.results.I[:15000]), 1000)

plt.clf()
plt.plot(problem.results.S[:15000], color="yellowgreen", marker="o", markevery=marker_grid, linestyle=":", label="S")
plt.plot(problem.results.I[:15000], color="darkorange", marker="o", markevery=marker_grid, label="I")
plt.title("S & I -- System With Reduction 0.6 at I = 0.05")
plt.ylabel("Proportion")
plt.xlabel("Simulation time units")
plt.axhline(y=1/3, color="blue", linestyle="--", xmin=0, xmax=1000, label="S Herd Immunity")
plt.axhline(y=0.1, color="red", linestyle="--", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_baseline4.eps')

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()
problem.max_lockdowns_allowed = np.inf
problem.kappa = 0.5
problem.threshold_up = 0.1
problem.threshold_down = 0.1
problem.full_output = True
problem.simulate_policy()

num_lockdowns = 0
num_lockdowns_ls = [0]

for i in range(1, len(problem.results.x1)):
    num_lockdowns += (problem.results.x1[i-1] == 0 and problem.results.x1[i] == 1)
    num_lockdowns_ls.append(num_lockdowns)

plt.clf()
plt.scatter(np.arange(len(problem.results.x1)), problem.results.x1, color="blue", marker="^", label="Stage Number")
plt.plot(np.arange(len(problem.results.x1)), num_lockdowns_ls/num_lockdowns, color="gold", label="Scaled Number of Lockdowns")
plt.title("Bang-Bang Policy for Reduction 0.5")
plt.ylabel("Stage Number / Scaled Number of Lockdowns")
plt.xlabel("Simulation time units")
plt.legend()
plt.savefig('constraint1e-1_bangbang.eps')
plt.show()

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()
problem.max_lockdowns_allowed = 1
problem.kappa = 0
problem.threshold_up = 1
problem.threshold_down = 1
problem.full_output = True
problem.simulate_policy()

marker_grid = np.arange(1000, len(problem.results.I[:15000]), 1000)

problem2 = SIR.ProblemInstance()
problem2.max_lockdowns_allowed = 1
problem2.kappa = 0.5
problem2.threshold_up = 0.05
problem2.threshold_down = 0.05
problem2.full_output = True
problem2.simulate_policy()

problem3 = SIR.ProblemInstance()
problem3.max_lockdowns_allowed = 1
problem3.kappa = 0.5
problem3.threshold_up = 0.075
problem3.threshold_down = 0.075
problem3.full_output = True
problem3.simulate_policy()

problem4 = SIR.ProblemInstance()
problem4.max_lockdowns_allowed = 1
problem4.kappa = 0.5
problem4.threshold_up = 0.1
problem4.threshold_down = 0.1
problem4.full_output = True
problem4.simulate_policy()

# midnightblue, mediumblue, mediumpurple, hotpink

plt.clf()
plt.plot(problem.results.I[:15000], color="hotpink", linestyle="-.", markevery=marker_grid, label="I, No Intervention")
plt.plot(problem2.results.I[:15000], color="midnightblue", marker="D", markevery=marker_grid, label="I, Threshold 0.05")
plt.plot(problem3.results.I[:15000], color="mediumblue", marker="h", markevery=marker_grid, label="I, Threshold 0.075")
plt.plot(problem4.results.I[:15000], color="mediumpurple", marker="o", markevery=marker_grid, label="I, Threshold 0.1")
plt.title("Infected -- Reduction 0.5, 1 Max Lockdown")
plt.ylabel("Proportion Infected")
plt.xlabel("Simulation time units")
plt.axhline(y=0.1, color="red", linestyle="--", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_interventions_1max.eps')

plt.show()
plt.clf()

breakpoint()

problem = SIR.ProblemInstance()
problem.max_lockdowns_allowed = np.inf
problem.kappa = 0
problem.threshold_up = 1
problem.threshold_down = 1
problem.full_output = True
problem.simulate_policy()

marker_grid = np.arange(1000, len(problem.results.I[:15000]), 1000)

problem2 = SIR.ProblemInstance()
problem2.max_lockdowns_allowed = np.inf
problem2.kappa = 0.5
problem2.threshold_up = 0.05
problem2.threshold_down = 0.05
problem2.full_output = True
problem2.simulate_policy()

problem3 = SIR.ProblemInstance()
problem3.max_lockdowns_allowed = np.inf
problem3.kappa = 0.5
problem3.threshold_up = 0.075
problem3.threshold_down = 0.075
problem3.full_output = True
problem3.simulate_policy()

problem4 = SIR.ProblemInstance()
problem4.max_lockdowns_allowed = np.inf
problem4.kappa = 0.5
problem4.threshold_up = 0.1
problem4.threshold_down = 0.1
problem4.full_output = True
problem4.simulate_policy()

# midnightblue, mediumblue, mediumpurple, hotpink

plt.clf()
plt.plot(problem.results.I[:15000], color="hotpink", linestyle="-.", markevery=marker_grid, label="I, No Intervention")
plt.plot(problem2.results.I[:15000], color="midnightblue", marker="D", markevery=marker_grid, label="I, Threshold 0.05")
plt.plot(problem3.results.I[:15000], color="mediumblue", marker="h", markevery=marker_grid, label="I, Threshold 0.075")
plt.plot(problem4.results.I[:15000], color="mediumpurple", marker="o", markevery=marker_grid, label="I, Threshold 0.1")
plt.title("Infected -- Reduction 0.5, No Max Lockdowns")
plt.ylabel("Proportion Infected")
plt.xlabel("Simulation time units")
plt.axhline(y=0.1, color="red", linestyle="--", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_interventions_nomax.eps')

plt.show()
plt.clf()

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()

problem.full_output = True
problem.kappa = 0.57
problem.threshold_up = 0.096
problem.threshold_down = 0.096
problem.max_lockdowns_allowed = np.inf
problem.simulate_policy()
problem.results()

plt.plot(problem.results.I, color="goldenrod", marker="D", markevery=np.arange(1000, len(problem.results.I), 1000), label="No max lockdown")

problem.full_output = True
problem.kappa = 0.57
problem.threshold_up = 0.096
problem.threshold_down = 0.096
problem.max_lockdowns_allowed = 1
problem.simulate_policy()
problem.results()
plt.plot(problem.results.I, color="violet", marker="o", markevery=np.arange(1000, len(problem.results.I), 1000), label="1 max lockdown")
plt.title("Infected: 1 max lockdown vs no max, threshold_up 0.096, threshold_down 0.096")
plt.ylabel("Proportion infected")
plt.xlabel("Simulation time")
plt.axhline(y=0.1, color="red", xmin=0, xmax=1000, linestyle="--", label="Constraint")
plt.legend()

plt.savefig('constraint1e-1_max.eps')

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

marker_grid = np.arange(1000, len(problem.results.I), 1000)

plt.plot(problemA.results.S, color="yellowgreen", marker="o", markevery=marker_grid, linestyle=":", label="S, kappa 0.72")
plt.plot(problemB.results.S, color="lightseagreen", marker="D", markevery=marker_grid, linestyle=":", label="S, kappa 0.55")
plt.plot(problemA.results.I, color="darkorange", marker="o", markevery=marker_grid, label="I, kappa 0.72")
plt.plot(problemB.results.I, color="chocolate", marker="D", markevery=marker_grid, label="I, kappa 0.55")
plt.title("S & I -- one lockdown allowed, threshold_up 0.091, threshold_down 0.06")
plt.ylabel("Proportion")
plt.xlabel("Simulation time units")
plt.axhline(y=0.1, color="red", xmin=0, xmax=1000, linestyle="--", label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_kappa_is_weird.eps')

plt.show()

breakpoint()

marker_grid = np.arange(1000, len(problem.results.I[:10000]), 1000)

plt.clf()
plt.plot(problemA.results.S[:10000], color="yellowgreen", marker="o", markevery=marker_grid, linestyle=":", label="S, kappa 0.72")
plt.plot(problemB.results.S[:10000], color="lightseagreen", marker="D", markevery=marker_grid, linestyle=":", label="S, kappa 0.55")
plt.plot(problemA.results.I[:10000], color="darkorange", marker="o", markevery=marker_grid, label="I, kappa 0.72")
plt.plot(problemB.results.I[:10000], color="chocolate", marker="D", markevery=marker_grid, label="I, kappa 0.55")
plt.title("S & I -- max 1 lockdown, up 0.091, down 0.06")
plt.ylabel("Proportion")
plt.xlabel("Simulation time units")
plt.axhline(y=0.1, color="red", linestyle="--", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_kappa_is_weird_zoom_in.eps')

breakpoint()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

problem = SIR.ProblemInstance()
problem.kappa = 0
problem.threshold_up = 1
problem.threshold_down = 1
problem.full_output = True
problem.simulate_policy()

marker_grid = np.arange(1000, len(problem.results.I[:15000]), 1000)

plt.clf()
plt.plot(problem.results.S[:15000], color="yellowgreen", marker="o", markevery=marker_grid, linestyle=":", label="S")
plt.plot(problem.results.I[:15000], color="darkorange", marker="o", markevery=marker_grid, label="I")
plt.title("S & I -- System Without Intervention")
plt.ylabel("Proportion")
plt.xlabel("Simulation time units")
plt.axhline(y=1/3, color="blue", linestyle="--", xmin=0, xmax=1000, label="S Herd Immunity")
plt.axhline(y=0.1, color="red", linestyle="--", xmin=0, xmax=1000, label="Constraint")
plt.legend()
plt.savefig('constraint1e-1_baseline2.eps')

plt.show()
plt.clf()

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