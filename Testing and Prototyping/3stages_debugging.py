# Script for debugging and testing scriptA_cluster_inertia_3stages.py
import SIR_det_3stages as SIR
import matplotlib.pyplot as plt
import numpy as np

def compute_stage_changes(x1, x2):
    stage_history = x1 + 3 * x2
    return np.sum(stage_history[:-1] != stage_history[1:])

# Examples of attributes to toggle:
#   threshold_up
#   threshold_down
#   inertia
#   beta0
#   kappa
#   time_end
problem = SIR.ProblemInstance()

# ('3stages_highR0on_half_inertia10', (0.025, 0.065), 514580.0, 2)
problem.I_constraint = 0.1
problem.inertia = 10 * 100
problem.medium_threshold = 0.025
problem.high_threshold = 0.065
problem.beta0 = 3/10.0
problem.medium_kappa = 0.25
problem.high_kappa = 0.5
problem.full_output = True
problem.simulate_policy()

plt.clf()
plt.plot(problem.results.I/problem.I_constraint, label="I/I_constraint")
plt.plot(problem.results.x1 + 2*problem.results.x2, alpha=0.4, color="grey", label="Stage")
plt.legend()
plt.savefig("3stages_highR0on_half_inertia10.png", dpi=1200)

plt.clf()
plt.plot(problem.results.S, label="S")
plt.plot(problem.results.I, label="I")
plt.plot(problem.results.R, label="R")
plt.legend()
plt.savefig("SIR_3stages_highR0on_half_inertia10.png", dpi=1200)

print(compute_stage_changes(problem.results.x1, problem.results.x2))

# ('3stages_highR0on_close_inertia10', (0.01, 0.095), 264490.0, 2)
problem.I_constraint = 0.1
problem.inertia = 10 * 100
problem.medium_threshold = 0.01
problem.high_threshold = 0.095
problem.beta0 = 3/10.0
problem.medium_kappa = 0.4
problem.high_kappa = 0.5
problem.full_output = True
problem.simulate_policy()

plt.clf()
plt.plot(problem.results.I/problem.I_constraint, label="I/I_constraint")
plt.plot(problem.results.x1 + 2*problem.results.x2, alpha=0.4, color="grey", label="Stage")
plt.legend()
plt.savefig("3stages_highR0on_close_inertia10.png", dpi=1200)

plt.clf()
plt.plot(problem.results.S, label="S")
plt.plot(problem.results.I, label="I")
plt.plot(problem.results.R, label="R")
plt.legend()
plt.savefig("SIR_3stages_highR0on_close_inertia10.png", dpi=1200)

print(compute_stage_changes(problem.results.x1, problem.results.x2))

# ('3stages_lowR0on_half_inertia10', (0.055, 0.09), 160340.0, 2)

problem.I_constraint = 0.1
problem.inertia = 10 * 100
problem.medium_threshold = 0.055
problem.high_threshold = 0.09
problem.beta0 = 3/10.0
problem.medium_kappa = 0.4
problem.high_kappa = 0.8
problem.full_output = True
problem.simulate_policy()

plt.clf()
plt.plot(problem.results.I/problem.I_constraint, label="I/I_constraint")
plt.plot(problem.results.x1 + 2*problem.results.x2, alpha=0.4, color="grey", label="Stage")
plt.legend()
plt.savefig("3stages_lowR0on_half_inertia10.png", dpi=1200)

plt.clf()
plt.plot(problem.results.S, label="S")
plt.plot(problem.results.I, label="I")
plt.plot(problem.results.R, label="R")
plt.legend()
plt.savefig("SIR_3stages_lowR0on_half_inertia10.png", dpi=1200)

print(compute_stage_changes(problem.results.x1, problem.results.x2))

# ('3stages_lowR0on_close_inertia10', (0.035, 0.1), 103530.0, 1)

problem.I_constraint = 0.1
problem.inertia = 10 * 100
problem.medium_threshold = 0.035
problem.high_threshold = 0.1
problem.beta0 = 3/10.0
problem.medium_kappa = 0.6
problem.high_kappa = 0.8
problem.full_output = True
problem.simulate_policy()

plt.clf()
plt.plot(problem.results.I/problem.I_constraint, label="I/I_constraint")
plt.plot(problem.results.x1 + 2*problem.results.x2, alpha=0.4, color="grey", label="Stage")
plt.legend()
plt.savefig("3stages_lowR0on_close_inertia10.png", dpi=1200)

plt.clf()
plt.plot(problem.results.S, label="S")
plt.plot(problem.results.I, label="I")
plt.plot(problem.results.R, label="R")
plt.legend()
plt.savefig("SIR_3stages_lowR0on_close_inertia10.png", dpi=1200)

print(compute_stage_changes(problem.results.x1, problem.results.x2))

breakpoint()

# Results are stored as an attribute
plt.clf()
plt.plot(problem.results.S, label="S")
plt.plot(problem.results.I, label="I")
plt.plot(problem.results.R, label="R")
plt.legend()
plt.show()

breakpoint()