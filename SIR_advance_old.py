import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib

import scipy.integrate

matplotlib.use("TkAgg")

time_end = 10000

param_tau = 10
param_i_hospital = 0.1
param_steps = 100.0

param_s_start = 1 - 0.0025
param_i_start = 0.0025
param_beta0 = 3.0 / 10.0
param_x1_start = 0
param_x2_start = 0


def thresholds_generator(stage1_info, stage2_info):
    """
    :param stage1_info: [3-tuple] with elements corresponding to
        start point, end point, and step size (all must be integers)
        for candidate values for stage 2
    :param stage2_info: see stage1_infor
    :return: [array] of 2-tuples
    """

    # Create an array (grid) of potential thresholds for each stage
    stage1_options = np.arange(stage1_info[0], stage1_info[1], stage1_info[2])
    stage2_options = np.arange(stage2_info[0], stage2_info[1], stage2_info[2])

    # Using Cartesian products, create a list of 5-tuple combos
    stage_options = [stage1_options, stage2_options]
    candidate_feasible_combos = []
    for combo in itertools.product(*stage_options):
        candidate_feasible_combos.append(combo)

    # Eliminate 5-tuples that do not satisfy monotonicity constraint
    # However, ties in thresholds are allowed
    feasible_combos = []
    for combo in candidate_feasible_combos:
        if np.all(np.diff(combo) >= 0):
            feasible_combos.append(combo)

    return feasible_combos


def cost_func_linear(x1_vector, x2_vector, lockdown_cost):
    return np.sum(x1_vector) + lockdown_cost * np.sum(x2_vector)


def simulate_policy(c, kappa, lockdown_cost, full_output=False):

    var_cutoff_1 = c[0]
    var_cutoff_2 = c[1]

    param_kappa_1 = kappa[0]
    param_kappa_2 = kappa[1]

    feasible = True
    cost = np.inf

    s = np.zeros(time_end)
    i = np.zeros(time_end)
    beta = np.zeros(time_end)
    x1 = np.zeros(time_end)
    x2 = np.zeros(time_end)

    s[0] = param_s_start
    i[0] = param_i_start
    x1[0] = param_x1_start
    x2[0] = param_x2_start

    for t in range(1, time_end):

        beta[t] = param_beta0 * (1 - param_kappa_1 * x1[t - 1] - param_kappa_2 * x2[t - 1])

        s[t] = s[t - 1] - (beta[t] * s[t - 1] * i[t - 1]) * 1 / param_steps

        i[t] = i[t - 1] + (beta[t] * s[t - 1] * i[t - 1] - i[t - 1] / param_tau) * \
               1 / param_steps

        if i[t] >= var_cutoff_2:
            x2[t] = 1
        elif var_cutoff_2 > i[t] >= var_cutoff_1:
            x1[t] = 1

        if i[t] >= param_i_hospital:
            feasible = False

    if feasible:
        cost = cost_func_linear(x1, x2, lockdown_cost)

    if full_output == False:
        return cost, x1, x2
    else:
        return cost, x1, x2, s, i


def simulate_many_policies(solutions, kappa, lockdown_cost):
    cost_history = []
    total_x1_history = []
    total_x2_history = []

    num_solutions = len(solutions)

    for i in range(num_solutions):
        c = solutions[i]

        cost, x1, x2 = simulate_policy(c, kappa, lockdown_cost)

        cost_history.append(cost)
        total_x1_history.append(np.sum(x1))
        total_x2_history.append(np.sum(x2))

    return cost_history, total_x1_history, total_x2_history

breakpoint()

plt.clf()
cost, x1, x2, s, i = simulate_policy((0.05, 0.06), (0.5, 1), 10, True)
plt.plot(i, label="0.05")
print(np.sum(x1))
print(np.sum(x2))
cost, x1, x2, s, i = simulate_policy((0.02, 0.06), (0.5, 1), 10, True)
plt.plot(i, label="0.02")
print(np.sum(x1))
print(np.sum(x2))
plt.legend()
plt.show()

breakpoint()

plt.clf()
cost, x1, x2, s, i = simulate_policy((0.08, 0.09), (0.5, 0.75), 2, True)
x1 = np.array(x1)
x1 = x1 == 1
x2 = np.array(x2)
x2 = x2 == 1
i = np.array(i)
x0 = x1 + x2 == 0
# plt.plot(i[x0], linestyle="None", marker="o", color="green")
# plt.plot(i[x1], linestyle="None", marker="o", color="blue")
# plt.plot(i[x2], linestyle="None", marker="o", color="red")

plt.plot(i[x0], color="green")
plt.plot(i[x1], color="blue")
plt.plot(i[x2], color="red")
plt.show()

breakpoint()

optimal_num_days_x1 = []
optimal_num_days_x2 = []
optimal_thresholds = []

solutions = thresholds_generator((0, param_i_hospital, 0.01), (0, param_i_hospital, 0.01))

plt.clf()

for lockdown_cost in np.concatenate((np.arange(1, 11), [100, 1000, 10000])):
    cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (0.5, 0.75), lockdown_cost)
    best = np.argmin(cost_history)
    optimal_thresholds.append(solutions[best])
    optimal_num_days_x1.append(total_x1_history[best])
    optimal_num_days_x2.append(total_x2_history[best])

    cost, x1, x2, s, i = simulate_policy(solutions[best], (0.5, 0.75), lockdown_cost, True)
    plt.plot(i, label="m = " + str(lockdown_cost))
    # plt.plot(x1)

plt.legend()
plt.show()
    # plt.show()

print(optimal_num_days_x1)
print(optimal_num_days_x2)
print(optimal_thresholds)

breakpoint()

cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (0.5, 0.75), 1)
plt.clf()
plt.scatter(total_x1_history, total_x2_history, c=np.array(cost_history), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (0.5, 0.75), 10000)
plt.clf()
plt.scatter(total_x1_history, total_x2_history, c=np.array(cost_history), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

breakpoint()

for kappa_candidate in np.arange(1, 11)/10:
    cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (kappa_candidate, 1), 1)
    print(kappa_candidate)
    print(solutions[np.argmin(cost_history)])

breakpoint()

cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (0.5, 1), 1)
plt.clf()
plt.scatter(total_x1_history, total_x2_history, c=np.array(cost_history), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

breakpoint()

# cost, total_x1, total_x2, s, i = simulate_policy((np.inf, np.inf), (0, 0), 0, True)

c = np.arange(0,5)/50
kappa = np.arange(0,5)/5
colors = ["green", "blue", "yellow", "orange", "red"]
colors.reverse()

for l in range(5):
    plt.clf()
    for j in range(5):
        cost, total_x1, total_x2, s, i = simulate_policy((c[j], np.inf), (kappa[l], 0), 0, True)
        plt.plot(i, color=colors[j], label=str(c[j]))
        plt.title("Kappa " + str(kappa[l]))

    plt.legend()
    plt.show()

breakpoint()

plt.clf()

for kappa in np.arange(0, 10)/10:
    cost, total_x1, total_x2, s, i = simulate_policy((0, np.inf), (kappa, 0), 0, True)
    if np.sum(i < param_i_hospital) == len(i):
        plt.plot(i, color="green", label=str(kappa))

for kappa in np.arange(0, 10)/10:
    cost, total_x1, total_x2, s, i = simulate_policy((0.02, np.inf), (kappa, 0), 0, True)
    if np.sum(i < param_i_hospital) == len(i):
        plt.plot(i, color="blue", label=str(kappa))

for kappa in np.arange(0, 10)/10:
    cost, total_x1, total_x2, s, i = simulate_policy((0.04, np.inf), (kappa, 0), 0, True)
    if np.sum(i < param_i_hospital) == len(i):
        plt.plot(i, color="yellow", label=str(kappa))

for kappa in np.arange(0, 10)/10:
    cost, total_x1, total_x2, s, i = simulate_policy((0.06, np.inf), (kappa, 0), 0, True)
    if np.sum(i < param_i_hospital) == len(i):
        plt.plot(i, color="orange", label=str(kappa))

for kappa in np.arange(0, 10)/10:
    cost, total_x1, total_x2, s, i = simulate_policy((0.08, np.inf), (kappa, 0), 0, True)
    # if np.sum(i < param_i_hospital) == len(i):
    #    plt.plot(i, color="red", label=str(kappa))

# plt.legend()
plt.show()

breakpoint()

optimal_num_days_x1 = []
optimal_num_days_x2 = []
optimal_thresholds = []

solutions = thresholds_generator((0, param_i_hospital, 0.01), (0, param_i_hospital, 0.01))

breakpoint()

# fig, ax = plt.subplots()
# ax.pcolormesh(total_x1_history, total_x2_history, cost_history)

# plt.scatter(total_x1_history, total_x2_history, c=np.array(cost_history), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()

fig, ax = plt.subplots()
ax.scatter(total_x1_history, total_x2_history, c=np.array(cost_history), cmap=plt.cm.jet)
# ax.legend()
# ax.grid(True)
# ax.set_xlabel("Policy's Days in Stage 1")
# ax.set_ylabel("Policy's Days in Stage 2")
# cbar = plt.colorbar()
# cbar.ax.set_ylabel('# of contacts', rotation=270)
plt.show()
