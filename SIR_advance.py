import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib

import scipy.integrate
import scipy.optimize

import math


def build_beta_func(tau, s0, ihat):
    def beta_func(beta):
        return 1 - 1 / (beta * tau) - math.log(beta * tau * s0) / (beta * tau) - ihat

    return beta_func


beta_func = build_beta_func(10, 0.99, 0.1)

# breakpoint()

# for beta in np.arange(1, 1000)/1000:
    # print(beta_func(beta), beta)

# results = scipy.optimize.root(beta_func, 0.5)

# breakpoint()

matplotlib.use("TkAgg")

time_end = 1000000

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


def cost_func_linear(x1_vector, x2_vector, m1, m2):
    return m1 * np.sum(x1_vector) + m2 * np.sum(x2_vector)


def simulate_policy(c, kappa, m, full_output=False):
    var_cutoff_1 = c[0]
    var_cutoff_2 = c[1]

    param_kappa_1 = kappa[0]
    param_kappa_2 = kappa[1]

    m1 = m[0]
    m2 = m[1]

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
        cost = cost_func_linear(x1, x2, m1, m2)

    if full_output == False:
        return cost, x1, x2
    else:
        return cost, x1, x2, s, i


def simulate_many_policies(solutions, kappa, m):
    cost_history = []
    total_x1_history = []
    total_x2_history = []

    num_solutions = len(solutions)

    for i in range(num_solutions):
        c = solutions[i]

        cost, x1, x2 = simulate_policy(c, kappa, m)

        cost_history.append(cost)
        total_x1_history.append(np.sum(x1))
        total_x2_history.append(np.sum(x2))

    return cost_history, total_x1_history, total_x2_history


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plt.clf()

# c, kappa, m


for kappa1 in (1.0, 1.0):
    plt.clf()
    for c1 in (0.02, 0.04, 0.06, 0.08):
        cost, x1, x2, s, i = simulate_policy((0, c1), (0, kappa1), (1, 1), full_output=True)
        # print("~~~~~~~~~~")
        # print(np.sum(x1))
        # print(np.argmax(i))
        # times_above_c1 = np.argwhere(i >= c1)
        # print(times_above_c1[0] + (times_above_c1[-1] - times_above_c1[0])/2)
        for j in range(len(s)):
            if s[j] < 1/3:
                print(i[j])
                break
        plt.plot(i, label=str(c1))
        # plt.plot(s, "-,", label=str(c1))
    plt.legend()
    plt.show()
    # breakpoint()

breakpoint()

# for reduction in np.arange(30, 100+1)/100:

reduction = 0.5
solutions = thresholds_generator((0, param_i_hospital + 0.001, 0.001), (2, 2.001, 0.01))
cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (reduction, 1), (1, 1))

# print(np.diff(total_x1_history))

breakpoint()

c1 = []
best_cost = []
total_x1 = []

for reduction in np.arange(30, 100 + 1) / 100:
    # for weight in (1, 10, 100):

    solutions = thresholds_generator((0, param_i_hospital + 0.001, 0.001), (2, 2.001, 0.01))
    cost_history, total_x1_history, total_x2_history = simulate_many_policies(solutions, (reduction, 1), (1, 1))
    best = np.argmin(cost_history)
    c1.append(solutions[best][0])
    total_x1.append(total_x1_history[best])
    # best_cost.append(cost_history[best])

breakpoint()
