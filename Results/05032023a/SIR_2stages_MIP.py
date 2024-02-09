from __future__ import division
import pyomo.environ as pyo

import numpy as np
import matplotlib
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt

time_end = 10

model = pyo.ConcreteModel()

model.T = pyo.RangeSet(0, time_end)
model.T_minus_start = pyo.RangeSet(1, time_end)

model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.i = pyo.Var(model.T, domain=pyo.NonNegativeReals)

model.alpha = pyo.Var(model.T, domain=pyo.Binary)
# model.alpha = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,1))
model.cutoff = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,1), initialize=0.005)

model.s_0 = 1 - 0.0025
model.i_0 = 0.0025
model.beta_0 = 3/10
model.tau = 10
model.i_hospital = 0.01
model.alpha_0 = 0
model.timesteps = 1.0

model.bigM = 10

model.kappa = 1.0

def cost_func(m):
    return pyo.summation(m.alpha)

model.obj_func = pyo.Objective(rule=cost_func)

model.constraints = pyo.ConstraintList()

model.constraints.add(model.s[0] == model.s_0)
model.constraints.add(model.i[0] == model.i_0)
model.constraints.add(model.alpha[0] == model.alpha_0)

for t in model.T_minus_start:
    model.constraints.add(
        model.s[t] == model.s[t - 1] - (model.beta_0 * (1 - model.kappa * model.alpha[t - 1]) * model.s[t - 1] * model.i[t - 1]) * 1/model.timesteps)
    model.constraints.add(
        model.i[t] == model.i[t - 1] + (model.beta_0 * (1 - model.kappa * model.alpha[t - 1]) * model.s[t - 1] * model.i[t - 1] - model.i[
            t - 1] / model.tau) * 1/model.timesteps)

for t in model.T:
    model.constraints.add(model.i[t] <= model.i_hospital)
    model.constraints.add(model.i[t] <= model.cutoff + model.bigM * model.alpha[t])
    model.constraints.add(model.cutoff - model.bigM * (1 - model.alpha[t]) <= model.i[t])
    # model.constraints.add(model.alpha[t] * (1-model.alpha[t]) == 0)

msolver = pyo.SolverFactory('mindtpy')
solution = msolver.solve(model)
# data = solution.Problem._list

s_vals = []
i_vals = []
alpha_vals = []
triggered_vals = []

for t in model.T:
    s_vals.append(model.s[t].value)
    i_vals.append(model.i[t].value)
    alpha_vals.append(model.alpha[t].value)
    # triggered_vals.append((model.i[t].value >= model.cutoff.value))
    
s_vals = np.array(s_vals)
i_vals = np.array(i_vals)
alpha_vals = np.array(alpha_vals)
triggered_vals = np.array(triggered_vals)

print(alpha_vals)
print(model.cutoff.value)

breakpoint()

plt.clf()
plt.plot(model.beta_0 * model.tau * (1 - np.array(alpha_vals)) * np.array(s_vals), "--")
plt.show()

plt.clf()
plt.plot(np.array(i_vals)/model.i_hospital, "o")
plt.plot(alpha_vals, "+")
plt.show()
# plt.plot(1 - np.array(s_vals) - np.array(i_vals), "+")

breakpoint()

# optimal_s_values = [pyo.value(model.s[key]) for key in model.s]
# optimal_i_values = [pyo.value(model.i[key]) for key in model.i]
# optimal_alpha_values = [pyo.value(model.alpha[key]) for key in model.alpha]

# breakpoint()

# log_infeasible_constraints(model, log_expression=True, log_variables=True)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

# model = model.create_instance()