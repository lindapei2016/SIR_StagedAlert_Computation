import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
import numpy as np

from jax.test_util import check_grads

import matplotlib.pyplot as plt

# jax._src.config.traceback_filtering

import SIR_det_2stages as SIR_2stages
import time

eps = 1e-6

ODE_steps = 10.0
max_t = 10 * 100
beta0 = 3/10
tau = 10.0

def SIR(kappa):

    S = 0.99
    I = 0.01
    R = 0

    for t in range(1, max_t):

        beta = beta0 * (1 - kappa)
        S = S - (beta * S * I) * 1 / ODE_steps
        I = I + (beta * S * I - I / tau) * 1 / ODE_steps
        R = R + (I / tau) * (1 / ODE_steps)

    return I

print(grad(SIR)(0.1))
print((SIR(0.1+eps)-SIR(0.1-eps))/(2*eps))

def SIR_I_peak(kappa):

    S = 0.99
    I = 0.01
    R = 0

    I_history = []

    for t in range(1, max_t):

        beta = beta0 * (1 - kappa)
        S = S - (beta * S * I) * 1 / ODE_steps
        I = I + (beta * S * I - I / tau) * 1 / ODE_steps
        R = R + (I / tau) * (1 / ODE_steps)

        I_history.append(I)

    return max(I_history)

print(grad(SIR_I_peak)(0.1))
print((SIR_I_peak(0.1+eps)-SIR_I_peak(0.1-eps))/(2*eps))

def SIR_I_history(kappa):

    S = 0.99
    I = 0.01
    R = 0

    I_history = []

    for t in range(1, max_t):

        beta = beta0 * (1 - kappa)
        S = S - (beta * S * I) * 1 / ODE_steps
        I = I + (beta * S * I - I / tau) * 1 / ODE_steps
        R = R + (I / tau) * (1 / ODE_steps)

        I_history.append(I)

    return I_history

# print(grad(SIR_I_peak)(0.1))
# print((SIR_I_peak(0.1+eps)-SIR_I_peak(0.1-eps))/(2*eps))

# Also would like to test if can take derivative of SIR problem with
#   2-stage threshold

def problem_wrapper_function(threshold_up):
    problem = SIR_2stages.ProblemInstance()
    problem.threshold_up = threshold_up
    problem.threshold_down = threshold_up
    problem.simulate_policy()
    return max(problem.results.I)

eps = 1e-4

start = time.time()
print(grad(problem_wrapper_function)(0.04))
print((problem_wrapper_function(0.04+eps)-problem_wrapper_function(0.04-eps))/(2*eps))
print(time.time() - start)

breakpoint()

