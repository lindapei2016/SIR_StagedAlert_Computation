import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
import numpy as np

from jax.test_util import check_grads

# jax._src.config.traceback_filtering

ODE_steps = 10.0
max_t = 100
beta0 = 3/10
tau = 10.0

def SIR(kappa):

    S = 0.1
    I = 0.9
    R = 0

    for t in range(1, max_t):

        beta = beta0 * (1 - kappa)
        S = S - (beta * S * I) * 1 / ODE_steps
        I = I + (beta * S * I - I / tau) * 1 / ODE_steps
        R = R + (I / tau) * (1 / ODE_steps)

    return I

SIR(0.1)

eps = 0.001
print((SIR(0.1+eps)-SIR(0.1-eps))/(2*eps))
