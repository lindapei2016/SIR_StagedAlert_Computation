from scipy.optimize import rosen, rosen_der, rosen_hess
from cyipopt import minimize_ipopt
from scipy.sparse import coo_array
import numpy as np

# Examples from
# https://cyipopt.readthedocs.io/en/stable/tutorial.html

def con(x):
    return np.array([ 10 -x[1]**2 - x[2], 100.0 - x[4]**2 ])

def con_jac(x):
    # Dense Jacobian:
    # J = (0  -2*x[1]   -1   0     0     )
    #         (0   0         0   0   -2*x[4] )
    # Sparse Jacobian (COO)
    rows = np.array([0, 0, 1])
    cols = np.array(([1, 2, 4]))
    data = np.array([-2*x[1], -1, -2*x[4]])
    return coo_array((data, (rows, cols))).toarray()

def con_hess(x, _lambda):
    H1 = np.array([
        [0,  0, 0, 0, 0],
        [0, -2, 0, 0, 0 ],
        [0,  0, 0, 0, 0 ],
        [0,  0, 0, 0, 0 ],
        [0,  0, 0, 0, 0 ]
    ])

    H2 = np.array([
        [0, 0, 0, 0,  0],
        [0, 0, 0, 0,  0],
        [0, 0, 0, 0,  0],
        [0, 0, 0, 0,  0],
        [0, 0, 0, 0, -2]
    ])
    return _lambda[0] * H1 + _lambda[1] * H2

constr = {'type': 'ineq', 'fun': con, 'jac': con_jac, 'hess': con_hess}

# initial guess
x0 = np.array([1.1, 1.1, 1.1, 1.1, 1.1])

# solve the problem
res = minimize_ipopt(rosen, jac=rosen_der, hess=rosen_hess, x0=x0, constraints=constr)

breakpoint()