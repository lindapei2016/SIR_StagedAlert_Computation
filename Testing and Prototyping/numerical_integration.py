import numpy as np
import SIR_det_2stages as SIR
from scipy import optimize
import matplotlib.pyplot as plt


# Check analytically to see if a threshold is feasible
#   (for two stages, a "normal" stage and a lockdown stage).
# Use a root-finding method to find the value of S

# One version of the equation has I0_val + S0_val simply replaced by 1
#   but this assumes that I0_val + S0_val = 1. This assumption
#   does not hold for our use-case of max infections.
def compute_max_infections(I0_val, S0_val, R0_val):
    return I0_val + S0_val - 1 / R0_val - np.log(R0_val * S0_val) / R0_val


def build_sol_solution_path_eq(I0_val, I_val, S0_val, R0_val):
    def sol_solution_path_eq(S_val):
        '''
        Solution curve is I = I0 + S0 - S + log(S/S0)/R0
            (see Hethcote 2000 "The Mathematics of
            Infectious Diseases").
        Solution curve equation: for a given proportion
            susceptible, returns difference between
            pre-specified proportion infected and
            actual proportion infected when proportion
            susceptible equals
        Want to find root of this equation to find
            S that corresponds with I
        :param S_val: [scalar in [0,1]] proportion susceptible.
        :return: [scalar] actual proportion infected when
            proportion susceptible equals S_val minus I_val
        '''
        return I0_val + S0_val - I_val - S_val + np.log(S_val / S0_val) / R0_val

    return sol_solution_path_eq


# eps = 1e-6
eps = 1e-4

problem = SIR.ProblemInstance()
problem.beta0 = 3 / 10.0
problem.max_lockdowns_allowed = np.inf
problem.I_constraint = 0.1
problem.kappa = 5 / 10.0
problem.threshold_up = 0.05
problem.threshold_down = 0.05
problem.simulate_policy()

s_at_threshold_up = 1
s_at_threshold_down = 1

first_guess = 0.9
second_guess = 1 / (problem.beta0 * problem.tau)

first_solution_path_I_start = problem.I_start
first_solution_path_S_start = problem.S_start

# Need to figure something out -- something is weird here...
# Oh the goofiness is happening because I have an arbitrary number of lockdowns
#   and the bouncy-bounce is happening towards the end

while s_at_threshold_up >= 1 / (problem.beta0 * problem.tau) + eps and \
        s_at_threshold_down >= 1 / (problem.beta0 * problem.tau) + eps:
    first_solution_path = build_sol_solution_path_eq(first_solution_path_I_start,
                                                     problem.threshold_up,
                                                     first_solution_path_S_start,
                                                     problem.beta0 * problem.tau)
    s_at_threshold_up = optimize.fsolve(first_solution_path, first_guess)
    print("Susceptibles at time of entering lockdown " + str(s_at_threshold_up))
    print("Max infections " + str(compute_max_infections(problem.threshold_up,
                                                         s_at_threshold_up,
                                                         problem.beta0 * problem.tau * (1 - problem.kappa))))
    second_solution_path = build_sol_solution_path_eq(problem.threshold_up,
                                                      problem.threshold_up,
                                                      s_at_threshold_up,
                                                      problem.beta0 * problem.tau * (1 - problem.kappa))
    s_at_threshold_down = optimize.newton(second_solution_path, second_guess)
    print("Susceptibles at time of leaving lockdown " + str(s_at_threshold_down))
    first_solution_path_I_start = problem.threshold_up
    first_solution_path_S_start = s_at_threshold_down
    first_guess = second_guess
    breakpoint()

plt.clf()
plt.plot(problem.results.S)
plt.plot(problem.results.I)
plt.show()

breakpoint()
