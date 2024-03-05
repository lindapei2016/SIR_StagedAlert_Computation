###############################################################################

# Contains various routines for parsing data generated from
#   large-scale cluster runs.

# Based off of Dave 1:1 discussion on 02/27/2024 to get
#   more insight into structure of optimal solution
# Look at 2-stage system with only 1 max allowed lockdown

###############################################################################

import numpy as np
import glob
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import optimize

plt.rcParams.update({"font.size": 14})
plt.rcParams.update({"lines.linewidth": 2.0})

import SIR_det_2stages as SIR

eps = 1e-6


###############################################################################

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


###############################################################################

# TODO: change summary output file so it is not a .txt file that
#   must be parsed like this -- just write directly as an organized .csv file!
# 5 columns
# Col 0 has form max_lockdowns + "_beta0_" + beta0 + "_kappa_" + kappa
#   where max_lockdowns is a string in ("1max", "nomax"),
#   beta0 is a string of an integer between 0 and 100, and
#   kappa is a string of an integer between 0 and 100
# Col 1 has form "(0." + float_string where float_string is a
#   string form of a float between 0 and 1
# Col 2 has form float_string + ".0)" where float_string is
#   analogous to above
# Col 3 is the cost (as string)
# Col 4 is number of lockdowns and has form
#   integer_string + ")" where integer_string is a
#   string of an integer
# df = pd.read_csv("SIR_det_2stages_scriptA_inertia.txt", header=None)
df = pd.read_csv("SIR_det_2stages_scriptA.csv", header=None)

# Convert the original .txt/.csv file into a pandas dataframe with the columns
#   "Threshold Up" [float], "Threshold Down" [float], "Cost" [float],
#   "Num Lockdowns" [float], "Max Lockdowns" [float], "Beta0" [int],
#   "Kappa" [int]
df = df.rename(columns={0: "Type", 1: "Threshold Up", 2: "Threshold Down", 3: "Cost", 4: "Num Lockdowns"})
df["Max Lockdowns"] = df["Type"].str.split("_", expand=True)[0].str.split("'", expand=True)[1]
df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("1max", "1", regex=True)
df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("nomax", "inf", regex=True)
df["Max Lockdowns"] = df["Max Lockdowns"].astype("float")
df["Beta0"] = df["Type"].str.split("_", expand=True)[2].astype("int")
df["Kappa"] = df["Type"].str.split("_", expand=True)[4].str.replace("'", "", regex=True).astype("int")
df["Threshold Up"] = df["Threshold Up"].str.split("(", expand=True)[1].astype("float")
df["Threshold Down"] = df["Threshold Down"].str.split(")", expand=True)[0].astype("float")
df["Cost"] = df["Cost"].astype("float")
df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "", regex=True).astype("int")
df.drop(columns="Type", inplace=True)

# Accidentally simulated too many kappas and they are non-sensical --
#   ignore any output from simulations with kappa more than 100%
df = df[df["Kappa"] <= 100]

# Divide by ODE discretization steps
df["Num Lockdowns"] = df["Num Lockdowns"]

df["R0"] = df["Beta0"] / 10  # divide Beta0 by 100 (go from percentage to decimal) then multiply by tau

###############################################################################

# Separate into df for 1 max lockdown and df for no max lockdowns

df_1max = df[df["Max Lockdowns"] == 1]
df_1max = df_1max.sort_values(["R0", "Kappa"])
df_1max_feasible = df_1max[df_1max["Cost"] < np.inf]

df_nomax = df[df["Max Lockdowns"] == np.inf]
df_nomax = df_nomax.sort_values(["R0", "Kappa"])
df_nomax_feasible = df_nomax[df_nomax["Cost"] < np.inf]

###############################################################################

# Get S,I information at start of lockdown, peak under lockdown, and
#   end of lockdown
# Can also simulate this (and can simulate this to check)
# But using numerical integration to solve for these 3 timepoints
#   (note that it actually might be 2 timepoints because
#   there might be no peak under lockdown if the transmission reduction
#   is powerful enough)

df_1max_full_output = df_1max_feasible[df_1max_feasible["Cost"] < max(df_1max_feasible["Cost"])]
df_1max_full_output = df_1max_full_output[df_1max_full_output["Cost"] > 0]

s_at_threshold_up_array = []
s_at_peak_array = []
s_at_threshold_down_array = []

max_infections_array = []

problem = SIR.ProblemInstance()

# Note -- this only makes sense for systems that actually go into lockdown!
# I actually think this works decently well
for i in range(len(df_1max_full_output)):
    row = df_1max_full_output.iloc[i]
    threshold = row["Threshold Up"]
    beta0 = row["Beta0"] / 100.0
    kappa = row["Kappa"] / 100.0

    first_guess = 0.9
    second_guess = 1 / (beta0 * problem.tau)

    up_solution_path_I_start = problem.I_start
    up_solution_path_S_start = problem.S_start

    up_solution_path = build_sol_solution_path_eq(up_solution_path_I_start,
                                                  threshold,
                                                  up_solution_path_S_start,
                                                  beta0 * problem.tau)
    s_at_threshold_up = optimize.fsolve(up_solution_path, first_guess)
    s_at_threshold_up_array.append(s_at_threshold_up)
    # print("Susceptibles at time of entering lockdown " + str(s_at_threshold_up))

    max_infections = compute_max_infections(threshold,
                                            s_at_threshold_up,
                                            beta0 * problem.tau * (1 - kappa))
    peak_solution_path = build_sol_solution_path_eq(threshold,
                                                    max_infections,
                                                    s_at_threshold_up,
                                                    beta0 * problem.tau * (1-kappa))
    s_at_peak = optimize.newton(peak_solution_path, first_guess)
    s_at_peak_array.append(s_at_peak)
    max_infections_array.append(max_infections)
    # print("Max infections " + str(max_infections))

    down_solution_path = build_sol_solution_path_eq(threshold,
                                                    threshold,
                                                    s_at_threshold_up,
                                                    beta0 * problem.tau * (1 - kappa))
    s_at_threshold_down = optimize.newton(down_solution_path, second_guess)
    s_at_threshold_down_array.append(s_at_threshold_down)
    # print("Susceptibles at time of leaving lockdown " + str(s_at_threshold_down))

df_1max_full_output["S Up"] = np.asarray(s_at_threshold_up_array)
df_1max_full_output["S Peak"] = np.asarray(s_at_peak_array)
df_1max_full_output["S Down"] = np.asarray(s_at_threshold_down_array)
df_1max_full_output["I Peak"] = np.asarray(max_infections_array)

# Effective reproduction number
df_1max_full_output["Re Up"] = df_1max_full_output["S Up"] * df_1max_full_output["Beta0"]/100.0 * problem.tau
df_1max_full_output["Re Peak"] = df_1max_full_output["S Peak"] * df_1max_full_output["Beta0"]/100.0 * problem.tau * \
                               (1-df_1max_full_output["Kappa"]/100.0)
df_1max_full_output["Re Down"] = df_1max_full_output["S Down"] * df_1max_full_output["Beta0"]/100.0 * problem.tau

# df_subset = df_1max_full_output[df_1max_full_output["Kappa"] == 60]
# plt.plot(df_subset["R0"], df_subset["Re Up"], label="Effective R, Nominal Transmission, Lockdown Start")
# plt.plot(df_subset["R0"], df_subset["Re Peak"], label="Effective R, Reduced Transmission, Infections Peak")
# plt.plot(df_subset["R0"], df_subset["Re Down"], label="Effective R, Nominal Transmission, Lockdown End")
# plt.legend()

# Let's get the effective reproduction number through time
# Say, for kappa = 60
S_matrix = []
I_matrix = []
R_matrix = []
Re_matrix = []

# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

c1 = "white"
c2 = "red"

df_subset = df_1max_full_output[(df_1max_full_output["Kappa"] == 60) & (df_1max_full_output["R0"] >= 3.8)]
for i in range(len(df_subset)):
    row = df_subset.iloc[i]
    problem = SIR.ProblemInstance()
    problem.threshold_up = row["Threshold Up"]
    problem.threshold_down = row["Threshold Down"]
    problem.beta0 = row["Beta0"]/100.0
    problem.kappa = row["Kappa"]/100.0
    problem.max_lockdowns_allowed = 1
    problem.full_output = True
    problem.simulate_policy()
    S_matrix.append(problem.results.S)
    I_matrix.append(problem.results.I)
    R_matrix.append(problem.results.R)
    re = np.asarray(problem.results.S) * problem.beta0 * problem.tau * (1 - (np.asarray(problem.results.I) > problem.threshold_up - 5e-2) * problem.kappa)
    Re_matrix.append(re)
    start = np.argwhere((np.asarray(problem.results.I) > problem.threshold_up - 5e-4))[0]
    peak = np.argwhere(np.asarray(problem.results.I) == np.max(problem.results.I))[0]
    end = np.argwhere((np.asarray(problem.results.I) > problem.threshold_up - 5e-4))[-1]
    A = problem.results.R[end] - problem.results.R[peak]
    B = problem.results.R[peak] - problem.results.R[start]
    print(A)
    print(B)
    print(A / B)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # plt.plot(re, color=colorFader(c1, c2, mix=i/len(df_subset)))
    plt.plot(np.asarray(problem.results.R), color=colorFader(c1, c2, mix=i / len(df_subset)))

breakpoint()

for kappa in (50, 60, 70):
    df_subset = df_1max_full_output[df_1max_full_output["Kappa"] == kappa]
    plt.clf()
    plt.plot(df_subset["R0"], df_subset["S Up"], label="S at Lockdown Start")
    plt.plot(df_subset["R0"], df_subset["S Down"], label="S at Lockdown End")
    plt.plot(df_subset["R0"], df_subset["S Peak"], label="S at I Peak")
    plt.plot(df_subset["R0"], df_subset["I Peak"], color="red", label="I Peak")
    plt.xlabel("Basic Reproduction Number R0")
    plt.ylabel("Proportion")
    plt.title("Kappa " + str(kappa) + ", Sample Paths of Optimal Thresholds")
    plt.legend()
    plt.savefig("sample_paths_" + str(kappa) + ".png", dpi=1200)

    plt.clf()
    df_subset = df_1max_full_output[df_1max_full_output["Kappa"] == kappa]
    plt.xlabel("Basic Reproduction Number R0")
    plt.ylabel("Optimal Threshold")
    plt.title("Kappa " + str(kappa) + ", Optimal Thresholds")
    plt.plot(df_subset["R0"], df_subset["Threshold Up"])
    plt.savefig("optimal_threshold_" + str(kappa) + ".png", dpi=1200)

breakpoint()

# Fix kappa...

###############################################################################

plt.clf()

optimal_kappas_1max = []

for r in df_1max_feasible["R0"].unique():
    optimal_kappas_1max.append(df_1max_feasible[df_1max_feasible["R0"] == r].min()["Kappa"])

breakpoint()

# Heatmap -- for beta0 and kappa, what is optimal threshold? (for 1 max lockdown
#   and no max lockdowns)
# Similarly, what is optimal cost?
# Optimal lockdowns?

# Lineplot similar to Fujiwara -- x-axis is beta0, y-axis is
#   optimal value of kappa (that gives lowest cost out of all kappas)

###############################################################################
