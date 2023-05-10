###############################################################################

# Contains various routines for parsing data generated from
#   large-scale cluster runs.

# Note -- a different/conflicting copy of script_parsing.py may exist
#   in the results folder containing .csv files due to scp from
#   the cluster. Make sure the correct script_parsing.py is run
#   (from the outer directory).

###############################################################################

import numpy as np
import glob
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
plt.rcParams.update({"lines.linewidth":2.0})

import SIR_det_2stages as SIR

eps = 1e-6


###############################################################################


def aggregate_cost_brute_force_cluster_csvs(policies, glob_expr, split_expr, df_name_prefix):
    '''
    Called to aggregate cost_history .csv files generated from
        running brute force optimization (testing all combos on
        a fine grid) on the cluster
    Called on cluster output after running SIR_det_2stages.find_optimum()
    Exports aggregated .csv file with columns "Kappa", "Thresholds", "Cost"
        corresponding to optimal (min cost) thresholds for that
        value of kappa (transmission reduction under lockdown)
    :param policies: [list-like] list/array/tuple of tuples -- ith element
        corresponds to ith policy -- should be in the same order as
        the list of tuples given to the optimization -- used to match
        cluster output with
    :param glob_expr: [str] path with pattern matching to select
        filenames of interest
    :param split_expr: [str] substring of glob_expr immediately before
        kappa ID -- used to split each filename of interest and
        process filename for meaningful information
    :param df_name_prefix: [str] prefix for .csv file that
        stores aggregated pandas dataframe
    :return: [None]
    '''

    # SIR_det_2stages.find_optimum() creates .csvs with the common term "cost_history"
    #   so select those .csvs to aggregate cost information
    filenames = glob.glob(glob_expr)

    cost_history_df = pd.DataFrame(columns=["Kappa", "Thresholds", "Cost"])

    # Issue is that sorting the filenames does not create the proper numerical order
    # Extract numerical ID of filename
    # ID / 100 is the kappa2 used

    for filename in filenames:
        cost_history = np.genfromtxt(filename, delimiter=",")
        best_policy_ix = np.argmin(cost_history)
        thresholds = policies[best_policy_ix]
        best_cost = cost_history[best_policy_ix]
        kappa = int(filename.split(split_expr)[-1].split("_")[0]) / 100
        cost_history_df.loc[len(cost_history_df)] = [kappa, thresholds, best_cost]

    cost_history_df = cost_history_df.sort_values(by="Kappa")

    cost_history_df.to_csv(df_name_prefix + "_cost_history_df.csv", sep=",")


###############################################################################


def clean_brute_force_cluster_txt(filename, df_name_prefix):
    '''
    Called to clean summary .txt file generated from
        running brute force optimization (testing all combos on
        a fine grid) on the cluster
    Called on cluster output after running SIR_det_2stages.find_optimum()
    Exports cleaned .csv file with columns "Threshold Type", "Max Lockdowns", "Kappa",
        "Threshold Up", "Threshold Down", "Cost", "Num Lockdowns",
        where last 4 columns contain details of optimal (min cost) thresholds for
        particular threshold type, maximum number of lockdowns allowed, and
        kappa (transmission reduction under lockdown)
    :param filename: [str] filename of .txt file to clean
    :param df_name_prefix: [str] prefix for .csv file that
        stores cleaned pandas dataframe
    :return: [None]
    '''
    df = pd.read_csv(filename, header=None)
    df = df.rename(columns={0: "Type", 1: "Threshold Up", 2: "Threshold Down", 3: "Cost", 4: "Num Lockdowns"})
    df["Type"] = df["Type"].str.replace("(", "", regex=True)
    df[["Kappa", "Threshold Type", "Max Lockdowns"]] = df["Type"].str.split("_", expand=True)
    df = df.drop(columns="Type")
    df["Threshold Up"] = df["Threshold Up"].str.replace("(", "", regex=True)
    df["Threshold Down"] = df["Threshold Down"].str.replace(")", "", regex=True)
    df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "", regex=True)
    df["Kappa"] = df["Kappa"].str.replace("'", "", regex=True)
    df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("'", "", regex=True)
    df["Kappa"] = df["Kappa"].astype(float)
    df["Kappa"] = df["Kappa"] / 100
    df = df.sort_values(by=["Threshold Type", "Max Lockdowns", "Kappa"])
    df = df[["Threshold Type", "Max Lockdowns", "Kappa", "Threshold Up", "Threshold Down", "Cost", "Num Lockdowns"]]

    df = df[df["Cost"] != " inf"]
    df["Cost"] = df["Cost"].str.replace(" ", "", regex=True)

    # Have to convert to float before converting to int
    df["Cost"] = df["Cost"].astype("float")
    df["Cost"] = df["Cost"].astype("int")

    df["Threshold Up"] = df["Threshold Up"].astype(float)
    df["Threshold Down"] = df["Threshold Down"].astype(float)
    df = df.round({"Threshold Up": 3, "Threshold Down" : 3})

    df.to_csv(df_name_prefix + "_cleaned.csv", sep=",")


###############################################################################

need_clean = False

if need_clean == True:

    filename = "/Users/linda/Dropbox/RESEARCH/CurrentResearchProjects/StagedAlertTheory/" \
                "StagedAlertTheoryCode/Results/05032023a/SIR_det_2stages_scriptB.txt"

    df_name_prefix = "constraint0.1_inertia0"

    clean_brute_force_cluster_txt(filename, df_name_prefix)

    # Manual fix -- for some reason two processors think they are rank 57...
    #   so remove entries with kappa 0.57 until this problem is solved
    df = pd.read_csv("constraint0.1_inertia0_cleaned.csv")
    df = df[df["Kappa"] != 0.57]
    df.to_csv("constraint0.1_inertia0_cleaned.csv")

###############################################################################

problem = SIR.ProblemInstance()
problem.max_lockdowns_allowed = 1

# problem = ProblemInstance()
# problem.inertia = 0
# problem.kappa = 0.5
# problem.threshold_up = 0.067
# problem.threshold_down = 0.09
# problem.full_output = True
# problem.simulate_policy()

df = pd.read_csv("constraint0.1_inertia0_cleaned.csv")

compute_diff = False

if compute_diff == True:

    kappas = np.array(df["Kappa"])
    policies = list(zip(df["Threshold Up"], df["Threshold Down"]))

    S_left_during_lockdown = []
    I_left_during_lockdown = []

    for i in range(len(kappas)):

        problem.kappa = kappas[i]
        problem.threshold_up = policies[i][0]
        problem.threshold_down = policies[i][1]
        problem.full_output = True
        problem.simulate_policy()
        S = problem.results.S
        I = problem.results.I
        x0 = problem.results.x0
        S_diff_ix_start = np.argwhere(x0==0)[0] - 1
        S_diff_ix_end = np.argwhere(x0==0)[-1]
        S_left_during_lockdown.append(S[S_diff_ix_start]-S[S_diff_ix_end])
        I_left_during_lockdown.append(I[S_diff_ix_start] - I[S_diff_ix_end])

    df["S_left_during_lockdown"] = np.array(S_left_during_lockdown).flatten()
    df["I_left_during_lockdown"] = np.array(I_left_during_lockdown).flatten()

df.to_csv("constraint0.1_inertia0_cleaned.csv")

###############################################################################

asymmetric_max1 = df[(df["Threshold Type"] == "asymmetric") & (df["Max Lockdowns"] == "max1")]
symmetric_max1 = df[(df["Threshold Type"] == "symmetric") & (df["Max Lockdowns"] == "max1")]

a_m1_S_left = asymmetric_max1["S_left_during_lockdown"]
a_m1_I_left = asymmetric_max1["I_left_during_lockdown"]

a_m1_kappa = np.array(asymmetric_max1["Kappa"], dtype=float)
s_m1_kappa = np.array(asymmetric_max1["Kappa"], dtype=float)

S_m1_S_left = symmetric_max1["S_left_during_lockdown"]
S_m1_I_left = symmetric_max1["I_left_during_lockdown"]

marker_grid = np.arange(5, len(a_m1_kappa), 5)

plt.plot(a_m1_kappa, a_m1_S_left, color="yellowgreen", linestyle=":", marker="o", markevery=marker_grid, label="S Diff, Asymmetric")
plt.plot(s_m1_kappa, S_m1_S_left, color="lightseagreen", linestyle=":", marker="D", markevery=marker_grid, label="S Diff, Symmetric")
plt.plot(a_m1_kappa, a_m1_I_left, color="darkorange", marker="o", markevery=marker_grid, label="I Diff, Asymmetric")
plt.plot(s_m1_kappa, S_m1_I_left, color="red", marker="D", markevery=marker_grid, linestyle=":", label="I Diff, Symmetric")
plt.xlabel("Kappa (Transmission Reduction)")
plt.ylabel("Proportion")
plt.title("Differences Before and After Lockdown vs Kappa")
plt.legend()
plt.savefig('constraint1e-1_differences_kappa.eps')

plt.show()

plt.clf()

###############################################################################

asymmetric_max1 = df[(df["Threshold Type"] == "asymmetric") & (df["Max Lockdowns"] == "max1")]
symmetric_max1 = df[(df["Threshold Type"] == "symmetric") & (df["Max Lockdowns"] == "max1")]
asymmetric_nomax = df[(df["Threshold Type"] == "asymmetric") & (df["Max Lockdowns"] == "nomax")]
symmetric_nomax = df[(df["Threshold Type"] == "symmetric") & (df["Max Lockdowns"] == "nomax")]

# Threshold Down versus kappas

a_m1 = np.array(asymmetric_max1["Threshold Down"], dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"], dtype=float)
s_m1 = np.array(symmetric_max1["Threshold Down"], dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"], dtype=float)
a_nm = np.array(asymmetric_nomax["Threshold Down"], dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"], dtype=float)
s_nm = np.array(symmetric_nomax["Threshold Down"], dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"], dtype=float)

plt.plot(a_m1_kappa, a_m1, color="yellowgreen", linestyle=":", marker="o", markevery=marker_grid, label="Asymmetric, Max 1")
plt.plot(s_m1_kappa, s_m1, color="lightseagreen", linestyle=":", marker="D", markevery=marker_grid, label="Symmetric, Max 1")
plt.plot(a_nm_kappa, a_nm, color="darkorange", marker="o", markevery=marker_grid, label="Asymmetric, No Max")
plt.plot(s_nm_kappa, s_nm, color="chocolate", marker="D", markevery=marker_grid, label="Symmetric, No Max")
plt.title("Threshold Down vs Kappa")
plt.legend()
plt.ylabel("Threshold Down")
plt.xlabel("Kappa (Transmission Reduction)")
# plt.show()

plt.savefig('constraint1e-1_down_kappa.svg', format='svg', dpi=1200)
plt.savefig('constraint1e-1_down_kappa.eps')
plt.clf()

# Threshold Up versus kappas

a_m1 = np.array(asymmetric_max1["Threshold Up"], dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"], dtype=float)
s_m1 = np.array(symmetric_max1["Threshold Up"], dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"], dtype=float)
a_nm = np.array(asymmetric_nomax["Threshold Up"], dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"], dtype=float)
s_nm = np.array(symmetric_nomax["Threshold Up"], dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"], dtype=float)

plt.plot(a_m1_kappa, a_m1, color="yellowgreen", linestyle=":", marker="o", markevery=marker_grid, label="Asymmetric, Max 1")
# plt.plot(s_m1_kappa, s_m1, color="lightseagreen", linestyle=":", marker="D", markevery=marker_grid, label="Symmetric, Max 1")
plt.plot(a_nm_kappa, a_nm, color="darkorange", marker="o", markevery=marker_grid, label="Asymmetric, No Max")
# plt.plot(s_nm_kappa, s_nm, color="chocolate", marker="D", markevery=marker_grid, label="Symmetric, No Max")
plt.title("Threshold Up vs Kappa")
plt.legend()
plt.ylabel("Threshold Up")
plt.xlabel("Kappa (Transmission Reduction)")
# plt.show()

breakpoint()

plt.savefig('constraint1e-1_up_kappa.svg', format='svg', dpi=1200)
plt.savefig('constraint1e-1_up_kappa.eps')
plt.clf()

# Cost versus kappas

a_m1 = np.array(asymmetric_max1["Cost"], dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"], dtype=float)
s_m1 = np.array(symmetric_max1["Cost"], dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"], dtype=float)
a_nm = np.array(asymmetric_nomax["Cost"], dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"], dtype=float)
s_nm = np.array(symmetric_nomax["Cost"], dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"], dtype=float)

plt.plot(a_m1_kappa, np.log(a_m1), color="yellowgreen", linestyle=":", marker="o", markevery=marker_grid, label="Asymmetric, Max 1")
plt.plot(s_m1_kappa, np.log(s_m1), color="lightseagreen", linestyle=":", marker="D", markevery=marker_grid, label="Symmetric, Max 1")
plt.plot(a_nm_kappa, np.log(a_nm), color="darkorange", marker="o", markevery=marker_grid, label="Asymmetric, No Max")
plt.plot(s_nm_kappa, np.log(s_nm), color="chocolate", marker="D", markevery=marker_grid, label="Symmetric, No Max")
plt.title("Log Time in Lockdown vs Kappa")
plt.legend()
plt.ylabel("Log Time in Lockdown (Simulation Time Units)")
plt.xlabel("Kappa (Transmission Reduction)")
# plt.show()

plt.savefig('constraint1e-1_time_kappa.svg', format='svg', dpi=1200)
plt.savefig('constraint1e-1_time_kappa.eps')
plt.clf()

breakpoint()

###############################################################################

glob_expr = "/Users/linda/Dropbox/RESEARCH/CurrentResearchProjects/StagedAlertTheory/" \
            "StagedAlertTheoryCode/Results/05032023a/*asymmetric*nomax*cost*"

split_expr = "05032023a/"

problem = SIR.ProblemInstance()

asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                    (0, problem.I_constraint + eps, problem.grid_grain),
                                                    symmetric=False)

aggregate_cost_brute_force_cluster_csvs(policies=asymmetric_policies,
                                        glob_expr=glob_expr,
                                        split_expr=split_expr,
                                        df_name_prefix="asymmetric_nomax")

breakpoint()

###############################################################################


# Some old parsing -- rework -- can probably just use
#   clean_brute_force_cluster_txt() as helper function

# df = pd.read_csv("SIR_det_2stages_Iconstraint0.2.csv", header=None)
# df = df.rename(columns={0: "Type", 1: "Threshold Up", 2: "Threshold Down", 3: "Cost", 4: "Num Lockdowns"})
# df[["Kappa", "I Constraint", "Threshold Type", "Max Lockdowns"]] = df["Type"].str.split("_", expand=True)
# df = df[df["I Constraint"].str.contains("0.2")]
# df = df.drop(columns="Type")
# df = df.drop(columns="I Constraint")
# df["Threshold Up"] = df["Threshold Up"].str.replace("(", "", regex=True)
# df["Threshold Down"] = df["Threshold Down"].str.replace(")", "", regex=True)
# df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "", regex=True)
# df["Kappa"] = df["Kappa"].str.replace("'", "", regex=True)
# df["Kappa"] = df["Kappa"].str.replace("(", "", regex=True)
# df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("'", "", regex=True)
# df["Kappa"] = df["Kappa"].astype(float)
# df["Kappa"] = df["Kappa"] / 100
# df = df.sort_values(by=["Threshold Type", "Max Lockdowns", "Kappa"])
# df = df[["Threshold Type", "Max Lockdowns", "Kappa", "Threshold Up", "Threshold Down", "Cost", "Num Lockdowns"]]
# df.to_csv("SIR_det_2stages_Iconstraint0.2_cleaned.csv", sep=",")

###############################################################################
