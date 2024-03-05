###############################################################################

# Contains various routines for parsing data generated from
#   large-scale cluster runs.

# Note -- a different/conflicting copy of script_parsing.py may exist
#   in the results folder containing .csv files due to scp from
#   the cluster. Make sure the correct script_parsing.py is run
#   (from the outer directory).

# TODO:
# Inertia for 10 "days" and 20 "days"? I think the smaller inertia,
#   the lower the cost
# *** for one max lockdown, optimal changes that happen
#   during this lockdown! Optimal proportion of people infected during lockdown
#   and optimal proportion of people who recover during lockdown

###############################################################################

import numpy as np
import glob
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams.update({"font.size": 14})
plt.rcParams.update({"lines.linewidth":2.0})

import SIR_det_2stages as SIR

eps = 1e-6

###############################################################################

plot_heatmaps = True
plot_optimal_kappa = False
get_max_infected = False

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
df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "", regex=True).astype("float")
df.drop(columns="Type", inplace=True)

# Accidentally simulated too many kappas and they are non-sensical --
#   ignore any output from simulations with kappa more than 100%
df = df[df["Kappa"] <= 100]

# Divide by ODE discretization steps
# df["Cost"] = df["Cost"] / 1000
df["Num Lockdowns"] = df["Num Lockdowns"]

df["R0"] = df["Beta0"]/10 # divide Beta0 by 100 (go from percentage to decimal) then multiply by tau
# df["Kappa"] = df["Kappa"]/100

df_1max = df[df["Max Lockdowns"]==1]
df_1max = df_1max.sort_values(["R0","Kappa"])
df_1max_feasible = df_1max[df_1max["Cost"] < np.inf]

df_nomax = df[df["Max Lockdowns"]==np.inf]
df_nomax = df_nomax.sort_values(["R0","Kappa"])
df_nomax_feasible = df_nomax[df_nomax["Cost"] < np.inf]

breakpoint()

###############################################################################

if get_max_infected:

    # 1 max lockdown

    # df_1max_full_output = df_1max_feasible[df_1max_feasible["Cost"] < max(df_1max_feasible["Cost"])]
    #
    # max_infected = []
    #
    # for i in range(len(df_1max_full_output)):
    # # for i in [1]:
    #     policy = df_1max_full_output.iloc[i]
    #     problem = SIR.ProblemInstance()
    #     problem.full_output = True
    #     problem.beta0 = policy["Beta0"]/100
    #     problem.kappa = policy["Kappa"]/100
    #     problem.threshold_up = policy["Threshold Up"]
    #     problem.threshold_down = policy["Threshold Down"]
    #     problem.max_lockdowns_allowed = policy["Max Lockdowns"]
    #     problem.simulate_policy()
    #     max_infected.append(max(problem.results.I))
    #
    #     if i % 100 == 0:
    #         print(i)
    #
    # df_1max_full_output["Max Infected"] = np.array(max_infected)
    # df_1max_full_output.to_csv("df_1max_full_output.csv", header=True)

    # No max lockdowns
    
    df_nomax_full_output = df_nomax_feasible[df_nomax_feasible["Cost"] < max(df_nomax_feasible["Cost"])]

    max_infected = []

    for i in range(len(df_nomax_full_output)):
    # for i in [1]:
        policy = df_nomax_full_output.iloc[i]
        problem = SIR.ProblemInstance()
        problem.full_output = True
        problem.beta0 = policy["Beta0"]/100
        problem.kappa = policy["Kappa"]/100
        problem.threshold_up = policy["Threshold Up"]
        problem.threshold_down = policy["Threshold Down"]
        problem.max_lockdowns_allowed = policy["Max Lockdowns"]
        problem.simulate_policy()
        max_infected.append(max(problem.results.I))

        if i % 100 == 0:
            print(i)

    df_nomax_full_output["Max Infected"] = np.array(max_infected)
    df_nomax_full_output.to_csv("df_nomax_full_output.csv", header=True)

    breakpoint()

###############################################################################

breakpoint()

if plot_optimal_kappa:
    plt.clf()

    optimal_kappas_1max = []

    for r in df_1max_feasible["R0"].unique():
        optimal_kappas_1max.append(df_1max_feasible[df_1max_feasible["R0"] == r].min()["Kappa"])

breakpoint()

if plot_heatmaps:

    # 1 max lockdown

    # plt.clf()
    # df_1max_heatmap_df = df_1max_feasible[df_1max_feasible["Cost"] != 0].pivot_table(index="R0",columns="Kappa",values="Threshold Up")
    # sns.heatmap(df_1max_heatmap_df, cmap="viridis")
    # plt.xlabel("Kappa (Transmission Reduction) Percentage")
    # plt.ylabel("R0 (Basic Reproduction Number)")
    # plt.title("1 Max Lockdown, Optimal Threshold")
    # plt.savefig("1max_threshold.png", dpi=1200)
    #
    # plt.clf()
    # df_1max_heatmap_df = df_1max_feasible[df_1max_feasible["Cost"] != 0].pivot_table(index="R0",columns="Kappa",values="Cost")
    # sns.heatmap(df_1max_heatmap_df, cmap="viridis")
    # plt.xlabel("Kappa (Transmission Reduction) Percentage")
    # plt.ylabel("R0 (Basic Reproduction Number)")
    # plt.title("1 Max Lockdown, Optimal Days in Lockdown")
    # plt.savefig("1max_cost.png", dpi=1200)

    # No max lockdowns

    plt.clf()
    df_nomax_heatmap_df = df_nomax_feasible[df_nomax_feasible["Cost"] != 0].pivot_table(index="R0",columns="Kappa",values="Threshold Up")
    sns.heatmap(df_nomax_heatmap_df, cmap="viridis")
    plt.xlabel("Kappa (Transmission Reduction) Percentage")
    plt.ylabel("R0 (Basic Reproduction Number)")
    plt.title("No Max Lockdowns, Optimal Threshold")
    plt.savefig("nomax_threshold.png", dpi=1200)

    plt.clf()
    df_nomax_heatmap_df = df_nomax_feasible[df_nomax_feasible["Cost"] != 0].pivot_table(index="R0",columns="Kappa",values="Cost")
    sns.heatmap(df_nomax_heatmap_df, cmap="viridis")
    plt.xlabel("Kappa (Transmission Reduction) Percentage")
    plt.ylabel("R0 (Basic Reproduction Number)")
    plt.title("No Max Lockdowns, Optimal Cost")
    plt.savefig("nomax_cost.png", dpi=1200)

    plt.clf()
    df_nomax_heatmap_df = df_nomax_feasible[df_nomax_feasible["Cost"] != 0].pivot_table(index="R0",columns="Kappa",values="Num Lockdowns")
    sns.heatmap(df_nomax_heatmap_df, cmap="viridis")
    plt.xlabel("Kappa (Transmission Reduction) Percentage")
    plt.ylabel("R0 (Basic Reproduction Number)")
    plt.title("No Max Lockdowns, Number of Lockdowns")
    plt.savefig("nomax_lockdowns.png", dpi=1200)

breakpoint()

# Heatmap -- for beta0 and kappa, what is optimal threshold? (for 1 max lockdown
#   and no max lockdowns)
# Similarly, what is optimal cost?
# Optimal lockdowns?

# Lineplot similar to Fujiwara -- x-axis is beta0, y-axis is
#   optimal value of kappa (that gives lowest cost out of all kappas)




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
    df["Type"] = df["Type"].str.replace("('", "", regex=True)
    df["Threshold Up"] = df["Threshold Up"].str.replace("(", "", regex=True)
    df["Threshold Down"] = df["Threshold Down"].str.replace(")", "", regex=True)
    df["Num Lockdowns"] = df["Type"]
    df["Transmission Rate"] = df["Type"]
    df["Reduction"] = df["Type"]
    df = df.drop(columns="Type")
    df["Max Lockdowns"].str.split("_", expand=True)[0]
    breakpoint()

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

# Change filenames accordingly

need_clean = True

if need_clean:

    filename = "SIR_det_2stages_scriptA.txt"

    df_name_prefix = "SIR_det_2stages_scriptA"

    clean_brute_force_cluster_txt(filename, df_name_prefix)

breakpoint()

###############################################################################

glob_expr = "/Users/linda/Dropbox/RESEARCH/CurrentResearchProjects/StagedAlertTheory/" \
            "StagedAlertTheoryCode/*"

# nomax_beta0_57_kappa_70_num_lockdowns_history

split_expr = "nomax_beta0_57_kappa_"

problem = SIR.ProblemInstance()

asymmetric_policies = SIR.ProblemInstance.thresholds_generator((0, problem.I_constraint + eps, problem.grid_grain),
                                                    (0, problem.I_constraint + eps, problem.grid_grain),
                                                    symmetric=False)

aggregate_cost_brute_force_cluster_csvs(policies=asymmetric_policies,
                                        glob_expr=glob_expr,
                                        split_expr=split_expr,
                                        df_name_prefix="asymmetric_nomax")

breakpoint()