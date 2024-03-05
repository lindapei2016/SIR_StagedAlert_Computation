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

# breakpoint()

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
df = pd.read_csv("highkappak80.csv", header=None)

breakpoint()

# Convert the original .txt/.csv file into a pandas dataframe with the columns
#   "Medium Threshold" [float], "High Threshold" [float], "Cost" [float],
#   "Num Lockdowns" [float], "Max Lockdowns" [float], "Beta0" [int],
#   "Kappa" [int]
df = df.rename(columns={0: "Type", 1: "Medium Threshold", 2: "High Threshold", 3: "Cost", 4: "Num Lockdowns"})
df["Kappa"] = df["Type"].str.split("_", expand=True)[1].str.split("mediumkappa",expand=True)[1].astype("float").astype("int")
df["Beta0"] = df["Type"].str.split("_", expand=True)[2].str.split("beta0",expand=True)[1].astype("float").astype("int")
df["Medium Threshold"] = df["Medium Threshold"].str.split("(", expand=True)[1].astype("float")
df["High Threshold"] = df["High Threshold"].str.split(")", expand=True)[0].astype("float")
df["Cost"] = df["Cost"].astype("int")
df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "", regex=True).astype("int")
df.drop(columns="Type", inplace=True)

# Divide by ODE discretization steps
# df["Cost"] = df["Cost"]
# df["Num Lockdowns"] = df["Num Lockdowns"]

df["R0"] = df["Beta0"]/10 # divide Beta0 by 100 (go from percentage to decimal) then multiply by tau
# df["Kappa"] = df["Kappa"]/100

###############################################################################

plt.clf()
df_heatmap = df.pivot_table(index="R0",columns="Kappa",values="Medium Threshold")
sns.heatmap(df_heatmap, cmap="viridis")
plt.xlabel("Medium Kappa")
plt.ylabel("R0 (Basic Reproduction Number)")
plt.title("Optimal Medium Threshold")
plt.savefig("3_stages_medium_threshold.png", dpi=1200)

plt.clf()
df_heatmap = df.pivot_table(index="R0",columns="Kappa",values="High Threshold")
sns.heatmap(df_heatmap, cmap="viridis")
plt.xlabel("Medium Kappa")
plt.ylabel("R0 (Basic Reproduction Number)")
plt.title("Optimal High Threshold")
plt.savefig("3_stages_high_threshold.png", dpi=1200)

plt.clf()
df_heatmap = df.pivot_table(index="R0",columns="Kappa",values="Num Lockdowns")
sns.heatmap(df_heatmap, cmap="viridis")
plt.xlabel("Medium Kappa")
plt.ylabel("R0 (Basic Reproduction Number)")
plt.title("Optimal Num Lockdowns")
plt.savefig("3_stages_num_lockdowns.png", dpi=1200)

plt.clf()
df_heatmap = df.pivot_table(index="R0",columns="Kappa",values="Cost")
sns.heatmap(df_heatmap, cmap="viridis")
plt.xlabel("Medium Kappa")
plt.ylabel("R0 (Basic Reproduction Number)")
plt.title("Optimal Cost")
plt.savefig("3_stages_num_cost.png", dpi=1200)

breakpoint()

