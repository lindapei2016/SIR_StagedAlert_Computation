###############################################################################

# Contains various routines for parsing data generated from
#   large-scale cluster runs.

###############################################################################

import numpy as np
import glob
import pandas as pd

eps = 1e-6

###############################################################################

df = pd.read_csv("SIR_det_2stages_Iconstraint0.2.csv", header=None)
df = df.rename(columns={0:"Type", 1:"Threshold Up", 2:"Threshold Down", 3:"Cost", 4:"Num Lockdowns"})
df[["Kappa", "I Constraint", "Threshold Type", "Max Lockdowns"]] = df["Type"].str.split("_", expand=True)
df = df[df["I Constraint"].str.contains("0.2")]
df = df.drop(columns="Type")
df = df.drop(columns="I Constraint")
df["Threshold Up"] = df["Threshold Up"].str.replace("(", "")
df["Threshold Down"] = df["Threshold Down"].str.replace(")", "")
df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "")
df["Kappa"] = df["Kappa"].str.replace("'", "")
df["Kappa"] = df["Kappa"].str.replace("(", "")
df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("'", "")
df["Kappa"] = df["Kappa"].astype(float)
df["Kappa"] = df["Kappa"]/100
df = df.sort_values(by=["Threshold Type", "Max Lockdowns", "Kappa"])
df = df[["Threshold Type", "Max Lockdowns", "Kappa", "Threshold Up", "Threshold Down", "Cost", "Num Lockdowns"]]
df.to_csv("SIR_det_2stages_Iconstraint0.2_cleaned.csv", sep=",")

breakpoint()

###############################################################################

df = pd.read_csv("SIR_det_2stages_scriptB.csv", header=None)
df = df.rename(columns={0:"Type", 1:"Threshold Up", 2:"Threshold Down", 3:"Cost", 4:"Num Lockdowns"})
df["Type"] = df["Type"].str.replace("(","")
df[["Kappa", "Threshold Type", "Max Lockdowns"]] = df["Type"].str.split("_", expand=True)
df = df.drop(columns="Type")
df["Threshold Up"] = df["Threshold Up"].str.replace("(", "")
df["Threshold Down"] = df["Threshold Down"].str.replace(")", "")
df["Num Lockdowns"] = df["Num Lockdowns"].str.replace(")", "")
df["Kappa"] = df["Kappa"].str.replace("'", "")
df["Max Lockdowns"] = df["Max Lockdowns"].str.replace("'", "")
df["Kappa"] = df["Kappa"].astype(float)
df["Kappa"] = df["Kappa"]/100
df = df.sort_values(by=["Threshold Type", "Max Lockdowns", "Kappa"])
df = df[["Threshold Type", "Max Lockdowns", "Kappa", "Threshold Up", "Threshold Down", "Cost", "Num Lockdowns"]]
df.to_csv("SIR_det_2stages_scriptB_cleaned.csv", sep=",")


breakpoint()

###############################################################################

policies = np.arange(0, 0.1 + eps, 0.001)

filenames = glob.glob("/Users/lindapei/Dropbox/RESEARCH/CurrentResearchProjects/StagedAlertTheory/StagedAlertTheoryCode/04102023/*cost_history*")

cost_history_df = pd.DataFrame(columns=["Kappa2", "Threshold2", "Cost"])

# Issue is that sorting the filenames does not create the proper numerical order
# Extract numerical ID of filename
# ID / 100 is the kappa2 used
# kappa2 = int(filename.split("04102023/")[-1].split("_cost_history.csv")[0])/100

for filename in filenames:
    cost_history = np.genfromtxt(filename, delimiter=",")
    best_policy_ix = np.argmin(cost_history)
    threshold2 = policies[best_policy_ix]
    best_cost = cost_history[best_policy_ix]
    kappa2 = int(filename.split("04102023/")[-1].split("_cost_history.csv")[0]) / 100
    cost_history_df.loc[len(cost_history_df)] = [kappa2, threshold2, best_cost]
    # breakpoint()

cost_history_df = cost_history_df.sort_values(by="Kappa2")

cost_history_df.to_csv("cost_history_df.csv", sep=",")

print(cost_history_df)

breakpoint()