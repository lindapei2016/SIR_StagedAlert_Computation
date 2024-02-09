###############################################################################

# Contains various routines for parsing data generated from
#   large-scale cluster runs.

###############################################################################

import numpy as np
import glob
import pandas as pd

import matplotlib.pyplot as plt

eps = 1e-6

###############################################################################

policies = np.arange(0, 0.1 + eps, 0.001)

filenames = glob.glob("/Users/lindapei/Dropbox/RESEARCH/CurrentResearchProjects/StagedAlertTheory/StagedAlertTheoryCode/Results/05032023a/*.csv")

breakpoint()

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

df = df[df["Cost"] != " inf"]
asymmetric_max1 = df[(df["Threshold Type"] == "asymmetric") & (df["Max Lockdowns"] == "max1")]
symmetric_max1 = df[(df["Threshold Type"] == "symmetric") & (df["Max Lockdowns"] == "max1")]
asymmetric_nomax = df[(df["Threshold Type"] == "asymmetric") & (df["Max Lockdowns"] == "nomax")]
symmetric_nomax = df[(df["Threshold Type"] == "symmetric") & (df["Max Lockdowns"] == "nomax")]

asymmetric_max1 = asymmetric_max1[asymmetric_max1["Kappa"] != 0.57]
symmetric_max1 = symmetric_max1[symmetric_max1["Kappa"] != 0.57]
asymmetric_nomax = asymmetric_nomax[asymmetric_nomax["Kappa"] != 0.57]
symmetric_nomax = symmetric_nomax[symmetric_nomax["Kappa"] != 0.57]

# Threshold Down versus kappas

a_m1 = np.array(asymmetric_max1["Threshold Down"],dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"],dtype=float)
s_m1 = np.array(symmetric_max1["Threshold Down"],dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"],dtype=float)
a_nm = np.array(asymmetric_nomax["Threshold Down"],dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"],dtype=float)
s_nm = np.array(symmetric_nomax["Threshold Down"],dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"],dtype=float)

plt.plot(a_m1_kappa, a_m1, color="yellowgreen", label="Asymmetric, Max 1")
plt.plot(s_m1_kappa, s_m1, color="lightseagreen", label="Symmetric, Max 1")
plt.plot(a_nm_kappa, a_nm, color="darkorange", linestyle=":", label="Asymmetric, No Max")
plt.plot(s_nm_kappa, s_nm, color="red", linestyle=":", label="Symmetric, No Max")
plt.title("Threshold Down versus kappa, for symmetric (& asymmetric) policies and max 1 (& no max) lockdown")
plt.legend()
plt.ylabel("Threshold Down")
plt.xlabel("Kappa (Transmission Reduction)")
plt.show()

breakpoint()

# Threshold Up versus kappas

a_m1 = np.array(asymmetric_max1["Threshold Up"],dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"],dtype=float)
s_m1 = np.array(symmetric_max1["Threshold Up"],dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"],dtype=float)
a_nm = np.array(asymmetric_nomax["Threshold Up"],dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"],dtype=float)
s_nm = np.array(symmetric_nomax["Threshold Up"],dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"],dtype=float)

plt.plot(a_m1_kappa, a_m1, color="yellowgreen", label="Asymmetric, Max 1")
plt.plot(s_m1_kappa, s_m1, color="lightseagreen", label="Symmetric, Max 1")
plt.plot(a_nm_kappa, a_nm, color="darkorange", linestyle=":", label="Asymmetric, No Max")
plt.plot(s_nm_kappa, s_nm, color="red", linestyle=":", label="Symmetric, No Max")
plt.title("Threshold Up versus kappa, for symmetric (& asymmetric) policies and max 1 (& no max) lockdown")
plt.legend()
plt.ylabel("Threshold Up")
plt.xlabel("Kappa (Transmission Reduction)")
plt.show()

breakpoint()

# Cost versus kappas

plt.clf()

a_m1 = np.array(asymmetric_max1["Cost"],dtype=float)
a_m1_kappa = np.array(asymmetric_max1["Kappa"],dtype=float)
s_m1 = np.array(symmetric_max1["Cost"],dtype=float)
s_m1_kappa = np.array(symmetric_max1["Kappa"],dtype=float)
a_nm = np.array(asymmetric_nomax["Cost"],dtype=float)
a_nm_kappa = np.array(asymmetric_nomax["Kappa"],dtype=float)
s_nm = np.array(symmetric_nomax["Cost"],dtype=float)
s_nm_kappa = np.array(symmetric_nomax["Kappa"],dtype=float)
plt.plot(a_m1_kappa, np.log(a_m1), color="yellowgreen", label="Asymmetric, Max 1")
plt.plot(s_m1_kappa, np.log(s_m1), color="lightseagreen", label="Symmetric, Max 1")
plt.plot(a_nm_kappa, np.log(a_nm), color="darkorange", linestyle=":", label="Asymmetric, No Max")
plt.plot(s_nm_kappa, np.log(s_nm), color="red", linestyle=":", label="Symmetric, No Max")
plt.title("Log costs versus kappa, for symmetric (& asymmetric) policies and max 1 (& no max) lockdown")
plt.legend()
plt.ylabel("Log Costs (Time Units in Lockdown)")
plt.xlabel("Kappa (Transmission Reduction)")
plt.show()

breakpoint()

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