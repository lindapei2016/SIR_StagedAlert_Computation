import numpy as np
import glob
import pandas as pd

eps = 1e-6

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