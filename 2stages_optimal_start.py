###############################################################################

# Based off of Dave 1:1 discussion on 03/04/2024 -- see
#   Notability notes for that date
# Look at 2-stage system with only 1 max allowed lockdown

# Option A (wiggle room)
# - Step 1: find lockdown start time such that peak infections are below capacity
# - Step 2: find lockdown end time such that second peak is not worse than first
# - Note: because the Fujiwara et al. 2022 formula I am using (Formula S8)
#       is for the case when "the maximum appears during the intervention,"
#       it means the second peak is not worse than the first --
#       so Step 2 is unnecessary / redundant
# - Step 3: infer kappas to make this viable
#
# Option B (symmetric staged alert)
# - From Option A, infer the threshold --
#       we will get a more restricted range of kappas

###############################################################################

import numpy as np
import glob
import pandas as pd
from scipy import optimize
import SIR_det_2stages as SIR

###############################################################################

#
def recovered_lockdown_start(i_max):
    pass