# gets the equilibrium of simultaneous, and the bidders'
# covered-call distributions for use in gettin the
# descending

import numpy as np

from tools.timber.model import Timber
import tools.timber.fitcovcall as fitcovcall
from multiprocessing import Pool

outfile = "timber_equil.py"

num_fit_samps = 10000

# line j is a set of parameters for auction j
timber_params_list = [
  [3.582, 0.378, 0.576, 0.689, 2.05],
  [3.582, 0.378, 0.576, 0.689, 2.05],
  [3.582, 0.378, 0.576, 0.689, 2.05],
  [3.582, 0.378, 0.576, 0.689, 2.05],
  [3.582, 0.378, 0.576, 0.689, 2.05],
  [2.921, 0.378, 0.576, 0.689, 2.05],
  [4.243, 0.378, 0.576, 0.689, 2.05],
  [3.582, 0.169, 0.576, 0.689, 2.05],
  [3.582, 0.587, 0.576, 0.689, 2.05],
  [3.582, 0.378, 0.349, 0.689, 2.05],
  [3.582, 0.378, 0.804, 0.689, 2.05],
  [3.582, 0.378, 0.576, 0.505, 2.05],
  [3.582, 0.378, 0.576, 0.872, 2.05],
  [3.582, 0.378, 0.576, 0.689, 0.39],
  [3.582, 0.378, 0.576, 0.689, 3.72],
  [3.582, 0.378, 0.576, 0.689, 16.0]]

# line j is the (# of loggers, # of mills) bidding in auction j
ns_list = [
  [4,4],
  [4,1],
  [4,7],
  [0,4],
  [8,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4],
  [4,4]]

timber_list = [Timber(*p) for p in timber_params_list]

## ------------------------------------------------------------

with open(outfile, "w") as f:
  f.write("num_fit_samps = " + str(num_fit_samps) + "\n")
  f.write("timber_params_list = [\n")
  for timber_params in timber_params_list:
    f.write("  " + str(timber_params) + ",\n")
  f.write("  ]\n\n")
  f.write("ns_list = [\n")
  for ns in ns_list:
    f.write("   " + str(ns) + ",\n")
  f.write("  ]\n\n")

pool = Pool(len(timber_list))
input_list = zip(timber_list, [num_fit_samps]*len(timber_list))
fit_params_list = pool.starmap(fitcovcall.fit_lognorm, input_list)
pool.close()
pool.join()
fit_params_list = np.array(fit_params_list).tolist()
with open(outfile, "a") as f:
  f.write("fit_params_list = [\n")
  for p in fit_params_list:
    f.write("  " + str(p) + ",\n")
  f.write("  ]\n\n\n")
  f.write("solver_output_fp = [\n")
  for i in range(len(kinds_list)):
    f.write("  # scenario " + str(i) + "\n")
    f.write("   [],\n\n")
  f.write("  ]\n\n")


