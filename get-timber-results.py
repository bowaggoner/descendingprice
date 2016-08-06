# get the welfare and revenue of the auctions, given computed equilibria

import math,random
import numpy as np
from multiprocessing import Pool

from tools.timber.model import Timber
import tools.firstprice.getresults as fp

# data
from timber_equil import *

outfile = "timber_results.py"

num_trials = 10000000


with open(outfile, "w") as f:
  f.write("import numpy as np\n")
  f.write("import matplotlib.pyplot as plt\n\n")
  f.write("num_trials = " + str(num_trials) + "\n\n")
  f.write("timber_params_list = [\n")
  for timber_params in timber_params_list:
    f.write("  " + str(timber_params) + ",\n")
  f.write("  ]\n\n")
  f.write("ns_list = [\n")
  for ns in ns_list:
    f.write("   " + str(ns) + ",\n")
  f.write("  ]\n\n")


timber_list = [Timber(*p) for p in timber_params_list]

## -----------------------------------------------------------

# for auction i in list, return two lists:
# [desc_welfare, desc_revenue, opt_welfare, welf_of_random, frac_inspect_desc, frac_inspect_opt]
# and a list containing standard errors for each
PRINT_EVERY = int(num_trials / 10)
def get_results_for(i):
  bid_fs = get_bid_fs(i)
  # for computing rolling average and variance, from Knuth's Art of CS
  # (descend_welfare, descend_revenue, opt_welfare)
  avg = np.zeros( (6) )
  Ss = np.zeros( (len(avg)) )
  for t in range(num_trials):
    if t % PRINT_EVERY == 0:
      print("auction " + str(i) + " at trial " + str(t))
    samp = draw_samp(i)
    # winner's covered call, winner's bid, opt covered call
    result = np.array(fp.get_result(samp, bid_fs))
    diff = result - avg
    avg += diff / (t+1)
    newdiff = result - avg
    Ss += np.multiply(diff, newdiff)
  var = Ss / (num_trials - 1)
  std_err = [math.sqrt(v / num_trials) for v in var]
  return avg.tolist(), std_err

# draw a sample of a single auction (using i-th parameters)
# samp[k][j] = (strike, val) of j-th bidder of kind k
def draw_samp(i):
  samp = [[timber_list[i].gen_logger() for j in range(ns_list[i][0])],
          [timber_list[i].gen_mill() for j in range(ns_list[i][1])]]
  return np.array(samp)
 
# for descending auction, get the bid functions
# of each bidder kind using output of AuctionSolver
def get_bid_fs(i):
  if len(solver_output_fp[i]) == 0:
    return []
  return fp.get_lognorm_bids_FP(solver_output_fp[i], len(ns_list[i]))


## ------------------------------------------------------------

pool = Pool()
results_list = pool.map(get_results_for, [3]) #range(len(timber_list)))
pool.close()
pool.join()

with open(outfile, "a") as f:
  f.write("results = [\n")
  f.write("  # descend_welfare, descend_revenue, opt_welfare\n")
  for i in range(len(results_list)):
    f.write("  " + str(results_list[i][0]) + ",\n")
  f.write("  ]\n\n")
  f.write("max_std_error = " + str(max([max(r[1]) for r in results_list])) + "\n\n")
  f.write("std_errors = [\n")
  f.write("  # descend_welfare, descend_revenue, opt_welfare\n")
  for i in range(len(results_list)):
    f.write("  " + str(results_list[i][1]) + ",\n")
  f.write("  ]\n\n")

  f.write("# roberts-sweeting results\n")
  f.write("# (from running their matlab code)\n")
  f.write("""rs_results = [
  # sequential welfare, sequential revenue, simultaneous welfare, simultaneous revenue
  [101.1496, 72.5724,  99.4027,  71.2584],
  [77.9112,  52.3396,  76.5191,  50.958],
  [114.3238, 85.2241,  112.2773, 83.8897],
  [96.3176,  64.515,   95.4909,  64.4184],
  [104.6606, 77.6487,  102.4034, 75.8768],
  [49.0601,  35.5204,   47.5228,  34.0991],
  [198.9304, 143.7588, 196.8765, 142.7343],
  [85.582,   62.9647,  83.8372,  61.9163],
  [122.3738, 85.333,   120.7989, 83.9017],
  [71.865,   58.818,   70.3854,  57.1279],
  [143.9811, 91.6648,  142.2728, 91.1134],
  [102.9065, 72.3803,  101.078,  70.6343],
  [99.3422,  73.1008,  97.455,   71.9115],
  [108.4611, 75.7422,  108.0496, 75.7455],
  [95.6589,  69.3332,  92.8316,  66.6754],
  [71.0199,  50.6419,  61.036,   42.422],
  ]

print("Timber auctions results for each of 15 parameter settings:")
print("")
print("Welfare")
print("-------")
print("#   simul   sequential   descending (std err)   first-best (std err)")
for i in range(len(results)):
  print("%2d   %7.3f   %7.3f   %7.3f (%5.3f)   %7.3f (%5.3f)"
        % (i+1, rs_results[i][2], rs_results[i][0], results[i][0], std_errors[i][0], results[i][2], std_errors[i][2]))

print("")
print("Welfare as percent of first-best")
print("----------------------------------")
print("#   simul   sequential   descending")
for i in range(len(results)):
  opt = results[i][2]
  print("%2d   %7.2f   %7.2f   %7.2f"
        % (i+1, 100*rs_results[i][2]/opt, 100*rs_results[i][0]/opt, 100*results[i][0]/opt))

print("")
print("Revenue")
print("-------")
print("#   simul   sequential   descending (std err)")
for i in range(len(results)):
  print("%2d   %7.3f   %7.3f   %7.3f (%5.3f)"
        % (i+1, rs_results[i][3], rs_results[i][1], results[i][1], std_errors[i][1]))

print("")
print("Revenue as percent of descending")
print("----------------------------------")
print("#   simul   sequential")
for i in range(len(results)):
  desc = max(results[i][1], 0.01)
  print("%2d   %7.2f   %7.2f"
        % (i+1, 100*rs_results[i][3]/desc, 100*rs_results[i][1]/desc))

print("")
print("Percent of bidders inspecting")
print("------------------------------")
print("#   descending (std err)   opt (std err)")
for i in range(len(results)):
  print("%2d   %6.1f (%3.1f)   %6.1f (%3.1f)"
        % (i+1, 100*results[i][4], 100*std_errors[i][4], 100*results[i][5], 100*std_errors[i][5]))


width = 0.2
inds = np.arange(len(results))
one_inds = np.arange(1, len(results)+1)

## welfare plot
ax = plt.figure()
simul = plt.bar(one_inds, [rs_results[i][2] for i in inds], width, color='green')
seq = plt.bar(one_inds + width, [rs_results[i][0] for i in inds], width, color='black')
descend = plt.bar(one_inds + 2*width, [results[i][0] for i in inds], width, yerr=[std_errors[i][0] for i in inds], color='blue')
opt = plt.bar(one_inds + 3*width, [results[i][2] for i in inds], width, yerr=[std_errors[i][2] for i in inds], color='red')
plt.ylabel("welfare")
plt.xlabel("parameter set")
plt.legend([simul[0], seq[0], descend[0], opt[0]], ["Simultaneous", "Sequential", "Descending", "Opt"])
plt.xticks(one_inds)


## revenue plot
ax = plt.figure()
simul = plt.bar(one_inds, [rs_results[i][3] for i in inds], width, color='green')
seq = plt.bar(one_inds + width, [rs_results[i][1] for i in inds], width, color='black')
descend = plt.bar(one_inds + 2*width, [results[i][1] for i in inds], width, yerr=[std_errors[i][1] for i in inds], color='blue')
plt.ylabel("revenue")
plt.xlabel("parameter set")
plt.legend([simul[0], seq[0], descend[0]], ["Simultaneous", "Sequential", "Descending"])
plt.xticks(one_inds)

plt.show()
""") # end f.write()

