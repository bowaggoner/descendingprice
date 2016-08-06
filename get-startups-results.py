# get the welfare and revenue of the auctions, given computed equilibria

import math
import numpy as np
from multiprocessing import Pool

from tools.startups.model import BidderParams,Kind
import tools.startups.getresults as simul
import tools.firstprice.getresults as fp

# input file
from saved_startups_equil_1 import *

outfile = "saved_startups_results_1.py"

num_trials = 100000#0


with open(outfile, "w") as f:
  f.write("num_trials = " + str(num_trials) + "\n\n")
  f.write("num_fit_samps = " + str(num_fit_samps) + "\n")
  f.write("num_bid_pts = " + str(num_bid_pts) + "\n")
  f.write("num_mus = " + str(num_mus) + "\n\n")
  f.write("kinds_params_list = [\n")
  for kinds_params in kinds_params_list:
    f.write("  " + str(kinds_params) + ",\n")
  f.write("  ]\n\n")
  f.write("ns_params_list = [\n")
  for ns in ns_params_list:
    f.write("   " + str(ns) + ",\n")
  f.write("  ]\n\n")
  f.write("ns_list = [\n")
  for ns in ns_list:
    f.write("   " + str(ns) + ",\n")
  f.write("  ]\n\n")


kinds_list = [[Kind(*p) for p in auction] for auction in kinds_params_list]

## -----------------------------------------------------------

# Return two lists: a list of results and one of corresponding standard errors.
# Results in each list are, in order:
#   simul_welfare
#   simul_revenue
#   descend_welfare
#   descend_revenue
#   opt_welfare
#   fraction_who_inspect_in_simul
#   fraction_who_inspect_in_desc
#   fraction_who_inspect_in_opt
#   fraction_of_time_winner_inspected_in_simul
#   fraction_of_time_winner_in_simul_was_opt
#   never_inspect_welfare
#   never_inspect_revenue
#   random_bidder_welfare
# The latter two are hypothetical results if bidders never inspected
# but just bid E[value]-cost in a simultaneous auction.
def get_results_for(i):
  bid_fs = get_bid_fs(i)
  PRINT_EVERY = max(int(num_trials/4), 1)
  avg = np.zeros( (13) )
  # for computing rolling variance, from Knuth's Art of CS
  Ss = np.zeros( (len(avg)) )

  kinds = kinds_list[i]
  ns = ns_list[i]
  v1s = np.array(simul_v1s_list[i])
  cstars = np.array(simul_cstars_list[i])

  for t in range(num_trials):
    if t % PRINT_EVERY == 0:
      print(i," at trial ",t," of ",num_trials)
    samp = draw_samp(i)
    fp_samp = [[s[2:] for s in ss] for ss in samp]  # just (strike, val)
    simul_welf, simul_rev, simul_frac, winner_insp, winner_opt, ni_welf, ni_rev = simul.get_result(samp, kinds, ns, v1s, cstars)
    desc_welf, desc_rev, opt_welf, rand_welf, desc_frac, opt_frac = fp.get_result(fp_samp, bid_fs)

    result = np.array([simul_welf, simul_rev, desc_welf, desc_rev, opt_welf, simul_frac, desc_frac, opt_frac, winner_insp, winner_opt, ni_welf, ni_rev, rand_welf])
    diff = result - avg
    avg += diff / (t+1)
    newdiff = result - avg
    Ss += np.multiply(diff, newdiff)
  var = Ss / (num_trials - 1)
  std_err = [math.sqrt(v / num_trials) for v in var]
  return avg.tolist(), std_err

# draw a sample of a single auction (using i-th parameters)
# samp[k][j] = (mu, cost, strike, val) of j-th bidder of kind k
def draw_samp(i):
  return [[kind.gen_samp() for j in range(ns_list[i][k])] for k,kind in enumerate(kinds_list[i])]

# for descending auction, get the bid functions
# of each bidder kind using output of AuctionSolver
def get_bid_fs(i):
  if len(solver_output_fp[i]) == 0:
    return []
  return fp.get_lognorm_bids_FP(math.exp(solver_mu_shifts[i]) * np.array(solver_output_fp[i]), len(ns_list[i]))


## ------------------------------------------------------------
# Write everything to a file that prints and plots results

pool = Pool()
results_and_errs = pool.map(get_results_for,range(len(kinds_list)))
pool.close()
pool.join()


with open(outfile, "a") as f:
  f.write("import numpy as np\n")
  f.write("import matplotlib.pyplot as plt\n\n")
  f.write("results_list = [\n")
  f.write("  # simul_welfare, simul_revenue, descend_welfare, descend_revenue, opt_welfare, frac_inspect_simul, frac_simul_winners_inspect, frac_simul_winners_opt, never_inspect_welfare, never_inspect_revenue, random_welfare\n")
  for res in results_and_errs:
    f.write("  " + str(res[0]) + ",\n")
  f.write("  ]\n\n")
  f.write("max_std_error = " + str(max([max(r[1]) for r in results_and_errs])) + "\n\n")
  f.write("std_errors = [\n")
  for res in results_and_errs:
    f.write("  " + str(res[1]) + ",\n")
  f.write("  ]\n\n")
  f.write("""

print("Welfare")
print("-------")
print("#   simul (std err)   descending (std err)   first-best (std err)   |   never-inspect (std err)   random (std err)")
for i,res in enumerate(results_list):
  print("%2d   %7.3f (%5.3f)   %7.3f (%5.3f)   %7.3f (%5.3f)   |   %7.3f (%5.3f)   %7.3f (%5.3f)"
        % (i+1, res[0], std_errors[i][0], res[2], std_errors[i][2], res[4], std_errors[i][4], res[10], std_errors[i][10], res[12], std_errors[i][12]))

print("")
print("Welfare as percent of first-best")
print("--------------------------------")
print("#   simul   descending   |   never-inspect   random")
for i,res in enumerate(results_list):
  opt = res[4]
  if opt == 0.0:
    print("-----------")
  else:
    print("%2d   %7.2f   %7.2f   |   %7.2f   %7.2f"
          % (i+1, 100*res[0]/opt, 100*res[2]/opt, 100*res[10]/opt, 100*res[12]/opt))

print("")
print("Revenue")
print("-------")
print("#   simul (std err)   descending (std err)   |   never-inspect (std err)")
for i,res in enumerate(results_list):
   print("%2d   %7.3f (%5.3f)   %7.3f (%5.3f)   |   %7.3f (%5.3f)"
         % (i+1, res[1], std_errors[i][1], res[3], std_errors[i][3], res[11], std_errors[i][11]))

print("")
print("Revenue as percent of descending")
print("--------------------------------")
print("#   simul   |   never-inspect")
for i,res in enumerate(results_list):
  desc = res[3]
  if desc == 0.0:
    print("---------")
  else:
    print("%2d   %7.3f   |   %7.3f"
          % (i+1, 100*res[1]/desc, 100*res[11]/desc))

print("")
print("Percent of bidders inspecting")
print("-----------------------------")
print("#   simul (std err)   descending (std err)   opt (std err)")
for i,res in enumerate(results_list):
  print("%2d   %5.2f (%5.2f)   %5.2f (%5.2f)   %5.2f (%5.2f)"
        % (i+1, 100*res[5], 100*std_errors[i][5], 100*res[6], 100*std_errors[i][6], 100*res[7], 100*std_errors[i][7]))

print("")
print("Simultaneous auction stats")
print("--------------------------")
print("percent of times winner inspected and")
print("percent of times winner had highest covered call")
print("#   winner_insp (std err)   winner_was_opt (std err)")
for i,res in enumerate(results_list):
  print("%2d   %7.2f (%5.2f)   %7.2f (%5.2f)"
        % (i+1, 100*res[8], 100*std_errors[i][8], 100*res[9], 100*std_errors[i][9]))

width = 0.27
inds = np.arange(len(results_list))
one_inds = np.arange(1, len(results_list)+1)

## welfare plot
ax = plt.figure()
simul = plt.bar(one_inds, [r[0] for r in results_list], width, yerr=[s[0] for s in std_errors], color='green')
descend = plt.bar(one_inds + width, [r[2] for r in results_list], width, yerr=[s[2] for s in std_errors], color='blue')
opt = plt.bar(one_inds + 2*width, [r[4] for r in results_list], width, yerr=[s[4] for s in std_errors], color='red')
plt.ylabel("welfare")
plt.xlabel("parameter set")
plt.legend([simul[0], descend[0], opt[0]], ["Simultaneous", "Descending", "Opt"])
plt.xticks(one_inds)
plt.title("Welfare")

## revenue plot
ax = plt.figure()
simul = plt.bar(one_inds, [r[1] for r in results_list], width, yerr=[s[1] for s in std_errors], color='green')
descend = plt.bar(one_inds + width, [r[3] for r in results_list], width, yerr=[s[3] for s in std_errors], color='blue')
plt.ylabel("revenue")
plt.xlabel("parameter set")
plt.legend([simul[0], descend[0]], ["Simultaneous", "Descending"])
plt.xticks(one_inds)
plt.title("Revenue")

plt.show()
""") # end f.write()

