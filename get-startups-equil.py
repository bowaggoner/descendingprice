# 1. Generates a bunch of scenarios (a.k.a. "auctions", i.e. sets of bidders with
#    their parameters).
# 2. Fits a lognormal distribution to each bidder's covered-call distribution
#    in each auction.
# 3. Solves for equilibrium of simultaneous sealed-bid second-price auction.
# 4. Prints all these results to the file 'startups_equil.py'.

import math
import numpy as np
from multiprocessing import Pool

from tools.startups.model import BidderParams, Kind
import tools.startups.getequil as simul
import tools.startups.fitcovcall as fitcovcall
import tools.firstprice.getequil as firstprice
from tools.lognorm import lognorm_cdf, lognorm_ppf

np.seterr(all='raise')

# Write results to this file
outfile = "startups_equil.py"

# Control computational intensiveness and detail
num_fit_samps = 100000
num_bid_pts = 1200
num_mus = 1200
tol = 1e-7


# These parameters control how to generate bidders.
# See tools/startups/model.py for more on what parameters mean.
# To re-use bidder kinds that were already generated, see below.
bidder_params_list = [
  # mu_v, mu_c, sigma_v, sigma_c, a_0, a_1, rho
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, 0.7],    # baseline
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, 0.7],    # 2 bidders
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, 0.7],    # 10 bidders
 
  [0.0, -0.62, 1.5, 0.2, 0.1, 0.4, 0.7],    # high sigma_v
  [0.0, -0.62, 0.5, 0.2, 0.1, 0.4, 0.7],    # low sigma_v

  [0.0, -0.35, 1.0, 0.2, 0.1, 0.4, 0.7],    # high mu_c
  [0.0, -1.00, 1.0, 0.2, 0.1, 0.4, 0.7],    # low mu_c

  [0.0, -0.62, 1.0, 0.0, 0.1, 0.4, 0.7],    # low sigma_c
  [0.0, -0.62, 1.0, 1.0, 0.1, 0.4, 0.7],    # high sigma_c

  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, -0.5],   # negative correlation
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, 0.0],    # no correlation
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.4, 1.0],    # perfect correlation

  [0.0, -0.62, 1.0, 0.2, 0.5, 0.25, 0.7],   # change alpha (more heterogeneous)
  [0.0, -0.62, 1.0, 0.2, 0.5, 0.05, 0.7],   # change alpha (ditto, more uncertainty)
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.1, 0.7],    # change alpha (more uncertainty)
  [0.0, -0.62, 1.0, 0.2, 0.1, 0.7, 0.7],    # change alpha (less uncertainty)
  ]
  
# generate this many bidder kinds for each scenario
ns_params_list = [
  5,
  2,
 10,

  5,
  5,

  5,
  5,

  5,
  5,

  5,
  5,
  5,

  5,
  5,
  5,
  5,
  ]


# Now, generate the bidder kinds.
# To re-run scenarios that were already generated, comment out
# the following lines, which generate kinds_params_list and
# ns_list.
# Instead, copy/paste the old kinds_params_list and ns_list here, e.g.
#     kinds_params_list = [ ... ]
#     ns_list = [ ... ]

## BEGIN COMMENTING OUT HERE

# allow one bidder of each kind
ns_list = [[1]*num for num in ns_params_list]

# generate ns_params_list[i] different bidder kinds for scenario i,
kinds_params_list = []
for i,bp in enumerate(bidder_params_list):
  obj = BidderParams(*bp)
  kinds_params_list.append([obj.gen_kind_params() for j in range(len(ns_list[i]))])

## END COMMENTING OUT HERE

## ------------------------------------------------------------

# Use parameter lists to create Kind objects
kinds_list = [[Kind(*p) for p in pl] for pl in kinds_params_list]

# print each scenario in a on a separate line
def write_scenario_list(a, name_str, extra_newline=False):
  with open(outfile, "a") as f:
    f.write(name_str + " = [\n")
    for i,e in enumerate(a):
      f.write("  # scenario " + str(i) + "\n")
      f.write("  " + str(e) + ",\n")
      if extra_newline:
        f.write("\n")
    f.write("  ]\n\n")
    if extra_newline:
      f.write("\n")

# print each scenario, kind in a on a separate line
def write_scenario_and_kinds_list(a, name_str, comments=False):
  with open(outfile, "a") as f:
    f.write(name_str + " = [\n")
    for i,e in enumerate(a):
      f.write("  # scenario " + str(i) + "\n")
      f.write("  [\n")
      for i2,e2 in enumerate(e):
        if comments:
          f.write("    # kind " + str(i2) + "\n")
        f.write("    " + str(e2) + ",\n")
      f.write("  ],\n\n")
    f.write("  ]\n\n\n")




with open(outfile, "w") as f:
  f.write("num_fit_samps = " + str(num_fit_samps) + "\n")
  f.write("num_bid_pts = " + str(num_bid_pts) + "\n")
  f.write("num_mus = " + str(num_mus) + "\n")
  f.write("tol = " + str(tol) + "\n\n")

write_scenario_list(bidder_params_list, "bidder_params_list")
write_scenario_list(ns_params_list, "ns_params_list")
write_scenario_list(ns_list, "ns_list")
write_scenario_and_kinds_list(kinds_params_list, "kinds_params_list")

## ------------------------------------------------------------

print("\n----------------------\nFitting covered-call distributions...\n--------------------")

fit_params_list = []
for i,kinds in enumerate(kinds_list):
  pool = Pool()
  input_list = zip(kinds, [num_fit_samps]*len(kinds))
  fit_params = pool.starmap(fitcovcall.fit_lognorm, input_list)
  pool.close()
  pool.join()
  fit_params = np.array(fit_params).tolist()
  fit_params_list.append(fit_params)
write_scenario_and_kinds_list(fit_params_list, "fit_params_list")

# get some number so that, by adding it to all mus,
# they become nonnegative
def get_shift(params):
  smallest_mu = min(p[0] for p in params)
  shift = 0.0
  while smallest_mu + shift < 0.0:
    shift += 0.5
  return shift

solver_mu_shifts = list(map(get_shift, fit_params_list))
write_scenario_list(solver_mu_shifts, "solver_mu_shifts")

def get_upper(i):
  return int(max([lognorm_ppf(0.9995,p[0] + solver_mu_shifts[i],p[1]) for p in fit_params_list[i]]))

solver_upper_bounds = [get_upper(i) for i in range(len(fit_params_list))]
write_scenario_list(solver_upper_bounds, "solver_upper_bounds")

solver_shifted_params_list = []
for i,params in enumerate(fit_params_list):
  solver_shifted_params_list.append([[p[0] + solver_mu_shifts[i], p[1]] for p in params])
write_scenario_and_kinds_list(solver_shifted_params_list, "solver_shifted_params_list")
  

print("\n----------------------\nFit covered-call distributions.\n--------------------")
print("\n----------------------\nGetting simultaneous equilibrium...\n--------------------")

  
def do_one_simul_eq(i):
  ns = ns_list[i]
  res = simul.Equil(kinds_list[i], ns).compute_equilibrium(num_bid_pts, num_mus, tol=tol)
  if res is None:
    print("Failed to get equilibrium for " + str(i) + "\n")
    xs = []
    Gs_all = []
  else:
    xs, Gs_all, v1s_all, cstars_all = res
    print("Got equilibrium for " + str(i) + ":")
    print("xs = " + str(xs))
    print("Gs_all = " + str(Gs_all))
    print("v1s_all = " + str(v1s_all))
    print("cstars_all = " + str(cstars_all) + "\n\n")
  return res

pool = Pool()
simul_equil_list = pool.starmap(do_one_simul_eq, [[i] for i in range(len(kinds_list))])
pool.close()
pool.join()

write_scenario_list([s[0] for s in simul_equil_list], "simul_xs_list", True)
write_scenario_and_kinds_list([s[1] for s in simul_equil_list], "simul_Gs_list", True)
write_scenario_and_kinds_list([s[2] for s in simul_equil_list], "simul_v1s_list", True)
write_scenario_and_kinds_list([s[3] for s in simul_equil_list], "simul_cstars_list", True)
write_scenario_and_kinds_list([[] for i in range(len(kinds_list))], "solver_output_fp", True)


print("\n----------------------\nGot simultaneous equilibrium.\n--------------------")

