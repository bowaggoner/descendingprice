# Checks the startups equilibrium outpus.
#
# 1. Simultaneous auction.
#    For each scenario, plot the equilibrium bid distributions for each kind
#    of bidder.
#
# 2. Simultaneous auction (continued).
#    For each scenario, plot the equilibrium c* for each kind of bidder.
#    This plots, for each E[val] the bidder has (based on type), the
#    threshold cost c* such that inspect iff cost <= c*.
#
# 3. Descending auction.
#    For each scenario, it plots the empirical covered-call distribution of each
#    bidder along with its lognormal approximation, and the difference.
#    If the difference is very large especially in the upper-tail, then the
#    first-price equilibrium we compute on the lognormal approximation may
#    not accurately reflect the actual auction.


import math
import numpy as np
import matplotlib.pyplot as plt

from tools.startups.model import BidderParams,Kind
from tools.lognorm import lognorm_cdf

# get the parameters from the equilibrium
from bad_startups_equil_symmetric import *
#from startups_equil import *


kinds_list = [[Kind(*p) for p in ps] for ps in kinds_params_list]

print("plotting simultaneous bid distributions...")

for i in range(len(kinds_list)):
  plt.figure()
  for j in range(len(kinds_list[i])):
    plt.plot(simul_xs_list[i], simul_Gs_list[i][j])
  plt.xlabel("x")
  plt.ylabel("Pr[bid <= x]")
  plt.title("Simultaneous bid cdf for scenario " + str(i))

plt.show()


print("plotting c*s...")

for i in range(len(kinds_list)):
  plt.figure()
  for j in range(len(kinds_list[i])):
    plt.plot([math.exp(v) for v in simul_v1s_list[i][j]], simul_cstars_list[i][j])
  plt.xlabel("expected value (e^{mu_v + v1})")
  plt.ylabel("c*")
  plt.title("C*s for scenario " + str(i))

plt.show()

print("plotting covered-call distributions...")

num_samps = 10000

for i in range(len(kinds_list)):
  fig, ax = plt.subplots(len(kinds_list[i]))
  for k,kind in enumerate(kinds_list[i]):
    axis = ax[k] if len(kinds_list[i]) > 1 else ax
    covs = sorted([kind.gen_kappa() for t in range(num_samps)])
    empirical = [(i+1)/num_samps for i in range(num_samps)]
    axis.plot(covs, empirical)
    cdf = lambda x: lognorm_cdf(x, fit_params_list[i][k][0], fit_params_list[i][k][1])
    fitted = list(map(cdf, covs))
    axis.plot(covs, fitted)
    axis.plot(covs, abs(np.array(fitted) - np.array(empirical)))
    axis.legend(["empirical", "fitted", "difference"])
    axis.set_title('bidder kind ' + str(k))
    #axis.xlabel('covered call')
    #axis.ylabel('cdf')
  try:
    fig.set_title('Covered-call distributions for scenario ' + str(i))
  except:
    pass

plt.show()


