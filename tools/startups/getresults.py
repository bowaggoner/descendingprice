import math, random
import numpy as np
import sys
from bisect import bisect_left
import scipy.stats
import scipy.optimize

from tools.lognorm import *
import tools.startups.model as model
from tools.startups.getequil import approx_f_from_arr

# samp = (mu, cost, strike, val)
# Return list of:
#   simul_welfare
#   simul_revenue
#   frac_inspect
#   1 if winner inspected, 0 otherwise
#   1 if winner is opt winner, 0 otherwise
#   never_inspect_welfare
#   never_inspect_revenue
# The latter two are hypothetical results of an auction where
# bidders never inspect and just bid E[value]-cost.
def get_result(samp, kinds, ns, v1s, cstars):
  num_inspect = 0
  num_bidders = sum(ns)
  total_cost = 0.0

  top_bid = 0.0
  second_bid = 0.0
  did_winner_inspect = 0
  winner_inds = (-1,-1)
  top_net_util = 0.0
  second_net_util = 0.0
  top_covcall = -100000000
  top_covcall_inds = (-2,-2)

  for k in range(len(kinds)):
    sigma = kinds[k].insp_sigma
    for j in range(ns[k]):
      mu, cost, strike, val = samp[k][j]
      net = lognorm_mean(mu, sigma) - cost
      if net > top_net_util:
        second_net_util = top_net_util
        top_net_util = net
      elif net > second_net_util:
        second_net_util = net

      covcall = min(strike, val)
      if covcall > top_covcall:
        top_covcall = covcall
        top_covcall_inds = (k,j)

      if len(v1s) == 0:
        # we had failed to find an equilibrium
        continue

      # compute cstar from the equilibrium
      cstar = approx_f_from_arr(mu, v1s[k], cstars[k])
      if cost <= cstar:
        num_inspect += 1
        total_cost += cost
        bid = val
        this_bidder_inspected = 1
      else:
        bid = net
        this_bidder_inspected = 0

      if bid > top_bid:
        second_bid = top_bid
        top_bid = bid
        did_winner_inspect = this_bidder_inspected
        winner_inds = (k,j)
      elif bid > second_bid:
        second_bid = bid

  # net welfare is highest bid minus cost of all inspections so far
  # (this is true whether or not the winner has already inspected)
  # use a reserve of 0.0
  welf = top_bid - total_cost
  rev = second_bid
  welf_ni = top_net_util
  rev_ni = second_net_util
  winner_is_opt = 1 if winner_inds == top_covcall_inds else 0
  if not did_winner_inspect:
     num_inspect += 1  # winner inspects after getting the item
  return welf, rev, num_inspect/num_bidders, did_winner_inspect, winner_is_opt, welf_ni, rev_ni
 
# get the highest two elements with a reserve of 0
def get_top_two(arr):
  first = 0
  second = 0
  for x in arr:
    if x > first:
      second = first
      first = x
    elif x > second:
      second = x
  return first, second



