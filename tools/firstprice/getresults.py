# Computes welfare in a first-price auction
# Uses the output of AuctionSolver
# using the two ways it outputs equilibria.
#
# For the "fixed-point finite difference iterations" (FP)
# solver, it outputs an approximate inverse bid function
# as a list of data points x and bids b where inv_bid(b) = x
#
# For the "constrained strategic equilibrium" (CSE)
# solver, it outputs an x_transform function and
# an approximation of the inverse bid function by
# Chebyshev polynomials T0, T1, ...

import math
import numpy as np
import numpy.random
import scipy.optimize
from bisect import bisect_left
from multiprocessing import Pool

# simulate a single auction outcome
# where samp[k][j] = (strike, val) of j-th bidder of kind k
# and bid_fs[k](v) = bid of kind k with covered-call value v
# if bid_fs is an empty list, then skip the descending-price mech.
# Return:
#   winning covered-call
#   winning bid
#   max covered-call
#   average covered-call
#   fraction who inspect in descending
#   fraction who inspect in opt
def get_result(samp, bid_fs):
  win_bid = 0.0
  win_covcall = 0.0
  opt_covcall = 0.0
  total_covcall = 0.0
  inspect_times = []
  strikes = []
  num_bidders = sum(len(s) for s in samp)
  for k in range(len(samp)):
    for j in range(len(samp[k])):
      (strike, val) = samp[k][j]
      strikes.append(strike)
      covcall = min(strike, val)
      total_covcall += covcall
      opt_covcall = max(opt_covcall, covcall)
      if len(bid_fs) == 0:
        continue

      inspect_at = bid_fs[k](strike)
      inspect_times.append(inspect_at)
      bid = inspect_at if val >= strike else bid_fs[k](val)
      # break ties in favor of lower value, just to avoid
      # inflating efficiency of our mechanism
      if bid > win_bid or (bid == win_bid and covcall < win_covcall):
        win_bid = bid
        win_covcall = covcall
  num_inspect_desc = sum([1 for t in inspect_times if t >= win_bid])
  num_inspect_opt = sum([1 for s in strikes if s >= opt_covcall])
  frac_inspect_desc = num_inspect_desc / num_bidders
  frac_inspect_opt = num_inspect_opt / num_bidders
  return win_covcall, win_bid, opt_covcall, total_covcall/num_bidders, frac_inspect_desc, frac_inspect_opt



## ------------------------------------------------------------
# using output of AuctionSolver's finite difference

# bids is a vector of bids and data the corresponding values
# just approximate it in between the data points by a linear function
def approx_bid(v, bids, data):
  if v <= data[0]:
    return bids[0]
  if v >= data[-1]:
    return bids[-1]
  ind = bisect_left(data, v)
  if v == data[ind]:
    return bids[ind]
  alpha = (v - data[ind-1]) / (data[ind] - data[ind-1])
  return (1.0-alpha)*bids[ind-1] + alpha*bids[ind]


# take in the output of AuctionSolver for "fixed point finite difference"
def get_lognorm_bids_FP(output, num_kinds):
  bs = output[0::num_kinds+1]  # e.g. if num_kinds is 2, take every third element
  vals = [output[i::num_kinds+1] for i in range(1,num_kinds+1)]
  def get_bid_f(val_list):
    return lambda v: approx_bid(v, bs, val_list)
  bid_fs = [get_bid_f(val_list) for val_list in vals]
  return bid_fs

