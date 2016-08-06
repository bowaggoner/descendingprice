import math, random
import numpy as np
import sys
import scipy.stats
import scipy.optimize
from ..lognorm import *

import matplotlib.pyplot as plt

class Equil:
  def __init__(self, ns, val_cdfs, lo, hi):
    self.ns = ns
    self.num_kinds = len(ns)
    self.val_cdfs = val_cdfs
    self.lo = lo
    self.hi = hi

  def compute_equilibrium(self, num_pts, initial_block_size, learn_rate=0.15, tol=0.00001, max_rounds=1000):
    self.num_pts = num_pts
    self.xs = np.linspace(self.lo, self.hi, self.num_pts)
    self.dbid = self.xs[1] - self.xs[0]
    self.Gs = np.array([[cdf(x) for x in self.xs] for cdf in self.val_cdfs])
    self.vals = np.zeros( (self.num_kinds, self.num_pts) )

    if initial_block_size % 2 == 0:
      initial_block_size += 1
    for block_size in range(initial_block_size, 0, -2):
      print("BLOCK SIZE " + str(block_size))
      if (block_size+1) % 2 == 0:
        plt.figure()
        for Gs in self.Gs:
          plt.plot(self.xs, Gs)
        plt.title("Gs")
        plt.show()
        plt.figure()
        for val_list in self.vals:
          plt.plot(self.xs[:int(self.num_pts/8)], val_list[:int(self.num_pts/8)])
        plt.title("Vals")
        plt.show()
      success = self.compute_block_equilibrium(block_size, learn_rate, tol, max_rounds)
      if success is False:
        return [], []
    return self.xs, self.Gs

  def compute_block_equilibrium(self, block_size, learn_rate, tol, max_rounds):
    error = 100
    rounds = 0
    while error > tol:
      new_error = 0.0
      for k in range(self.num_kinds):
        br = self.compute_best_response(k, block_size)
        new_error += sum(abs(br - self.Gs[k])) / self.num_pts
        self.Gs[k] *= 1.0 - learn_rate
        self.Gs[k] += learn_rate * br
      new_error /= self.num_kinds
      print("round ", rounds, ": error ", new_error)
      if new_error > error:
        learn_rate *= 0.75
        print("cutting learning rate to ", learn_rate)
      error = new_error

      rounds += 1
      if rounds > max_rounds:
        print("Best-responses failed to converge!", file=sys.stderr)
        return False
    return True

  def compute_best_response(self, k, block_size):
    g = np.zeros( (self.num_pts) )  # best-response pdf
    vals = np.zeros( (self.num_pts) )
    opp_cdf = 0.0
    val = 0.0
    val_cdf = 0.0
    for bid_ind, bid in enumerate(self.xs):
      opp_cdf_2 = self.compute_opp_bid_cdf(bid_ind, k)
      opp_pmf = opp_cdf_2 - opp_cdf
      opp_cdf = opp_cdf_2

      new_val = self.compute_val_for_bid(bid, opp_pmf, opp_cdf)
      if new_val <= val:
        self.vals[k][bid_ind] = val
        continue
      val = new_val
      self.vals[k][bid_ind] = val
      val_cdf_2 = self.val_cdfs[k](new_val)
      val_pmf = val_cdf_2 - val_cdf
      val_cdf = val_cdf_2

      # spread out this pmf on this bid and surrounding ones
      # according to block_size
      perbid_prob = val_pmf / block_size
      start_ind = bid_ind - int(block_size / 2)
      start_ind = max(min(self.num_pts - block_size, start_ind), 0)
      for j in range(start_ind, start_ind + block_size):
        g[j] += perbid_prob
    return pdf_to_cdf(g)

  def compute_val_for_bid(self, bid, opp_pmf, opp_cdf):
    return bid + opp_cdf * self.dbid / opp_pmf

  def compute_opp_bid_cdf(self, ind, k):
    prob = 1.0
    for other_k in range(self.num_kinds):
      num = self.ns[other_k]
      if other_k == k:
        num -= 1
        if num == 0:
          continue
      prob *= self.Gs[other_k][ind] ** num
    return prob


def pdf_to_cdf(g):
  for i in range(1,len(g)):
    g[i] += g[i-1]
  g[-1] = 1.0
  return g

