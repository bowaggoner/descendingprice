import math
import numpy as np
import sys
from bisect import bisect_left
import scipy.optimize

from tools.startups.lazy_bisect import lazy_bisect
from tools.lognorm import *
import tools.startups.model as model

class Equil:
  def __init__(self, kinds, ns):
    self.kinds = kinds
    self.num_kinds = len(kinds)
    self.ns = ns

  def compute_equilibrium(self, num_pts, num_v1s, learn_rate=0.35, tol=0.00001, max_rounds=100):
    """ In a simultaneous-sealed-bid auction, compute an equilibrium
        in terms of distribution of bids made by each bidder.
        Use iterated best-responses.
        Outputs xs, Gs, v1s, cstars: a list of allowed bids, corresponding equilibrium
        bid cdfs for each bidder, list of discretized V1 (types) for each bidder, and
        corresponding cost thresholds (inspect at v1 iff cost <= threshold).
  
        kinds:      list of model.Kind objects of the bidders.
        ns:         list of num of bidders of each kind
        num_pts:    integer; discretize the bid spaces to this many points (suggestion 200)
        num_mus:    integer; how many values of mu in numerical integration (suggestion 200)
        learn_rate: float in [0,1]: how quickly to evolve bid distribution
        tol:        required average error in best-response to stop
                    tol=0.02 means <= 2% difference in best-response cdf at each point
        max_rounds: integer; give up after this many iterations
    """
    self.num_xs = num_pts
    self.num_v1s = num_v1s
    get_v1s = lambda kind: np.linspace(min_v1(kind), max_v1(kind), num_v1s if kind.type_sigma > 0.0 else 1)
    self.v1s = np.array([get_v1s(kind) for kind in self.kinds])
    max_bid = max([lognorm_ppf(0.9999, kind.mu_v, kind.sigma_v) for kind in self.kinds])
    self.xs = np.linspace(0.0, max_bid, self.num_xs)

    # precompute some useful stuff
    # exp_profits[k][mu_ind][x_ind] = E[(val - xs[x_ind])+]
    # for a bidder of kind k with mu = mus[k][mu_ind]
    self.exp_profits = np.array([[[expected_profit(b, v1, kind.insp_sigma)
      for b in self.xs] for v1 in self.v1s[k]] for k,kind in enumerate(self.kinds)])
    self.v1_cdfs = np.array([[norm_cdf(v1, self.kinds[k].mu_v, self.kinds[k].type_sigma)
       for v1 in my_v1s] for k,my_v1s in enumerate(self.v1s)])
    for k in range(len(self.v1s)):
      self.v1_cdfs[k][-1] = 1.0

    # set initial parameters
    self.cstars = np.array([[0.0 for v1 in my_v1s] for my_v1s in self.v1s])
    self.Gs = np.array([[lognorm_cdf(x, kind.mu_v, kind.sigma_v)
      for x in self.xs] for kind in self.kinds])
    for k in range(len(self.Gs)):
      self.Gs[k][-1] = 1.0
  
    error = 100
    rounds = 0
    while error > tol:
      new_error = 0.0
      for k in range(len(self.kinds)):
        br = self.compute_best_response(k)
        new_error += sum(abs(br - self.Gs[k])) / len(br)
        self.Gs[k] *= (1.0 - learn_rate)
        self.Gs[k] += learn_rate*br
      
      new_error /= self.num_kinds
      if new_error > error:
        # assume we're going too fast and overshooting
        learn_rate *= 0.75
        print("cutting learning rate to ", learn_rate)
      error = new_error
      print("round ",rounds, ":  error ", error)
      if rounds > max_rounds:
        print("Best-responses failed to converge", file=sys.stderr)
        return [], [], [], []
      rounds += 1
    return self.xs.tolist(), self.Gs.tolist(), self.v1s.tolist(), self.cstars.tolist()


  def compute_best_response(self, k):
    # pre-compute Pr[highest competing bid <= x] for x in xs
    opp_cdf = self.compute_opp_bid_cdf(k)
    g = np.zeros( (self.num_xs) )  # best-response pdf
  
    # integrate over different mus (outcomes of bidder's draw of V1)
    # and for each, fill in the distribution of bids when drawing that mu
    v1_cdf = 0.0
    for v1_ind,v1 in enumerate(self.v1s[k]):
      v1_cdf2 = self.v1_cdfs[k][v1_ind]
      v1_pmf = v1_cdf2 - v1_cdf
      v1_cdf = v1_cdf2
  
      # get a list of discretized costs from low to high
      ev = lognorm_mean(v1, self.kinds[k].insp_sigma)
      cs_list = np.array(list(reversed(ev - self.xs)))
  
      # inspect iff c <= curr_thresh
      guess = self.cstars[k][0] if v1_ind == 0 else self.cstars[k][v1_ind-1]
      cstar_ind, cstar = self.get_c_threshold(v1_ind, k, opp_cdf, ev, cs_list, guess)
      self.cstars[k][v1_ind] = cstar
      self.update_g(g, v1_pmf, v1_ind, v1, cstar_ind, cstar, k, opp_cdf, cs_list)
    return pdf_to_cdf(g)

  # find c* using cs_list
  def get_c_threshold(self, v1_ind, k, opp_cdf, ev, cs_list, guess):
    sigma = self.kinds[k].insp_sigma
    # gain from inspection = integral E[(v - b)+] dG_{-i}(b)
    # where G_{-i}(b) is cdf of highest opposing bid
    gain_insp = my_cdf_integral(opp_cdf, self.exp_profits[k][v1_ind])

    def diff(c):
      bid_if_dont = ev - c
      prob_win_if_dont = approx_f_from_arr(bid_if_dont, self.xs, opp_cdf)
      exp_pay_if_dont = my_cdf_integral(opp_cdf, self.xs, hi_cdf=prob_win_if_dont)
      util_if_dont = prob_win_if_dont * bid_if_dont - exp_pay_if_dont
      util_insp = gain_insp - c
      return util_if_dont - util_insp

    # if we don't have a guess, use binary search
    if guess is None:
      zero_index = index_of(0, cs_list)
      cstar_ind = min(len(cs_list)-1, lazy_bisect(0.0, cs_list, diff, lo=zero_index))
    else:
      # just linear search to the left or right
      cstar_ind = index_of(guess, cs_list)
      curr_d = diff(cs_list[cstar_ind])
      if curr_d == 0.0:
        return cstar_ind, cs_list[cstar_ind]
      elif curr_d < 0.0:
        while cstar_ind+1 < len(cs_list) and diff(cs_list[cstar_ind+1]) < 0.0:
          cstar_ind += 1
      else:
        while cstar_ind > 0 and cs_list[cstar_ind-1] >= 0 and curr_d > 0.0:
          cstar_ind -= 1
          curr_d = diff(cs_list[cstar_ind])
    if cstar_ind == 0 or cstar_ind == self.num_v1s-1 or cs_list[cstar_ind-1] < 0.0:
      return cstar_ind, cs_list[cstar_ind]
    res = scipy.optimize.bisect(diff, cs_list[cstar_ind], cs_list[cstar_ind+1])
    return cstar_ind, res

  def update_g(self, g, v1_pmf, v1_ind, v1, cstar_ind, cstar, k, opp_cdf, cs_list):
    if cstar <= 0.0:
      # skip the inspection case cause it never happens
      c_cdf = 0.0
    else:
      # When inspecting, just bid the value distribution
      # chopped by the probability of this v1 and small enough c
      c_cdf = self.kinds[k].cost_cdf_given_v1(cstar, v1)
      prob = v1_pmf * c_cdf
      val_cdf = 0.0
      sigma = self.kinds[k].insp_sigma
      for i in range(self.num_xs):
        val_cdf2 = 1.0 if i == self.num_xs-1 else lognorm_cdf(self.xs[i], v1, sigma)
        val_pmf = val_cdf2 - val_cdf
        val_cdf = val_cdf2
        g[i] += prob * val_pmf
  
    # now cases when not inspecting; for each c
    # from cstar up to max, bid E[val]-c when getting that c
    start = min(cstar_ind+1, self.num_xs-1)
    for i in range(start,self.num_xs):
      c = cs_list[i]
      c_cdf2 = 1.0 if i==self.num_xs-1 else self.kinds[k].cost_cdf_given_v1(c, v1)
      c_pmf = c_cdf2 - c_cdf
      c_cdf = c_cdf2
      # due to reversing, when we have this cost, this is our bid
      g[self.num_xs - 1 - i] += v1_pmf * c_pmf
    # done updating

  # compute a cdf of the highest bid faced by a bidder of kind k
  # for each x in xs[k]
  def compute_opp_bid_cdf(self, k):
    cdf = np.zeros( (self.num_xs) )
    for x_ind, x in enumerate(self.xs):
      prob = 1.0
      for i in range(self.num_kinds):
        num = self.ns[i] - (1 if i==k else 0)
        prob *= self.Gs[i][x_ind] ** num
      cdf[x_ind] = prob
    cdf[-1] = 1.0
    return cdf

def pdf_to_cdf(g):
  for i in range(1,len(g)):
    g[i] += g[i-1]
  g[-1] = 1.0
  return g


# given a list of points and function values,
# approximate the value of the function at some x
def approx_f_from_arr(x, pts, vals):
  ind = index_of(x, pts)
  if ind == len(pts)-1:
    return vals[-1]
  alpha = (x - pts[ind])/(pts[ind+1]-pts[ind])
  return (1.0-alpha)*vals[ind] + alpha*vals[ind+1]

# round x (down) to an index in my_xs
# works very fast if my_xs is evenly spaced;
# is correct but slow otherwise
def index_of(x, my_xs):
  x_one = my_xs[1]
  if x < x_one:
    return 0
  x_zero = my_xs[0]
  skip = x_one - x_zero
  guess = int( (x-x_zero)/skip )

  max_ind = len(my_xs)-1
  guess = min(guess, max_ind)
  if x >= my_xs[guess]:
    while x >= my_xs[guess] and guess < max_ind:
      guess += 1
    return guess-1
  while x < my_xs[guess]:
    guess -= 1
  return guess


# get lower / upper bounds on v1 = mu_v + V1
def min_v1(kind):
  return norm_ppf(1e-8, kind.mu_v, kind.type_sigma)
def max_v1(kind):
  return norm_ppf(1.0 - 1e-8, kind.mu_v, kind.type_sigma)

# using a trapezoid rule to integrate
# array of cdf against array of vals
def my_cdf_integral(cdf, vals, hi_cdf=None):
  g0 = cdf[0]
  if hi_cdf is not None and g0 > hi_cdf:
    return 0.0
  v0 = vals[0]
  total = g0 * v0
  i = 0
  for i in range(1,len(vals)):
    g1 = cdf[i]
    v1 = vals[i]
    if hi_cdf is not None and g1 > hi_cdf:
      alpha = (hi_cdf - g0) / (g1 - g0)
      vhi = alpha * v1 + (1.0 - alpha) * v0
      total += (hi_cdf - g0) * (v0 + vhi) / 2.0
      break
    total += (g1 - g0) * (v0 + v1) / 2.0
    v0 = v1
    g0 = g1
  return total

