import math
import numpy as np
import numpy.random
import scipy.optimize

from ..lognorm import *

""" In an auction, a bidder's "kind" determines its type distribution.
0. We draw some number of bidder "kinds" according to BidderParams.
   We choose how many bidders of each kind arrive.
   ex ante, the kind of each bidder is common knowledge. 
1. Each bidder draws a private type according to its kind.
   This determines its cost and also its value distribution.
2. Bidders may inspect and draw their true value from its distribution.
"""

class BidderParams:
  """ A bidder's value is lognormal(mu_v, sigma_v) and cost lognormal(mu_c, sigma_c).
        log(val)  = mu_v + V0 + V1 + V2
        log(cost) = mu_c + C0 + C1
      Each variable on the right is a scalar mean-zero normal random variable.
      V0 has variance a_0 * sigma_v^2; V1 has variance a_1 * sigma_v^2;
      V2 has variance (1-a_0-a_1) * sigma_v^2.
      C0 has variance sigma_c^2 * a_0/(a_0+a_1); C1 has variance sigma_c^2 * a1/(a0+a1)
      Correlation:
        (V0, C0) are drawn jointly and independently
        (V1, C1) are drawn jointly and independently
        V2 is drawn independently.
      The correlations of (V0,C0) and (V1,C1) are governed by rho with rho=0
      uncorrelated, rho=-1 completely anti-correlated, rho=1 completely correlated.
  """
  def __init__(self, mu_v, mu_c, sigma_v, sigma_c, a_0, a_1, rho):
    self.mu_v = mu_v
    self.mu_c = mu_c
    self.sigma_v = sigma_v
    self.sigma_c = sigma_c
    self.a_0 = a_0
    self.a_1 = a_1
    self.rho = rho

  def __str__(self):
    return str([self.mu_v, self.mu_c, self.sigma_v, self.sigma_c, self.a_0, self.a_1, self.rho])
  def __repr__(self):
    return str(self)

  def gen_kind_params(self):
    mean = np.array( (self.mu_v, self.mu_c) )
    cov_term = self.rho * self.a_0 * self.sigma_v * self.sigma_c / math.sqrt(self.a_0 + self.a_1)
    cov = np.matrix( ((self.a_0 * self.sigma_v**2, cov_term),
                      (cov_term, self.a_0 * self.sigma_c**2 / (self.a_0 + self.a_1))) )
    (new_mu_v, new_mu_c) = numpy.random.multivariate_normal(mean, cov)
    new_sigma_v = math.sqrt(1.0 - self.a_0)*self.sigma_v
    new_alpha = self.a_1 / (1.0 - self.a_0)
    new_sigma_c = math.sqrt(self.a_1 / (self.a_0 + self.a_1)) * self.sigma_c
    return [new_mu_v, new_mu_c, new_sigma_v, new_sigma_c, new_alpha, self.rho]


class Kind:
  """ This is equivalent to a BidderParams bidder who has observed and conditioned on (C0,V0).
      Here, mu_v,sigma_v,mu_c,sigma_c mean something different from in BidderParams (sorry!).
      They are the new means and std devs conditioned on (C0,V0).

      So, the value is lognormal(mu_v, sigma_v) and cost lognormal(mu_c, sigma_c).
        log(val)  = mu_v + V1 + V2
        log(cost) = mu_c + C1
      (V1, C1) and V2 are all distributed exactly as in BidderParams; only the mus and sigmas
      have been updated/changed.
      In terms of these new variables: alpha = a_1 / (1 - a_0).
      V1 is normal with mean zero and variance alpha*sigma_v^2.
      (We use type_sigma for the corresponding standard deviation.)
      C1 is normal with mean zero and variance sigma_c^2.
      Their correlation is determined by rho.
      V2 is normal with mean zero and variance (1-alpha)*sigma_v^2.
      (We use insp_sigma for the corresponding standard deviation.)
  """
  def __init__(self, mu_v, mu_c, sigma_v, sigma_c, alpha, rho):
    self.mu_v = mu_v
    self.mu_c = mu_c
    self.sigma_v = sigma_v
    self.sigma_c = sigma_c
    self.alpha = alpha
    self.rho = rho
    # sigma parameter of type mu draw (i.e. Y1)
    self.type_sigma = math.sqrt(alpha) * sigma_v
    # sigma parameter of value draw conditioned on type (i.e. Y2)
    self.insp_sigma = math.sqrt(1.0 - alpha) * sigma_v

  def __str__(self):
    return str([self.mu_v, self.mu_c, self.sigma_v, self.sigma_c, self.alpha, self.rho])
  def __repr__(self):
    return str(self)

  # generate type as a pair (mu_v+V1, log(c)) of correlated normals
  def gen_type(self):
    mean = np.array( (self.mu_v, self.mu_c) )
    cov_term = self.rho * math.sqrt(self.alpha) * self.sigma_v * self.sigma_c
    cov = np.matrix( ((self.alpha * self.sigma_v**2, cov_term),
                      (cov_term, self.sigma_c**2)) )
    return numpy.random.multivariate_normal(mean, cov)

  # generate value, cost pair from a type pair
  def gen_valcost(self, my_type):
    logval = numpy.random.normal(my_type[0], self.insp_sigma)
    return (math.exp(logval), math.exp(my_type[1]))
  
  # return a covered call value only
  def gen_kappa(self):
    (mu_v, logc) = self.gen_type()
    strike = strike_price(mu_v, self.insp_sigma, math.exp(logc))
    logval = numpy.random.normal(mu_v, self.insp_sigma)
    return min(math.exp(logval), strike)

  # generate (mu, cost, strike, val)
  def gen_samp(self):
    my_type = self.gen_type()
    mu = my_type[0]
    val, cost = self.gen_valcost(my_type)
    strike = strike_price(mu, self.insp_sigma, cost)
    return mu, cost, strike, val

  # Pr[cost <= c | mu_v + V1 = mu]
  # In other words, if a bidder draws type with first entry mu,
  # return the conditional cdf of the cost
  def cost_cdf_given_v1(self, c, mu):
    if c <= 0:
      return 0.0
    if self.type_sigma <= 0.0:  # V1 is a constant zero
      return lognorm_cdf(c, self.mu_c, self.sigma_c)
    my_mu = self.mu_c + (self.rho*self.sigma_c / self.type_sigma) * (mu - self.mu_v)
    my_sigma = (1.0 - self.rho)*self.sigma_c
    if my_sigma <= 0.0:
      return 1.0 if my_mu <= math.log(c) else 0.0
    return lognorm_cdf(c, my_mu, my_sigma)

