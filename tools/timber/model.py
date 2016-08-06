#!/usr/bin/python3
#
# The model of Roberts+Sweeting AER
# "When Should Sellers Use Auctions",
#
# Values are drawn lognormal(mu_v, sigma_v),
# then signals are drawn with log(signal) ~ Normal(log(value), sigma_eps).
# Hence posterior is lognormal with parameters
#     alpha*mu_v + (1-alpha)*log(signal), sqrt(alpha)*sigma_v
# where alpha = sigma_eps^2/(sigma_v^2 + sigma_eps^2).
#

# Parameters of bidder distributions
# ----------------------------------
# mu_logger: parameter for logger value distribution
# mu_diff: difference between parameter of loggers and sawmills
# sigma_v: parameter for value distributions (same for both types)
# alpha: controls sigma_eps, namely alpha = sigma_eps^2/(sigma_v^2 + sigma_eps^2)
# cost: inspection cost (fixed, same for both types, denoted K in RS)
# 
# Drawing the parameters
# ----------------------
# In the following, TRN(mu,sigma,lo,hi) = normal distribution truncated to [lo, hi]
#
# mu_logger ~ Normal(X_a*beta_1, omega_mu_logger^2)
# sigma_v ~ TRN(X_a*beta_2, omega_sigma^2, 0.01, infty)
# mu_diff ~ TRN(X_a*beta_3, omega_mu_diff^2, 0, infty)
# alpha ~ TRN(X_a*beta_4, omega_alpha^2, 0, 1)
# cost ~ TRN(X_a*beta_g, omega_cost^2, 0, infty)
#

import math,random
import sys
import numpy as np
import numpy.random
import scipy.optimize
import scipy.stats
from multiprocessing import Pool
import matplotlib.pyplot as plt

from ..lognorm import *

class Timber:
  def __init__(self, mu_logger, mu_diff, sigma_v, alpha, cost):
    self.mu_logger = mu_logger
    self.mu_diff = mu_diff
    self.sigma_v = sigma_v
    self.alpha = alpha
    self.cost = cost

    self.mu_mill = mu_logger + mu_diff
    self.sigma_eps = sigma_v * math.sqrt(alpha / (1.0 - alpha))

  def __str__(self):
    return "Timber" + str((self.mu_logger, self.mu_diff, self.sigma_v, self.alpha, self.cost))
  def __repr__(self):
    return str(self)

  def gen_logger(self):
    return self.gen_samp(self.mu_logger)

  def gen_mill(self):
    return self.gen_samp(self.mu_mill)

  # return [strike, val]
  def gen_samp(self, mu):
    logval = np.random.normal(mu, self.sigma_v)
    logsignal = np.random.normal(logval, self.sigma_eps)
    conditional_mu = self.alpha*mu + (1.0 - self.alpha)*logsignal
    conditional_sigma = math.sqrt(self.alpha) * self.sigma_v
    strike = strike_price(conditional_mu, conditional_sigma, self.cost)
    return strike, math.exp(logval)
   

