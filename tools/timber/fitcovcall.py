# fit distributions to bidders' covered call distributions

import sys
import scipy.optimize
import numpy as np

from tools.timber.model import Timber
from ..lognorm import *

def fit(timber, my_cdf, initial_params, num_samps):
  """ Attempt to fit a distribution to the covered call value distribution.
      timber: a model.Timber
      my_cdf: a cumulative distribution function for the class of distributions to fit.
              my_cdf(x, params) = Pr[outcome <= x ; params]
      initial_params: starting point for fit
      num_samps: number of empirical covered calls to draw
  """
  logger_covs = np.array([min(timber.gen_logger()) for i in range(num_samps)])
  mill_covs = np.array([min(timber.gen_mill()) for i in range(num_samps)])
  logger_params = do_fit(sorted(logger_covs), my_cdf, initial_params)
  mill_params = do_fit(sorted(mill_covs), my_cdf, initial_params)
  return logger_params, mill_params

# given sorted list of sampled covered call values, fit parameterized distribution
def do_fit(covs, my_cdf, initial_params):
  def loss(p):
    total = 0.0
    for i,s in enumerate(covs):
      # i/len(covs) fraction of samples are below s
      total += (my_cdf(s, p) - i/len(covs))**2.0
    return total
  res = scipy.optimize.minimize(loss, initial_params, method = "Nelder-Mead")
  if not res.success:
    print("Fitting CDF of covered call distribution failed!", file=sys.stderr)
    print(res.message, file=sys.stderr)
  return res.x

def my_lognorm_cdf(x, params):
  return lognorm_cdf(x, params[0], params[1])

def fit_lognorm(timber, num_trials, initial_params=[4,0.8]):
  return fit(timber, my_lognorm_cdf, initial_params, num_trials)

