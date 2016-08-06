# fit distributions to bidders' covered call distributions

import sys
import scipy.optimize
from ..lognorm import *

def fit(kind, cdf_func, initial_params, num_trials):
  samps = sorted([kind.gen_kappa() for t in range(num_trials)])
  def loss(p):
    total = 0.0
    for i,s in enumerate(samps):
      # i/num_trials fraction of samps are below s
      total += (cdf_func(s, p) - i/num_trials)**2.0
    return total
  res = scipy.optimize.minimize(loss, initial_params, method="Nelder-Mead")
  if not res.success:
    print("Fitting CDF of covered call distribution failed!", file=sys.stderr)
    print(res.message, file=sys.stderr)
  return res.x

def my_lognorm_cdf(x, params):
  return lognorm_cdf(x, params[0], params[1])

def fit_lognorm(kind, num_trials, initial_params=[1,1]):
  return fit(kind, my_lognorm_cdf, initial_params, num_trials)

