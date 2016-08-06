# useful functions for the lognormal distribution

import math
from scipy.optimize import brentq

def norm_cdf(x, mu, sigma):
  if sigma == 0.0:
    return 1.0 if mu <= x else 0.0
  return 0.5 + 0.5*math.erf((x - mu) / (math.sqrt(2.0)*sigma))

def lognorm_cdf(x, mu, sigma):
  if x <= 0.0:
    return 0.0
  return norm_cdf(math.log(x), mu, sigma)

# inverse cdf: return the x such that Pr[X <= x] = q
# when X ~ Normal(mu, sigma)
def norm_ppf(q, mu, sigma, tol=1e-12):
  if sigma == 0.0:
    return mu
  if q == 0.5:
    return mu

  # get bounds a <= result <= b, then use scipy.optimize.brentq
  diff = sigma
  if q < 0.5:
    b = mu
    a = mu - diff
    cdf = norm_cdf(a, mu, sigma)
    while cdf > q:
      b = a
      diff *= 2
      a -= diff
      cdf = norm_cdf(a, mu, sigma)
      if q < tol and cdf < tol:
        return a
  else:
    a = mu
    b = mu + diff
    cdf = norm_cdf(b, mu, sigma)
    while cdf < q:
      a = b
      diff *= 2
      b += diff
      cdf = norm_cdf(b, mu, sigma)
      if q > 1.0 - tol and cdf > 1.0 - tol:
        return b

  return brentq(lambda x: norm_cdf(x, mu, sigma) - q, a, b, xtol=tol)

# inverse cdf: return the x such that Pr[X <= x] = q
# when X ~ lognormal(mu, sigma)
def lognorm_ppf(q, mu, sigma, tol=1e-12):
  return math.exp(norm_ppf(q, mu, sigma, tol=tol))

def lognorm_mean(mu, sigma):
  return math.exp(mu + 0.5*sigma**2.0)

# integral from x to infinity of z f(z) dz, for lognormal
def partial_expectation(x, mu, sigma):
  mean = lognorm_mean(mu, sigma)
  if x <= 0.0:
    return mean
  # closed-form solution in terms of normal cdf
  return mean * norm_cdf(mu + sigma**2 - math.log(x), 0.0, sigma)

# E[(val - strike_price)+]
# when value is drawn lognormal(mu, sigma)
def expected_profit(strike, mu, sigma):
  first_term = partial_expectation(strike, mu, sigma)
  second_term = strike * (1.0 - lognorm_cdf(strike, mu, sigma))
  return first_term - second_term

# same as expected_profit, but broken up into
# both terms
def expected_gain_pay(strike, mu, sigma):
  first_term = partial_expectation(strike, mu, sigma)
  second_term = strike * (1.0 - lognorm_cdf(strike, mu, sigma))
  return first_term, second_term
  
# return the s such that E[(val - s)+] = cost
def strike_price(mu, sigma, cost, lo=None, hi=None):
  def difference(strike):
    return expected_profit(strike, mu, sigma) - cost
  if lo is None:
    lo = lognorm_mean(mu, sigma) - cost
  while difference(lo) < 0.0:
    if lo >= 0.0:
      lo = -1.0
    else:
      lo *= 2
  if hi is None:
    hi = lognorm_ppf(0.995, mu, sigma)
  while difference(hi) > 0.0:
    lo = hi
    hi *= 2
  return brentq(difference, lo, hi)

