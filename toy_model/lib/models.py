__description__ = "a module that houses our models for signals and noise distributions"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np

from scipy.stats import chi2
from scipy.stats import ncx2
from scipy.stats import pareto

#-------------------------------------------------

# degrees of freedom for additive noise distributions
__noise_df = 2

# default distribution parameters
__alpha = 2
__beta = 1e-4
__Rdt = 1e-5

# default monte-carlo integration parameters
__num_mc = 100

#-------------------------------------------------

def __logaddexp(array_like, axis=0):
    '''
    does the same thing as np.logaddexp, but sums all elements in the array instead of summing 2 arrays.
    returns a float
    if there is more than one index, we sum over the axis specified by "axis"
    '''
    if len(np.shape(array_like))==1:
        m = np.max(array_like)
        return m + np.log(np.sum(np.exp(np.array(array_like)-m)))
    else:
        m = np.max(array_like, axis=axis)
        return m + np.log(np.sum(np.exp(np.array(array_like)-m), axis=axis))

#-------------------------------------------------

#### noise distributions

def draw_noiseData(Nsamp=1, **kwargs):
    """
    draw an SNR from the noise distribution
    """
    return chi2.rvs(__noise_df, size=Nsamp)

def noiseData_lnpdf(x, **kwargs):
    """
    evaluate the noise probability density function at x
    """
    return chi2.logpdf(x, __noise_df)

def noiseData_lncdf(x, **kwargs):
    """
    evaluate the noise cumulative density function for data<=x
    """
    return chi2.logcdf(x, __noise_df)

#-------------------------------------------------

### signal distributions

# pareto distribution for latent variable (actual SNR)
def __draw_truncatedPareto(Nsamp=1, alpha=__alpha, beta=__beta, **kwargs):
    return beta*((1-np.random.rand(Nsamp))**(-1./alpha) - 1) ### 

def __truncatedPareto_lnpdf(x, alpha=__alpha, beta=__beta, **kwargs):
    return np.log(alpha/beta) - (alpha+1)*np.log(1+x/beta)

def __truncatedPareto_lncdf(x, alpha=__alpha, beta=__beta, **kwargs):
    return np.log(1 - (1-x/beta)**(-alpha))

# distribution of noisy observations, marginalized over latent variable
def draw_signalData(Nsamp=1, alpha=__alpha, beta=__beta, **kwargs):
    """
    draw an SNR from the signal distribution
    """
    return np.array([ncx2.rvs(__noise_df, nc) for nc in __draw_truncatedPareto(Nsamp, alpha=alpha, beta=beta)])

def signalData_lnpdf(x, alpha=__alpha, beta=__beta, num_mc=__num_mc, **kwargs):
    """
    evaluate the signal probability density function at x
    this is done by monte carlo sampling from p(y|alpha, beta) and approximating the integral of ncx2.pdf(x, __noise_df, y)
    """
    y = __draw_truncatedPareto(Nsamp=num_mc, alpha=alpha, beta=beta) ### draw monte carlo samples from p(y|alpha, beta)
    return __logaddexp([ncx2.logpdf(x, __noise_df, _) for _ in y]) - np.log(num_mc) ### approximate the integral via importance sampling

def signalData_lncdf(x, alpha=__alpha, beta=__beta, num_mc=__num_mc, **kwargs):
    """
    evaluate the signal cumulative density function for data<=x
    this is done by monte carlo sampling from p(y|alpha, beta) and approximating the integral of ncx2.cdf(x, __noise_df, y)
    """
    y = __draw_truncatedPareto(Nsamp=num_mc, alpha=alpha, beta=beta)
    return __logaddexp([ncx2.logcdf(x, __noise_df, _) for _ in y]) - np.log(num_mc)

#-------------------------------------------------

### bernoulli trials based on rate to determine whether there is a signal present

def draw_signalPresence(Nsamp=1, Rdt=__Rdt, **kwargs):
    """
    returns True if there is a signal, False if there isn't
    """
    return (np.random.rand(Nsamp) > np.exp(-Rdt))

def signalPresence_prob(Rdt=__Rdt, **kwargs):
    return np.log(1-np.exp(-Rdt))

#-------------------------------------------------

print """\
WARING:
also need to define models that have non-uniform distributions over time
    -> check whether non-observations are informative in the likelihood
"""
