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

#-------------------------------------------------

#### noise distributions

def draw_noiseData(Nsamp=1, **kwargs):
    """
    draw an SNR from the noise distribution
    """
    return chi2.rvs(__noise_df, size=Nsamp)

def noiseData_pdf(x, **kwargs):
    """
    evaluate the noise probability density function at x
    """
    return chi2.pdf(x, __noise_df)

def noiseData_cdf(x, **kwargs):
    """
    evaluate the noise cumulative density function for data<=x
    """
    return chi2.cdf(x, __noise_df)

#-------------------------------------------------

### signal distributions

# pareto distribution for latent variable (actual SNR)
def __draw_truncatedPareto(Nsamp=1, alpha=__alpha, beta=__beta, **kwargs):
    return beta*((1-np.random.rand(Nsamp))**(-1./alpha) - 1) ### 

def __truncatedPareto_pdf(x, alpha=__alpha, beta=__beta, **kwargs):
    return (alpha/beta)*(1+x/beta)**(-alpha-1)

def __truncatedPareto_cdf(x, alpha=__alpha, beta=__beta, **kwargs):
    return 1 - (1-x/beta)**(-alpha)

# distribution of noisy observations, marginalized over latent variable
def draw_signalData(Nsamp=1, alpha=__alpha, beta=__beta, **kwargs):
    """
    draw an SNR from the signal distribution
    """
    return np.array([ncx2.rvs(__noise_df, nc) for nc in __truncatedPareto_rvs(Nsamp, alpha=alpha, beta=beta)])

def signalData_pdf(x, alpha=__alpha, beta=__beta, **kwargs):
    """
    evaluate the signal probability density function at x
    """
    raise NotImplementedError, 'need to figure out how to calculate the marginalization over the latent variable quickly...'

def signalData_cdf(x, alpha=__alpha, beta=__beta, **kwargs):
    """
    evaluate the signal cumulative density function for data<=x
    """
    raise NotImplementedError, 'need to figure out how to calculate this quickly...'

#-------------------------------------------------

### bernoulli trials based on rate to determine whether there is a signal present

def draw_signalPresence(Nsamp=1, Rdt=__Rdt, **kwargs):
    """
    returns True if there is a signal, False if there isn't
    """
    return (np.random.rand(Nsamp) > np.exp(-Rdt))

def signalPresence_prob(Rdt=__Rdt, **kwargs):
    return 1-np.exp(-Rdt) 
