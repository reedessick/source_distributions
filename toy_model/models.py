__description__ = "a module that houses our models for signals and noise distributions"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np

from scipy.stats import chi2
from scipy.stats import ncx2
from scipy.stats import pareto

#-------------------------------------------------
__noise_df = 2
#### noise distributions
def draw_noise(Nsamp=1):
    """
    draw an SNR from the noise distribution
    """
    return chi2.rvs(__noise_df, size=Nsamp)

def noise_pdf(x):
    """
    evaluate the noise probability density function at x
    """
    return chi2.pdf(x, __noise_df)

def noise_logpdf(x):
    return chi2.logpdf(x, __noise_df)

def noise_cdf(x):
    """
    evaluate the noise cumulative density function for data<=x
    """
    return chi2.cdf(x, __noise_df)

def noise_logcdf(x):
    return chi2.logcdf(x, __noise_df)

#-------------------------------------------------

### signal distributions
def draw_signal(Nsamp=1, index=2):
    """
    draw an SNR from the signal distribution
    """
    return np.array([ncx2.rvs(__noise_df, nc) for nc in pareto.rvs(index, size=Nsamp)]) ### may be a faster way to do this?

def signal_pdf(x, index=2):
    """
    evaluate the signal probability density function at x
    """
    raise NotImplementedError

def signal_cdf(x, index=2):
    """
    evaluate the signal cumulative density function for data<=x
    """
    raise NotImplementedError
