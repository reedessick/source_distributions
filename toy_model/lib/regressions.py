__description__ = "a module that houses methods for regressing parameters using different methods"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import models

import emcee

#-------------------------------------------------

__known_methods = [
    'above_threshold_is_signal',               ### a naive approach in which you assume everything above threshold is a signal 
    'renormalized_likelihood_above_threshold', ### based of a private communication Maya Fishbach sent me
                                               ###   "renormalizes" the likelihood with the constraint that data is above threshold
    'all_data_all_models',                     ### based on Messenger&Veitch (2013)
                                               ###   considers all possible models (both signal and noise) explicitly for every datum.
                                               ###   marginalizes over unobserved data below some threshold
    'above_threshold_all_models',              ### a modified version of Messenger&Veitch (2013)
                                               ###   explicitly include both signal and noise models, but only for data above threshold
]

__known_engines = [
    'emcee',
#    'pymultinest', ### I don't have this installed yet...
]

__default_engine = 'emcee'
assert __default_engine in __known_engines

#------------------------

__thr = 10 ### default threshold

### default prior ranges
__min_alpha = 1
__max_alpha = 3

__min_beta = 1e-5
__max_beta = 1e-3

__min_Rdt = 1e-8
__max_Rdt = 1e-3

### engine technical parameters
__num_steps = 1000
__num_walkers = 50
__num_threads = 1

#-------------------------------------------------

def __above_threshold_is_signal_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, **kwargs):
    """
    only include data that is above thr and treats them all as signals
    """
    truth = data[:,0]>=thr
    return np.sum(np.log(models.signalData_pdf(data[truth][:,0], alpha=alpha, beta=beta)))

def __renormalized_likelihood_above_threshold_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, **kwargs):
    """
    only include data that is above thr, treats them all as signals, and renormalizes the likelihood so that it only covers "detectable data"
    """
    norm = 1-models.signalData_cdf(thr, alpha=alpha, beta=beta) ### normalization of likelihood for data above threshold
    return np.sum(np.log(models.signalData_pdf(data[data[:,0]>=thr][:,0], alpha=alpha, beta=beta)) - np.log(norm))

def __above_threshold_all_models_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, **kwargs):
    """
    only include data that is above a threshold and treats them all as signals
    """
    selected = data[data[:,0]>=thr]
    pRdt = models.signalPresence_prob(Rdt=Rdt) ### prior for signal model (based on rate)

    ### work with Logs for accuracies sake...
    lnpNoise = np.log(models.noiseData_pdf(selected[:,0])) + np.log(1-pRdt)
    lnpSignal = np.log(models.signalData_pdf(selected[:,0], alpha=alpha, beta=beta)) + np.log(pRdt)

    ### add models together
    return np.sum(np.logaddexp(lnpNoise, lnpSignal))

def __all_data_all_models_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, **kwargs):
    """
    explicitly include all models in the likelihood and marginalize over "unobserved" data that is below thr
    """
    truth = data[:,0]>=thr
    above = data[truth]
    below = data[np.logical_not(truth)]
    pRdt = models.signalPresence_prob(Rdt=Rdt) ### prior for signal model (based on rate)

    a = np.log(1-pRdt)
    b = np.log(pRdt)

    ### data above threshold
    lnpNoiseAbove = np.log(models.noiseData_pdf(above[:,0])) + a
    lnpSignalAbove = np.log(models.signalData_pdf(above[:,0], alpha=alpha, beta=beta)) + b

    ### (marginalized) data below threshold
    Nbelow = len(below)
    lnpNoiseBelow = np.log(models.noiseData_cdf(thr)) + a
    lnpSignalBelow = np.log(models.signalData_cdf(thr, alpha=alpha, beta=beta)) + b

    ### add everything together
    return np.sum(np.logaddexp(lnpNoiseAbove, lnpSignalAbove)) + Nbelow*np.logaddexp(lnpNoiseBelow, lnpSignalBelow)

#--- a single routing function, because that's how we're doing things (abstract the library away from the executable)

def lnlikelihood(data, method, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, **kwargs):
    """
    compute the likelihood
    """
    if method == 'above_threshold_is_signal':
        return __above_threshold_is_signal_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, **kwargs)

    elif method == 'above_threshod_all_models':
        return __above_threshold_all_models_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, **kwargs)

    elif method == "renormalized_likelihood_above_threshold":
        return __renormalized_likelihood_above_threshold_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, **kwargs)

    elif method == "all_data_all_models":
        return __all_data_all_models_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, **kwargs)

    else:
        raise ValueError, 'method=%s not understood'%method

#-------------------------------------------------

def lnprior(alpha, beta, Rdt, min_alpha=__min_alpha, max_alpha=__max_alpha, min_beta=__min_beta, max_beta=__max_beta, min_Rdt=__min_Rdt, max_Rdt=__max_Rdt, **kwargs):
    """
    top hat prior
    """
    if (min_alpha <= alpha <= max_alpha) and (min_beta <= beta <= max_beta) and (min_Rdt <= Rdt <= max_Rdt):
        return -np.log(max_alpha-min_alpha) - np.log(max_beta - min_beta) - np.log(max_Rdt - min_Rdt)
    else:
        return -np.infty ### top-hat priors

#-------------------------------------------------

def lntarget(data, method, **kwargs):
    """
    lnlikelihood + lnprior
    the target distribution from which we sample
    """
    return lnlikelihood(data, method, **kwargs) + lnprior(**kwargs)

def regress(
      data, 
      method, 
      engine, 
      outpath,
      thr=__thr,
      Rdt=None,
      min_Rdt = __min_Rdt,
      max_Rdt = __max_Rdt,
      alpha = None,
      min_alpha = __min_alpha,
      max_alpha = __max_alpha,
      beta = None,
      min_beta = __min_beta,
      max_beta = __max_beta,
      num_mc = models.__num_mc,
      num_steps = __num_steps,
      num_walkers = __num_walkers,
      num_threads = __num_threads,
      verbose=False,
      **kwargs
    ):
    """
    a single function call that will set up the regression and execute it
    output is written in a common data format for all engines and placed in outpath
    """
    raise NotImplementedError


print """\
WARING:
also need to define models that have non-uniform distributions over time
    -> check whether non-observations are informative in the likelihood
"""
