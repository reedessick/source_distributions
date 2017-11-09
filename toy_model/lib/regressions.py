__description__ = "a module that houses methods for regressing parameters using different methods"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import sys
import gzip
import numpy as np
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

def __above_threshold_is_signal_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, num_mc=models.__num_mc, **kwargs):
    """
    only include data that is above thr and treats them all as signals
    """
    truth = data[:,0]>=thr
    if np.any(truth):
        return np.sum(models.signalData_lnpdf(data[truth][:,0], alpha=alpha, beta=beta, num_mc=num_mc))

    else:
        return 0

def __renormalized_likelihood_above_threshold_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, num_mc=models.__num_mc, **kwargs):
    """
    only include data that is above thr, treats them all as signals, and renormalizes the likelihood so that it only covers "detectable data"
    """
    truth = data[:,0]>=thr
    if np.any(truth):
        norm = 1-models.signalData_cdf(thr, alpha=alpha, beta=beta, num_mc=num_mc) ### normalization of likelihood for data above threshold
        return np.sum(models.signalData_lnpdf(data[truth][:,0], alpha=alpha, beta=beta, num_mc=num_mc) - np.log(norm))

    else:
        return 0

def __above_threshold_all_models_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, num_mc=models.__num_mc, **kwargs):
    """
    only include data that is above a threshold and treats them all as signals
    """
    selected = data[data[:,0]>=thr]
    if len(selected):
        pRdt = models.signalPresence_prob(Rdt=Rdt) ### prior for signal model (based on rate)

        ### work with Logs for accuracies sake...
        lnpNoise = models.noiseData_lnpdf(selected[:,0]) + np.log(1-pRdt)
        lnpSignal = models.signalData_lnpdf(selected[:,0], alpha=alpha, beta=beta, num_mc=num_mc) + np.log(pRdt)

        ### add models together
        return np.sum(np.logaddexp(lnpNoise, lnpSignal))

    else:
        return 0

def __all_data_all_models_lnlikelihood(data, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, num_mc=models.__num_mc, **kwargs):
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
    lnpNoiseAbove = models.noiseData_lnpdf(above[:,0]) + a
    lnpSignalAbove = models.signalData_lnpdf(above[:,0], alpha=alpha, beta=beta, num_mc=num_mc) + b

    ### (marginalized) data below threshold
    Nbelow = len(below)
    lnpNoiseBelow = models.noiseData_lncdf(thr) + a
    lnpSignalBelow = models.signalData_lncdf(thr, alpha=alpha, beta=beta, num_mc=num_mc) + b

    ### add everything together
    return np.sum(np.logaddexp(lnpNoiseAbove, lnpSignalAbove)) + Nbelow*np.logaddexp(lnpNoiseBelow, lnpSignalBelow)

#--- a single routing function, because that's how we're doing things (abstract the library away from the executable)

def lnlikelihood(data, method, thr=__thr, alpha=models.__alpha, beta=models.__beta, Rdt=models.__Rdt, num_mc=models.__num_mc, **kwargs):
    """
    compute the likelihood
    """
    if method == 'above_threshold_is_signal':
        return __above_threshold_is_signal_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, num_mc=num_mc, **kwargs)

    elif method == 'above_threshod_all_models':
        return __above_threshold_all_models_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, num_mc=num_mc, **kwargs)

    elif method == "renormalized_likelihood_above_threshold":
        return __renormalized_likelihood_above_threshold_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, num_mc=num_mc, **kwargs)

    elif method == "all_data_all_models":
        return __all_data_all_models_lnlikelihood(data, alpha=alpha, beta=beta, Rdt=Rdt, num_mc=num_mc, **kwargs)

    else:
        raise ValueError, 'method=%s not understood'%method

#-------------------------------------------------

def lnprior(alpha, beta, Rdt, min_alpha=__min_alpha, max_alpha=__max_alpha, min_beta=__min_beta, max_beta=__max_beta, min_Rdt=__min_Rdt, max_Rdt=__max_Rdt, **kwargs):
    """
    top hat prior
    """
    ans = 0
    if (min_alpha!=None and max_alpha!=None):
        if (min_alpha <= alpha <= max_alpha):
            ans -= np.log(max_alpha-min_alpha)
        else:
            return -np.infty

    if (min_beta!=None and max_beta!=None):
        if (min_beta <= beta <= max_beta):
            ans -= np.log(max_beta - min_beta)
        else:
            return -np.infty

    if (min_Rdt!=None and max_Rdt!=None):
        if (min_Rdt <= Rdt <= max_Rdt):
            ans -= np.log(max_Rdt - min_Rdt)
        else:
            return np.infty

    return ans

def sample_from_prior(prior_min, prior_max, Nsamp=1):
    """
    sample from the top hat prior
    """
    return prior_min + (prior_max-prior_min)*np.random.rand(Nsamp)

#-------------------------------------------------

def __lntarget(x, lnL, lnP):
    """
    a wrapper function that first evaluates lnP to make sure parameters are sane and only then evaluates lnL
    """
    ans = lnP(x)
    if ans > -np.infty:
        ans += lnL(x)
    return ans

def __emcee_regress(
      data,
      method,
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
    run emcee
    """
    # define the target distribution on the fly
    # also sample from prior distributions
    sample_alpha = alpha==None
    sample_beta = beta==None
    sample_Rdt = Rdt==None

    ### neglect params from prior if frozen
    if not sample_alpha:
        min_alpha = max_alpha = None
    if not sample_beta:
        min_beta = max_beta = None
    if not sample_Rdt:
        min_Rdt = max_Rdt = None

    prior_kwargs = dict(
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        min_beta=min_beta,
        max_beta=max_beta,
        min_Rdt=min_Rdt,
        max_Rdt=max_Rdt
    )

    # set up initial positions
    if sample_alpha and sample_beta and sample_Rdt:
        if verbose:
            print( 'sampling : alpha, beta, Rdt')
        lnL = lambda var_alpha, var_beta, var_Rdt : lnlikelihood(data, method, thr=thr, alpha=var_alpha, beta=var_beta, Rdt=var_Rdt, num_mc=num_mc)
        lnP = lambda var_alpha, var_beta, var_Rdt : lnprior(var_alpha, var_beta, var_Rdt, **prior_kwargs)
        template = "%.9e %.9e %.9e"
        num_dim = 3
        p0 = np.array([
            sample_from_prior(min_alpha, max_alpha, Nsamp=num_walkers), 
            sample_from_prior(min_beta, max_beta, Nsamp=num_walkers), 
            sample_from_prior(min_Rdt, max_Rdt, Nsamp=num_walkers),
        ]).transpose()

    elif sample_alpha and sample_beta:
        if verbose:
            print( 'sampling : alpha, beta')
        lnL = lambda var_alpha, var_beta : lnlikelihood(data, method, thr=thr, alpha=var_alpha, beta=var_beta, Rdt=Rdt, num_mc=num_mc)
        lnP = lambda var_alpha, var_beta : lnprior(var_alpha, var_beta, Rdt, **prior_kwargs)
        template = "%.9e %.9e "+"%.9e"%Rdt
        num_dim = 2
        p0 = np.array([
            sample_from_prior(min_alpha, max_alpha, Nsamp=num_walkers),
            sample_from_prior(min_beta, max_beta, Nsamp=num_walkers),
        ]).transpose()

    elif sample_alpha and sample_Rdt:
        if verbose:
            print( 'sampling : alpha, Rdt')
        lnL = lambda var_alpha, var_Rdt : lnlikelihood(data, method, thr=thr, alpha=var_alpha, beta=beta, Rdt=var_Rdt, num_mc=num_mc)
        lnP = lambda var_alpha, var_Rdt : lnprior(var_alpha, beta, var_Rdt, **prior_kwargs)
        template = "%.9e "+"%.9e"%beta+" %.9e"
        num_dim = 2
        p0 = np.array([
            sample_from_prior(min_alpha, max_alpha, Nsamp=num_walkers),
            sample_from_prior(min_Rdt, max_Rdt, Nsamp=num_walkers),
        ]).transpose()

    elif sample_beta and sample_Rdt:
        if verbose:
            print( 'sampling : beta, Rdt')
        lnL = lambda var_beta, var_Rdt : lnlikelihood(data, method, thr=thr, alpha=alpha, beta=var_beta, Rdt=var_Rdt, num_mc=num_mc)
        lnP = lambda var_beta, var_Rdt : lnprior(alpha, var_beta, var_Rdt, **prior_kwargs)
        template = "%.9e"%alpha+" %.9e %.9e"
        num_dim = 2
        p0 = np.array([
            sample_from_prior(min_beta, max_beta, Nsamp=num_walkers),
            sample_from_prior(min_Rdt, max_Rdt, Nsamp=num_walkers),
        ]).transpose()

    elif sample_alpha:
        if verbose:
            print( 'sampling : alpha')
        lnL = lambda (var_alpha,) : lnlikelihood(data, method, thr=thr, alpha=var_alpha, beta=beta, Rdt=Rdt, num_mc=num_mc)
        lnP = lambda (var_alpha,) : lnprior(var_alpha, beta, Rdt, **prior_kwargs)
        template = "%.9e "+"%.9e %.9e"%(beta, Rdt)
        num_dim = 1
        p0 = np.array([
            sample_from_prior(min_alpha, max_alpha, Nsamp=num_walkers),
        ]).transpose()

    elif sample_beta:
        if verbose:
            print( 'sampling : beta')
        lnL = lambda (var_beta,) : lnlikelihood(data, method, thr=thr, alpha=alpha, beta=var_beta, Rdt=Rdt, num_mc=num_mc) 
        lnP = lambda (var_beta,) : lnprior(alpha, var_beta, Rdt, **prior_kwargs)
        template = "%.9e"%alpha+" %.9e "+"%.9e"%Rdt
        num_dim = 1
        p0 = np.array([
            sample_from_prior(min_beta, max_beta, Nsamp=num_walkers),
        ]).transpose()

    elif sample_Rdt:
        if verbose:
            print( 'sampling : Rdt')
        lnL = lambda (var_Rdt,) : lnlikelihood(data, method, thr=thr, alpha=alpha, beta=beta, Rdt=var_Rdt, num_mc=num_mc)
        lnP = lambda (var_Rdt,) : lnprior(alpha, beta, var_Rdt, **prior_kwargs)
        template = "%.9e %.9e "%(alpha, beta)+"%.9e"
        num_dim = 1
        p0 = np.array([
            sample_from_prior(min_Rdt, max_Rdt, Nsamp=num_walkers),
        ]).transpose()

    else:
        raise ValueError, 'all model parameters pinned. Sampling does not make sense...'

    template = "%04d %.9e "+template

    # define sampler object
    sampler = emcee.EnsembleSampler(
        num_walkers, 
        num_dim, 
        __lntarget, 
        args = [lnL, lnP],
        threads=num_threads,
    )

    ### run the sampler
    if outpath.endswith('gz'):
        file_obj = gzip.open(outpath, 'w')
    else:
        file_obj = open(outpath, 'w')
    print >> file_obj, '%4s %14s %14s %14s %14s'%tuple('id lnprob alpha beta Rdt'.split())

    if verbose: ### print a progress bar while sampling
        status_width = 100 ### hard code this because we don't really need to change it
        progress = '\r[%s%s] %.2f'
        sys.stdout.write(progress%('', ' '*status_width, 0))
        sys.stdout.flush()
        for i, (pos, lnpost, _) in enumerate(sampler.sample(p0, iterations=num_steps)):
            ### report state
            for walker in xrange(num_walkers):
                print >> file_obj, template%tuple([walker, lnpost[walker]]+list(pos[walker]))
            file_obj.flush() ### only flush output if we're Verbose

            ### print progress
            f = (i+1.)/num_steps
            n = int(status_width*f)
            sys.stdout.write(progress%('-'*n, ' '*(status_width-n), f*100))
            sys.stdout.flush()
        sys.stdout.write("\n")

    else: ### just sample silently
        for pos, lnpost, _ in sampler.sample(p0, iterations=num_steps):
            ### report state
            for walker in xrange(num_walkers):
                params = pos[walker]
                print >> file_obj, template%tuple([walker, lnpost[walker]]+list(pos[walker]))

    print """\
WARNING:
 add in a checkpoint/revcovery option?
"""

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
    if engine == "emcee":
        __emcee_regress(
              data,
              method,
              outpath,
              thr=thr,
              Rdt=Rdt,
              min_Rdt = min_Rdt,
              max_Rdt = max_Rdt,
              alpha = alpha,
              min_alpha = min_alpha,
              max_alpha = max_alpha,
              beta = beta,
              min_beta = min_beta,
              max_beta = max_beta,
              num_mc = num_mc,
              num_steps = num_steps,
              num_walkers = num_walkers,
              num_threads = num_threads,
              verbose = verbose,
              **kwargs
        )
    else:
        raise ValueError, 'engine=%s not understood'%engine
