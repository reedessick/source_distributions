__description__ = "a module that houses methods for regressing parameters using different methods"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

__known_methods = [
    'above_threshold',                         ### a naive approach in which you assume everything above threshold is a signal 
    'renormalized_likelihood_above_threshold', ### based of a private communication Maya Fishbach sent me
                                               ###   "renormalizes" the likelihood with the constraint that data is above threshold
    'all_data_all_models',                     ### based on Messenger&Veitch (2013)
                                               ###   considers all possible models (both signal and noise) explicitly for every datum.
                                               ###   marginalizes over unobserved data below some threshold
]

__known_engines = [
    'pymultinest',
    'emcee',
]

__default_engine = 'pymultinest'
assert __default_engine in __known_engines

#-------------------------------------------------
