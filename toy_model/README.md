# toy model for inference of population parameters

This is a toy model in which we compare different approaches for handling observational selection effects. 
These include
  * explicitly including all terms from all models (both signal and noise) for all observations (both detection and non-detection)
  * "renormalizing" the likelihood in Bayes theorem by considering only the values of "data" that are above a detection threshold and only considering events that are detected (i.e.: ignoring all times for which the data is below the detection threshold). This may introduce a dependence on the model parameters within the normalization.
Both these approaches can be thought of as discretizing time into a sequence of approximately independently identically distributed segments. 
The differences are in which segments are retained and how Bayes theorem is formulated using those segments.

In particular, we will consider the following models.
  * noise: a chi2 distribution with 2 degrees of freedom.
  * signal: a non-central chi2 distribution with 2 degrees of freedom and the non-centrality parameter drawn from a Pareto distribution.
We will attempt to infer the index of the Pareto distribution using both methods, investigating possible biases and sensitivity to detection threshold in each.

We may want to add more latent variables (i.e.: a mass parameter drawn from an unknown distribution) and either regress that out or marginalize over it. 
This situation will almost certainly arise with real data, and therefore we need to determine whether the effects are non-trivial (if they exist).

## model selection

We will extend the model to consider time-dependent rates of signals. 
Our two models will assume
    * that signals are uniformly distributed in time, or
    * some "Heaviside-like" distribution of signals in time (e.g.: signals are only found in the second half of the experiment).
In particular, we are interested to see the relative performance of our approaches in distinguishing between the two models, with specific emphasis on whether the incorporation of segments without detection actually provides informative data to the likelihood so that the Odds Ratio between the models is not dominated only by the prior volume of each model.
