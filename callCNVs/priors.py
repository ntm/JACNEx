############################################################################################
# Copyright (C) Nicolas Thierry-Mieg and Amandine Septier, 2021-2025
#
# This file is part of JACNEx, written by Nicolas Thierry-Mieg and Amandine Septier
# (CNRS, France)  {Nicolas.Thierry-Mieg,Amandine.Septier}@univ-grenoble-alpes.fr
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
############################################################################################


import logging
import numba
import numpy

# silence numba when we are in DEBUG mode
logging.getLogger('numba').setLevel(logging.INFO)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# calcPriors:
# calculate prior probabilities for each CN state.
# Algorithm starts by counting the states that maximize the likelihood, and
# then iteratively recalculates the priors by counting the states that maximize
# the posterior probability (ie prior * likelihood).
# This procedure ends when convergence is reached (approximately, ie using 2
# decimals in scentific notation for each proba) or after maxIter iterations.
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (NOCALL exons must have all likelihoods == -1)
#
# Returns priors (ndarray of nbStates floats): prior probabilities for each state.
def calcPriors(likelihoods):
    # max number of iterations, hard-coded
    maxIter = 50

    nbStates = likelihoods.shape[2]
    # priors start at 1 (ie we will initially count states with max likelihood)
    priors = numpy.ones(nbStates, dtype=numpy.float64)

    # to check for (approximate) convergence
    formattedPriorsPrev = ""
    # text including priors at each step, for logging in case we don't converge
    noConvergeString = ""
    # number of iterations to converge, 0 if we never converged
    converged = 0

    for i in range(maxIter):
        priors = calcPosteriors(likelihoods, priors)
        formattedPriors = " ".join(["%.5f" % x for x in priors])
        debugString = "Priors at iteration " + str(i + 1) + ":\t" + formattedPriors
        noConvergeString += "\n" + debugString
        # Check for convergence
        if formattedPriors == formattedPriorsPrev:
            converged = i + 1
            break
        else:
            formattedPriorsPrev = formattedPriors

    if converged:
        logger.debug("Priors converged after %i iterations:%s", converged, noConvergeString)
    else:
        logger.warning("Priors did not converge after %i iterations:%s", maxIter, noConvergeString)
        logger.warning("Try increasing maxIter in calcPriors(), and/or file an issue on github")
    # whether we converged or not, return last computed priors
    return priors


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################
# calcPosteriors:
# Given likelihoods and prior probabilities for each state, calculate the
# posterior probabilities for each state. These can then be considered as an
# updated vector of priors.
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
# - priors (ndarray of nbStates floats): initial prior probabilities for each state
#
# Returns posteriors, same type as priors.
@numba.njit
def calcPosteriors(likelihoods, priors):
    # init counts with a pseudo-count of one (avoid issues if no counts at all), this
    # won't matter for haploids since their CN1 likelihoods are zero
    countsPerState = numpy.ones(len(priors), dtype=numpy.uint64)

    calledExons = likelihoods[0, :, 0] >= 0
    calledExonLikelihoods = likelihoods[:, calledExons, :]
    bestStates = (priors * calledExonLikelihoods).argmax(axis=2)
    for bs in bestStates.ravel():
        countsPerState[bs] += 1

    posteriors = countsPerState.astype(numpy.float64) / countsPerState.sum()
    return(posteriors)
