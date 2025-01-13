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
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# buildBaseTransMatrix
# Build the base transition matrix, using the states that maximize
# the posterior probability (ie prior * likelihood) for each
# pair of called exons that are "close enough" (maxIED).
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (NOCALL exons must have all likelihoods == -1)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - priors (ndarray dim nbStates): prior probabilities for each state
# - maxIED (int): max inter-exon distance for a pair of consecutive called exons
#   to contribute to the baseTransMatrix.
#
# Returns transMatrix (ndarray[floats] dim nbStates*nbStates): base transition
# probas between states
def buildBaseTransMatrix(likelihoods, exons, priors, maxIED):
    nbStates = len(priors)
    # count transitions between valid, close-enough exons in all samples, and
    # init counts with a pseudo-count of one (avoid issues if no counts at all),
    #  this won't matter for haploids since their CN1 likelihoods are zero
    countsAllSamples = numpy.ones((nbStates, nbStates), dtype=numpy.uint64)

    bestStates = (priors * likelihoods).argmax(axis=2)
    prevChrom = ""
    prevEnd = 0
    prevExind = 0
    for ei in range(len(exons)):
        # ignore NOCALL (ie all likelihoods == -1) exons
        if likelihoods[0, ei, 0] < 0:
            continue
        else:
            if exons[ei][0] != prevChrom:
                # changed chrom
                prevChrom = exons[ei][0]
            elif exons[ei][1] - prevEnd <= maxIED:
                for si in range(likelihoods.shape[0]):
                    countsAllSamples[bestStates[si, prevExind], bestStates[si, ei]] += 1
            # in all cases, update prevs
            prevEnd = exons[ei][2]
            prevExind = ei

    # Normalize each row to obtain the transition matrix
    baseTransMat = countsAllSamples.astype(numpy.float64)
    for i in range(nbStates):
        rowSum = countsAllSamples[i, :].sum()
        if rowSum > 0:
            baseTransMat[i, :] /= rowSum
        # else the row is all-zeroes, nothing to do
    return(baseTransMat)


######################################
# adjustTransMatrix
# When distance between exons increases, the correlation between CN states of
# consecutive exons diminishes until at some point we simply expect the priors.
# Formally we implement this as follows: transition probabilities are smoothed
# from base probas to prior probas following a power law, until reaching the
# prior at d = dmax.
#
# Args:
# - transMatrix (ndarray dim 4*4): base transition probabilities between states
# - priors (ndarray dim 4): prior probabilities for each state
# - d [int]: distance between exons
# - dmax (float): distance where prior is reached
#
# Returns: adjusted transition probability matrix (ndarray dim 4*4)
def adjustTransMatrix(transMatrix, priors, d, dmax):
    # hardcoded power of the power law, values between 3-6 should be reasonable
    N = 5

    newTrans = numpy.zeros_like(transMatrix)

    if d >= dmax:
        # use priors
        for prevState in range(len(transMatrix)):
            newTrans[prevState] = priors
    else:
        # newTrans[p->i](x) = trans[p->i] + (prior[i] - trans[p->i]) * (x/d_max)^N
        for state in range(len(transMatrix)):
            newTrans[:, state] = (transMatrix[:, state] +
                                  (priors[state] - transMatrix[:, state]) * (d / dmax) ** N)

    return newTrans
